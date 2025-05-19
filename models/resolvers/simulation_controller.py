import json
import threading
from typing import List, Dict, Tuple, Any

import networkx as nx
import osmnx as ox
from PyQt5 import QtCore

from models.entities.disruption import Disruption
from models.resolvers.actions import DisruptionAction
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance


class SimulationController(QtCore.QObject):
    route_update_available = QtCore.pyqtSignal()
    deliveries_reassigned = QtCore.pyqtSignal(int, int, list)
    delivery_skipped = QtCore.pyqtSignal(int, int)
    wait_added = QtCore.pyqtSignal(int, float)
    disruption_activated = QtCore.pyqtSignal(object)

    def __init__(self, graph, warehouse_location, delivery_points, drivers,
                 solution, disruption_service, resolver=None, route_update_queue=None):
        super().__init__()
        self.G = graph
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.drivers = drivers
        self.current_solution = solution
        self.disruption_service = disruption_service
        self.resolver = resolver

        self.simulation_time = 8 * 3600
        self.driver_positions = {}
        self.driver_routes = {}
        self.completed_deliveries = set()
        self.skipped_deliveries = set()

        self.original_estimated_time = 0
        self.current_estimated_time = 0
        self.disruption_count = 0
        self.action_count = 0
        self.recalculation_count = 0

        self._route_cache = {}
        self.pending_actions = {}

        self.actions_taken = []
        self.route_update_queue = route_update_queue
        self._driver_routes_lock = threading.Lock()

        self.pending_deliveries = {}
        self.disruption_end_handlers = {}

    def set_resolver(self, resolver: DisruptionResolver):
        self.resolver = resolver

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def update_simulation_time(self, new_time):
        old_time = self.simulation_time
        self.simulation_time = new_time

        for driver_id, pendings in list(self.pending_deliveries.items()):
            for i, (delivery_idx, disruption_id, end_time, *rest) in reversed(
                    list(enumerate(pendings))):
                if old_time < end_time <= new_time:
                    handler = self.disruption_end_handlers.get(disruption_id)
                    if handler:
                        print(f"Found handler for disruption {disruption_id}, executing")
                        handler()
                        del self.disruption_end_handlers[disruption_id]
                    else:
                        print(f"No handler found for disruption {disruption_id}")

    def initialize_simulation(self):
        for driver in self.drivers:
            self.driver_positions[driver.id] = self.warehouse_location

        for driver in self.drivers:
            if driver.id not in self.driver_routes:
                assignment = next((a for a in self.current_solution if a.driver_id == driver.id),
                                  None)

                self.driver_routes[driver.id] = {
                    'points': [self.warehouse_location, self.warehouse_location],
                    'times': [0],
                    'delivery_indices': [],
                    'segments': [],
                    'assignment': assignment
                }

        self._calculate_initial_routes()

        all_drivers = [d.id for d in self.drivers]
        routes_drivers = list(self.driver_routes.keys())
        missing = [d for d in all_drivers if d not in routes_drivers]
        if missing:
            print(f"WARNING: Missing routes for drivers: {missing}")

        self.simulation_time = 8 * 3600
        self.completed_deliveries = set()
        self.skipped_deliveries = set()
        self.actions_taken = []
        self.disruption_count = 0
        self.action_count = 0
        self.recalculation_count = 0

    def _calculate_initial_routes(self):
        self.driver_routes = {}

        for driver in self.drivers:
            assignment = next((a for a in self.current_solution if a.driver_id == driver.id), None)

            if assignment is None:
                print(
                    f"Warning: No assignment found for driver {driver.id} - creating empty assignment")
                from models.entities.delivery import DeliveryAssignment
                assignment = DeliveryAssignment(
                    driver_id=driver.id,
                    delivery_indices=[],
                    total_weight=0.0,
                    total_volume=0.0
                )
                self.current_solution.append(assignment)

            self.driver_routes[driver.id] = {
                'points': [self.warehouse_location, self.warehouse_location],
                'times': [0],
                'delivery_indices': [],
                'segments': [],
                'assignment': assignment
            }

        for assignment in self.current_solution:
            driver_id = assignment.driver_id

            if assignment.delivery_indices:
                try:
                    route_details = self._calculate_route_details(assignment)
                    self.driver_routes[driver_id] = route_details
                except Exception as e:
                    print(f"Error calculating route for driver {driver_id}: {e}")

        self.original_estimated_time = self._calculate_total_time()
        self.current_estimated_time = self.original_estimated_time

    def _calculate_route_details(self, assignment) -> Dict[str, Any]:
        delivery_indices = []

        route_points = [self.warehouse_location]

        for idx in assignment.delivery_indices:
            try:
                if idx < 0 or idx >= len(self.delivery_points):
                    print(f"Warning: Invalid delivery point index {idx}")
                    continue

                lat, lon, _, _ = self.delivery_points[idx]
                route_points.append((lat, lon))
            except Exception as e:
                print(f"Error processing delivery index {idx}: {e}")
                continue

        route_points.append(self.warehouse_location)

        if len(route_points) < 2:
            return {
                'points': [self.warehouse_location, self.warehouse_location],
                'times': [0],
                'delivery_indices': [],
                'segments': [],
                'assignment': assignment
            }

        detailed_points = []
        detailed_times = []
        segment_info = []

        for i in range(len(route_points) - 1):
            try:
                start = route_points[i]
                end = route_points[i + 1]

                segment = self._get_path_between(start, end)
                segment_info.append(segment)

                if i > 0 and detailed_points and 'points' in segment and segment['points']:
                    segment_points = segment['points'][1:]
                else:
                    segment_points = segment.get('points', [])

                detailed_points.extend(segment_points)

                segment_times = segment.get('times', [])
                if len(segment_times) < len(segment_points) - 1:
                    segment_times.extend([60.0] * (len(segment_points) - 1 - len(segment_times)))

                detailed_times.extend(segment_times)

                if 0 < i < len(route_points) - 1:
                    delivery_indices.append(len(detailed_points) - 1)
            except Exception as e:
                print(f"Error processing route segment {i}: {e}")
                continue

        return {
            'points': detailed_points,
            'times': detailed_times,
            'delivery_indices': delivery_indices,
            'segments': segment_info,
            'assignment': assignment
        }

    def _get_path_between(self, start, end) -> Dict[str, Any]:
        cache_key = (start, end)

        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        try:
            start_node = ox.nearest_nodes(self.G, X=float(start[1]), Y=float(start[0]))
            end_node = ox.nearest_nodes(self.G, X=float(end[1]), Y=float(end[0]))

            path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')

            points = []
            times = []
            total_time = 0
            total_distance = 0

            for i in range(len(path)):
                node = path[i]
                point = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                points.append(point)

                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_data = self.G.get_edge_data(node, next_node, 0)

                    if 'travel_time' in edge_data:
                        segment_time = edge_data['travel_time']
                    elif 'length' in edge_data:
                        length = edge_data['length']
                        speed = edge_data.get('speed', 50)
                        segment_time = length / (speed * 1000 / 3600)
                    else:
                        next_point = (self.G.nodes[next_node]['y'], self.G.nodes[next_node]['x'])
                        distance = calculate_haversine_distance(point, next_point)
                        segment_time = distance / (50 * 1000 / 3600)

                    times.append(segment_time)
                    total_time += segment_time

                    if 'length' in edge_data:
                        total_distance += edge_data['length']

            result = {
                'points': points,
                'times': times,
                'total_time': total_time,
                'total_distance': total_distance
            }

            if len(self._route_cache) < 10000:
                self._route_cache[cache_key] = result

            return result

        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            direct_distance = calculate_haversine_distance(start, end)
            direct_time = direct_distance / (50 * 1000 / 3600)

            result = {
                'points': [start, end],
                'times': [direct_time],
                'total_time': direct_time,
                'total_distance': direct_distance
            }

            if len(self._route_cache) < 10000:
                self._route_cache[cache_key] = result

            return result

    def _calculate_total_time(self):
        max_time = 0

        for driver_id, route in self.driver_routes.items():
            if 'times' in route:
                route_time = sum(route['times'])
                max_time = max(max_time, route_time)

        return max_time

    def get_current_state(self):
        try:
            active_disruptions = self.disruption_service.get_active_disruptions()

            driver_assignments = {}
            for assignment in self.current_solution:
                remaining_deliveries = [
                    idx for idx in assignment.delivery_indices
                    if idx not in self.completed_deliveries and idx not in self.skipped_deliveries
                ]
                if remaining_deliveries:
                    driver_assignments[assignment.driver_id] = remaining_deliveries

            state = DeliverySystemState(
                drivers=self.drivers,
                deliveries=self.delivery_points,
                disruptions=active_disruptions,
                simulation_time=self.simulation_time,
                graph=self.G,
                warehouse_location=self.warehouse_location,
                driver_assignments=driver_assignments,
                driver_positions=self.driver_positions.copy(),
                driver_routes=self.driver_routes
            )

            return state

        except Exception as e:
            print(f"Error getting current state: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_driver_route(self, driver_id: int, new_route: List[Tuple[float, float]],
                            rerouted_segment_start: int = None, rerouted_segment_end: int = None,
                            next_delivery_index: int = None) -> bool:
        try:
            with self._driver_routes_lock:
                assignment = None
                for a in self.current_solution:
                    if a.driver_id == driver_id:
                        assignment = a
                        break
                if not assignment:
                    print(f"No assignment found for driver {driver_id}")
                    return False
                if driver_id not in self.driver_routes:
                    print(f"Driver {driver_id} not found in driver routes - initializing")
                    self.driver_routes[driver_id] = {
                        'points': [self.warehouse_location, self.warehouse_location],
                        'times': [0],
                        'delivery_indices': [],
                        'segments': [],
                        'assignment': assignment
                    }

                route_delivery_indices = []
                if assignment:
                    for idx, point in enumerate(new_route):
                        for delivery_idx in assignment.delivery_indices:
                            if delivery_idx < len(self.delivery_points):
                                delivery_lat, delivery_lon = self.delivery_points[delivery_idx][0:2]
                                distance = calculate_haversine_distance(point, (delivery_lat,
                                                                                delivery_lon))
                                if distance < 15:
                                    route_delivery_indices.append(idx)
                                    break

                route_details = {
                    'points': new_route,
                    'times': [],
                    'delivery_indices': route_delivery_indices,
                    'segments': [],
                    'assignment': assignment,
                    'rerouted_segment': (rerouted_segment_start, rerouted_segment_end)
                    if rerouted_segment_start is not None else None,
                    'next_delivery_index': next_delivery_index
                }

                for i in range(len(new_route) - 1):
                    start = new_route[i]
                    end = new_route[i + 1]
                    try:
                        path_details = self._get_path_between(start, end)
                        time_seconds = path_details.get('total_time', 60.0)
                        route_details['times'].append(max(1.0, time_seconds))
                    except Exception as e:
                        print(f"Error calculating travel time between {start} and {end}: {e}")
                        try:
                            distance = calculate_haversine_distance(start, end)
                            time_seconds = distance / (30 * 1000 / 3600)
                            route_details['times'].append(max(1.0, time_seconds))
                        except Exception as dist_e:
                            print(f"Fallback distance calculation failed: {dist_e}")
                            route_details['times'].append(60.0)

                self.driver_routes[driver_id] = route_details
                self.current_estimated_time = self._calculate_total_time()

                route_data = {
                    'type': 'full_route_update',
                    'points': ";".join([f"{lat:.6f},{lon:.6f}" for lat, lon in new_route]),
                    'times': route_details['times'],
                    'rerouted_segment_start': rerouted_segment_start,
                    'rerouted_segment_end': rerouted_segment_end,
                    'next_delivery_index': next_delivery_index,
                    'completed_deliveries': list(self.completed_deliveries),
                    'delivery_indices': route_delivery_indices
                }
                route_string = json.dumps(route_data)

                if self.route_update_queue is not None:
                    update_data = (driver_id, route_string)
                    self.route_update_queue.put(update_data)
                    self.route_update_available.emit()
                else:
                    print("Route update queue not available")
                    return False
                return True

        except Exception as e:
            print(f"Error in update_driver_route: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_driver_position(self, driver_id, position):
        self.driver_positions[driver_id] = position

        if self.disruption_service:
            newly_activated = self.disruption_service.check_drivers_near_disruptions(
                self.driver_positions, self.driver_routes)

            for disruption in newly_activated:
                self.disruption_activated.emit({'disruption_id': disruption.id})

    def process_disruption_activation(self, disruption):
        try:
            print(f"Processing disruption activation: {disruption}")
            if not hasattr(disruption, 'metadata'):
                disruption.metadata = {}
            if 'driver_id' in disruption and disruption.driver_id is not None:
                disruption.metadata['triggered_by_driver'] = disruption.driver_id
                print(f"Disruption {disruption.id} was triggered by driver {disruption.driver_id}")
            self.disruption_activated.emit(disruption)
        except Exception as e:
            print(f"Error processing disruption activation: {e}")
            import traceback
            traceback.print_exc()

    def handle_disruption(self, disruption: Disruption) -> List[DisruptionAction]:
        try:
            if not self.resolver:
                print("Warning: No resolver available to handle disruption")
                return []

            if not isinstance(disruption, Disruption):
                print(f"Invalid disruption object: {type(disruption)}")
                return []

            try:
                state = self.get_current_state()
            except Exception as e:
                print(f"Error getting current state: {e}")
                return []

            self.disruption_count += 1

            try:
                should_recalc = self.resolver.should_recalculate(state, disruption)
            except Exception as e:
                print(f"Error in recalculation check: {e}")
                should_recalc = False

            if not should_recalc:
                return []

            try:
                actions = self.resolver.on_disruption_detected(disruption, state)
                if not actions:
                    return []

                successful_actions = []
                for action in actions:
                    try:
                        success = action.execute(self)
                        if success:
                            successful_actions.append(action)
                            self.action_count += 1

                            if hasattr(action,
                                       'action_type') and action.action_type.name == 'REROUTE_BASIC':
                                print(
                                    f"handle_disruption: RerouteBasicAction executed successfully for driver {getattr(action, 'driver_id', 'N/A')}.")
                    except Exception as e:
                        print(f"Error executing action: {e}")
                        continue

                return successful_actions

            except Exception as e:
                print(f"Error in handle_disruption: {e}")
                return []

        except Exception as e:
            print(f"Fatal error in handle_disruption: {e}")
            return []

    def get_pending_actions_for_driver(self, driver_id):
        actions = self.pending_actions.get(driver_id, [])

        if actions:
            print(f"Pending actions for driver {driver_id}: {len(actions)}")
            for i, action in enumerate(actions):
                print(f"  Action {i}: {action.action_type.name}")
                if hasattr(action, 'rerouted_segment_start') and hasattr(action,
                                                                         'original_segment_end'):
                    print(
                        f"    Original Segment: {action.rerouted_segment_start}-{action.original_segment_end}")
                if hasattr(action, 'detour_points'):
                    print(f"    Detour Points: {len(action.detour_points)}")
                if hasattr(action, 'delivery_indices'):
                    print(f"    Delivery indices: {action.delivery_indices}")

        result = []
        for action in actions:
            action_dict = action.to_dict()
            result.append(action_dict)

        if driver_id in self.pending_actions:
            self.pending_actions[driver_id] = []

        return result

    def execute_actions(self, actions: List[DisruptionAction]) -> int:
        successful = 0
        for action in actions:
            if action.execute(self):
                successful += 1
                self.actions_taken.append(action)
                self.action_count += 1

        return successful
