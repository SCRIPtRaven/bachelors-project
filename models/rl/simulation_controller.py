import math
from typing import List, Dict, Tuple, Any

import networkx as nx
import osmnx as ox
from PyQt5 import QtCore

from models.entities.disruption import Disruption
from models.rl.actions import DisruptionAction
from models.rl.resolver import DisruptionResolver
from models.rl.state import DeliverySystemState


class SimulationController(QtCore.QObject):
    """
    Controller that interfaces between the disruption resolver and the simulation.
    Handles state management, action execution, and communication with the UI.
    """

    route_updated = QtCore.pyqtSignal(int, list)  # driver_id, new_route
    route_recalculated = QtCore.pyqtSignal(object)  # complete solution
    deliveries_reassigned = QtCore.pyqtSignal(int, int, list)  # from_driver, to_driver, deliveries
    delivery_skipped = QtCore.pyqtSignal(int, int)  # driver_id, delivery_index
    wait_added = QtCore.pyqtSignal(int, float)  # driver_id, wait_time
    action_log = QtCore.pyqtSignal(str)  # Human-readable action description
    disruption_activated = QtCore.pyqtSignal(object)

    def __init__(self, graph, warehouse_location, delivery_points, drivers,
                 solution, disruption_service, resolver=None):
        super().__init__()
        self.G = graph
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.drivers = drivers
        self.current_solution = solution
        self.disruption_service = disruption_service
        self.resolver = resolver

        # Working state
        self.simulation_time = 8 * 3600  # Start at 8:00 AM in seconds
        self.driver_positions = {}  # Tracks each driver's current position
        self.driver_routes = {}  # Detailed routes for each driver
        self.completed_deliveries = set()  # Set of completed delivery indices
        self.skipped_deliveries = set()  # Set of skipped delivery indices

        # Performance tracking
        self.original_estimated_time = 0
        self.current_estimated_time = 0
        self.disruption_count = 0
        self.action_count = 0
        self.recalculation_count = 0

        # Caches
        self._route_cache = {}
        self.pending_actions = {}

        # Action log
        self.actions_taken = []

    def set_resolver(self, resolver: DisruptionResolver):
        """Set the disruption resolver to use"""
        self.resolver = resolver

    def initialize_simulation(self):
        """Initialize the simulation state before starting"""
        # Set initial driver positions to warehouse
        for driver in self.drivers:
            self.driver_positions[driver.id] = self.warehouse_location

        # Calculate initial routes
        self._calculate_initial_routes()

        # Reset tracking
        self.simulation_time = 8 * 3600
        self.completed_deliveries = set()
        self.skipped_deliveries = set()
        self.actions_taken = []
        self.disruption_count = 0
        self.action_count = 0
        self.recalculation_count = 0

    def _calculate_initial_routes(self):
        """Calculate the initial detailed routes for all drivers"""
        self.driver_routes = {}

        for assignment in self.current_solution:
            driver_id = assignment.driver_id
            if not assignment.delivery_indices:
                continue

            route_details = self._calculate_route_details(assignment)
            self.driver_routes[driver_id] = route_details

        # Calculate the original estimated completion time
        self.original_estimated_time = self._calculate_total_time()
        self.current_estimated_time = self.original_estimated_time

    def _calculate_route_details(self, assignment) -> Dict[str, Any]:
        """Calculate detailed route information for an assignment"""
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

        # Add warehouse at the end of the route
        route_points.append(self.warehouse_location)

        # If we don't have at least start and end points, return a minimal valid route
        if len(route_points) < 2:
            return {
                'points': [self.warehouse_location, self.warehouse_location],
                'times': [0],
                'delivery_indices': [],
                'segments': [],
                'assignment': assignment
            }

        # Calculate detailed route with node-by-node paths
        detailed_points = []
        detailed_times = []
        segment_info = []

        for i in range(len(route_points) - 1):
            try:
                start = route_points[i]
                end = route_points[i + 1]

                # Get or calculate path between points
                segment = self._get_path_between(start, end)
                segment_info.append(segment)

                # Add to detailed points (avoid duplicating connecting points)
                if i > 0 and detailed_points and 'points' in segment and segment['points']:
                    # Skip the first point as it's the same as the last point of previous segment
                    segment_points = segment['points'][1:]
                else:
                    segment_points = segment.get('points', [])

                detailed_points.extend(segment_points)

                # Ensure times are properly handled
                segment_times = segment.get('times', [])
                if len(segment_times) < len(segment_points) - 1:
                    # Fill in missing times with reasonable defaults
                    segment_times.extend([60.0] * (len(segment_points) - 1 - len(segment_times)))

                detailed_times.extend(segment_times)

                # Mark delivery points
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
        """Get or calculate the path between two points"""
        cache_key = (start, end)

        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        try:
            start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
            end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])

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
                        # Estimate using haversine distance
                        next_point = (self.G.nodes[next_node]['y'], self.G.nodes[next_node]['x'])
                        distance = self._haversine_distance(point, next_point)
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

            # Cache the result
            if len(self._route_cache) < 10000:
                self._route_cache[cache_key] = result

            return result

        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            # Fall back to direct path if no route found
            direct_distance = self._haversine_distance(start, end)
            direct_time = direct_distance / (50 * 1000 / 3600)  # 50 km/h

            result = {
                'points': [start, end],
                'times': [direct_time],
                'total_time': direct_time,
                'total_distance': direct_distance
            }

            # Cache the result
            if len(self._route_cache) < 10000:
                self._route_cache[cache_key] = result

            return result

    def _haversine_distance(self, point1, point2):
        """Calculate the Haversine distance between two points in meters"""
        lat1, lon1 = point1
        lat2, lon2 = point2

        # Convert to radians
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters

        return c * r

    def _calculate_total_time(self):
        """Calculate the estimated total time for all routes"""
        max_time = 0

        for driver_id, route in self.driver_routes.items():
            if 'times' in route:
                route_time = sum(route['times'])
                max_time = max(max_time, route_time)

        return max_time

    def get_current_state(self):
        """Get the current state of the delivery system"""
        try:
            # Get active disruptions
            active_disruptions = self.disruption_service.get_active_disruptions(self.simulation_time)

            # Get current driver assignments from solution
            driver_assignments = {}
            for assignment in self.current_solution:
                # Only include remaining deliveries (not completed/skipped)
                remaining_deliveries = [
                    idx for idx in assignment.delivery_indices
                    if idx not in self.completed_deliveries and idx not in self.skipped_deliveries
                ]
                if remaining_deliveries:  # Only include drivers with remaining work
                    driver_assignments[assignment.driver_id] = remaining_deliveries

            # Create state object with current information
            state = DeliverySystemState(
                drivers=self.drivers,
                deliveries=self.delivery_points,
                disruptions=active_disruptions,
                simulation_time=self.simulation_time,
                graph=self.G,
                warehouse_location=self.warehouse_location,
                driver_assignments=driver_assignments,  # Pass actual assignments
                driver_positions=self.driver_positions.copy()  # Pass actual positions
            )

            return state

        except Exception as e:
            print(f"Error getting current state: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_driver_route(self, driver_id: int, new_route: List[Tuple[float, float]]) -> bool:
        """
        Update a driver's route with new points

        Args:
            driver_id: ID of the driver to update
            new_route: List of (lat, lon) coordinates for the new route

        Returns:
            True if successful, False otherwise
        """
        if driver_id not in self.driver_routes:
            return False

        # Find the assignment for this driver
        assignment = None
        for a in self.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment:
            return False

        # Recalculate the route details
        route_details = self._calculate_route_details(assignment)
        self.driver_routes[driver_id] = route_details

        # Update the estimated completion time
        self.current_estimated_time = self._calculate_total_time()

        # Emit signal for UI update
        self.route_updated.emit(driver_id, new_route)

        # Log the action
        self.action_log.emit(f"Updated route for Driver {driver_id}")

        return True

    def reassign_deliveries(self, from_driver_id: int, to_driver_id: int, delivery_indices: List[int]) -> bool:
        """
        Transfer deliveries from one driver to another

        Args:
            from_driver_id: ID of the driver giving up deliveries
            to_driver_id: ID of the driver receiving deliveries
            delivery_indices: List of delivery indices to transfer

        Returns:
            True if successful, False otherwise
        """
        # Find the assignments
        from_assignment = None
        to_assignment = None

        for a in self.current_solution:
            if a.driver_id == from_driver_id:
                from_assignment = a
            if a.driver_id == to_driver_id:
                to_assignment = a

        if not from_assignment or not to_assignment:
            return False

        # Check if the deliveries exist in from_assignment
        for idx in delivery_indices:
            if idx not in from_assignment.delivery_indices:
                return False

        # Check capacity constraints
        to_driver = next((d for d in self.drivers if d.id == to_driver_id), None)
        if not to_driver:
            return False

        additional_weight = 0
        additional_volume = 0

        for idx in delivery_indices:
            _, _, weight, volume = self.delivery_points[idx]
            additional_weight += weight
            additional_volume += volume

        new_weight = to_assignment.total_weight + additional_weight
        new_volume = to_assignment.total_volume + additional_volume

        if new_weight > to_driver.weight_capacity or new_volume > to_driver.volume_capacity:
            return False

        # Transfer the deliveries
        for idx in delivery_indices:
            from_assignment.delivery_indices.remove(idx)
            to_assignment.delivery_indices.append(idx)

        # Update weights and volumes
        from_assignment.total_weight -= additional_weight
        from_assignment.total_volume -= additional_volume
        to_assignment.total_weight = new_weight
        to_assignment.total_volume = new_volume

        # Recalculate routes
        if from_assignment.delivery_indices:
            self.driver_routes[from_driver_id] = self._calculate_route_details(from_assignment)

        self.driver_routes[to_driver_id] = self._calculate_route_details(to_assignment)

        # Update the estimated completion time
        self.current_estimated_time = self._calculate_total_time()

        # Emit signal for UI update
        self.deliveries_reassigned.emit(from_driver_id, to_driver_id, delivery_indices)

        # Log the action
        delivery_str = ", ".join([str(i) for i in delivery_indices])
        self.action_log.emit(
            f"Reassigned deliveries {delivery_str} from Driver {from_driver_id} to Driver {to_driver_id}")

        return True

    def skip_delivery(self, driver_id: int, delivery_index: int) -> bool:
        """
        Mark a delivery as skipped and update the route

        Args:
            driver_id: ID of the driver
            delivery_index: Index of the delivery to skip

        Returns:
            True if successful, False otherwise
        """
        # Find the assignment
        assignment = None

        for a in self.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or delivery_index not in assignment.delivery_indices:
            return False

        # Mark as skipped
        self.skipped_deliveries.add(delivery_index)

        # Remove from assignment
        assignment.delivery_indices.remove(delivery_index)

        # Update weight and volume
        _, _, weight, volume = self.delivery_points[delivery_index]
        assignment.total_weight -= weight
        assignment.total_volume -= volume

        # Recalculate route
        if assignment.delivery_indices:
            self.driver_routes[driver_id] = self._calculate_route_details(assignment)
        else:
            # Empty route
            self.driver_routes[driver_id] = {
                'points': [self.warehouse_location, self.warehouse_location],
                'times': [0],
                'delivery_indices': [],
                'segments': [],
                'assignment': assignment
            }

        # Update the estimated completion time
        self.current_estimated_time = self._calculate_total_time()

        # Emit signal for UI update
        self.delivery_skipped.emit(driver_id, delivery_index)

        # Log the action
        self.action_log.emit(f"Skipped delivery {delivery_index} for Driver {driver_id}")

        return True

    def add_driver_wait_time(self, driver_id: int, wait_time: float) -> bool:
        """
        Add wait time to a driver's current activity

        Args:
            driver_id: ID of the driver
            wait_time: Time to wait in seconds

        Returns:
            True if successful, False otherwise
        """
        if driver_id not in self.driver_routes:
            return False

        # Add a special wait marker to the route
        route = self.driver_routes[driver_id]

        if 'wait_times' not in route:
            route['wait_times'] = {}

        # Use the current time as the key
        route['wait_times'][self.simulation_time] = wait_time

        # Update the estimated completion time
        self.current_estimated_time += wait_time

        # Emit signal for UI update
        self.wait_added.emit(driver_id, wait_time)

        # Log the action
        minutes = wait_time / 60
        self.action_log.emit(f"Added {minutes:.1f} minute wait time for Driver {driver_id}")

        return True

    def update_driver_position(self, driver_id, position):
        """
        Update a driver's position and check for nearby disruptions
        """
        # Update the driver's position
        self.driver_positions[driver_id] = position

        # Check if any disruptions should be activated
        if self.disruption_service:
            newly_activated = self.disruption_service.check_drivers_near_disruptions(
                self.driver_positions, self.driver_routes)

            # Process newly activated disruptions
            for disruption in newly_activated:
                # Log the disruption activation
                self.action_log.emit(f"Disruption activated: {disruption.type.value} (ID: {disruption.id})")

                # Emit signal for disruption activation
                self.disruption_activated.emit({'disruption_id': disruption.id})

                # Handle the disruption if we have a resolver
                if self.resolver:
                    actions = self.handle_disruption(disruption)
                    if actions:
                        self.action_count += len(actions)
                        self.action_log.emit(f"Took {len(actions)} actions in response to disruption")

    def process_disruption_activation(self, disruption):
        """Process a newly activated disruption"""
        # Log the disruption activation
        self.action_log.emit(f"Disruption activated: {disruption.type.value} (ID: {disruption.id})")

        # Emit disruption activation signal
        self.disruption_activated.emit(disruption)

        # Handle the disruption if we have a resolver
        if self.resolver:
            actions = self.handle_disruption(disruption)
            if actions:
                self.action_count += len(actions)

    def reorder_deliveries(self, driver_id: int, new_order: List[int]) -> bool:
        try:
            # Find the assignment
            assignment = None
            for a in self.current_solution:
                if a.driver_id == driver_id:
                    assignment = a
                    break

            if not assignment:
                return False

            # Validate new order safely
            if not isinstance(new_order, list) or not new_order:
                return False

            # Check if the new order contains valid indices
            if not all(isinstance(idx, int) for idx in new_order):
                return False

            # Check if the indices are valid
            for idx in new_order:
                if idx not in assignment.delivery_indices:
                    return False

            # Update the order
            assignment.delivery_indices = new_order.copy()  # Make a copy to be safe

            # Recalculate route - protect from exceptions
            try:
                self.driver_routes[driver_id] = self._calculate_route_details(assignment)
            except Exception as e:
                print(f"Error recalculating route: {e}")
                # Create a minimal valid route to avoid crashes
                self.driver_routes[driver_id] = {
                    'points': [self.warehouse_location, self.warehouse_location],
                    'times': [0],
                    'delivery_indices': [],
                    'segments': [],
                    'assignment': assignment
                }

            # Update the estimated completion time - protect from exceptions
            try:
                self.current_estimated_time = self._calculate_total_time()
            except Exception as e:
                print(f"Error calculating time: {e}")
                # Continue with partial update

            # Emit signal for UI update - use a try/except block with more validation
            try:
                if driver_id in self.driver_routes and 'points' in self.driver_routes[driver_id]:
                    points = self.driver_routes[driver_id]['points']
                    if points and isinstance(points, list):
                        self.route_updated.emit(driver_id, points)
                    else:
                        print(f"Warning: Invalid points for driver {driver_id}")
                else:
                    print(f"Warning: Missing route data for driver {driver_id}")
            except Exception as e:
                print(f"Error emitting route update: {e}")
                # Continue with partial update

            # Log the action
            self.action_log.emit(f"Reordered deliveries for Driver {driver_id}")

            return True

        except Exception as e:
            print(f"Error in reorder_deliveries: {e}")
            return False

    def handle_disruption(self, disruption: Disruption) -> List[DisruptionAction]:
        try:
            if not self.resolver:
                print("Warning: No resolver available to handle disruption")
                return []

            if not isinstance(disruption, Disruption):
                print(f"Invalid disruption object: {type(disruption)}")
                return []

            # Get current state with validation
            try:
                state = self.get_current_state()
            except Exception as e:
                print(f"Error getting current state: {e}")
                return []

            self.disruption_count += 1
            self.action_log.emit(f"Detected {disruption.type.value} disruption (ID: {disruption.id})")

            # Use resolver with error handling
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
                        if action.execute(self):
                            successful_actions.append(action)
                            self.action_count += 1
                    except Exception as e:
                        print(f"Error executing action: {e}")
                        continue

                return successful_actions

            except Exception as e:
                print(f"Error in disruption resolution: {e}")
                return []

        except Exception as e:
            print(f"Fatal error in handle_disruption: {e}")
            return []

    def get_pending_actions_for_driver(self, driver_id):
        """Get and clear pending actions for a driver"""
        actions = self.pending_actions.get(driver_id, [])
        if driver_id in self.pending_actions:
            self.pending_actions[driver_id] = []  # Clear after retrieving

        # Convert actions to dictionaries for JSON serialization
        return [action.to_dict() for action in actions]

    def execute_actions(self, actions: List[DisruptionAction]) -> int:
        """
        Execute a list of actions and return the number of successful actions

        Args:
            actions: List of actions to execute

        Returns:
            Number of successfully executed actions
        """
        successful = 0
        for action in actions:
            if action.execute(self):
                successful += 1
                self.actions_taken.append(action)
                self.action_count += 1

        return successful
