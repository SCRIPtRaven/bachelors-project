import math
from typing import List, Set, Optional, Tuple

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.rl.actions import (
    DisruptionAction, RerouteAction, ReassignDeliveriesAction
)
from models.rl.resolver import DisruptionResolver
from models.rl.state import DeliverySystemState


class RuleBasedResolver(DisruptionResolver):
    """
    Rule-based implementation of DisruptionResolver that uses predefined
    rules to handle different types of disruptions.
    """

    def __init__(self, graph, warehouse_location, max_computation_time=1.0):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.max_computation_time = max_computation_time

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost
        """
        if disruption.type == DisruptionType.ROAD_CLOSURE:
            return True

        elif disruption.type == DisruptionType.TRAFFIC_JAM:
            if disruption.severity >= 0.3:
                return True

            affected_drivers = self._get_affected_drivers(disruption, state)
            return len(affected_drivers) > 0

        elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            affected_drivers = self._get_affected_drivers(disruption, state)
            return len(affected_drivers) > 0

        return disruption.severity >= 0.4

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """Determine actions to take in response to disruptions"""
        actions = []

        for disruption in active_disruptions:
            try:
                affected_drivers = self._get_affected_drivers(disruption, state)

                print(f"Processing {disruption.type.value} affecting {len(affected_drivers)} drivers")

                if disruption.type == DisruptionType.ROAD_CLOSURE or disruption.type == DisruptionType.TRAFFIC_JAM:
                    for driver_id in affected_drivers:
                        reroute_action = self._create_reroute_action(driver_id, disruption, state)
                        if reroute_action:
                            actions.append(reroute_action)

                elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                    for driver_id in affected_drivers:
                        delivery_idx = self._find_affected_delivery(driver_id, disruption, state)
                        if delivery_idx is not None:
                            from models.rl.actions import RecipientUnavailableAction
                            action = RecipientUnavailableAction(
                                driver_id=driver_id,
                                delivery_index=delivery_idx,
                                disruption_id=disruption.id,
                                duration=disruption.duration
                            )
                            actions.append(action)

            except Exception as e:
                print(f"Error handling disruption {disruption.id}: {e}")
                continue

        return actions

    def _create_simple_reassignment(self, broken_driver_id, state):
        """Create a simplified reassignment that always works"""
        actions = []

        broken_assignments = state.driver_assignments.get(broken_driver_id, [])
        if not broken_assignments:
            return actions

        for driver_id in state.driver_assignments:
            if driver_id != broken_driver_id:
                action = ReassignDeliveriesAction(
                    from_driver_id=broken_driver_id,
                    to_driver_id=driver_id,
                    delivery_indices=broken_assignments
                )
                actions.append(action)
                break

        return actions

    def _create_delivery_reordering(self, driver_id, disruption, state):
        """Create a delivery reordering action to handle recipient unavailability"""
        try:
            delivery_idx = disruption.metadata.get("delivery_point_index")
            if delivery_idx is None:
                return None

            assignments = state.driver_assignments.get(driver_id, [])
            if delivery_idx not in assignments or len(assignments) <= 1:
                return None

            new_order = [idx for idx in assignments if idx != delivery_idx]
            new_order.append(delivery_idx)

            from models.rl.actions import PrioritizeDeliveryAction
            return PrioritizeDeliveryAction(
                driver_id=driver_id,
                delivery_indices=new_order
            )
        except Exception as e:
            print(f"Error creating delivery reordering: {e}")
            return None

    def _get_affected_drivers(self, disruption: Disruption, state: DeliverySystemState) -> Set[int]:
        """Determine which drivers are affected by a disruption"""
        try:
            affected_drivers = set()

            if disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                if "delivery_point_index" in disruption.metadata:
                    delivery_idx = disruption.metadata["delivery_point_index"]
                    for driver_id, assignments in state.driver_assignments.items():
                        if delivery_idx in assignments:
                            affected_drivers.add(driver_id)
                            break

            effective_radius = disruption.affected_area_radius * 2

            for driver_id, assignments in state.driver_assignments.items():
                if not assignments:
                    continue

                position = state.driver_positions.get(driver_id)
                if position:
                    try:
                        distance = self._calculate_distance(position, disruption.location)
                        if distance <= effective_radius:
                            affected_drivers.add(driver_id)
                            continue
                    except Exception:
                        pass

                for delivery_idx in assignments:
                    try:
                        if delivery_idx < len(state.deliveries):
                            delivery = state.deliveries[delivery_idx]
                            lat, lon = delivery[0], delivery[1]

                            distance = self._calculate_distance((lat, lon), disruption.location)
                            if distance <= effective_radius:
                                affected_drivers.add(driver_id)
                                break
                    except Exception:
                        continue

            if not affected_drivers and state.driver_assignments:
                for driver_id, assignments in state.driver_assignments.items():
                    if assignments:
                        affected_drivers.add(driver_id)
                        break

            return affected_drivers

        except Exception as e:
            print(f"Error in _get_affected_drivers: {e}")
            if state.driver_assignments:
                for driver_id, assignments in state.driver_assignments.items():
                    if assignments:
                        return {driver_id}
            return set()

    def _find_affected_delivery(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        int]:
        """
        Find which delivery for a driver is affected by a recipient unavailable disruption

        Returns the delivery index if found, None otherwise
        """
        if disruption.type != DisruptionType.RECIPIENT_UNAVAILABLE:
            return None

        assignments = state.driver_assignments.get(driver_id, [])

        for delivery_idx in assignments:
            if delivery_idx < len(state.deliveries):
                delivery = state.deliveries[delivery_idx]
                lat, lon = delivery[0], delivery[1]

                distance_to_disruption = self._calculate_distance(
                    (lat, lon),
                    disruption.location
                )

                if distance_to_disruption <= disruption.affected_area_radius:
                    return delivery_idx

        return None

    def _is_major_route_disruption(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if a disruption affects a major route
        """
        try:
            affected_drivers = self._get_affected_drivers(disruption, state)
            if len(affected_drivers) >= 1:
                return True

            try:
                nearest_node = ox.nearest_nodes(
                    self.G,
                    X=disruption.location[1],
                    Y=disruption.location[0]
                )

                degree = len(list(self.G.neighbors(nearest_node)))

                return degree >= 2

            except Exception as e:
                print(f"Error analyzing road importance: {e}")
                return disruption.severity > 0.3
        except Exception as e:
            print(f"Error in _is_major_route_disruption: {e}")
            return False

    def _create_reroute_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteAction]:
        try:
            position = state.driver_positions.get(driver_id)
            if not position:
                return None

            original_detailed_route = []
            if driver_id in state.driver_routes and 'points' in state.driver_routes[driver_id]:
                original_detailed_route = state.driver_routes[driver_id]['points']

            if not original_detailed_route or len(original_detailed_route) < 2:
                print(f"No detailed route found for driver {driver_id}")
                return None

            current_segment_start = 0
            min_distance = float('inf')

            for i, point in enumerate(original_detailed_route):
                try:
                    distance = self._calculate_distance(position, point)
                    if distance < min_distance:
                        min_distance = distance
                        current_segment_start = i
                except Exception as e:
                    print(f"Error calculating distance: {e}")

            print(f"Driver {driver_id} is near point {current_segment_start} of {len(original_detailed_route)}")

            look_ahead = 10

            affected_segment_start = -1
            affected_segment_end = -1

            for i in range(current_segment_start,
                           min(current_segment_start + look_ahead, len(original_detailed_route) - 1)):
                current_point = original_detailed_route[i]
                next_point = original_detailed_route[i + 1]

                if self._segment_near_disruption(current_point, next_point, disruption):
                    affected_segment_start = i
                    for j in range(i + 1, min(i + look_ahead, len(original_detailed_route) - 1)):
                        if not self._segment_near_disruption(original_detailed_route[j], original_detailed_route[j + 1],
                                                             disruption):
                            affected_segment_end = j + 1
                            break

                    if affected_segment_end == -1:
                        affected_segment_end = min(i + look_ahead, len(original_detailed_route) - 1)

                    break

            if affected_segment_start == -1:
                print(f"No affected segment found for driver {driver_id} near disruption {disruption.id}")
                return None

            print(
                f"Found affected segment {affected_segment_start}-{affected_segment_end} out of {len(original_detailed_route)}")

            start_point = original_detailed_route[affected_segment_start]
            end_point = original_detailed_route[affected_segment_end]

            detour_points = self._find_path_avoiding_disruption(state.graph, start_point, end_point, disruption)

            if not detour_points or len(detour_points) < 2:
                print(f"Failed to find detour for driver {driver_id}")
                return None

            new_route = []

            new_route.extend(original_detailed_route[:affected_segment_start])

            if new_route and detour_points and new_route[-1] == detour_points[0]:
                new_route.extend(detour_points[1:])
            else:
                new_route.extend(detour_points)

            if new_route and affected_segment_end + 1 < len(original_detailed_route):
                if new_route[-1] == original_detailed_route[affected_segment_end + 1]:
                    new_route.extend(original_detailed_route[affected_segment_end + 2:])
                else:
                    new_route.extend(original_detailed_route[affected_segment_end + 1:])

            if len(new_route) < 2:
                print(f"Resulting route has too few points for driver {driver_id}")
                return None

            print(f"Created new route with {len(new_route)} points, original had {len(original_detailed_route)}")

            active_assignments = state.driver_assignments.get(driver_id, [])

            next_delivery_idx = None
            if active_assignments:
                next_delivery_idx = active_assignments[0]

            delivery_indices = []
            for idx, point in enumerate(new_route):
                for delivery_idx in active_assignments:
                    if delivery_idx < len(state.deliveries):
                        delivery_lat, delivery_lon = state.deliveries[delivery_idx][0:2]
                        delivery_point = (delivery_lat, delivery_lon)

                        if self._calculate_distance(point, delivery_point) < 15:  # Within 15m
                            delivery_indices.append(idx)
                            break

            print(f"Found {len(delivery_indices)} delivery points in the new route")

            return RerouteAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_start + len(detour_points) - 1,
                next_delivery_index=next_delivery_idx,
                delivery_indices=delivery_indices
            )

        except Exception as e:
            print(f"Error creating reroute action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _segment_near_disruption(self, start, end, disruption):
        """
        Check if a route segment passes near a disruption

        Args:
            start: (lat, lon) tuple of segment start
            end: (lat, lon) tuple of segment end
            disruption: Disruption object

        Returns:
            bool: True if segment passes within disruption radius
        """
        try:
            if (self._calculate_distance(start, disruption.location) <= disruption.affected_area_radius or
                    self._calculate_distance(end, disruption.location) <= disruption.affected_area_radius):
                return True

            closest_distance = self._point_to_segment_distance(disruption.location, start, end)
            return closest_distance <= disruption.affected_area_radius
        except Exception as e:
            print(f"Error in _segment_near_disruption: {e}")
            return True

    def _point_to_segment_distance(self, point, line_start, line_end):
        """
        Calculate the shortest distance from a point to a line segment

        Args:
            point: (lat, lon) tuple of the point
            line_start: (lat, lon) tuple of segment start
            line_end: (lat, lon) tuple of segment end

        Returns:
            float: Distance in meters
        """
        try:
            p_lat, p_lon = point
            s_lat, s_lon = line_start
            e_lat, e_lon = line_end

            if s_lat == e_lat and s_lon == e_lon:
                return self._calculate_distance(point, line_start)

            line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

            if line_length_sq < 1e-10:
                return self._calculate_distance(point, line_start)

            t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                               (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

            closest_lat = s_lat + t * (e_lat - s_lat)
            closest_lon = s_lon + t * (e_lon - s_lon)

            return self._calculate_distance(point, (closest_lat, closest_lon))
        except Exception as e:
            print(f"Error in distance calculation: {e}")
            try:
                return self._calculate_distance(point, line_start)
            except:
                return float('inf')

    def _find_path_avoiding_disruption(self, graph, start_point, end_point, disruption):
        """
        Find a path between points that avoids a disruption area
        """
        try:
            if not graph:
                print("No graph available for path finding")
                return [start_point, end_point]

            try:
                start_node = ox.nearest_nodes(graph, X=start_point[1], Y=start_point[0])
                end_node = ox.nearest_nodes(graph, X=end_point[1], Y=end_point[0])
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return [start_point, end_point]

            G_mod = graph.copy()

            disruption_nodes = []
            affected_edges = 0
            disruption_radius = disruption.affected_area_radius

            search_radius = disruption_radius * 1.2

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = self._calculate_distance(node_point, disruption.location)
                if distance <= search_radius:
                    disruption_nodes.append(node)

            weight_multiplier = 10.0
            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (9.0 * disruption.severity)
            elif disruption.type.value == 'road_closure':
                weight_multiplier = 100.0
            elif disruption.type.value == 'recipient_unavailable':
                weight_multiplier = 5.0

            for node in disruption_nodes:
                for neighbor in list(G_mod.neighbors(node)):
                    if G_mod.has_edge(node, neighbor):
                        for edge_key in list(G_mod[node][neighbor].keys()):
                            if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                G_mod[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier
                                affected_edges += 1

            print(f"Modified {affected_edges} edges for disruption avoidance")

            try:
                path = nx.shortest_path(G_mod, start_node, end_node, weight='travel_time')

                route_points = []
                for node in path:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = G_mod.nodes[node]['y']
                        lon = G_mod.nodes[node]['x']
                        route_points.append((lat, lon))

                if len(route_points) >= 2:
                    route_points[0] = start_point
                    route_points[-1] = end_point
                    return route_points
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            try:
                center_lat, center_lon = disruption.location
                start_lat, start_lon = start_point
                end_lat, end_lon = end_point

                mid_lat = (start_lat + end_lat) / 2
                mid_lon = (start_lon + end_lon) / 2

                dlat = mid_lat - center_lat
                dlon = mid_lon - center_lon

                mag = math.sqrt(dlat ** 2 + dlon ** 2)
                if mag > 0:
                    norm_lat = dlat / mag
                    norm_lon = dlon / mag
                    detour_distance = disruption_radius * 2

                    waypoint_lat = center_lat + norm_lat * detour_distance
                    waypoint_lon = center_lon + norm_lon * detour_distance

                    try:
                        waypoint_node = ox.nearest_nodes(graph, X=waypoint_lon, Y=waypoint_lat)

                        path1 = nx.shortest_path(graph, start_node, waypoint_node, weight='travel_time')
                        path2 = nx.shortest_path(graph, waypoint_node, end_node, weight='travel_time')

                        combined_path = path1[:-1] + path2

                        route_points = []
                        for node in combined_path:
                            if 'y' in graph.nodes[node] and 'x' in graph.nodes[node]:
                                lat = graph.nodes[node]['y']
                                lon = graph.nodes[node]['x']
                                route_points.append((lat, lon))

                        if len(route_points) >= 2:
                            route_points[0] = start_point
                            route_points[-1] = end_point
                            return route_points
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            except Exception as e:
                print(f"Error in detour calculation: {e}")

            print("All routing attempts failed, using direct path with intermediate points")

            direct_route = [start_point]

            steps = 3
            for i in range(1, steps):
                t = i / steps
                lat = start_point[0] * (1 - t) + end_point[0] * t
                lon = start_point[1] * (1 - t) + end_point[1] * t

                perpendicular_offset = disruption_radius * 0.8 * math.sin(math.pi * t)

                dx = end_point[1] - start_point[1]
                dy = end_point[0] - start_point[0]

                length = math.sqrt(dx ** 2 + dy ** 2)
                if length > 0:
                    dx, dy = -dy / length, dx / length

                    lat += perpendicular_offset * dy
                    lon += perpendicular_offset * dx

                    direct_route.append((lat, lon))

            direct_route.append(end_point)
            return direct_route

        except Exception as e:
            print(f"Error in _find_path_avoiding_disruption: {e}")
            import traceback
            traceback.print_exc()
            return [start_point,
                    ((start_point[0] + end_point[0]) / 2 + 0.0005, (start_point[1] + end_point[1]) / 2 + 0.0005),
                    end_point]

    def _create_reassignment_actions(self, broken_driver_id: int, state: DeliverySystemState) -> List[DisruptionAction]:
        """Create actions to reassign deliveries from a broken-down driver to others"""
        actions = []

        broken_assignments = state.driver_assignments.get(broken_driver_id, [])
        if not broken_assignments:
            print(f"No assignments for broken driver {broken_driver_id}")
            return actions

        print(f"Found {len(broken_assignments)} assignments for broken driver {broken_driver_id}")

        available_drivers = []
        for driver_id in state.driver_assignments:
            if driver_id != broken_driver_id:
                weight_capacity = 1000.0
                volume_capacity = 10.0

                if driver_id in state.driver_capacities:
                    try:
                        weight_capacity, volume_capacity = state.driver_capacities[driver_id]
                    except:
                        pass

                distance = float('inf')
                current_pos = state.driver_positions.get(driver_id)
                broken_pos = state.driver_positions.get(broken_driver_id)

                if current_pos and broken_pos:
                    try:
                        distance = self._calculate_distance(current_pos, broken_pos)
                    except:
                        pass

                available_drivers.append({
                    'id': driver_id,
                    'distance': distance,
                    'weight_capacity': weight_capacity,
                    'volume_capacity': volume_capacity
                })

        if not available_drivers:
            print(f"No available drivers found")
            return actions

        available_drivers.sort(key=lambda d: d['distance'])
        print(f"Found {len(available_drivers)} available drivers")

        delivery_data = []
        for idx in broken_assignments:
            weight = 1.0
            volume = 0.01

            try:
                if idx < len(state.deliveries):
                    delivery = state.deliveries[idx]

                    if isinstance(delivery, (list, tuple)):
                        if len(delivery) > 2:
                            weight = float(delivery[2]) if delivery[2] is not None else weight
                        if len(delivery) > 3:
                            volume = float(delivery[3]) if delivery[3] is not None else volume
                    elif hasattr(delivery, 'weight') and hasattr(delivery, 'volume'):
                        weight = float(delivery.weight)
                        volume = float(delivery.volume)
            except:
                pass

            delivery_data.append({
                'index': idx,
                'weight': weight,
                'volume': volume
            })

        if available_drivers and delivery_data:
            remaining_deliveries = set(broken_assignments)

            for driver in available_drivers:
                if not remaining_deliveries:
                    break

                assignable_indices = []
                assigned_weight = 0
                assigned_volume = 0

                for delivery in delivery_data:
                    if delivery['index'] in remaining_deliveries:
                        new_weight = assigned_weight + delivery['weight']
                        new_volume = assigned_volume + delivery['volume']

                        if (new_weight <= driver['weight_capacity'] and
                                new_volume <= driver['volume_capacity']):
                            assignable_indices.append(delivery['index'])
                            assigned_weight = new_weight
                            assigned_volume = new_volume

                if assignable_indices:
                    action = ReassignDeliveriesAction(
                        from_driver_id=broken_driver_id,
                        to_driver_id=driver['id'],
                        delivery_indices=assignable_indices
                    )
                    actions.append(action)
                    print(
                        f"Created reassignment action from {broken_driver_id} to {driver['id']} with {len(assignable_indices)} deliveries")

                    remaining_deliveries.difference_update(assignable_indices)

            if not actions and available_drivers:
                first_delivery = broken_assignments[0]
                closest_driver = available_drivers[0]['id']

                action = ReassignDeliveriesAction(
                    from_driver_id=broken_driver_id,
                    to_driver_id=closest_driver,
                    delivery_indices=[first_delivery]
                )
                actions.append(action)
                print(
                    f"Created minimum fallback action: reassign delivery {first_delivery} from {broken_driver_id} to {closest_driver}")

        return actions

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the Haversine distance between two points in meters
        """
        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000

        return c * r

    def _distance_point_to_line(self, point, line_start, line_end):
        """
        Calculate the shortest distance from a point to a line segment
        """
        try:
            p_lat, p_lon = point
            s_lat, s_lon = line_start
            e_lat, e_lon = line_end

            if s_lat == e_lat and s_lon == e_lon:
                return self._calculate_distance(point, line_start)

            line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

            if line_length_sq < 1e-10:
                return self._calculate_distance(point, line_start)

            t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                               (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

            closest_lat = s_lat + t * (e_lat - s_lat)
            closest_lon = s_lon + t * (e_lon - s_lon)

            return self._calculate_distance(point, (closest_lat, closest_lon))
        except Exception as e:
            print(f"Error in distance calculation: {e}")
            try:
                return self._calculate_distance(point, line_start)
            except:
                return float('inf')
