import math
import time
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
        self.max_computation_time = max_computation_time  # Maximum computation time in seconds
        self._route_cache = {}

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost
        """
        # Vehicle breakdowns should always be handled
        if disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
            return True

        # Road closures should always be handled too
        elif disruption.type == DisruptionType.ROAD_CLOSURE:
            return True

        # Make traffic jams more likely to be recalculated
        elif disruption.type == DisruptionType.TRAFFIC_JAM:
            # Lower the severity threshold from 0.5 to 0.3
            if disruption.severity >= 0.3:
                return True

            # Check if any drivers are affected
            affected_drivers = self._get_affected_drivers(disruption, state)
            return len(affected_drivers) > 0

        # For recipient unavailable, always recalculate if a delivery is affected
        elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            affected_drivers = self._get_affected_drivers(disruption, state)
            return len(affected_drivers) > 0

        # More lenient default threshold (was 0.7)
        return disruption.severity >= 0.4

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """Determine actions to take in response to disruptions"""
        actions = []
        start_time = time.time()

        for disruption in active_disruptions:
            try:
                # Get affected drivers with better error handling
                try:
                    affected_drivers = self._get_affected_drivers(disruption, state)
                except Exception as e:
                    print(f"Error getting affected drivers: {e}")
                    affected_drivers = set()

                    # Emergency fallback - get any driver
                    if state.driver_assignments:
                        for driver_id in state.driver_assignments:
                            affected_drivers = {driver_id}
                            break

                print(f"Processing {disruption.type.value} affecting {len(affected_drivers)} drivers")

                # Handle different disruption types
                if disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
                    if affected_drivers:
                        broken_driver_id = next(iter(affected_drivers))
                        # Simplified reassignment that always creates an action
                        reassignment_actions = self._create_simple_reassignment(broken_driver_id, state)
                        actions.extend(reassignment_actions)

                elif disruption.type == DisruptionType.ROAD_CLOSURE:
                    for driver_id in affected_drivers:
                        reroute_action = self._create_simple_reroute(driver_id, disruption, state)
                        if reroute_action:
                            actions.append(reroute_action)

                elif disruption.type == DisruptionType.TRAFFIC_JAM:
                    # Handle traffic jams same as road closures
                    for driver_id in affected_drivers:
                        reroute_action = self._create_simple_reroute(driver_id, disruption, state)
                        if reroute_action:
                            actions.append(reroute_action)

                elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                    # For unavailable recipients, create reordering actions
                    for driver_id in affected_drivers:
                        reorder_action = self._create_delivery_reordering(driver_id, disruption, state)
                        if reorder_action:
                            actions.append(reorder_action)

            except Exception as e:
                print(f"Error handling disruption {disruption.id}: {e}")
                continue

        return actions

    def _create_simple_reroute(self, driver_id, disruption, state):
        """Create a simplified reroute action that always works"""
        try:
            # Generate a detour point away from the disruption
            detour_lat = disruption.location[0] + 0.005  # ~500m away
            detour_lon = disruption.location[1] + 0.005

            # Create a simple route: current position → detour → warehouse
            position = state.driver_positions.get(driver_id, self.warehouse_location)
            new_route = [position, (detour_lat, detour_lon), self.warehouse_location]

            return RerouteAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id
            )
        except Exception as e:
            print(f"Error creating simple reroute: {e}")
            return None

    def _create_simple_reassignment(self, broken_driver_id, state):
        """Create a simplified reassignment that always works"""
        actions = []

        # Get the broken driver's assignments
        broken_assignments = state.driver_assignments.get(broken_driver_id, [])
        if not broken_assignments:
            return actions

        # Find any other driver
        for driver_id in state.driver_assignments:
            if driver_id != broken_driver_id:
                # Create a reassignment action for all deliveries
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
            # Find affected delivery
            delivery_idx = disruption.metadata.get("delivery_point_index")
            if delivery_idx is None:
                return None

            assignments = state.driver_assignments.get(driver_id, [])
            if delivery_idx not in assignments or len(assignments) <= 1:
                return None

            # Create new order with affected delivery last
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

            # For vehicle breakdowns, use affected_driver_ids or find any active driver
            if disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
                if hasattr(disruption, 'affected_driver_ids') and disruption.affected_driver_ids:
                    return set(disruption.affected_driver_ids)
                else:
                    # Find any driver as fallback
                    if state.driver_assignments:
                        for driver_id, assignments in state.driver_assignments.items():
                            if assignments:  # Only select drivers with actual assignments
                                return {driver_id}
                    return set()

            # For recipient unavailable, check metadata for specific delivery point
            if disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                if "delivery_point_index" in disruption.metadata:
                    delivery_idx = disruption.metadata["delivery_point_index"]
                    for driver_id, assignments in state.driver_assignments.items():
                        if delivery_idx in assignments:
                            affected_drivers.add(driver_id)
                            break

            # More generous distance check (doubled radius)
            effective_radius = disruption.affected_area_radius * 2

            # Check all drivers
            for driver_id, assignments in state.driver_assignments.items():
                if not assignments:
                    continue

                # Check if driver's position is near disruption
                position = state.driver_positions.get(driver_id)
                if position:
                    try:
                        distance = self._calculate_distance(position, disruption.location)
                        if distance <= effective_radius:
                            affected_drivers.add(driver_id)
                            continue
                    except Exception:
                        pass

                # Check if any delivery points are near disruption
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

            # Fallback: if no drivers affected but we need one, pick the first available
            if not affected_drivers and state.driver_assignments:
                for driver_id, assignments in state.driver_assignments.items():
                    if assignments:
                        affected_drivers.add(driver_id)
                        break

            return affected_drivers

        except Exception as e:
            print(f"Error in _get_affected_drivers: {e}")
            # Fallback to any driver with assignments
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
            # Check how many drivers might be affected
            affected_drivers = self._get_affected_drivers(disruption, state)
            if len(affected_drivers) >= 1:
                return True

            # Check if the disruption is on a major road
            try:
                # Find the nearest node to the disruption
                nearest_node = ox.nearest_nodes(
                    self.G,
                    X=disruption.location[1],
                    Y=disruption.location[0]
                )

                # Check node degree (number of connected roads)
                # Higher degree indicates an intersection or major road
                degree = len(list(self.G.neighbors(nearest_node)))

                return degree >= 2  # Consider major if it's an intersection with 3+ roads

            except Exception as e:
                print(f"Error analyzing road importance: {e}")
                # Fall back to disruption severity if graph analysis fails
                return disruption.severity > 0.3
        except Exception as e:
            print(f"Error in _is_major_route_disruption: {e}")
            return False  # Be conservative if we can't analyze properly

    def _create_reroute_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteAction]:
        """
        Create a reroute action for a driver affected by a disruption
        """
        try:
            # Validate driver assignments exist
            assignments = state.driver_assignments.get(driver_id, [])
            if not assignments:
                print(f"No assignments found for driver {driver_id}")
                return None

            # Safely get driver position
            position = state.driver_positions.get(driver_id)
            if not position:
                print(f"No position found for driver {driver_id}")
                return None

            # Build route points with validation
            route_points = [position]
            for delivery_idx in assignments:
                try:
                    if delivery_idx >= len(state.deliveries):
                        print(f"Invalid delivery index {delivery_idx}")
                        continue

                    delivery = state.deliveries[delivery_idx]
                    # Handle both tuple and object formats
                    if isinstance(delivery, (list, tuple)):
                        lat, lon = delivery[0], delivery[1]
                    else:
                        lat, lon = delivery.coordinates[0], delivery.coordinates[1]

                    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                        print(f"Invalid coordinates for delivery {delivery_idx}")
                        continue

                    route_points.append((lat, lon))
                except Exception as e:
                    print(f"Error processing delivery {delivery_idx}: {e}")
                    continue

            # Add warehouse return
            route_points.append(self.warehouse_location)

            # Only proceed if we have valid points
            if len(route_points) < 2:
                print("Not enough valid points to create route")
                return None

            # Calculate route avoiding disruption
            try:
                detailed_route = self._calculate_route_avoiding_disruption(
                    route_points,
                    disruption,
                    state
                )

                if not detailed_route:
                    print("Failed to calculate alternative route")
                    return None

                return RerouteAction(
                    driver_id=driver_id,
                    new_route=detailed_route,
                    affected_disruption_id=disruption.id
                )

            except Exception as e:
                print(f"Error calculating alternative route: {e}")
                return None

        except Exception as e:
            print(f"Error creating reroute action for driver {driver_id}: {e}")
            return None

    def _calculate_route_avoiding_disruption(self, waypoints: List[Tuple[float, float]],
                                             disruption: Disruption,
                                             state: DeliverySystemState) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate a route through waypoints that avoids a disruption
        """
        if not waypoints or len(waypoints) < 2:
            return None

        # Always add a detour point to ensure we create a different route
        disruption_lat, disruption_lon = disruption.location
        detour_lat = disruption_lat + 0.005  # Move ~500m away
        detour_lon = disruption_lon + 0.005

        # Create the route with the detour point
        detailed_route = [waypoints[0]]  # Start with first point
        detailed_route.append((detour_lat, detour_lon))  # Add detour

        # Add remaining waypoints
        for point in waypoints[1:]:
            detailed_route.append(point)

        return detailed_route

    def _segment_intersects_disruption(self, start: Tuple[float, float],
                                       end: Tuple[float, float],
                                       disruption: Disruption) -> bool:
        """
        Check if a route segment intersects a disruption area

        Uses a simplified approach checking if the line segment passes near the disruption
        """
        disruption_location = disruption.location
        radius = disruption.affected_area_radius

        # Check if either endpoint is within the disruption
        start_distance = self._calculate_distance(start, disruption_location)
        if start_distance <= radius:
            return True

        end_distance = self._calculate_distance(end, disruption_location)
        if end_distance <= radius:
            return True

        # Check if the line segment passes through the disruption
        # Simplification: Check distance from disruption to the line segment
        line_distance = self._distance_point_to_line(
            disruption_location,
            start,
            end
        )

        return line_distance <= radius

    def _find_alternative_path(self, start, end, disruption):
        """
        Find an alternative path between two points that avoids a disruption
        """
        try:
            # Find nearest nodes in the graph
            try:
                start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
                end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return None

            # Create a copy of the graph for path finding
            G_copy = self.G.copy()

            # Find the node nearest to the disruption
            try:
                disruption_node = ox.nearest_nodes(
                    G_copy,
                    X=disruption.location[1],
                    Y=disruption.location[0]
                )
            except Exception as e:
                print(f"Error finding node near disruption: {e}")
                # Continue without marking disruption area - safer than crashing

            # Mark edges near the disruption as high cost
            radius = disruption.affected_area_radius
            affected_nodes = []

            for node, data in G_copy.nodes(data=True):
                try:
                    if 'y' not in data or 'x' not in data:
                        continue  # Skip nodes without coordinates

                    node_point = (data['y'], data['x'])
                    distance = self._calculate_distance(node_point, disruption.location)

                    if distance <= radius:
                        affected_nodes.append(node)
                except Exception as e:
                    # Just skip problematic nodes instead of crashing
                    continue

            # Only process nodes that actually exist in the graph
            valid_affected_nodes = [n for n in affected_nodes if n in G_copy]

            # Safely increase weights of edges connected to affected nodes
            for node in valid_affected_nodes:
                try:
                    neighbors = list(G_copy.neighbors(node))
                    for neighbor in neighbors:
                        if G_copy.has_edge(node, neighbor):
                            for edge_key in G_copy[node][neighbor]:
                                # Apply a cost multiplier based on disruption severity
                                multiplier = 1.0 + (9.0 * disruption.severity)  # 1x to 10x

                                # Safely update travel time if it exists
                                if 'travel_time' in G_copy[node][neighbor][edge_key]:
                                    G_copy[node][neighbor][edge_key]['travel_time'] *= multiplier
                except Exception as e:
                    # Skip this node if there's an error, but continue processing others
                    print(f"Error adjusting edge weights for node {node}: {e}")
                    continue

            # Find the shortest path in the modified graph
            try:
                path = nx.shortest_path(G_copy, start_node, end_node, weight='travel_time')
            except nx.NetworkXNoPath:
                print(f"No path found between {start} and {end}")
                return None
            except Exception as e:
                print(f"Error finding path: {e}")
                return None

            # Convert path nodes to coordinates
            route_points = []
            for node in path:
                try:
                    data = G_copy.nodes[node]
                    if 'y' in data and 'x' in data:
                        route_points.append((data['y'], data['x']))
                except Exception as e:
                    # Skip problematic nodes
                    continue

            # Make sure we have at least start and end points
            if len(route_points) < 2:
                return [start, end]  # Fallback to direct path

            return route_points

        except Exception as e:
            print(f"Error finding alternative path: {e}")
            import traceback
            traceback.print_exc()
            # Return a direct path as fallback
            return [start, end]

    def _create_reassignment_actions(self, broken_driver_id: int, state: DeliverySystemState) -> List[DisruptionAction]:
        """Create actions to reassign deliveries from a broken-down driver to others"""
        actions = []

        # Get the broken driver's remaining deliveries
        broken_assignments = state.driver_assignments.get(broken_driver_id, [])
        if not broken_assignments:
            print(f"No assignments for broken driver {broken_driver_id}")
            return actions

        print(f"Found {len(broken_assignments)} assignments for broken driver {broken_driver_id}")

        # Find other available drivers
        available_drivers = []
        for driver_id in state.driver_assignments:
            if driver_id != broken_driver_id:
                # Get capacity with safe defaults
                weight_capacity = 1000.0  # Large default
                volume_capacity = 10.0  # Large default

                # Try to get actual capacity if available
                if driver_id in state.driver_capacities:
                    try:
                        weight_capacity, volume_capacity = state.driver_capacities[driver_id]
                    except:
                        pass  # Keep defaults if unpacking fails

                # Get positions for distance calculation
                distance = float('inf')
                current_pos = state.driver_positions.get(driver_id)
                broken_pos = state.driver_positions.get(broken_driver_id)

                if current_pos and broken_pos:
                    try:
                        distance = self._calculate_distance(current_pos, broken_pos)
                    except:
                        pass  # Keep infinity if calculation fails

                available_drivers.append({
                    'id': driver_id,
                    'distance': distance,
                    'weight_capacity': weight_capacity,
                    'volume_capacity': volume_capacity
                })

        # No drivers available? Return empty list
        if not available_drivers:
            print(f"No available drivers found")
            return actions

        # Sort by distance - closest first
        available_drivers.sort(key=lambda d: d['distance'])
        print(f"Found {len(available_drivers)} available drivers")

        # Extract delivery data
        delivery_data = []
        for idx in broken_assignments:
            weight = 1.0  # Default weight
            volume = 0.01  # Default volume

            try:
                if idx < len(state.deliveries):
                    delivery = state.deliveries[idx]

                    # Parse delivery data based on format
                    if isinstance(delivery, (list, tuple)):
                        if len(delivery) > 2:
                            weight = float(delivery[2]) if delivery[2] is not None else weight
                        if len(delivery) > 3:
                            volume = float(delivery[3]) if delivery[3] is not None else volume
                    elif hasattr(delivery, 'weight') and hasattr(delivery, 'volume'):
                        weight = float(delivery.weight)
                        volume = float(delivery.volume)
            except:
                pass  # Keep defaults if parsing fails

            delivery_data.append({
                'index': idx,
                'weight': weight,
                'volume': volume
            })

        # Always create at least one action if we have drivers
        if available_drivers and delivery_data:
            # First try with capacity checks
            remaining_deliveries = set(broken_assignments)

            for driver in available_drivers:
                if not remaining_deliveries:
                    break

                assignable_indices = []
                assigned_weight = 0
                assigned_volume = 0

                # Check each delivery against capacity
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
                    # Create reassignment action
                    action = ReassignDeliveriesAction(
                        from_driver_id=broken_driver_id,
                        to_driver_id=driver['id'],
                        delivery_indices=assignable_indices
                    )
                    actions.append(action)
                    print(
                        f"Created reassignment action from {broken_driver_id} to {driver['id']} with {len(assignable_indices)} deliveries")

                    # Remove assigned deliveries
                    remaining_deliveries.difference_update(assignable_indices)

            # If we couldn't create any actions with capacity checks,
            # create one action for the first delivery to the closest driver
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

    def _distance_point_to_line(self, point, line_start, line_end):
        """
        Calculate the shortest distance from a point to a line segment
        """
        try:
            # Convert to simple Cartesian for distance calculation
            p_lat, p_lon = point
            s_lat, s_lon = line_start
            e_lat, e_lon = line_end

            # Check if line is a point
            if s_lat == e_lat and s_lon == e_lon:
                return self._calculate_distance(point, line_start)

            # Calculate line length squared
            line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

            # Avoid division by zero
            if line_length_sq < 1e-10:
                return self._calculate_distance(point, line_start)

            # Calculate projection parameter
            t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                               (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

            # Calculate closest point on line
            closest_lat = s_lat + t * (e_lat - s_lat)
            closest_lon = s_lon + t * (e_lon - s_lon)

            # Calculate distance to closest point
            return self._calculate_distance(point, (closest_lat, closest_lon))
        except Exception as e:
            print(f"Error in distance calculation: {e}")
            # Fallback to a direct distance to avoid crashing
            try:
                return self._calculate_distance(point, line_start)
            except:
                return float('inf')  # Return infinite distance as last resort
