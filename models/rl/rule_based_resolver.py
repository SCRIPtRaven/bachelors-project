import math
import time
from typing import List, Set, Optional, Tuple

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.rl.actions import (
    DisruptionAction, RerouteAction, ReassignDeliveriesAction,
    WaitAction, SkipDeliveryAction, PrioritizeDeliveryAction
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

        Uses simple rules based on disruption type, severity, and estimated impact
        """
        # Check disruption type (some types are more important than others)
        if disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
            # Always recalculate for vehicle breakdowns since they're critical
            return True

        elif disruption.type == DisruptionType.ROAD_CLOSURE:
            # For road closures, check if they're on a major route
            return self._is_major_route_disruption(state, disruption)

        elif disruption.type == DisruptionType.TRAFFIC_JAM:
            # For traffic jams, only recalculate if severe and affects multiple drivers
            if disruption.severity < 0.5:
                # Low severity traffic jams can be ignored
                return False

            affected_drivers = self._get_affected_drivers(disruption, state)
            return len(affected_drivers) >= 2  # Only recalculate if it affects multiple drivers

        elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            # Only recalculate for recipient unavailable if it's a time-sensitive delivery
            # or if the driver has many other deliveries that could be reordered
            affected_drivers = self._get_affected_drivers(disruption, state)

            if not affected_drivers:
                return False

            # Check if the affected driver has multiple deliveries
            driver_id = next(iter(affected_drivers))
            driver_assignments = state.driver_assignments.get(driver_id, [])

            return len(driver_assignments) >= 3  # Only worth reordering if multiple deliveries remain

        # Default fallback
        return disruption.severity > 0.7  # Only recalculate for high severity disruptions

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """
        Determine actions to take in response to disruptions

        Applies different strategies based on disruption type
        """
        actions = []
        start_time = time.time()

        for disruption in active_disruptions:
            # Respect computation time limit
            if time.time() - start_time > self.max_computation_time:
                break

            affected_drivers = self._get_affected_drivers(disruption, state)

            if disruption.type == DisruptionType.ROAD_CLOSURE:
                # Strategy: Reroute around the closure
                for driver_id in affected_drivers:
                    reroute_action = self._create_reroute_action(driver_id, disruption, state)
                    if reroute_action:
                        actions.append(reroute_action)

            elif disruption.type == DisruptionType.TRAFFIC_JAM:
                # Strategy: Either reroute or wait depending on severity and alternatives
                for driver_id in affected_drivers:
                    # Try to find an alternative route first
                    reroute_action = self._create_reroute_action(driver_id, disruption, state)

                    if reroute_action:
                        actions.append(reroute_action)
                    else:
                        # If no good alternative, sometimes it's better to wait
                        if disruption.severity > 0.7 and disruption.duration < 1800:  # < 30 minutes
                            wait_action = WaitAction(
                                driver_id=driver_id,
                                wait_time=disruption.duration,
                                disruption_id=disruption.id
                            )
                            actions.append(wait_action)

            elif disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
                # Strategy: Reassign deliveries to other drivers
                if len(affected_drivers) > 0:
                    broken_driver_id = next(iter(affected_drivers))
                    reassignment_actions = self._create_reassignment_actions(broken_driver_id, state)
                    actions.extend(reassignment_actions)

            elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                # Strategy: Skip or reschedule the delivery
                for driver_id in affected_drivers:
                    delivery_idx = self._find_affected_delivery(driver_id, disruption, state)

                    if delivery_idx is not None:
                        # Check if this is the driver's last delivery
                        driver_assignments = state.driver_assignments.get(driver_id, [])

                        if len(driver_assignments) <= 1:
                            # Just skip if it's the only delivery
                            skip_action = SkipDeliveryAction(
                                driver_id=driver_id,
                                delivery_index=delivery_idx
                            )
                            actions.append(skip_action)
                        else:
                            # Try to reorder deliveries to come back later
                            # Skip for now and reorder the remaining deliveries
                            new_order = [idx for idx in driver_assignments if idx != delivery_idx]
                            prioritize_action = PrioritizeDeliveryAction(
                                driver_id=driver_id,
                                delivery_indices=new_order
                            )
                            actions.append(prioritize_action)

        return actions

    def _get_affected_drivers(self, disruption: Disruption, state: DeliverySystemState) -> Set[int]:
        """
        Determine which drivers are affected by a disruption

        Returns a set of driver IDs
        """
        affected_drivers = set()

        if disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
            # Vehicle breakdowns affect specific drivers
            return disruption.affected_driver_ids

        # For other disruption types, check which drivers have nearby routes
        for driver_id, position in state.driver_positions.items():
            # Skip inactive drivers
            if driver_id not in state.driver_assignments:
                continue

            assignments = state.driver_assignments.get(driver_id, [])
            if not assignments:
                continue

            # Check if driver is within the affected area
            distance_to_disruption = self._calculate_distance(
                position,
                disruption.location
            )

            if distance_to_disruption <= disruption.affected_area_radius:
                affected_drivers.add(driver_id)
                continue

            # Check if any of the driver's pending deliveries are affected
            for delivery_idx in assignments:
                if delivery_idx < len(state.deliveries):
                    delivery = state.deliveries[delivery_idx]
                    lat, lon = delivery[0], delivery[1]

                    distance_to_disruption = self._calculate_distance(
                        (lat, lon),
                        disruption.location
                    )

                    if distance_to_disruption <= disruption.affected_area_radius:
                        affected_drivers.add(driver_id)
                        break

        return affected_drivers

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

        Major routes are those used by multiple drivers or on main roads
        """
        # Check how many drivers might be affected
        affected_drivers = self._get_affected_drivers(disruption, state)
        if len(affected_drivers) >= 2:
            return True

        # Check if the disruption is on a major road
        # This is a simplified approach - in reality, would check edge betweenness
        # or other network centrality measures
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

            return degree >= 3  # Consider major if it's an intersection with 3+ roads

        except Exception:
            # Fall back to disruption severity if graph analysis fails
            return disruption.severity > 0.6

    def _create_reroute_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteAction]:
        """
        Create a reroute action for a driver affected by a disruption

        Returns a RerouteAction if a better route is found, None otherwise
        """
        # Get driver's assignments
        assignments = state.driver_assignments.get(driver_id, [])
        if not assignments:
            return None

        # Get driver's current position
        position = state.driver_positions.get(driver_id)
        if not position:
            return None

        # Simplified: Create a route from current position to remaining deliveries to warehouse
        route_points = [position]

        for delivery_idx in assignments:
            if delivery_idx < len(state.deliveries):
                delivery = state.deliveries[delivery_idx]
                lat, lon = delivery[0], delivery[1]
                route_points.append((lat, lon))

        route_points.append(self.warehouse_location)

        # Create a detailed route that avoids the disruption
        detailed_route = self._calculate_route_avoiding_disruption(
            route_points,
            disruption,
            state
        )

        if detailed_route:
            return RerouteAction(
                driver_id=driver_id,
                new_route=detailed_route,
                affected_disruption_id=disruption.id
            )

        return None

    def _calculate_route_avoiding_disruption(self, waypoints: List[Tuple[float, float]],
                                             disruption: Disruption,
                                             state: DeliverySystemState) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate a route through waypoints that avoids a disruption

        Returns a list of (lat, lon) points, or None if no route found
        """
        if not waypoints or len(waypoints) < 2:
            return None

        detailed_route = []

        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]

            # Check if this segment intersects the disruption
            if self._segment_intersects_disruption(start, end, disruption):
                # Find alternative path for this segment
                alternative_path = self._find_alternative_path(start, end, disruption)

                if alternative_path:
                    # Add the alternative path (avoiding duplicating points)
                    if detailed_route and alternative_path:
                        detailed_route.extend(alternative_path[1:])
                    else:
                        detailed_route.extend(alternative_path)
                else:
                    # No alternative found, use direct path
                    if detailed_route and detailed_route[-1] == start:
                        detailed_route.append(end)
                    else:
                        detailed_route.extend([start, end])
            else:
                # Segment doesn't intersect, use direct path
                if detailed_route and detailed_route[-1] == start:
                    detailed_route.append(end)
                else:
                    detailed_route.extend([start, end])

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

    def _find_alternative_path(self, start: Tuple[float, float],
                               end: Tuple[float, float],
                               disruption: Disruption) -> Optional[List[Tuple[float, float]]]:
        """
        Find an alternative path between two points that avoids a disruption

        Uses the road network graph to find a path that avoids the disrupted area
        """
        try:
            # Find nearest nodes in the graph
            start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
            end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])

            # Create a copy of the graph for path finding
            G_copy = self.G.copy()

            # Find the node nearest to the disruption
            disruption_node = ox.nearest_nodes(
                G_copy,
                X=disruption.location[1],
                Y=disruption.location[0]
            )

            # Mark edges near the disruption as high cost
            radius = disruption.affected_area_radius
            affected_nodes = []

            for node, data in G_copy.nodes(data=True):
                node_point = (data['y'], data['x'])
                distance = self._calculate_distance(node_point, disruption.location)

                if distance <= radius:
                    affected_nodes.append(node)

            # Increase weights of edges connected to affected nodes
            for node in affected_nodes:
                for neighbor in G_copy.neighbors(node):
                    if G_copy.has_edge(node, neighbor):
                        for edge_key in G_copy[node][neighbor]:
                            # Apply a cost multiplier based on disruption severity
                            multiplier = 1.0 + (9.0 * disruption.severity)  # 1x to 10x

                            if 'travel_time' in G_copy[node][neighbor][edge_key]:
                                G_copy[node][neighbor][edge_key]['travel_time'] *= multiplier

                            if 'length' in G_copy[node][neighbor][edge_key]:
                                G_copy[node][neighbor][edge_key]['length'] *= multiplier

            # Find the shortest path in the modified graph
            path = nx.shortest_path(G_copy, start_node, end_node, weight='travel_time')

            # Convert path nodes to coordinates
            route_points = []
            for node in path:
                data = G_copy.nodes[node]
                route_points.append((data['y'], data['x']))

            return route_points

        except (nx.NetworkXNoPath, Exception) as e:
            print(f"Error finding alternative path: {e}")
            return None

    def _create_reassignment_actions(self, broken_driver_id: int, state: DeliverySystemState) -> List[DisruptionAction]:
        """
        Create actions to reassign deliveries from a broken-down driver to others

        Returns a list of ReassignDeliveriesAction objects
        """
        actions = []

        # Get the broken driver's remaining deliveries
        broken_assignments = state.driver_assignments.get(broken_driver_id, [])
        if not broken_assignments:
            return actions

        # Find other available drivers
        available_drivers = []

        for driver_id in state.driver_assignments:
            if driver_id != broken_driver_id:
                # Check if the driver has capacity
                if driver_id in state.driver_capacities:
                    weight_capacity, volume_capacity = state.driver_capacities[driver_id]

                    if weight_capacity > 0 and volume_capacity > 0:
                        # Calculate driver's distance to broken driver
                        if driver_id in state.driver_positions and broken_driver_id in state.driver_positions:
                            current_pos = state.driver_positions[driver_id]
                            broken_pos = state.driver_positions[broken_driver_id]

                            distance = self._calculate_distance(current_pos, broken_pos)

                            available_drivers.append({
                                'id': driver_id,
                                'distance': distance,
                                'weight_capacity': weight_capacity,
                                'volume_capacity': volume_capacity
                            })

        # Sort drivers by distance (closest first)
        available_drivers.sort(key=lambda d: d['distance'])

        # Calculate total weight and volume of remaining deliveries
        total_weight = 0
        total_volume = 0
        delivery_data = []

        for idx in broken_assignments:
            if idx < len(state.deliveries):
                delivery = state.deliveries[idx]
                weight = delivery[2] if len(delivery) > 2 else 1.0
                volume = delivery[3] if len(delivery) > 3 else 0.01

                total_weight += weight
                total_volume += volume

                delivery_data.append({
                    'index': idx,
                    'weight': weight,
                    'volume': volume
                })

        # Try to assign deliveries to available drivers
        remaining_deliveries = list(broken_assignments)

        for driver in available_drivers:
            # Skip if no deliveries left to assign
            if not remaining_deliveries:
                break

            # Calculate how many deliveries this driver can take
            assignable_indices = []
            assigned_weight = 0
            assigned_volume = 0

            for delivery in delivery_data:
                if delivery['index'] in remaining_deliveries:
                    if (assigned_weight + delivery['weight'] <= driver['weight_capacity'] and
                            assigned_volume + delivery['volume'] <= driver['volume_capacity']):
                        assignable_indices.append(delivery['index'])
                        assigned_weight += delivery['weight']
                        assigned_volume += delivery['volume']

            if assignable_indices:
                # Create a reassignment action
                action = ReassignDeliveriesAction(
                    from_driver_id=broken_driver_id,
                    to_driver_id=driver['id'],
                    delivery_indices=assignable_indices
                )

                actions.append(action)

                # Remove assigned deliveries from remaining list
                for idx in assignable_indices:
                    if idx in remaining_deliveries:
                        remaining_deliveries.remove(idx)

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

    def _distance_point_to_line(self, point: Tuple[float, float],
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float]) -> float:
        """
        Calculate the shortest distance from a point to a line segment

        Uses a simplified approach for geographic coordinates
        """
        # Convert to simple Cartesian for distance calculation
        # This is an approximation but works for short distances
        p_lat, p_lon = point
        s_lat, s_lon = line_start
        e_lat, e_lon = line_end

        # Check if line is a point
        if s_lat == e_lat and s_lon == e_lon:
            return self._calculate_distance(point, line_start)

        # Calculate line length squared
        line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

        # Calculate projection parameter
        t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                           (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

        # Calculate closest point on line
        closest_lat = s_lat + t * (e_lat - s_lat)
        closest_lon = s_lon + t * (e_lon - s_lon)

        # Calculate distance to closest point
        return self._calculate_distance(point, (closest_lat, closest_lon))
