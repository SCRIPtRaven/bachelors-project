import math
from typing import List, Set, Optional

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoRerouteAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction
)
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance


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

            if 'triggered_by_driver' in disruption.metadata:
                triggering_driver_id = disruption.metadata['triggered_by_driver']
                print(f"Disruption {disruption.id} was triggered by driver {triggering_driver_id}")
                affected_drivers.add(triggering_driver_id)

            for driver_id, assignments in state.driver_assignments.items():
                if not assignments:
                    continue

                position = state.driver_positions.get(driver_id)
                if position:
                    try:
                        distance = calculate_haversine_distance(position, disruption.location)
                        if distance <= effective_radius:
                            affected_drivers.add(driver_id)
                            continue
                    except Exception as e:
                        print(f"Error checking driver position: {e}")

                for delivery_idx in assignments:
                    try:
                        if delivery_idx < len(state.deliveries):
                            delivery = state.deliveries[delivery_idx]
                            lat, lon = delivery[0], delivery[1]

                            distance = calculate_haversine_distance((lat, lon), disruption.location)
                            if distance <= effective_radius:
                                affected_drivers.add(driver_id)
                                break
                    except Exception as e:
                        print(f"Error checking delivery point distance: {e}")
                        continue

                if driver_id in state.driver_routes and 'points' in state.driver_routes[driver_id]:
                    route_points = state.driver_routes[driver_id]['points']
                    if len(route_points) >= 2:
                        for i in range(len(route_points) - 1):
                            if self._segment_near_disruption(route_points[i], route_points[i + 1], disruption):
                                affected_drivers.add(driver_id)
                                #print(f"Driver {driver_id} has route segment near disruption {disruption.id}")
                                break

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

                distance_to_disruption = calculate_haversine_distance(
                    (lat, lon),
                    disruption.location
                )

                if distance_to_disruption <= disruption.affected_area_radius:
                    return delivery_idx

        return None

    def _create_no_reroute_action(self, driver_id: int, disruption: Disruption) -> Optional[NoRerouteAction]:
        """
        Create an action that explicitly chooses not to reroute a driver
        """
        try:
            #print(f"Creating no-reroute action for driver {driver_id} and disruption {disruption.id}")
            
            return NoRerouteAction(
                driver_id=driver_id,
                affected_disruption_id=disruption.id
            )
            
        except Exception as e:
            print(f"Error creating no-reroute action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_tight_avoidance_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteTightAvoidanceAction]:
        """
        Create a reroute action that minimally avoids the disruption
        """
        try:
            if driver_id not in state.driver_routes or 'points' not in state.driver_routes[driver_id]:
                #print(f"No route found for driver {driver_id}")
                return None
                
            route_points = state.driver_routes[driver_id]['points']
            if len(route_points) < 2:
                #print(f"Route too short for driver {driver_id}")
                return None
                
            # Find affected segment
            affected_segment_start = None
            affected_segment_end = None
            
            for i in range(len(route_points) - 1):
                if self._segment_near_disruption(route_points[i], route_points[i + 1], disruption):
                    affected_segment_start = i
                    affected_segment_end = i + 1
                    break
                    
            if affected_segment_start is None:
                #print(f"No affected segment found for driver {driver_id}")
                return None
                
            # Get the position and next delivery points
            position = state.driver_positions.get(driver_id)
            if not position:
                #print(f"No position found for driver {driver_id}")
                return None
                
            # Find next delivery
            next_delivery_index = None
            delivery_indices = state.driver_assignments.get(driver_id, [])
            
            if not delivery_indices:
                #print(f"No deliveries assigned to driver {driver_id}")
                return None
                
            # Find next delivery after current position
            for idx in delivery_indices:
                if idx < len(state.deliveries):
                    next_delivery_index = idx
                    break
            
            # Create a tight reroute that stays close to disruption boundary
            start_point = route_points[affected_segment_start]
            end_point = route_points[affected_segment_end]
            
            # Use a smaller buffer for tight avoidance (0.8 ratio)
            new_route = self._find_path_avoiding_disruption_tight(
                self.G, 
                start_point, 
                end_point, 
                disruption,
                buffer_ratio=0.8
            )
            
            # Combine the reroute with the original route
            full_route = []
            for i in range(affected_segment_start + 1):
                full_route.append(route_points[i])
                
            full_route.extend(new_route[1:-1])  # Skip first and last to avoid duplicates
            
            for i in range(affected_segment_end, len(route_points)):
                full_route.append(route_points[i])
            
            reroute_action = RerouteTightAvoidanceAction(
                driver_id=driver_id,
                new_route=full_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_end,
                next_delivery_index=next_delivery_index,
                delivery_indices=delivery_indices
            )
            
            return reroute_action
            
        except Exception as e:
            print(f"Error creating tight avoidance action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_wide_avoidance_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteWideAvoidanceAction]:
        """
        Create a reroute action that widely avoids the disruption
        """
        try:
            if driver_id not in state.driver_routes or 'points' not in state.driver_routes[driver_id]:
                #print(f"No route found for driver {driver_id}")
                return None
                
            route_points = state.driver_routes[driver_id]['points']
            if len(route_points) < 2:
                #print(f"Route too short for driver {driver_id}")
                return None
                
            # Find affected segment
            affected_segment_start = None
            affected_segment_end = None
            
            # Expand search radius for wide avoidance
            original_radius = disruption.affected_area_radius
            expanded_radius = original_radius * 2.0  # Double the radius for detecting affected segments
            
            try:
                # Temporarily increase the radius to find segments that should be widely avoided
                disruption.affected_area_radius = expanded_radius
                
                for i in range(len(route_points) - 1):
                    if self._segment_near_disruption(route_points[i], route_points[i + 1], disruption):
                        affected_segment_start = i
                        affected_segment_end = i + 1
                        break
            finally:
                # Restore original radius
                disruption.affected_area_radius = original_radius
                    
            if affected_segment_start is None:
                #print(f"No affected segment found for driver {driver_id}")
                return None
                
            # Get the position and next delivery points
            position = state.driver_positions.get(driver_id)
            if not position:
                #print(f"No position found for driver {driver_id}")
                return None
                
            # Find next delivery
            next_delivery_index = None
            delivery_indices = state.driver_assignments.get(driver_id, [])
            
            if not delivery_indices:
                #print(f"No deliveries assigned to driver {driver_id}")
                return None
                
            # Find next delivery after current position
            for idx in delivery_indices:
                if idx < len(state.deliveries):
                    next_delivery_index = idx
                    break
            
            # Create a wide reroute that stays far from disruption
            start_point = route_points[affected_segment_start]
            end_point = route_points[affected_segment_end]
            
            # Use a larger buffer for wide avoidance (2.0 ratio)
            new_route = self._find_path_avoiding_disruption_wide(
                self.G, 
                start_point, 
                end_point, 
                disruption,
                buffer_ratio=2.0
            )
            
            # Combine the reroute with the original route
            full_route = []
            for i in range(affected_segment_start + 1):
                full_route.append(route_points[i])
                
            full_route.extend(new_route[1:-1])  # Skip first and last to avoid duplicates
            
            for i in range(affected_segment_end, len(route_points)):
                full_route.append(route_points[i])
            
            reroute_action = RerouteWideAvoidanceAction(
                driver_id=driver_id,
                new_route=full_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_end,
                next_delivery_index=next_delivery_index,
                delivery_indices=delivery_indices
            )
            
            return reroute_action
            
        except Exception as e:
            print(f"Error creating wide avoidance action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_reroute_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        RerouteBasicAction]:
        try:
            position = state.driver_positions.get(driver_id)
            if not position:
                return None

            original_detailed_route = []
            if driver_id in state.driver_routes and 'points' in state.driver_routes[driver_id]:
                original_detailed_route = state.driver_routes[driver_id]['points']

            if not original_detailed_route or len(original_detailed_route) < 2:
                #print(f"No detailed route found for driver {driver_id}")
                return None

            current_segment_start = 0
            min_distance = float('inf')

            for i, point in enumerate(original_detailed_route):
                try:
                    distance = calculate_haversine_distance(position, point)
                    if distance < min_distance:
                        min_distance = distance
                        current_segment_start = i
                except Exception as e:
                    print(f"Error calculating distance: {e}")

            #print(f"Driver {driver_id} is near point {current_segment_start} of {len(original_detailed_route)}")

            look_ahead = 15  # Increased from 10 to check more of the route

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
                            affected_segment_end = j  # End is node before clear segment
                            break
                    if affected_segment_end == -1:
                        affected_segment_end = min(i + look_ahead, len(original_detailed_route) - 1)
                    break

            if affected_segment_start == -1:
                if 'triggered_by_driver' in disruption.metadata and disruption.metadata[
                    'triggered_by_driver'] == driver_id:
                    affected_segment_start = current_segment_start
                    affected_segment_end = min(current_segment_start + 5, len(original_detailed_route) - 1)
                    #print(
                    #    f"Driver {driver_id} triggered disruption {disruption.id}, forcing reroute at current position")
                else:
                    #print(f"No affected segment found for driver {driver_id} near disruption {disruption.id}")
                    return None

            #print(
            #    f"Found affected segment {affected_segment_start}-{affected_segment_end} out of {len(original_detailed_route)}")

            start_point = original_detailed_route[affected_segment_start]
            end_point_index = min(affected_segment_end + 1, len(original_detailed_route) - 1)
            end_point = original_detailed_route[end_point_index]

            detour_points = self._find_path_avoiding_disruption(state.graph, start_point, end_point, disruption)

            if not detour_points or len(detour_points) < 2:
                #print(f"Failed to find detour for driver {driver_id}")
                return None

            new_route = []
            new_route.extend(original_detailed_route[:affected_segment_start])
            if new_route and detour_points and calculate_haversine_distance(new_route[-1], detour_points[0]) < 1:
                new_route.extend(detour_points[1:])
            else:
                new_route.extend(detour_points)
            if end_point_index + 1 < len(original_detailed_route):
                remaining_route_start_point = original_detailed_route[end_point_index + 1]
                if new_route and calculate_haversine_distance(new_route[-1], remaining_route_start_point) < 1:
                    new_route.extend(original_detailed_route[end_point_index + 2:])
                else:
                    new_route.extend(original_detailed_route[end_point_index + 1:])
            if len(new_route) < 2: return None
            #print(f"Created new route with {len(new_route)} points.")

            next_delivery_index = None
            delivery_indices = []

            if 'delivery_indices' in state.driver_routes.get(driver_id, {}):
                delivery_indices = state.driver_routes[driver_id]['delivery_indices']
                if delivery_indices:
                    next_delivery_index = delivery_indices[0]  # First upcoming delivery

            reroute_action = RerouteBasicAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_end,
                next_delivery_index=next_delivery_index,
                delivery_indices=delivery_indices
            )

            return reroute_action

        except Exception as e:
            print(f"Error creating reroute action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _segment_near_disruption(self, start, end, disruption):
        """
        Check if a route segment is affected by a disruption
        """
        try:
            disruption_point = disruption.location
            min_distance = self._point_to_segment_distance(disruption_point, start, end)

            buffer_factor = 1.5
            return min_distance <= disruption.affected_area_radius * buffer_factor
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
                return calculate_haversine_distance(point, line_start)

            line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

            if line_length_sq < 1e-10:
                return calculate_haversine_distance(point, line_start)

            t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                               (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

            closest_lat = s_lat + t * (e_lat - s_lat)
            closest_lon = s_lon + t * (e_lon - s_lon)

            return calculate_haversine_distance(point, (closest_lat, closest_lon))
        except Exception as e:
            print(f"Error in distance calculation: {e}")
            try:
                return calculate_haversine_distance(point, line_start)
            except:
                return float('inf')

    def _find_path_avoiding_disruption(self, graph, start_point, end_point, disruption):
        """
        Find a path between points that avoids a disruption area
        """
        try:
            try:
                start_point = (float(start_point[0]), float(start_point[1]))
                end_point = (float(end_point[0]), float(end_point[1]))
                disruption_location = (float(disruption.location[0]), float(disruption.location[1]))
                disruption.location = disruption_location
            except (ValueError, TypeError) as e:
                print(f"Error converting coordinates to float: {e}")
                print(f"start_point: {start_point}, end_point: {end_point}, disruption_location: {disruption.location}")
                return [start_point, end_point]

            if not graph:
                print("No graph available for path finding")
                return [start_point, end_point]

            try:
                start_node = ox.nearest_nodes(graph, X=float(start_point[1]), Y=float(start_point[0]))
                end_node = ox.nearest_nodes(graph, X=float(end_point[1]), Y=float(end_point[0]))
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return [start_point, end_point]

            G_mod = graph.copy()

            disruption_nodes = []
            affected_edges = 0
            disruption_radius = float(disruption.affected_area_radius)

            search_radius = disruption_radius * 1.2

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = calculate_haversine_distance(node_point, disruption.location)
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
                                try:
                                    original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                    if not isinstance(original_time, (int, float)):
                                        original_time = float(original_time)

                                    G_mod[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier
                                    affected_edges += 1
                                except (TypeError, ValueError) as e:
                                    print(f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                    pass

            #print(f"Modified {affected_edges} edges for disruption avoidance")

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

    def _find_path_avoiding_disruption_tight(self, graph, start_point, end_point, disruption, buffer_ratio=0.8):
        """
        Find a path between points that minimally avoids a disruption area (tight avoidance)
        """
        try:
            try:
                start_point = (float(start_point[0]), float(start_point[1]))
                end_point = (float(end_point[0]), float(end_point[1]))
                disruption_location = (float(disruption.location[0]), float(disruption.location[1]))
                disruption.location = disruption_location
            except (ValueError, TypeError) as e:
                print(f"Error converting coordinates to float: {e}")
                print(f"start_point: {start_point}, end_point: {end_point}, disruption_location: {disruption.location}")
                return [start_point, end_point]

            if not graph:
                print("No graph available for path finding")
                return [start_point, end_point]

            try:
                start_node = ox.nearest_nodes(graph, X=float(start_point[1]), Y=float(start_point[0]))
                end_node = ox.nearest_nodes(graph, X=float(end_point[1]), Y=float(end_point[0]))
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return [start_point, end_point]

            G_mod = graph.copy()

            disruption_nodes = []
            affected_edges = 0
            disruption_radius = float(disruption.affected_area_radius)

            # Use a tighter search radius for the tight avoidance
            search_radius = disruption_radius * buffer_ratio * 1.1  # Just slightly larger than the buffer

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = calculate_haversine_distance(node_point, disruption.location)
                if distance <= search_radius:
                    disruption_nodes.append(node)

            # For tight avoidance, use a moderate weight multiplier
            weight_multiplier = 5.0
            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (4.0 * disruption.severity)  # Less severe multiplier
            elif disruption.type.value == 'road_closure':
                weight_multiplier = 50.0  # Still significant but less than wide avoidance

            for node in disruption_nodes:
                for neighbor in list(G_mod.neighbors(node)):
                    if G_mod.has_edge(node, neighbor):
                        for edge_key in list(G_mod[node][neighbor].keys()):
                            if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                try:
                                    original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                    if not isinstance(original_time, (int, float)):
                                        original_time = float(original_time)

                                    G_mod[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier
                                    affected_edges += 1
                                except (TypeError, ValueError) as e:
                                    print(f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                    pass

            #print(f"Modified {affected_edges} edges for tight disruption avoidance")

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

            # Fallback to geometric approach if network routing fails
            # For tight avoidance, use fewer steps and smaller perpendicular offset
            steps = 3  # Fewer intermediate points
            direct_route = [start_point]

            center_lat, center_lon = disruption.location
            start_lat, start_lon = start_point
            end_lat, end_lon = end_point

            # Calculate perpendicular direction for avoidance
            # For tight avoidance, use the direction that's just enough to avoid the disruption
            dlat = center_lat - ((start_lat + end_lat) / 2)
            dlon = center_lon - ((start_lon + end_lon) / 2)
            
            # Normalize the direction vector
            mag = math.sqrt(dlat ** 2 + dlon ** 2)
            if mag > 0:
                norm_dlat = dlat / mag
                norm_dlon = dlon / mag
            else:
                norm_dlat = 0
                norm_dlon = 0
                
            # Calculate perpendicular vector
            perp_dlat = -norm_dlon
            perp_dlon = norm_dlat

            for i in range(1, steps):
                t = i / steps
                lat = start_point[0] * (1 - t) + end_point[0] * t
                lon = start_point[1] * (1 - t) + end_point[1] * t

                # For tight avoidance, use a smaller offset
                perpendicular_offset = disruption_radius * buffer_ratio * math.sin(math.pi * t)

                # Apply the perpendicular offset in the direction away from the disruption
                offset_direction = -1 if (norm_dlat * perp_dlat + norm_dlon * perp_dlon) > 0 else 1
                
                lat += offset_direction * perpendicular_offset * perp_dlat
                lon += offset_direction * perpendicular_offset * perp_dlon

                direct_route.append((lat, lon))

            direct_route.append(end_point)
            return direct_route

        except Exception as e:
            print(f"Error in _find_path_avoiding_disruption_tight: {e}")
            import traceback
            traceback.print_exc()
            return [start_point, end_point]
    
    def _find_path_avoiding_disruption_wide(self, graph, start_point, end_point, disruption, buffer_ratio=2.0):
        """
        Find a path between points that widely avoids a disruption area
        """
        try:
            try:
                start_point = (float(start_point[0]), float(start_point[1]))
                end_point = (float(end_point[0]), float(end_point[1]))
                disruption_location = (float(disruption.location[0]), float(disruption.location[1]))
                disruption.location = disruption_location
            except (ValueError, TypeError) as e:
                print(f"Error converting coordinates to float: {e}")
                print(f"start_point: {start_point}, end_point: {end_point}, disruption_location: {disruption.location}")
                return [start_point, end_point]

            if not graph:
                print("No graph available for path finding")
                return [start_point, end_point]

            try:
                start_node = ox.nearest_nodes(graph, X=float(start_point[1]), Y=float(start_point[0]))
                end_node = ox.nearest_nodes(graph, X=float(end_point[1]), Y=float(end_point[0]))
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return [start_point, end_point]

            G_mod = graph.copy()

            disruption_nodes = []
            affected_edges = 0
            disruption_radius = float(disruption.affected_area_radius)

            # Use a wider search radius for the wide avoidance
            search_radius = disruption_radius * buffer_ratio

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = calculate_haversine_distance(node_point, disruption.location)
                if distance <= search_radius:
                    disruption_nodes.append(node)

            # For wide avoidance, use a more aggressive weight multiplier
            weight_multiplier = 20.0
            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (19.0 * disruption.severity)  # More severe multiplier
            elif disruption.type.value == 'road_closure':
                weight_multiplier = 200.0  # Very high to ensure wide avoidance

            for node in disruption_nodes:
                for neighbor in list(G_mod.neighbors(node)):
                    if G_mod.has_edge(node, neighbor):
                        for edge_key in list(G_mod[node][neighbor].keys()):
                            if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                try:
                                    original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                    if not isinstance(original_time, (int, float)):
                                        original_time = float(original_time)

                                    G_mod[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier
                                    affected_edges += 1
                                except (TypeError, ValueError) as e:
                                    print(f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                    pass

            #rint(f"Modified {affected_edges} edges for wide disruption avoidance")

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

            # Fallback to geometric approach if network routing fails
            # For wide avoidance, use more steps and larger perpendicular offset
            steps = 7  # More intermediate points for smoother wide curve
            direct_route = [start_point]

            center_lat, center_lon = disruption.location
            start_lat, start_lon = start_point
            end_lat, end_lon = end_point

            # Calculate perpendicular direction for avoidance
            dlat = center_lat - ((start_lat + end_lat) / 2)
            dlon = center_lon - ((start_lon + end_lon) / 2)
            
            # Normalize the direction vector
            mag = math.sqrt(dlat ** 2 + dlon ** 2)
            if mag > 0:
                norm_dlat = dlat / mag
                norm_dlon = dlon / mag
            else:
                norm_dlat = 0
                norm_dlon = 0
                
            # Calculate perpendicular vector
            perp_dlat = -norm_dlon
            perp_dlon = norm_dlat

            for i in range(1, steps):
                t = i / steps
                lat = start_point[0] * (1 - t) + end_point[0] * t
                lon = start_point[1] * (1 - t) + end_point[1] * t

                # For wide avoidance, use a larger offset
                perpendicular_offset = disruption_radius * buffer_ratio * math.sin(math.pi * t)

                # Apply the perpendicular offset in the direction away from the disruption
                offset_direction = -1 if (norm_dlat * perp_dlat + norm_dlon * perp_dlon) > 0 else 1
                
                lat += offset_direction * perpendicular_offset * perp_dlat
                lon += offset_direction * perpendicular_offset * perp_dlon

                direct_route.append((lat, lon))

            direct_route.append(end_point)
            return direct_route

        except Exception as e:
            print(f"Error in _find_path_avoiding_disruption_wide: {e}")
            import traceback
            traceback.print_exc()
            return [start_point, end_point]
