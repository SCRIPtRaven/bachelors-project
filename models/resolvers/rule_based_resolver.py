from typing import List, Set, Optional, Tuple

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction
)
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import find_closest_point_index_on_route


class RuleBasedResolver(DisruptionResolver):
    def __init__(self, graph, warehouse_location, max_computation_time=1.0):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.max_computation_time = max_computation_time

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
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

    def resolve_disruptions(self,
                            state: DeliverySystemState,
                            active_disruptions: List[Disruption],
                            force_process_driver_id: Optional[int] = None
                            ) -> List[DisruptionAction]:
        actions = []

        for disruption in active_disruptions:
            try:
                if force_process_driver_id is not None:
                    drivers_to_process = [force_process_driver_id]
                    print(f"Processing {disruption.type.value} for FORCED driver {force_process_driver_id}")
                else:
                    drivers_to_process = self._get_affected_drivers(disruption, state)
                    print(f"Processing {disruption.type.value} affecting {len(drivers_to_process)} drivers")

                if disruption.type == DisruptionType.ROAD_CLOSURE or disruption.type == DisruptionType.TRAFFIC_JAM:
                    for driver_id in drivers_to_process:
                        if driver_id not in state.driver_positions or driver_id not in state.driver_routes:
                            print(
                                f"Warning (RuleBased): Skipping driver {driver_id} for disruption {disruption.id} as they are missing from state dicts.")
                            continue
                        reroute_action = self._create_reroute_action(driver_id, disruption, state)
                        if reroute_action:
                            actions.append(reroute_action)

                elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                    for driver_id in drivers_to_process:
                        if driver_id not in state.driver_positions or driver_id not in state.driver_routes:
                            print(
                                f"Warning (RuleBased): Skipping driver {driver_id} for disruption {disruption.id} as they are missing from state dicts.")
                            continue
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
                                # print(f"Driver {driver_id} has route segment near disruption {disruption.id}")
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

    def _create_no_action(self, driver_id: int, disruption: Disruption) -> Optional[NoAction]:
        try:
            return NoAction(
                driver_id=driver_id,
                affected_disruption_id=disruption.id
            )

        except Exception as e:
            print(f"Error creating no-action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_affected_route_portion(self, driver_id: int, route_points: list, position: tuple,
                                     disruption: Disruption) -> Optional[Tuple[int, int]]:
        if not route_points or len(route_points) < 2:
            return None

        current_segment_start_idx = find_closest_point_index_on_route(route_points, position)
        if current_segment_start_idx == -1:
            current_segment_start_idx = 0

        look_ahead = 15
        affected_start_idx = -1
        affected_end_idx = -1

        for i in range(current_segment_start_idx, min(current_segment_start_idx + look_ahead, len(route_points) - 1)):
            current_point = route_points[i]
            next_point = route_points[i + 1]

            if self._segment_near_disruption(current_point, next_point, disruption):
                if affected_start_idx == -1:
                    affected_start_idx = i

                affected_end_idx = i + 1
            elif affected_start_idx != -1:
                break

        if affected_start_idx == -1:
            if 'triggered_by_driver' in disruption.metadata and disruption.metadata['triggered_by_driver'] == driver_id:
                affected_start_idx = current_segment_start_idx
                affected_end_idx = min(current_segment_start_idx + 1, len(route_points) - 1)
                # logger.info(f"Driver {driver_id} triggered disruption {disruption.id}, forcing reroute check at current segment {affected_start_idx}.")
            else:
                # logger.info(f"No affected segment found ahead of driver {driver_id} for disruption {disruption.id}")
                return None

        if affected_end_idx == -1 and affected_start_idx != -1:
            affected_end_idx = min(affected_start_idx + look_ahead, len(route_points) - 1)

        route_end_idx_to_replace = min(affected_end_idx, len(route_points) - 1)

        if affected_start_idx >= route_end_idx_to_replace:
            # logger.warning(f"Affected start index {affected_start_idx} >= end index {route_end_idx_to_replace}.")
            return None

        # logger.info(f"Found affected portion for driver {driver_id}: {affected_start_idx} to {route_end_idx_to_replace}")
        return affected_start_idx, route_end_idx_to_replace

    def _create_tight_avoidance_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> \
    Optional[
        RerouteTightAvoidanceAction]:
        try:
            if driver_id not in state.driver_routes or 'points' not in state.driver_routes[driver_id]:
                return None

            route_points = state.driver_routes[driver_id]['points']
            if len(route_points) < 2:
                return None

            affected_portion = self._find_affected_route_portion(driver_id, route_points,
                                                                 state.driver_positions.get(driver_id), disruption)

            if affected_portion is None:
                return None

            affected_segment_start, rerouted_segment_end_idx = affected_portion

            position = state.driver_positions.get(driver_id)
            if not position:
                return None

            next_delivery_index = None
            delivery_indices = state.driver_assignments.get(driver_id, [])

            if not delivery_indices:
                return None

            for idx in delivery_indices:
                if idx < len(state.deliveries):
                    next_delivery_index = idx
                    break

            start_point = route_points[affected_segment_start]
            end_point = route_points[rerouted_segment_end_idx]

            new_route = self._find_path_avoiding_disruption_tight(
                self.G,
                start_point,
                end_point,
                disruption,
                buffer_ratio=0.8
            )

            if new_route is None:
                # logger.warning(f"Tight avoidance pathfinding failed for driver {driver_id}")
                return None

            full_route = []
            for i in range(affected_segment_start + 1):
                full_route.append(route_points[i])

            full_route.extend(new_route[1:-1])

            for i in range(rerouted_segment_end_idx, len(route_points)):
                full_route.append(route_points[i])

            reroute_action = RerouteTightAvoidanceAction(
                driver_id=driver_id,
                new_route=full_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=rerouted_segment_end_idx,
                next_delivery_index=next_delivery_index,
                delivery_indices=delivery_indices
            )

            return reroute_action

        except Exception as e:
            print(f"Error creating tight avoidance action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_wide_avoidance_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> \
    Optional[
        RerouteWideAvoidanceAction]:
        try:
            if driver_id not in state.driver_routes or 'points' not in state.driver_routes[driver_id]:
                # print(f"No route found for driver {driver_id}")
                return None

            route_points = state.driver_routes[driver_id]['points']
            if len(route_points) < 2:
                # print(f"Route too short for driver {driver_id}")
                return None

            original_radius = disruption.affected_area_radius
            expanded_radius = original_radius * 2.0

            # Temporarily increase the radius to find segments that should be widely avoided
            # NOTE: This wide avoidance check still uses the simpler segment check, not the look-ahead.
            # This could be inconsistent. Consider standardizing on _find_affected_route_portion?
            # For now, keeping original logic for wide avoidance segment finding.
            affected_segment_start_simple = None
            affected_segment_end_simple = None
            try:
                disruption.affected_area_radius = expanded_radius
                for i in range(len(route_points) - 1):
                    if self._segment_near_disruption(route_points[i], route_points[i + 1], disruption):
                        affected_segment_start_simple = i
                        affected_segment_end_simple = i + 1
                        break
            finally:
                disruption.affected_area_radius = original_radius

            if affected_segment_start_simple is None:
                # logger.info(f"No segment found near wide radius for driver {driver_id}")
                return None

            affected_portion = self._find_affected_route_portion(driver_id, route_points,
                                                                 state.driver_positions.get(driver_id), disruption)

            if affected_portion is None:
                # logger.info(f"No affected portion found for wide avoidance (look-ahead) for driver {driver_id}")
                return None

            affected_segment_start, rerouted_segment_end_idx = affected_portion

            position = state.driver_positions.get(driver_id)
            if not position:
                # print(f"No position found for driver {driver_id}")
                return None

            next_delivery_index = None
            delivery_indices = state.driver_assignments.get(driver_id, [])

            if not delivery_indices:
                # print(f"No deliveries assigned to driver {driver_id}")
                return None

            for idx in delivery_indices:
                if idx < len(state.deliveries):
                    next_delivery_index = idx
                    break

            start_point = route_points[affected_segment_start]
            end_point = route_points[rerouted_segment_end_idx]

            new_route = self._find_path_avoiding_disruption_wide(
                self.G,
                start_point,
                end_point,
                disruption,
                buffer_ratio=1.5
            )

            if new_route is None:
                # logger.warning(f"Wide avoidance pathfinding failed for driver {driver_id}")
                return None

            full_route = []
            for i in range(affected_segment_start + 1):
                full_route.append(route_points[i])

            full_route.extend(new_route[1:-1])

            for i in range(rerouted_segment_end_idx, len(route_points)):
                full_route.append(route_points[i])

            reroute_action = RerouteWideAvoidanceAction(
                driver_id=driver_id,
                new_route=full_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=rerouted_segment_end_idx,
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

            route_points = []
            if driver_id in state.driver_routes and 'points' in state.driver_routes[driver_id]:
                route_points = state.driver_routes[driver_id]['points']

            if not route_points or len(route_points) < 2:
                # logger.info(f"No detailed route found for basic reroute driver {driver_id}")
                return None

            affected_portion = self._find_affected_route_portion(driver_id, route_points, position, disruption)

            if affected_portion is None:
                # logger.info(f"No affected portion found for basic reroute for driver {driver_id}")
                return None

            affected_segment_start, rerouted_segment_end_idx = affected_portion

            start_point = route_points[affected_segment_start]
            end_point = route_points[rerouted_segment_end_idx]

            detour_points = self._find_path_avoiding_disruption(state.graph, start_point, end_point, disruption)

            if detour_points is None:
                # logger.warning(f"Basic reroute pathfinding failed for driver {driver_id}")
                return None

            new_route = []
            new_route.extend(route_points[:affected_segment_start + 1])

            if len(detour_points) > 1:
                new_route.extend(detour_points[1:-1])
            elif len(detour_points) == 1:
                if calculate_haversine_distance(new_route[-1], detour_points[0]) > 1:
                    new_route.append(detour_points[0])

            if rerouted_segment_end_idx < len(route_points):
                if calculate_haversine_distance(new_route[-1], route_points[rerouted_segment_end_idx]) > 1:
                    new_route.append(route_points[rerouted_segment_end_idx])
                new_route.extend(route_points[rerouted_segment_end_idx + 1:])

            if len(new_route) < 2:
                # logger.warning("Basic reroute resulted in route < 2 points after stitching.")
                return None

                # logger.info(f"Created basic reroute with {len(new_route)} points.")

            next_delivery_index = None
            delivery_indices = []
            if 'delivery_indices' in state.driver_routes.get(driver_id, {}):
                delivery_indices = state.driver_routes[driver_id]['delivery_indices']
                current_delivery_idx_in_list = -1
                original_route_deliveries = state.driver_routes[driver_id].get('delivery_route_indices', {})
                for i in range(affected_segment_start, rerouted_segment_end_idx + 1):
                    if i in original_route_deliveries:
                        try:
                            current_delivery_idx_in_list = delivery_indices.index(original_route_deliveries[i])
                        except ValueError:
                            pass
                if current_delivery_idx_in_list != -1 and current_delivery_idx_in_list + 1 < len(delivery_indices):
                    next_delivery_index = delivery_indices[current_delivery_idx_in_list + 1]
                elif not delivery_indices:
                    next_delivery_index = None
                elif delivery_indices:
                    next_delivery_index = delivery_indices[0]

            reroute_action = RerouteBasicAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=rerouted_segment_end_idx,
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
        try:
            disruption_point = disruption.location
            min_distance = self._point_to_segment_distance(disruption_point, start, end)

            buffer_factor = 1.5
            return min_distance <= disruption.affected_area_radius * buffer_factor
        except Exception as e:
            print(f"Error in _segment_near_disruption: {e}")
            return True

    def _point_to_segment_distance(self, point, line_start, line_end):
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
            nodes_to_remove = []
            affected_edges = 0

            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (9.0 * disruption.severity)
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass
            elif disruption.type.value == 'road_closure':
                strict_radius = float(disruption.affected_area_radius)
                for node in disruption_nodes:
                    node_data = G_mod.nodes[node]
                    if 'y' in node_data and 'x' in node_data:
                        node_point = (node_data['y'], node_data['x'])
                        distance = calculate_haversine_distance(node_point, disruption.location)
                        if distance <= strict_radius:
                            nodes_to_remove.append(node)
            else:
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass

            if nodes_to_remove:
                # print(f"Removing {len(nodes_to_remove)} nodes for road closure (basic reroute).")
                G_mod.remove_nodes_from(nodes_to_remove)

            # print(f"Modified {affected_edges} edges or removed {len(nodes_to_remove)} nodes for basic disruption avoidance")

            try:
                path_nodes = nx.shortest_path(G_mod, start_node, end_node, weight='travel_time')

                route_points = []
                for node in path_nodes:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = G_mod.nodes[node]['y']
                        lon = G_mod.nodes[node]['x']
                        route_points.append((lat, lon))
                    else:
                        # logger.warning(f"Node data missing for {node} in basic path reconstruction.")
                        return None

                if len(route_points) >= 2:
                    route_points[0] = start_point
                    route_points[-1] = end_point
                    return route_points
                else:
                    # logger.warning("Path found but too short after point reconstruction.")
                    return None

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # logger.info("No network path found for basic avoidance.")
                return None

        except Exception as e:
            # logger.error(f"Error in _find_path_avoiding_disruption: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_path_avoiding_disruption_tight(self, graph, start_point, end_point, disruption, buffer_ratio=0.8):
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
            disruption_radius = float(disruption.affected_area_radius)

            search_radius = disruption_radius * buffer_ratio * 1.1

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = calculate_haversine_distance(node_point, disruption.location)
                if distance <= search_radius:
                    disruption_nodes.append(node)

            weight_multiplier = 5.0
            nodes_to_remove = []
            affected_edges = 0

            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (4.0 * disruption.severity)
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass
            elif disruption.type.value == 'road_closure':
                strict_radius = float(disruption.affected_area_radius)
                for node in disruption_nodes:
                    node_data = G_mod.nodes[node]
                    if 'y' in node_data and 'x' in node_data:
                        node_point = (node_data['y'], node_data['x'])
                        distance = calculate_haversine_distance(node_point, disruption.location)
                        if distance <= strict_radius:
                            nodes_to_remove.append(node)
            else:
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass

            if nodes_to_remove:
                # print(f"Removing {len(nodes_to_remove)} nodes for road closure (tight).")
                G_mod.remove_nodes_from(nodes_to_remove)

            # print(f"Modified {affected_edges} edges or removed {len(nodes_to_remove)} nodes for tight disruption avoidance")

            try:
                path_nodes = nx.shortest_path(G_mod, start_node, end_node, weight='travel_time')

                route_points = []
                for node in path_nodes:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = G_mod.nodes[node]['y']
                        lon = G_mod.nodes[node]['x']
                        route_points.append((lat, lon))
                    else:
                        # logger.warning(f"Node data missing for {node} in tight path reconstruction.")
                        return None

                if len(route_points) >= 2:
                    route_points[0] = start_point
                    route_points[-1] = end_point
                    return route_points
                else:
                    # logger.warning("Tight path found but too short after point reconstruction.")
                    return None

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # logger.info("No network path found for tight avoidance.")
                return None

        except Exception as e:
            # logger.error(f"Error in _find_path_avoiding_disruption_tight: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_path_avoiding_disruption_wide(self, graph, start_point, end_point, disruption, buffer_ratio=2.0):
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
            disruption_radius = float(disruption.affected_area_radius)

            search_radius = disruption_radius * buffer_ratio

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = calculate_haversine_distance(node_point, disruption.location)
                if distance <= search_radius:
                    disruption_nodes.append(node)

            weight_multiplier = 20.0
            nodes_to_remove = []
            affected_edges = 0

            if disruption.type.value == 'traffic_jam':
                weight_multiplier = 1.0 + (19.0 * disruption.severity)
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass
            elif disruption.type.value == 'road_closure':
                strict_radius = float(disruption.affected_area_radius)
                for node in disruption_nodes:
                    node_data = G_mod.nodes[node]
                    if 'y' in node_data and 'x' in node_data:
                        node_point = (node_data['y'], node_data['x'])
                        distance = calculate_haversine_distance(node_point, disruption.location)
                        if distance <= strict_radius:
                            nodes_to_remove.append(node)
            else:
                for node in disruption_nodes:
                    for neighbor in list(G_mod.neighbors(node)):
                        if G_mod.has_edge(node, neighbor):
                            for edge_key in list(G_mod[node][neighbor].keys()):
                                if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                    try:
                                        original_time = G_mod[node][neighbor][edge_key]['travel_time']
                                        if not isinstance(original_time, (int, float)):
                                            original_time = float(original_time)

                                        G_mod[node][neighbor][edge_key][
                                            'travel_time'] = original_time * weight_multiplier
                                        affected_edges += 1
                                    except (TypeError, ValueError) as e:
                                        print(
                                            f"Warning: Could not process travel_time for edge ({node}, {neighbor}): {e}")
                                        pass

            if nodes_to_remove:
                # print(f"Removing {len(nodes_to_remove)} nodes for road closure (wide).")
                G_mod.remove_nodes_from(nodes_to_remove)

            # print(f"Modified {affected_edges} edges or removed {len(nodes_to_remove)} nodes for wide disruption avoidance")

            try:
                path_nodes = nx.shortest_path(G_mod, start_node, end_node, weight='travel_time')

                route_points = []
                for node in path_nodes:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = G_mod.nodes[node]['y']
                        lon = G_mod.nodes[node]['x']
                        route_points.append((lat, lon))
                    else:
                        # logger.warning(f"Node data missing for {node} in wide path reconstruction.")
                        return None

                if len(route_points) >= 2:
                    route_points[0] = start_point
                    route_points[-1] = end_point
                    return route_points
                else:
                    # logger.warning("Wide path found but too short after point reconstruction.")
                    return None

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # logger.info("No network path found for wide avoidance.")
                return None

        except Exception as e:
            # logger.error(f"Error in _find_path_avoiding_disruption_wide: {e}")
            import traceback
            traceback.print_exc()
            return None
