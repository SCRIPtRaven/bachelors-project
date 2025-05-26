import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from typing import List, Tuple, Optional, Dict, Any

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import find_closest_point_index_on_route

class MLFeatureExtractor:
    def __init__(self, graph: nx.MultiDiGraph, travel_time_matrix: Optional[Dict[Tuple[int, int], float]] = None):
        self.graph = graph
        self.travel_time_matrix = travel_time_matrix
        self._node_pos_cache = {}

    def set_travel_time_matrix(self, ttm: Optional[Dict[Tuple[int, int], float]]):
        self.travel_time_matrix = ttm

    def _get_node_coordinates(self, node_id: int) -> Optional[Tuple[float, float]]:
        if node_id in self._node_pos_cache:
            return self._node_pos_cache[node_id]
        if self.graph and node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if 'y' in node_data and 'x' in node_data:
                coords = (float(node_data['y']), float(node_data['x']))
                self._node_pos_cache[node_id] = coords
                return coords
        return None

    def _get_current_route_segment_and_progress(
        self,
        driver_id: int,
        state: DeliverySystemState
    ) -> Tuple[Optional[int], float, List[Tuple[float, float]], List[int], List[float]]:
        driver_pos = state.driver_positions.get(driver_id)
        route_data = state.driver_routes.get(driver_id)

        if not driver_pos or not route_data:
            return None, 0.0, [], [], []

        route_points = route_data.get('points', [])
        route_nodes = route_data.get('nodes', [])
        route_segment_times = route_data.get('times', [])

        if not route_points or len(route_points) < 2:
            return None, 0.0, route_points, route_nodes, route_segment_times
        
        current_segment_idx = find_closest_point_index_on_route(route_points, driver_pos)
        if current_segment_idx == -1 or current_segment_idx >= len(route_points) -1:
             current_segment_idx = len(route_points) - 2 if len(route_points) >=2 else 0

        segment_start_coord = route_points[current_segment_idx]
        segment_end_coord = route_points[current_segment_idx + 1]
        
        dist_to_segment_start = calculate_haversine_distance(driver_pos, segment_start_coord)
        segment_length = calculate_haversine_distance(segment_start_coord, segment_end_coord)
        
        progress_on_segment = 0.0
        if segment_length > 1e-6:
            vec_segment = (segment_end_coord[0] - segment_start_coord[0], segment_end_coord[1] - segment_start_coord[1])
            vec_driver = (driver_pos[0] - segment_start_coord[0], driver_pos[1] - segment_start_coord[1])
            dot_product = vec_segment[0]*vec_driver[0] + vec_segment[1]*vec_driver[1]
            progress_on_segment = dot_product / (segment_length * segment_length)
            progress_on_segment = max(0.0, min(1.0, progress_on_segment))

        return current_segment_idx, progress_on_segment, route_points, route_nodes, route_segment_times


    def _get_distance_along_route_to_idx(
        self,
        current_segment_idx: int,
        progress_on_current_segment: float,
        target_point_idx_on_route_points: int,
        route_segment_times: List[float],
        route_points: List[Tuple[float,float]]
    ) -> float:
        if current_segment_idx >= target_point_idx_on_route_points:
            return 0.0

        total_dist_time = 0.0

        if current_segment_idx < len(route_segment_times):
             total_dist_time += route_segment_times[current_segment_idx] * (1.0 - progress_on_current_segment)
        
        for i in range(current_segment_idx + 1, target_point_idx_on_route_points):
            if i < len(route_segment_times):
                total_dist_time += route_segment_times[i]
            else:
                if i < len(route_points) -1:
                    total_dist_time += calculate_haversine_distance(route_points[i], route_points[i+1]) / 10
                else:
                    total_dist_time += 60

        return total_dist_time

    def _find_point_on_route_entering_disruption(
        self,
        route_points: List[Tuple[float, float]],
        start_search_idx_on_route_points: int,
        disruption: Disruption
    ) -> Optional[int]:
        for i in range(start_search_idx_on_route_points, len(route_points)):
            point_coord = route_points[i]
            dist_to_disruption_center = calculate_haversine_distance(point_coord, disruption.location)
            if dist_to_disruption_center <= disruption.affected_area_radius:
                return i
        return None

    def _get_next_delivery_on_route(
        self,
        driver_id: int,
        state: DeliverySystemState,
        current_point_idx_on_route_points: int
    ) -> Optional[int]:
        route_data = state.driver_routes.get(driver_id)
        if not route_data:
            return None
        
        delivery_indices_on_route = route_data.get('delivery_indices', []) 
        
        for delivery_idx_on_path in sorted(delivery_indices_on_route):
            if delivery_idx_on_path > current_point_idx_on_route_points:
                original_assignment = route_data.get('assignment')
                if original_assignment:
                    return delivery_idx_on_path
        return None


    def _get_nodes_in_radius(self, center_location: Tuple[float, float], radius: float, graph: nx.MultiDiGraph) -> List[int]:
        nodes_in_radius = []
        if not graph: return nodes_in_radius
        for node, data in graph.nodes(data=True):
            if 'y' in data and 'x' in data:
                try:
                    node_lat, node_lon = float(data['y']), float(data['x'])
                    if calculate_haversine_distance((node_lat, node_lon), center_location) <= radius:
                        nodes_in_radius.append(node)
                except (ValueError, TypeError):
                    continue
        return nodes_in_radius


    def extract_features(self, driver_id: int, disruption: Disruption, state: DeliverySystemState, graph: nx.MultiDiGraph) -> Optional[pd.DataFrame]:
        features = {}

        features['disruption_type_road_closure'] = 1 if disruption.type == DisruptionType.ROAD_CLOSURE else 0
        features['disruption_type_traffic_jam'] = 1 if disruption.type == DisruptionType.TRAFFIC_JAM else 0

        features['disruption_severity'] = disruption.severity

        driver_pos = state.driver_positions.get(driver_id)
        if not driver_pos: return None

        current_segment_idx_on_points, progress_on_segment, route_points, route_nodes, route_segment_times = \
            self._get_current_route_segment_and_progress(driver_id, state)

        if current_segment_idx_on_points is None or not route_points:
            features['distance_to_disruption_center'] = calculate_haversine_distance(driver_pos, disruption.location)
            features['remaining_deliveries'] = 0
            features['distance_along_route_to_disruption'] = 99999
            features['distance_to_next_delivery_along_route'] = 99999
            features['next_delivery_before_disruption'] = 0
            features['urban_density'] = len(self._get_nodes_in_radius(disruption.location, 500, graph)) / (np.pi * (0.5**2))
            features['alternative_route_density'] = features['urban_density'] * 0.5
            return pd.DataFrame([features])


        features['distance_to_disruption_center'] = calculate_haversine_distance(driver_pos, disruption.location)

        driver_assignment = state.driver_routes.get(driver_id, {}).get('assignment')
        remaining_deliveries_count = 0
        if driver_assignment:
            for delivery_idx_orig in driver_assignment.delivery_indices:
                if delivery_idx_orig not in state.completed_deliveries and delivery_idx_orig not in state.skipped_deliveries:
                    remaining_deliveries_count += 1
        features['remaining_deliveries'] = remaining_deliveries_count
        
        effective_current_point_idx = current_segment_idx_on_points 
        if progress_on_segment > 0.5 and current_segment_idx_on_points < len(route_points) -1:
             effective_current_point_idx = current_segment_idx_on_points + 1


        disruption_entry_idx_on_route = self._find_point_on_route_entering_disruption(
            route_points, effective_current_point_idx, disruption
        )
        
        dist_to_disruption_along_route = 99999.0
        if disruption_entry_idx_on_route is not None:
            dist_to_disruption_along_route = self._get_distance_along_route_to_idx(
                current_segment_idx_on_points, progress_on_segment, disruption_entry_idx_on_route, route_segment_times, route_points
            )
        features['distance_along_route_to_disruption'] = dist_to_disruption_along_route

        next_delivery_idx_on_route = self._get_next_delivery_on_route(driver_id, state, effective_current_point_idx)
        
        dist_to_next_delivery_along_route = 99999.0
        if next_delivery_idx_on_route is not None:
            dist_to_next_delivery_along_route = self._get_distance_along_route_to_idx(
                current_segment_idx_on_points, progress_on_segment, next_delivery_idx_on_route, route_segment_times, route_points
            )
        features['distance_to_next_delivery_along_route'] = dist_to_next_delivery_along_route

        next_delivery_is_before = 0
        if next_delivery_idx_on_route is not None and disruption_entry_idx_on_route is not None:
            if next_delivery_idx_on_route < disruption_entry_idx_on_route:
                next_delivery_is_before = 1
        elif next_delivery_idx_on_route is not None and disruption_entry_idx_on_route is None:
            next_delivery_is_before = 1
            
        features['next_delivery_before_disruption'] = next_delivery_is_before

        nodes_near_disruption = self._get_nodes_in_radius(disruption.location, 500, graph)
        area_km_sq = np.pi * (0.5**2) 
        features['urban_density'] = len(nodes_near_disruption) / area_km_sq if area_km_sq > 0 else 0

        features['alternative_route_density'] = features['urban_density'] * 0.75

        return pd.DataFrame([features])


