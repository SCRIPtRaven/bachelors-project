from colorsys import hsv_to_rgb
from typing import List, Tuple, Optional

import networkx as nx
import osmnx as ox

from config.config import RouteConfig
from models.entities.disruption import Disruption, DisruptionType
from utils.geo_utils import calculate_haversine_distance


class RouteColorManager:

    def __init__(self):
        self.golden_ratio = RouteConfig.GOLDEN_RATIO
        self.base_hues = RouteConfig.BASE_HUES
        self.saturation_levels = RouteConfig.SATURATION_LEVELS
        self.brightness_levels = RouteConfig.BRIGHTNESS_LEVELS
        self.patterns = RouteConfig.LINE_PATTERNS

        self.style_cache = {}
        self.used_combinations = set()

    def get_route_style(self, index, total_routes):
        cache_key = (index, total_routes)
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]

        hue_index = int((index * self.golden_ratio) * len(self.base_hues)) % len(self.base_hues)
        base_hue = self.base_hues[hue_index]

        sat_index = (index // len(self.base_hues)) % len(self.saturation_levels)
        bright_index = (index // (len(self.base_hues) * len(self.saturation_levels))) % len(self.brightness_levels)

        pattern_index = index % len(self.patterns)

        hsv_color = (
            base_hue,
            self.saturation_levels[sat_index],
            self.brightness_levels[bright_index]
        )

        rgb_color = hsv_to_rgb(*hsv_color)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255)
        )

        pattern = self.patterns[pattern_index]

        style = {
            'color': hex_color,
            'weight': pattern['weight'],
            'dash_array': pattern['dash_array'],
            'opacity': 0.8 if pattern['type'] != 'solid' else 0.9
        }

        self.style_cache[cache_key] = style
        return style


def calculate_route_length(route: List[Tuple[float, float]]) -> float:
    if not route or len(route) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(route) - 1):
        start_point = route[i]
        end_point = route[i + 1]
        segment_length = calculate_haversine_distance(start_point, end_point)
        total_length += segment_length

    return total_length


def calculate_travel_time(route: List[Tuple[float, float]], graph: Optional[nx.Graph] = None,
                          disruption: Optional[Disruption] = None) -> float:
    if not route or len(route) < 2:
        return 0.0

    if graph is not None and disruption is not None and disruption.type == DisruptionType.ROAD_CLOSURE:
        disruption_radius = disruption.affected_area_radius
        for point in route:
            if calculate_haversine_distance(point, disruption.location) <= disruption_radius:
                return float('inf')

    if graph is not None:
        total_time = 0.0
        for i in range(len(route) - 1):
            start_point = route[i]
            end_point = route[i + 1]

            try:
                start_node = ox.nearest_nodes(graph, X=start_point[1], Y=start_point[0])
                end_node = ox.nearest_nodes(graph, X=end_point[1], Y=end_point[0])

                if start_node not in graph or end_node not in graph:
                    return float('inf')

                segment_time = nx.shortest_path_length(graph, source=start_node, target=end_node, weight='travel_time')
                total_time += segment_time

            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return float('inf')
            except Exception as e:
                return float('inf')

        return total_time

    total_length = calculate_route_length(route)
    average_speed = 8.33
    if average_speed <= 0: return float('inf')
    return total_length / average_speed


def get_distance_along_route(route_points: List[Tuple[float, float]], target_index: int) -> float:
    if not route_points or target_index <= 0:
        return 0.0

    return calculate_route_length(route_points[:target_index + 1])


def find_closest_point_index_on_route(route_points: List[Tuple[float, float]], location: Tuple[float, float]) -> int:
    if not route_points:
        return -1

    min_dist = float('inf')
    closest_idx = -1
    for i, point in enumerate(route_points):
        dist = calculate_haversine_distance(point, location)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx


def find_route_enter_disruption_index(route_points: List[Tuple[float, float]], disruption_location: Tuple[float, float],
                                      disruption_radius: float, start_index: int = 0) -> int:
    if not route_points:
        return -1

    for i in range(start_index, len(route_points)):
        if calculate_haversine_distance(route_points[i], disruption_location) <= disruption_radius:
            return i
    return -1
