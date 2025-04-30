from colorsys import hsv_to_rgb
import math
import networkx as nx
import osmnx as ox
from typing import List, Tuple, Optional

from config.config import RouteConfig
from utils.geo_utils import calculate_haversine_distance


class RouteColorManager:
    """
    Advanced color management system that ensures maximum visual distinction between routes,
    especially for neighboring routes, using perceptual color spacing and contrast optimization.
    """

    def __init__(self):
        self.golden_ratio = RouteConfig.GOLDEN_RATIO
        self.base_hues = RouteConfig.BASE_HUES
        self.saturation_levels = RouteConfig.SATURATION_LEVELS
        self.brightness_levels = RouteConfig.BRIGHTNESS_LEVELS
        self.patterns = RouteConfig.LINE_PATTERNS

        self.style_cache = {}
        self.used_combinations = set()

    def get_route_style(self, index, total_routes):
        """
        Generates a visually distinct style for each route, ensuring neighboring routes
        are easily distinguishable.
        """
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
    """
    Calculate the total length of a route in meters
    
    Args:
        route: List of (latitude, longitude) points
        
    Returns:
        Total route length in meters
    """
    if not route or len(route) < 2:
        return 0.0
        
    total_length = 0.0
    for i in range(len(route) - 1):
        start_point = route[i]
        end_point = route[i + 1]
        segment_length = calculate_haversine_distance(start_point, end_point)
        total_length += segment_length
        
    return total_length


def calculate_travel_time(route: List[Tuple[float, float]], graph: Optional[nx.Graph] = None) -> float:
    """
    Calculate the estimated travel time for a route in seconds
    
    Args:
        route: List of (latitude, longitude) points
        graph: Road network graph (optional, used if available for more accurate estimates)
        
    Returns:
        Estimated travel time in seconds
    """
    if not route or len(route) < 2:
        return 0.0
    
    # If graph is provided, try to use it for more accurate travel time
    if graph is not None:
        try:
            # Try to map route points to graph nodes
            total_time = 0.0
            
            for i in range(len(route) - 1):
                start_point = route[i]
                end_point = route[i + 1]
                
                # Get nearest nodes in graph
                start_node = None
                end_node = None
                
                # Try to find nodes directly
                try:
                    start_node = ox.nearest_nodes(graph, X=start_point[1], Y=start_point[0])
                    end_node = ox.nearest_nodes(graph, X=end_point[1], Y=end_point[0])
                except Exception:
                    pass
                
                # If nodes found, calculate path and time
                if start_node is not None and end_node is not None:
                    try:
                        path = nx.shortest_path(graph, start_node, end_node, weight='travel_time')
                        if path:
                            # Sum travel time along path
                            segment_time = 0.0
                            for j in range(len(path) - 1):
                                u, v = path[j], path[j + 1]
                                edge_data = graph.get_edge_data(u, v, 0)  # Get edge data
                                if edge_data and 'travel_time' in edge_data:
                                    segment_time += edge_data['travel_time']
                                else:
                                    # Fallback to distance / average speed
                                    if 'length' in edge_data:
                                        # Assume 30 km/h average speed
                                        segment_time += edge_data['length'] / (30 * 1000 / 3600)
                            
                            total_time += segment_time
                            continue
                    except Exception:
                        pass
                
                # Fallback to simple distance-based estimate if graph calculation fails
                distance = calculate_haversine_distance(start_point, end_point)
                # Assume average speed of 30 km/h for driving (~8.33 m/s)
                segment_time = distance / 8.33
                total_time += segment_time
            
            return total_time
            
        except Exception:
            # Fallback to simple calculation below if anything fails
            pass
    
    # Simple calculation based on straight-line distances and average speed
    total_length = calculate_route_length(route)
    
    # Assume average speed of 30 km/h for driving (~8.33 m/s)
    average_speed = 8.33  # m/s
    
    return total_length / average_speed
