import math
from functools import lru_cache

import networkx as nx
import osmnx as ox


@lru_cache(maxsize=32)
def get_city_coordinates(city_name):
    try:
        location = ox.geocode(city_name)
        zoom = calculate_zoom_level(city_name)
        return (location[0], location[1]), zoom
    except Exception as e:
        print(f"Warning: Could not geocode {city_name}. Using default coordinates. Error: {e}")
        return (54.8985, 23.9036), 12


def calculate_zoom_level(city_name):
    try:
        gdf = ox.geocode_to_gdf(city_name)

        bounds = gdf.total_bounds
        width = abs(bounds[2] - bounds[0])

        if width > 0.5:  # Large city
            return 11
        elif width > 0.2:  # Medium city
            return 12
        else:  # Small city
            return 13
    except Exception as e:
        print(f"Warning: Could not calculate zoom level for {city_name}. Using default. Error: {e}")
        return 12


def find_accessible_node(G, lat, lon, center_node=None, search_radius=1000):
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

    max_radius = 5000
    while search_radius <= max_radius:
        try:
            node_id = ox.nearest_nodes(G, X=float(lon), Y=float(lat))
            if node_id in G.nodes:
                if center_node is None or nx.has_path(G, node_id, center_node):
                    node = G.nodes[node_id]
                    return node_id, (node['y'], node['x'])
        except Exception as e:
            print(f"Error finding node at ({lat:.6f}, {lon:.6f}): {str(e)}")

        search_radius += 500
        print(f"Expanding search radius to {search_radius}m")

    raise ValueError(f"No accessible node found near ({lat:.6f}, {lon:.6f})")


def calculate_haversine_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2

    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth radius in meters

    return c * r
