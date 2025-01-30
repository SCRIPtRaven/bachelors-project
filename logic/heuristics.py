import math


def euclidean_heuristic(G, u, v):
    """
    Simple Euclidean-distance heuristic in meters.
    G: a NetworkX Graph with node attributes x, y
    u, v: node IDs in G
    """
    lat1 = G.nodes[u]['y']
    lon1 = G.nodes[u]['x']
    lat2 = G.nodes[v]['y']
    lon2 = G.nodes[v]['x']

    R = 6371000
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    lon1_rad, lon2_rad = math.radians(lon1), math.radians(lon2)
    x = (lon2_rad - lon1_rad) * math.cos((lat1_rad + lat2_rad) / 2)
    y = lat2_rad - lat1_rad
    return math.sqrt(x * x + y * y) * R
