import networkx as nx
import osmnx as ox


def find_accessible_node(G, lat, lon, center_node=None, search_radius=1000):
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

    max_radius = 5000
    while search_radius <= max_radius:
        try:
            node_id = ox.nearest_nodes(G, X=lon, Y=lat)
            if node_id in G.nodes:
                if center_node is None or nx.has_path(G, node_id, center_node):
                    node = G.nodes[node_id]
                    return node_id, (node['y'], node['x'])
        except Exception as e:
            print(f"Error finding node at ({lat:.6f}, {lon:.6f}): {str(e)}")

        search_radius += 500
        print(f"Expanding search radius to {search_radius}m")

    raise ValueError(f"No accessible node found near ({lat:.6f}, {lon:.6f})")
