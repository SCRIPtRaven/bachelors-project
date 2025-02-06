import time

import networkx as nx
import osmnx as ox

from logic.tsp_solver import solve_tsp
from utils.distance_utils import haversine_distance


def compute_shortest_route(G, origin_latlng, destination_latlng):
    """
    Compute a single shortest path in G from origin to destination,
    returns (route_nodes, total_travel_time, total_distance, computation_time, cumulative_times, cumulative_distances).
    """
    start_time = time.time()
    orig_node = ox.nearest_nodes(G, X=origin_latlng[1], Y=origin_latlng[0])
    dest_node = ox.nearest_nodes(G, X=destination_latlng[1], Y=destination_latlng[0])

    route = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')
    total_travel_time = nx.shortest_path_length(G, orig_node, dest_node, weight='travel_time')

    computation_time = time.time() - start_time

    route_nodes = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

    cumulative_times = [0]
    cumulative_distances = [0]
    cumulative_time = 0
    cumulative_distance = 0
    total_distance = 0

    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if isinstance(edge_data, dict):
            edge_data = edge_data[list(edge_data.keys())[0]]

        travel_time = edge_data.get('travel_time', 0)
        length = edge_data.get('length', 0)

        cumulative_time += travel_time
        cumulative_distance += length

        cumulative_times.append(cumulative_time)
        cumulative_distances.append(cumulative_distance)

        total_distance += length

    return (route_nodes,
            total_travel_time,
            total_distance,
            computation_time,
            cumulative_times,
            cumulative_distances)


def find_tsp_route(G, delivery_points, center=None):
    """
    Find the optimal TSP route using OR-Tools solver.

    This function takes a road network graph and a set of delivery points, then finds
    the most efficient route that visits all points and returns to the starting location.
    It uses the Google OR-Tools library for solving the TSP and detailed path planning
    through the road network.

    Args:
        G: NetworkX graph of the road network
        delivery_points: List of (lat, lon) tuples for delivery locations
        center: Tuple of (lat, lon) for the starting/ending point

    Returns:
        route_coords: List of (lat, lon) coordinates forming the complete route
        total_travel_time: Total time in seconds
        total_distance: Total distance in meters
        computation_time: Time taken to compute the route
        snapped_nodes: List of node IDs in the route order
    """
    start_time = time.time()

    if center is None:
        lats = [data['y'] for _, data in G.nodes(data=True)]
        lons = [data['x'] for _, data in G.nodes(data=True)]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

    all_points = [center] + delivery_points

    ordered_points, total_travel_time = solve_tsp(G, all_points)

    route_coords = []
    snapped_nodes = []
    total_distance = 0

    for i in range(len(ordered_points) - 1):
        start = ordered_points[i]
        end = ordered_points[i + 1]

        start_node = ox.nearest_nodes(G, X=start[1], Y=start[0])
        end_node = ox.nearest_nodes(G, X=end[1], Y=end[0])
        snapped_nodes.append(start_node)

        path = nx.shortest_path(G, start_node, end_node, weight='travel_time')

        for node in path[:-1]:
            route_coords.append((G.nodes[node]['y'], G.nodes[node]['x']))
            if len(route_coords) > 1:
                last = route_coords[-2]
                current = route_coords[-1]
                total_distance += haversine_distance(
                    last[0], last[1],
                    current[0], current[1]
                )

    last_node = ox.nearest_nodes(G, X=ordered_points[-1][1], Y=ordered_points[-1][0])
    snapped_nodes.append(last_node)
    route_coords.append((G.nodes[last_node]['y'], G.nodes[last_node]['x']))

    computation_time = time.time() - start_time

    return route_coords, total_travel_time, total_distance, computation_time, snapped_nodes


def get_largest_connected_component(G):
    """
    Returns the largest strongly connected component of the graph.

    In a road network, a strongly connected component is a subset of nodes where
    every node can be reached from every other node. This is important for
    ensuring valid routes can be found between any two points.

    Args:
        G: NetworkX graph representing the road network

    Returns:
        NetworkX graph containing only the largest connected component
    """
    if G.is_directed():
        connected_components = list(nx.strongly_connected_components(G))
    else:
        connected_components = list(nx.connected_components(G))

    largest_component = max(connected_components, key=len)

    return G.subgraph(largest_component).copy()
