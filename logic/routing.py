import time

import networkx as nx
import osmnx as ox

from logic.heuristics import euclidean_heuristic


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

    # Convert route nodes to (lat, lon)
    route_nodes = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

    cumulative_times = [0]
    cumulative_distances = [0]
    cumulative_time = 0
    cumulative_distance = 0
    total_distance = 0

    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        # in case there's parallel edges
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


def find_tsp_route(G, delivery_points, center=(54.8985, 23.9036)):
    """
    Solve the TSP for all delivery_points plus the city center on G using A* for pairwise paths.
    Returns (final_route_coords, total_travel_time, total_distance, compute_time, snapped_nodes).
    """

    # Insert center as first point
    delivery_points.insert(0, center)

    # Snap each lat-lon to graph node
    snapped_nodes = []
    for lat, lon in delivery_points:
        snapped = ox.nearest_nodes(G, X=lon, Y=lat)
        snapped_nodes.append(snapped)

    # Build all-pairs A* travel_time
    node_count = len(snapped_nodes)
    T = {}
    all_paths = {}

    for i in range(node_count):
        T[i] = {}
        all_paths[i] = {}
        for j in range(node_count):
            if i == j:
                T[i][j] = 0
                all_paths[i][j] = [snapped_nodes[i]]
            else:
                try:
                    path = nx.astar_path(
                        G=G,
                        source=snapped_nodes[i],
                        target=snapped_nodes[j],
                        heuristic=lambda u, v: euclidean_heuristic(G, u, v),
                        weight='travel_time'
                    )
                    travel_time_sum = 0
                    for u, v in zip(path[:-1], path[1:]):
                        edge_data = G.get_edge_data(u, v)
                        edge_data = edge_data[list(edge_data.keys())[0]]
                        travel_time_sum += edge_data.get('travel_time', 999999)
                    T[i][j] = travel_time_sum
                    all_paths[i][j] = path
                except nx.NetworkXNoPath:
                    T[i][j] = float('inf')
                    all_paths[i][j] = []

    # Build TSP graph
    tsp_graph = nx.DiGraph()
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                cost = T[i][j]
                tsp_graph.add_edge(i, j, weight=cost)

    # Solve TSP (approx)
    t0 = time.time()
    tsp_node_path = nx.approximation.traveling_salesman_problem(
        tsp_graph, cycle=True, weight='weight'
    )
    compute_time = time.time() - t0

    total_distance = 0.0
    total_travel_time = 0.0
    final_path_nodes = []

    for idx in range(len(tsp_node_path) - 1):
        i = tsp_node_path[idx]
        j = tsp_node_path[idx + 1]
        segment = all_paths[i][j]
        final_path_nodes.extend(segment[:-1])

        for u, v in zip(segment[:-1], segment[1:]):
            e_data = G.get_edge_data(u, v)
            e_data = e_data[list(e_data.keys())[0]]
            dist = e_data.get('length', 0)
            ttime = e_data.get('travel_time', 0)
            total_distance += dist
            total_travel_time += ttime

    # Add the last node
    if tsp_node_path and len(tsp_node_path) > 1:
        last_idx = tsp_node_path[-1]
        last_segment = all_paths[tsp_node_path[-2]][last_idx]
        if last_segment:
            final_path_nodes.append(last_segment[-1])

    # Convert nodes to lat-lon for drawing on Folium
    final_route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in final_path_nodes]

    return final_route_coords, total_travel_time, total_distance, compute_time, snapped_nodes
