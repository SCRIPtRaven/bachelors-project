import networkx as nx
import numpy as np
import osmnx as ox
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def create_distance_matrix(G, points):
    """
    Creates a distance matrix for the OR-Tools solver using our graph.

    Args:
        G: NetworkX graph containing our road network
        points: List of (lat, lon) coordinates

    Returns:
        2D numpy array representing distances between all points
    """
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                origin = points[i]
                destination = points[j]
                try:
                    orig_node = ox.nearest_nodes(G, X=origin[1], Y=origin[0])
                    dest_node = ox.nearest_nodes(G, X=destination[1], Y=destination[0])
                    distance = nx.shortest_path_length(
                        G,
                        orig_node,
                        dest_node,
                        weight='travel_time'
                    )
                    distance_matrix[i][j] = int(distance * 100)
                except:
                    distance_matrix[i][j] = 1000000

    return distance_matrix


def solve_tsp(G, points):
    """
    Solves the TSP problem using Google OR-Tools.

    Args:
        G: NetworkX graph containing our road network
        points: List of (lat, lon) coordinates where first point is depot

    Returns:
        ordered_points: List of points in the order they should be visited
        total_time: Total travel time in seconds
    """
    distance_matrix = create_distance_matrix(G, points)

    manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 5  # Reasonable timeout

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        ordered_indices = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            ordered_indices.append(node_index)
            index = solution.Value(routing.NextVar(index))
        ordered_indices.append(0)

        ordered_points = [points[i] for i in ordered_indices]
        total_time = solution.ObjectiveValue() / 100

        return ordered_points, total_time
    else:
        raise Exception("No solution found for TSP")
