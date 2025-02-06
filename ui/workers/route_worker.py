import networkx as nx
from PyQt5 import QtCore

from logic.routing import compute_shortest_route


class ComputeRouteWorker(QtCore.QThread):
    """
    This worker calls logic.routing.compute_shortest_route in a background thread.
    """
    finished = QtCore.pyqtSignal(
        object,  # route_nodes
        float,  # total_travel_time
        float,  # total_distance
        float,  # computation_time
        bool,  # success
        str  # error_message
    )

    def __init__(self, G, origin, destination):
        super().__init__()
        self.G = G
        self.origin = origin
        self.destination = destination

    def run(self):
        try:
            (route_nodes,
             total_travel_time,
             total_distance,
             computation_time,
             cumulative_times,
             cumulative_distances) = compute_shortest_route(self.G, self.origin, self.destination)

            self.finished.emit(
                (route_nodes, cumulative_times, cumulative_distances),
                total_travel_time,
                total_distance,
                computation_time,
                True,
                ""
            )
        except nx.NetworkXNoPath:
            self.finished.emit(None, 0, 0, 0, False, "No path could be found between the selected points.")
        except Exception as e:
            self.finished.emit(None, 0, 0, 0, False, f"An error occurred while computing the route:\n{e}")
