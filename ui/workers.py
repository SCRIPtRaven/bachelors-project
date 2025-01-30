import networkx as nx
from PyQt5 import QtCore

from data.graph_manager import download_and_save_graph
from logic.routing import compute_shortest_route


class DownloadGraphWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, str)

    def run(self):
        try:
            download_and_save_graph()
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, f"An error occurred while downloading data:\n{e}")


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

            # On success
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
