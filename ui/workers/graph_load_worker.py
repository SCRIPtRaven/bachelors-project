import os

from PyQt5 import QtCore

from config.paths import get_graph_file_path, get_travel_times_path
from data import graph_manager
from data.graph_manager import download_and_save_graph
from logic.routing import get_largest_connected_component


class GraphLoadWorker(QtCore.QThread):
    """
    Worker thread for loading graph data asynchronously.
    Handles both loading existing graphs and downloading new ones.
    """
    finished = QtCore.pyqtSignal(bool, str, object, str)

    def __init__(self, city_name):
        super().__init__()
        self.city_name = city_name

    def run(self):
        try:
            graph_path = get_graph_file_path(self.city_name)
            travel_times_path = get_travel_times_path(self.city_name)

            try:
                G = graph_manager.load_graph(filename=graph_path)
                G = get_largest_connected_component(G)

                if os.path.isfile(travel_times_path):
                    graph_manager.update_travel_times_from_csv(G, travel_times_path)

                self.finished.emit(True, "Graph loaded successfully", G, self.city_name)

            except FileNotFoundError:
                success = download_and_save_graph(self.city_name)
                if success:
                    G = graph_manager.load_graph(filename=graph_path)
                    G = get_largest_connected_component(G)
                    self.finished.emit(True, "Graph downloaded and loaded successfully", G, self.city_name)
                else:
                    self.finished.emit(False, "Failed to download graph", None, self.city_name)

        except Exception as e:
            self.finished.emit(False, str(e), None, self.city_name)
