import os

from PyQt5 import QtCore

from config.config import PathsConfig
from models.services import graph
from models.services.graph import download_and_save_graph, get_largest_connected_component


class GraphLoadWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, str, object, str)

    def __init__(self, city_name):
        super().__init__()
        self.city_name = city_name

    def run(self):
        try:
            graph_path = PathsConfig.get_graph_file_path(self.city_name)
            travel_times_path = PathsConfig.get_travel_times_path(self.city_name)

            try:
                G = graph.load_graph(filename=graph_path)
                G = get_largest_connected_component(G)

                # Ensure G.graph['name'] is set
                if not hasattr(G, 'graph') or not isinstance(G.graph, dict):
                    G.graph = {}
                if 'name' not in G.graph or G.graph['name'] != self.city_name:
                    G.graph['name'] = self.city_name

                if os.path.isfile(travel_times_path):
                    graph.update_travel_times_from_csv(G, travel_times_path)

                self.finished.emit(True, "Graph loaded successfully", G, self.city_name)

            except FileNotFoundError:
                success = download_and_save_graph(self.city_name)
                if success:
                    G = graph.load_graph(filename=graph_path)
                    G = get_largest_connected_component(G)
                    # Ensure G.graph['name'] is set
                    if not hasattr(G, 'graph') or not isinstance(G.graph, dict):
                        G.graph = {}
                    if 'name' not in G.graph or G.graph['name'] != self.city_name:
                        G.graph['name'] = self.city_name
                    self.finished.emit(True, "Graph downloaded and loaded successfully", G, self.city_name)
                else:
                    self.finished.emit(False, "Failed to download graph", None, self.city_name)

        except Exception as e:
            self.finished.emit(False, str(e), None, self.city_name)
