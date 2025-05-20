import os

from PyQt5 import QtCore

from config.config import PathsConfig
from models.services.graph_service import (download_and_save_graph, get_largest_connected_component,
                                           load_graph, update_travel_times_from_csv)


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
                graph = load_graph(filename=graph_path)
                graph = get_largest_connected_component(graph)

                if not hasattr(graph, 'graph') or not isinstance(graph.graph, dict):
                    graph.graph = {}
                if 'name' not in graph.graph or graph.graph['name'] != self.city_name:
                    graph.graph['name'] = self.city_name

                if os.path.isfile(travel_times_path):
                    update_travel_times_from_csv(graph, travel_times_path)

                self.finished.emit(True, "Graph loaded successfully", graph, self.city_name)

            except FileNotFoundError:
                success = download_and_save_graph(self.city_name)
                if success:
                    graph = load_graph(filename=graph_path)
                    graph = get_largest_connected_component(graph)
                    if not hasattr(graph, 'graph') or not isinstance(graph.graph, dict):
                        graph.graph = {}
                    if 'name' not in graph.graph or graph.graph['name'] != self.city_name:
                        graph.graph['name'] = self.city_name
                    self.finished.emit(True, "Graph downloaded and loaded successfully", graph,
                                       self.city_name)
                else:
                    self.finished.emit(False, "Failed to download graph", None, self.city_name)

        except Exception as e:
            self.finished.emit(False, str(e), None, self.city_name)
