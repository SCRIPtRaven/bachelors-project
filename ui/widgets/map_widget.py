import osmnx as ox
from PyQt5 import QtCore, QtWidgets

from ui.map.controllers.delivery_controller import DeliveryController
from ui.map.controllers.driver_controller import DriverController
from ui.map.controllers.visualization_controller import VisualizationController
from ui.map.core.map_base import BaseMapWidget
from ui.map.utils.visualization_queue import VisualizationQueue
from ui.workers.graph_load_worker import GraphLoadWorker
from utils.geolocation import get_city_coordinates
from utils.route_color_manager import RouteColorManager


class MapWidget(BaseMapWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.route_color_manager = RouteColorManager()
        self.delivery_controller = DeliveryController(self)
        self.driver_controller = DriverController(self)
        self.visualization_controller = VisualizationController(self)

        self.visualization_queue = VisualizationQueue(self.visualization_controller)
        self.driver_controller.set_visualization_queue(self.visualization_queue)

        self.is_loading = False
        self.is_computing = False

    def set_stats_labels(self, time_label, travel_time_label, distance_label):
        self.time_label = time_label
        self.travel_time_label = travel_time_label
        self.distance_label = distance_label

    def load_graph_data(self, city_name):
        if self.is_loading:
            return

        self.is_loading = True
        self.setEnabled(False)

        self.load_worker = GraphLoadWorker(city_name)
        self.load_worker.finished.connect(self.on_graph_loaded)
        self.load_worker.start()

    def on_graph_loaded(self, success, message, graph, city_name):
        self.is_loading = False
        self.setEnabled(True)

        if success:
            self.G = graph
            self.current_city = city_name

            center, zoom = get_city_coordinates(city_name)
            self.init_map(center, zoom)
            self.load_map()

            QtWidgets.QMessageBox.information(self, "Success", message)
            self.load_completed.emit(True, message)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            self.load_completed.emit(False, message)

        self.load_worker.deleteLater()

    def find_shortest_route(self):
        if not self._validate_optimization_prerequisites():
            return

        try:
            if hasattr(self, 'optimization_thread'):
                if self.optimization_thread.isRunning():
                    self.optimization_thread.quit()
                    self.optimization_thread.wait()
                self.optimization_thread.deleteLater()

            self.optimization_thread = QtCore.QThread()

            self.visualization_controller.prepare_optimization(
                self.delivery_drivers,
                self.delivery_controller.snapped_delivery_points,
                self
                )

            self.visualization_controller.moveToThread(self.optimization_thread)
            self.optimization_thread.started.connect(
                self.visualization_controller.run_optimization
            )
            self.setEnabled(False)
            self.optimization_thread.start()

        except Exception as e:
            self.setEnabled(True)
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Route planning error: {str(e)}"
            )
            print(f"Detailed error: {e}")

    def generate_delivery_points(self, num_points):
        self.delivery_controller.generate_points(num_points)

    def generate_delivery_drivers(self, num_drivers):
        self.driver_controller.generate_drivers(num_drivers)

    def _validate_optimization_prerequisites(self):
        if (self.G is None or
                not self.delivery_controller.snapped_delivery_points or
                not self.driver_controller.delivery_drivers):
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Data",
                "Please ensure graph data, delivery points, and drivers are all loaded."
            )
            return False
        return True

    def get_main_window(self):
        """Get the MainWindow instance"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QWidget) and hasattr(parent, 'solution_switch'):
                return parent
            parent = parent.parent()
        return None

    def get_warehouse_location(self):
        """Calculate the center of the graph to place the warehouse"""
        if not self.G:
            return None

        lats = []
        lons = []
        for _, data in self.G.nodes(data=True):
            lats.append(data['y'])
            lons.append(data['x'])

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        warehouse_node = ox.nearest_nodes(self.G, X=center_lon, Y=center_lat)
        warehouse_coords = (self.G.nodes[warehouse_node]['y'], self.G.nodes[warehouse_node]['x'])

        return warehouse_coords

    @property
    def snapped_delivery_points(self):
        return self.delivery_controller.snapped_delivery_points

    @property
    def delivery_drivers(self):
        return self.driver_controller.delivery_drivers

    @property
    def selected_driver_id(self):
        return self.driver_controller.selected_driver_id

    @property
    def current_solution(self):
        return getattr(self.visualization_controller, 'current_solution', None)

    @property
    def unassigned_deliveries(self):
        return getattr(self.visualization_controller, 'unassigned_deliveries', None)

    @QtCore.pyqtSlot(object)
    def addToVisualizationQueue(self, data):
        """Safe method to add items to visualization queue from other threads"""
        self.visualization_queue.append(data)
