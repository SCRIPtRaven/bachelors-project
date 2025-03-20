import osmnx as ox
from PyQt5 import QtCore

from models.services.geolocation import GeolocationService
from utils.geo_utils import find_accessible_node
from viewmodels.viewmodel_messenger import MessageType


class DeliveryViewModel(QtCore.QObject):
    delivery_points_generated = QtCore.pyqtSignal(list)
    delivery_points_processed = QtCore.pyqtSignal(list, int, int)
    request_show_message = QtCore.pyqtSignal(str, str, str)
    warehouse_location_changed = QtCore.pyqtSignal(tuple)

    def __init__(self, messenger=None):
        super().__init__()
        self.snapped_delivery_points = []
        self.messenger = messenger
        self._graph = None

        if self.messenger:
            self.messenger.subscribe(MessageType.GRAPH_LOADED, self.handle_graph_loaded)

    def set_graph(self, graph):
        """Set the graph data"""
        self._graph = graph

        if self.messenger:
            self.messenger.send(MessageType.GRAPH_UPDATED, {'graph': self._graph})

        warehouse_location = self.get_warehouse_location()
        if warehouse_location:
            self.warehouse_location_changed.emit(warehouse_location)
            if self.messenger:
                self.messenger.send(MessageType.WAREHOUSE_LOCATION_UPDATED,
                                    {'location': warehouse_location})

    def generate_points(self, num_points):
        """Generate delivery points"""
        if self._graph is None:
            self.request_show_message.emit(
                "Graph Not Loaded",
                "Please load the graph data first.",
                "warning"
            )
            return

        try:
            node_coords = [(data['y'], data['x'])
                           for _, data in self._graph.nodes(data=True)]
            bounds = (
                min(lat for lat, _ in node_coords),
                max(lat for lat, _ in node_coords),
                min(lon for _, lon in node_coords),
                max(lon for _, lon in node_coords)
            )

            delivery_points = GeolocationService.generate_delivery_points(bounds, num_points)
            self._process_delivery_points(delivery_points)

            if self.messenger:
                self.messenger.send(MessageType.DELIVERY_POINTS_UPDATED,
                                    {'points': self.snapped_delivery_points})

        except Exception as e:
            self.request_show_message.emit(
                "Error",
                f"Error generating deliveries: {str(e)}",
                "critical"
            )
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()

    def _process_delivery_points(self, delivery_points):
        """Process and snap delivery points to the road network"""
        self.snapped_delivery_points = []
        successful_points = 0
        skipped_points = 0

        for point in delivery_points:
            try:
                lat, lon = point.coordinates
                node_id, (snapped_lat, snapped_lon) = find_accessible_node(
                    self._graph, lat, lon
                )

                self.snapped_delivery_points.append(
                    (snapped_lat, snapped_lon, point.weight, point.volume)
                )
                successful_points += 1

            except ValueError:
                print(f"Skipping inaccessible point ({lat:.6f}, {lon:.6f})")
                skipped_points += 1
            except Exception as e:
                print(f"Error processing point ({lat:.6f}, {lon:.6f}): {str(e)}")
                skipped_points += 1

        self.delivery_points_processed.emit(
            self.snapped_delivery_points,
            successful_points,
            skipped_points
        )

    def get_warehouse_location(self):
        """Calculate the center of the graph to place the warehouse"""
        if not self._graph:
            return None

        lats = []
        lons = []
        for _, data in self._graph.nodes(data=True):
            lats.append(data['y'])
            lons.append(data['x'])

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        warehouse_node = ox.nearest_nodes(self._graph, X=center_lon, Y=center_lat)
        warehouse_coords = (self._graph.nodes[warehouse_node]['y'],
                            self._graph.nodes[warehouse_node]['x'])

        return warehouse_coords

    def handle_graph_loaded(self, data):
        """Handle graph loaded messages from other ViewModels"""
        if 'graph' in data:
            self.set_graph(data['graph'])

    def validate_and_generate_points(self, num_deliveries_text):
        """Validate input and generate delivery points"""
        if not num_deliveries_text.isdigit():
            return False, "Please enter a valid number of delivery points."

        num_deliveries = int(num_deliveries_text)
        self.generate_points(num_deliveries)
        return True, ""
