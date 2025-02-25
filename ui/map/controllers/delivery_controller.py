from PyQt5 import QtWidgets, QtCore

from services.geolocation_service import GeolocationService
from ui.map.utils.map_utils import find_accessible_node


class DeliveryController(QtCore.QObject):
    def __init__(self, base_map):
        super().__init__()
        self.base_map = base_map
        self.snapped_delivery_points = []

    def generate_points(self, num_points):
        if self.base_map.G is None:
            QtWidgets.QMessageBox.warning(
                self.base_map,
                "Graph Not Loaded",
                "Please load the graph data first."
            )
            return

        try:
            node_coords = [(data['y'], data['x'])
                           for _, data in self.base_map.G.nodes(data=True)]
            bounds = (
                min(lat for lat, _ in node_coords),
                max(lat for lat, _ in node_coords),
                min(lon for _, lon in node_coords),
                max(lon for _, lon in node_coords)
            )

            delivery_points = GeolocationService.generate_delivery_points(bounds, num_points)
            self._process_delivery_points(delivery_points)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.base_map,
                "Error",
                f"Error generating deliveries: {str(e)}"
            )
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()

    def _process_delivery_points(self, delivery_points):
        """Process and snap delivery points to the road network"""
        self.snapped_delivery_points = []

        self.base_map.clear_layer("deliveries")

        successful_points = 0
        skipped_points = 0

        for point in delivery_points:
            try:
                lat, lon = point.coordinates
                node_id, (snapped_lat, snapped_lon) = find_accessible_node(
                    self.base_map.G, lat, lon
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

        if self.snapped_delivery_points:
            self.base_map.add_delivery_points(self.snapped_delivery_points)

        if skipped_points > 0:
            self._show_generation_results(successful_points, skipped_points)

    def _show_generation_results(self, successful, skipped):
        """Show a message with the results of delivery point generation"""
        QtWidgets.QMessageBox.information(
            self.base_map,
            "Delivery Points Generated",
            f"Successfully placed {successful} delivery points.\n"
            f"Skipped {skipped} inaccessible points."
        )
