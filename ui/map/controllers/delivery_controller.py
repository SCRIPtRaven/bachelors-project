import folium
from PyQt5 import QtWidgets, QtCore

from services.geolocation_service import GeolocationService
from ui.map.utils.map_utils import find_accessible_node
from utils.geolocation import get_city_coordinates


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

    def _process_delivery_points(self, delivery_points):
        self.snapped_delivery_points = []
        center, zoom = get_city_coordinates(self.base_map.current_city or "Kaunas, Lithuania")
        self.base_map.init_map(center, zoom)

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

                self._add_delivery_marker(
                    successful_points,
                    snapped_lat,
                    snapped_lon,
                    point.weight,
                    point.volume
                )

                successful_points += 1

            except ValueError:
                print(f"Skipping inaccessible point ({lat:.6f}, {lon:.6f})")
                skipped_points += 1
            except Exception as e:
                print(f"Error processing point ({lat:.6f}, {lon:.6f}): {str(e)}")
                skipped_points += 1

        if skipped_points > 0:
            self._show_generation_results(successful_points, skipped_points)

        self.base_map.load_map()

    def _add_delivery_marker(self, index, lat, lon, weight, volume):
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.7,
            popup=f'Delivery Point {index + 1}<br>'
                  f'Weight: {weight} kg<br>'
                  f'Volume: {volume} m³'
        ).add_to(self.base_map.map)

    def _show_generation_results(self, successful, skipped):
        QtWidgets.QMessageBox.information(
            self.base_map,
            "Delivery Points Generated",
            f"Successfully placed {successful} delivery points.\n"
            f"Skipped {skipped} inaccessible points."
        )
