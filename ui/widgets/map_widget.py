import os

import folium
import osmnx as ox
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebChannel

from config.paths import MAP_HTML, get_graph_file_path, get_travel_times_path
from config.settings import ROUTE_COLORS
from data import graph_manager
from logic.routing import find_tsp_route
from services.geolocation_service import GeolocationService
from ui.workers.download_worker import DownloadGraphWorker
from ui.workers.route_worker import ComputeRouteWorker
from utils.geolocation import get_city_coordinates


class CustomWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console: {level} - {message} (Line: {lineNumber})")


class WebBridge(QtCore.QObject):
    pointClicked = QtCore.pyqtSignal(float, float)

    @QtCore.pyqtSlot(float, float)
    def receivePointClicked(self, lat, lng):
        self.pointClicked.emit(lat, lng)


class MapWidget(QtWebEngineWidgets.QWebEngineView):
    load_completed = QtCore.pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.G = None
        self.map = None
        self.origin = None
        self.destination = None
        self.current_city = None

        self.time_label = None
        self.travel_time_label = None
        self.distance_label = None

        self.snapped_delivery_points = []

        self.channel = QtWebChannel.QWebChannel()
        self.bridge = WebBridge()
        self.channel.registerObject('bridge', self.bridge)

        self.custom_page = CustomWebEnginePage(self)
        self.setPage(self.custom_page)
        self.custom_page.setWebChannel(self.channel)

        self.bridge.pointClicked.connect(self.handle_point_clicked)

        initial_coords, initial_zoom = get_city_coordinates("Kaunas, Lithuania")
        self.init_map(initial_coords, initial_zoom)
        self.load_map()

    def set_stats_labels(self, time_label, travel_time_label, distance_label):
        self.time_label = time_label
        self.travel_time_label = travel_time_label
        self.distance_label = distance_label

    # ----------------- MAP / FOLIUM -----------------
    def init_map(self, center=None, zoom=None):
        """
        Initialize or reinitialize the Folium map with specific coordinates and zoom level.

        This method creates a new Folium map instance centered on the specified location.
        If no location is provided, it defaults to Kaunas as a fallback. The method is
        designed to work both during initial widget creation and when switching cities.

        Args:
            center (tuple, optional): Latitude and longitude coordinates (lat, lon)
                for the map center. Defaults to None, which will use Kaunas coordinates.
            zoom (int, optional): Initial zoom level for the map. Defaults to None,
                which will use zoom level 12 as a sensible default for most cities.
        """
        if center is None:
            center, zoom = get_city_coordinates("Kaunas, Lithuania")

        if zoom is None:
            zoom = 12

        self.map = folium.Map(location=center, zoom_start=zoom)

        self.add_click_listener()

    def add_click_listener(self):
        map_name = self.map.get_name()
        click_js = f"""
        function onMapClick(e) {{
            var lat = e.latlng.lat;
            var lng = e.latlng.lng;
            bridge.receivePointClicked(lat, lng);
        }}

        function addClickListener() {{
            {map_name}.on('click', onMapClick);
        }}

        if (typeof {map_name} !== 'undefined') {{
            addClickListener();
        }} else {{
            document.addEventListener("DOMContentLoaded", function(event) {{
                addClickListener();
            }});
        }}
        """
        self.map.get_root().script.add_child(folium.Element(click_js))

    def load_map(self):
        self.map.save(MAP_HTML)
        self.add_qwebchannel_script(MAP_HTML)
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(MAP_HTML))
        self.load(url)

    def add_qwebchannel_script(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        qwebchannel_setup = '''
        <script type="text/javascript" src="qwebchannel.js"></script>
        <script type="text/javascript">
            var bridge = null;
            new QWebChannel(qt.webChannelTransport, function(channel) {
                bridge = channel.objects.bridge;
            });
        </script>
        '''
        html = html.replace('</body>', qwebchannel_setup + '</body>')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    @QtCore.pyqtSlot(float, float)
    def handle_point_clicked(self, lat, lng):
        """
        First click sets origin, second sets destination, third resets, etc.
        When resetting, we maintain the current city's view rather than defaulting back.
        """
        if not self.origin:
            self.origin = (lat, lng)
            folium.Marker(location=self.origin, popup='Origin', icon=folium.Icon(color='green')).add_to(self.map)
            self.load_map()
        elif not self.destination:
            self.destination = (lat, lng)
            folium.Marker(location=self.destination, popup='Destination', icon=folium.Icon(color='red')).add_to(
                self.map)
            self.load_map()
        else:
            self.origin = (lat, lng)
            self.destination = None
            center, zoom = get_city_coordinates(self.current_city or "Kaunas, Lithuania")
            self.init_map(center, zoom)
            folium.Marker(location=self.origin, popup='Origin', icon=folium.Icon(color='green')).add_to(self.map)
            self.load_map()

    # ----------------- WORKERS & LOAD/SAVE -----------------
    def load_graph_data(self, city_name):
        """
        Load or download graph data for the specified city.
        After loading, reinitialize the map to center on the new city.
        """
        if self.G is not None:
            QtWidgets.QMessageBox.information(self, "Information", "Graph data is already loaded.")
            self.load_completed.emit(True, "Graph already loaded")
            return

        try:
            center, zoom = get_city_coordinates(city_name)

            graph_path = get_graph_file_path(city_name)
            travel_times_path = get_travel_times_path(city_name)

            self.G = graph_manager.load_graph(filename=graph_path)

            self.current_city = city_name
            self.init_map(center, zoom)

            if os.path.isfile(travel_times_path):
                graph_manager.update_travel_times_from_csv(self.G, travel_times_path)

            self.load_map()
            QtWidgets.QMessageBox.information(self, "Information", "Graph data loaded successfully.")
            self.load_completed.emit(True, "Graph loaded successfully")

        except FileNotFoundError:
            reply = QtWidgets.QMessageBox.question(
                self,
                "File Not Found",
                f"Graph data file for {city_name} not found. Would you like to download it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )

            if reply == QtWidgets.QMessageBox.Yes:
                self.worker = DownloadGraphWorker(city_name)
                self.worker.finished.connect(
                    lambda success, msg: self.on_download_and_load_finished(success, msg, city_name))
                self.worker.start()
            else:
                self.load_completed.emit(False, "User cancelled download")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading data:\n{e}")
            self.load_completed.emit(False, str(e))

    def on_download_and_load_finished(self, success, message, city_name):
        """Modified to handle city-specific loading"""
        if success:
            try:
                graph_path = get_graph_file_path(city_name)
                travel_times_path = get_travel_times_path(city_name)

                self.G = graph_manager.load_graph(filename=graph_path)
                self.current_city = city_name

                if os.path.isfile(travel_times_path):
                    graph_manager.update_travel_times_from_csv(self.G, travel_times_path)

                QtWidgets.QMessageBox.information(self, "Success", "Graph data downloaded and loaded successfully.")
                self.load_completed.emit(True, "Graph downloaded and loaded successfully")
            except Exception as e:
                error_msg = f"Error loading downloaded data:\n{e}"
                QtWidgets.QMessageBox.critical(self, "Error", error_msg)
                self.load_completed.emit(False, error_msg)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            self.load_completed.emit(False, message)

    def compute_route(self):
        if self.G is None:
            QtWidgets.QMessageBox.warning(self, "Data Not Loaded", "Please load or download the graph data first.")
            return
        if not self.origin or not self.destination:
            QtWidgets.QMessageBox.warning(self, "Points Not Selected",
                                          "Please select both origin and destination points.")
            return

        self.worker = ComputeRouteWorker(self.G, self.origin, self.destination)
        self.worker.finished.connect(self.on_route_computed)
        self.worker.start()

    def on_route_computed(self, route_data, total_travel_time, total_distance,
                          computation_time, success, message):
        if success:
            route_nodes, cumulative_times, cumulative_distances = route_data
            folium.PolyLine(locations=route_nodes, color=ROUTE_COLORS['normal'], weight=5, opacity=0.7).add_to(self.map)

            for idx, point in enumerate(route_nodes):
                folium.CircleMarker(
                    location=point,
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7,
                    popup=(
                        f'Step {idx + 1}<br>'
                        f'Cumulative Time: {cumulative_times[idx] / 60:.2f} min<br>'
                        f'Cumulative Distance: {cumulative_distances[idx]:.2f} m'
                    )
                ).add_to(self.map)

            if self.time_label:
                self.time_label.setText(f"Time to compute route: {computation_time:.2f}s")
            if self.travel_time_label:
                self.travel_time_label.setText(f"Total travel time: {total_travel_time / 60:.2f} min")
            if self.distance_label:
                self.distance_label.setText(f"Total distance: {total_distance / 1000:.2f} km")

            self.load_map()
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)

    # ----------------- TSP -----------------
    def find_shortest_route(self):
        """
        Collect all folium.Marker's lat-lng as "delivery points"
        and run TSP on them plus the center.
        """
        if self.G is None:
            QtWidgets.QMessageBox.warning(self, "Graph Not Loaded", "Please load the graph data first.")
            return

        delivery_points = []
        for child in self.map._children.values():
            if isinstance(child, folium.Marker):
                delivery_points.append((child.location[0], child.location[1]))

        if not delivery_points:
            QtWidgets.QMessageBox.warning(self, "No Deliveries", "No delivery points found on the map.")
            return

        try:
            city_center, _ = get_city_coordinates(self.current_city or "Kaunas, Lithuania")

            route_coords, total_travel_time, total_distance, compute_time, snapped_nodes = find_tsp_route(
                self.G,
                delivery_points,
                center=city_center
            )

            center, zoom = get_city_coordinates(self.current_city or "Kaunas, Lithuania")
            self.init_map(center, zoom)
            folium.PolyLine(
                locations=route_coords,
                color=ROUTE_COLORS['tsp'],
                weight=5,
                opacity=0.7,
                tooltip="A* TSP Route"
            ).add_to(self.map)

            for i, node_id in enumerate(snapped_nodes):
                lat, lon = self.G.nodes[node_id]['y'], self.G.nodes[node_id]['x']
                color = 'blue' if i != 0 else 'red'
                icon = 'info-sign' if i != 0 else 'home'
                popup_text = 'Center' if i == 0 else f'Delivery {i}'
                folium.Marker(
                    location=(lat, lon),
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(self.map)

            if self.time_label:
                self.time_label.setText(f"TSP solved in {compute_time:.2f} s")
            if self.travel_time_label:
                self.travel_time_label.setText(f"Total travel time: {total_travel_time / 60:.2f} min")
            if self.distance_label:
                self.distance_label.setText(f"Total distance: {total_distance / 1000:.2f} km")

            self.load_map()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"TSP error: {e}")

    def generate_delivery_points(self, num_points):
        """
        Generate random delivery points using geolocation service, snap them, and add markers.
        """
        if self.G is None:
            QtWidgets.QMessageBox.warning(self, "Graph Not Loaded", "Please load the graph data first.")
            return

        try:
            node_coords = [(data['y'], data['x']) for _, data in self.G.nodes(data=True)]
            min_lat = min(lat for lat, _ in node_coords)
            max_lat = max(lat for lat, _ in node_coords)
            min_lon = min(lon for _, lon in node_coords)
            max_lon = max(lon for _, lon in node_coords)

            points = GeolocationService.generate_delivery_points(
                (min_lat, max_lat, min_lon, max_lon), num_points
            )

            self.snapped_delivery_points = []
            center, zoom = get_city_coordinates(self.current_city or "Kaunas, Lithuania")
            self.init_map(center, zoom)

            for lat, lon in points:
                try:
                    nearest_node = ox.nearest_nodes(self.G, X=lon, Y=lat)
                    snapped_lat = self.G.nodes[nearest_node]['y']
                    snapped_lon = self.G.nodes[nearest_node]['x']
                    self.snapped_delivery_points.append((snapped_lat, snapped_lon))
                    folium.Marker(
                        location=(snapped_lat, snapped_lon),
                        popup='Delivery Point',
                        icon=folium.Icon(color='orange', icon='info-sign')
                    ).add_to(self.map)
                except Exception as e:
                    print(f"Skipping point ({lat}, {lon}): {e}")

            self.load_map()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error generating deliveries: {e}")
