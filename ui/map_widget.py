import math
import os
import random

import folium
import osmnx as ox
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebChannel

from data import graph_manager
from logic.routing import find_tsp_route
from ui.workers import DownloadGraphWorker, ComputeRouteWorker


class CustomWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console: {level} - {message} (Line: {lineNumber})")


class WebBridge(QtCore.QObject):
    pointClicked = QtCore.pyqtSignal(float, float)

    @QtCore.pyqtSlot(float, float)
    def receivePointClicked(self, lat, lng):
        self.pointClicked.emit(lat, lng)


class MapWidget(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.G = None
        self.map = None
        self.origin = None
        self.destination = None

        # Optional: store label references for stats
        self.time_label = None
        self.travel_time_label = None
        self.distance_label = None

        # Keep track of random delivery points
        self.snapped_delivery_points = []

        # Setup bridging
        self.channel = QtWebChannel.QWebChannel()
        self.bridge = WebBridge()
        self.channel.registerObject('bridge', self.bridge)

        self.custom_page = CustomWebEnginePage(self)
        self.setPage(self.custom_page)
        self.custom_page.setWebChannel(self.channel)

        self.bridge.pointClicked.connect(self.handle_point_clicked)

        self.init_map()
        self.load_map()

    def set_stats_labels(self, time_label, travel_time_label, distance_label):
        self.time_label = time_label
        self.travel_time_label = travel_time_label
        self.distance_label = distance_label

    # ----------------- MAP / FOLIUM -----------------
    def init_map(self):
        center = (54.8985, 23.9036)  # Kaunas
        self.map = folium.Map(location=center, zoom_start=12)
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
        self.map.save('resources/map.html')
        self.add_qwebchannel_script('resources/map.html')
        url = QtCore.QUrl.fromLocalFile(os.path.abspath('resources/map.html'))
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
            # reset
            self.origin = (lat, lng)
            self.destination = None
            self.init_map()
            folium.Marker(location=self.origin, popup='Origin', icon=folium.Icon(color='green')).add_to(self.map)
            self.load_map()

    # ----------------- WORKERS & LOAD/SAVE -----------------
    def download_graph_data(self):
        self.worker = DownloadGraphWorker()
        self.worker.finished.connect(self.on_download_finished)
        self.worker.start()

    def on_download_finished(self, success, message):
        if success:
            QtWidgets.QMessageBox.information(self, "Success", "Graph data downloaded and saved successfully.")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)

    def load_graph_data(self):
        if self.G is not None:
            QtWidgets.QMessageBox.information(self, "Information", "Graph data is already loaded.")
            return
        try:
            self.G = graph_manager.load_graph()
            # Optional: load adjusted times
            csv_path = "resources/adjusted_travel_times.csv"
            if os.path.isfile(csv_path):
                graph_manager.update_travel_times_from_csv(self.G, csv_path)
            QtWidgets.QMessageBox.information(self, "Information", "Graph data loaded successfully.")
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "File Not Found",
                                          "Graph data file not found. Please download the data.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading data:\n{e}")

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
            # Draw route on the map
            folium.PolyLine(locations=route_nodes, color='blue', weight=5, opacity=0.7).add_to(self.map)

            # Optionally add small step markers
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

            # Update labels
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

        # gather marker points
        delivery_points = []
        for child in self.map._children.values():
            if isinstance(child, folium.Marker):
                delivery_points.append((child.location[0], child.location[1]))

        if not delivery_points:
            QtWidgets.QMessageBox.warning(self, "No Deliveries", "No delivery points found on the map.")
            return

        try:
            route_coords, total_travel_time, total_distance, compute_time, snapped_nodes = find_tsp_route(
                self.G, delivery_points
            )

            # re-init the map
            self.init_map()
            # draw the TSP route
            folium.PolyLine(
                locations=route_coords,
                color='purple',
                weight=5,
                opacity=0.7,
                tooltip="A* TSP Route"
            ).add_to(self.map)

            # re-add markers
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

            # update stats
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
        Generate random delivery points inside/outside a bounding box,
        snap them, and add markers.
        """
        if self.G is None:
            QtWidgets.QMessageBox.warning(self, "Graph Not Loaded", "Please load the graph data first.")
            return

        try:
            self.init_map()

            node_coords = [(data['y'], data['x']) for _, data in self.G.nodes(data=True)]
            min_lat = min(lat for lat, _ in node_coords)
            max_lat = max(lat for lat, _ in node_coords)
            min_lon = min(lon for _, lon in node_coords)
            max_lon = max(lon for _, lon in node_coords)

            lat_mid = (min_lat + max_lat) / 2
            lon_mid = (min_lon + max_lon) / 2
            inner_lat_range = (max_lat - min_lat) * 0.5
            inner_lon_range = (max_lon - min_lon) * 0.5

            inner_min_lat = lat_mid - inner_lat_range / 2
            inner_max_lat = lat_mid + inner_lat_range / 2
            inner_min_lon = lon_mid - inner_lon_range / 2
            inner_max_lon = lon_mid + inner_lon_range / 2

            inner_count = int(num_points * 0.75)
            outer_count = num_points - inner_count

            inner_points = self.generate_randomized_grid_points(
                inner_min_lat, inner_max_lat, inner_min_lon, inner_max_lon, inner_count
            )

            outer_points = []
            while len(outer_points) < outer_count:
                lat = random.uniform(min_lat, max_lat)
                lon = random.uniform(min_lon, max_lon)
                # only accept if it's not in the "inner" bounding box
                if not (inner_min_lat <= lat <= inner_max_lat and inner_min_lon <= lon <= inner_max_lon):
                    outer_points.append((lat, lon))

            points = inner_points + outer_points
            snapped_points = []

            for lat, lon in points:
                try:
                    nearest_node = ox.nearest_nodes(self.G, X=lon, Y=lat)
                    snapped_points.append((self.G.nodes[nearest_node]['y'], self.G.nodes[nearest_node]['x']))
                except Exception as e:
                    print(f"Skipping point ({lat}, {lon}): {e}")

            self.snapped_delivery_points = snapped_points

            for lat, lon in snapped_points:
                folium.Marker(
                    location=(lat, lon),
                    popup='Delivery Point',
                    icon=folium.Icon(color='orange', icon='info-sign')
                ).add_to(self.map)

            self.load_map()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error generating deliveries: {e}")

    def generate_randomized_grid_points(self, min_lat, max_lat, min_lon, max_lon, num_points):
        try:
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            grid_size = math.ceil(math.sqrt(num_points))

            step_lat = lat_range / grid_size
            step_lon = lon_range / grid_size

            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) >= num_points:
                        break
                    lat = min_lat + i * step_lat + random.uniform(0, step_lat)
                    lon = min_lon + j * step_lon + random.uniform(0, step_lon)
                    points.append((lat, lon))
                if len(points) >= num_points:
                    break

            return points
        except Exception as e:
            print(f"Error generating grid points: {e}")
            return []
