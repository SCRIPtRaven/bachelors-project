import os
import time

import folium
import networkx as nx
import osmnx as ox
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebChannel

from config.paths import MAP_HTML, get_graph_file_path, get_travel_times_path
from config.settings import ROUTE_COLORS
from data import graph_manager
from logic.delivery_optimizer import SimulatedAnnealingOptimizer
from logic.routing import get_largest_connected_component
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


# TODO : Implement better separation of concerns for this gargantuan class
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
        self.delivery_drivers = []

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
            self.G = get_largest_connected_component(self.G)

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

    def find_accessible_node(self, lat, lon, search_radius=1000):
        """
        Finds the nearest accessible node to the given coordinates within the graph.

        This function implements a robust node-finding algorithm that:
        1. Searches for nodes within an expanding radius
        2. Verifies that found nodes are actually in our graph
        3. Ensures nodes are connected to the main road network
        4. Handles edge cases and errors gracefully

        Args:
            lat: Latitude of the desired location
            lon: Longitude of the desired location
            search_radius: Initial search radius in meters, expands if needed

        Returns:
            tuple: (node_id, (node_lat, node_lon)) for the found node

        Raises:
            ValueError: If no suitable node can be found within maximum search radius
        """
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

        if not hasattr(self, 'center_node'):
            try:
                center_y = sum(data['y'] for _, data in self.G.nodes(data=True)) / len(self.G)
                center_x = sum(data['x'] for _, data in self.G.nodes(data=True)) / len(self.G)
                self.center_node = ox.nearest_nodes(self.G, X=center_x, Y=center_y)
            except Exception as e:
                print(f"Warning: Could not establish center node: {e}")
                self.center_node = None

        max_radius = 5000

        while search_radius <= max_radius:
            try:
                node_id = ox.nearest_nodes(self.G, X=lon, Y=lat)

                if node_id in self.G.nodes:
                    if self.center_node is None or nx.has_path(self.G, node_id, self.center_node):
                        node = self.G.nodes[node_id]
                        return node_id, (node['y'], node['x'])
                    else:
                        print(f"Found node {node_id} is not connected to network center")
                else:
                    print(f"Found node {node_id} is not in our graph")

            except Exception as e:
                print(f"Error finding node at ({lat:.6f}, {lon:.6f}): {str(e)}")

            search_radius += 500
            print(f"Expanding search radius to {search_radius}m")

        raise ValueError(f"No accessible node found near ({lat:.6f}, {lon:.6f})")

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
    def get_folium_colors(self):
        """
        Returns a list of folium-supported colors for routes and markers.
        """
        return [
            'blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred',
            'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink'
        ]

    def find_shortest_route(self):
        if self.G is None or not self.snapped_delivery_points or not self.delivery_drivers:
            QtWidgets.QMessageBox.warning(self, "Missing Data",
                                          "Please ensure graph data, delivery points, and drivers are all loaded.")
            return

        try:
            if hasattr(self, 'optimization_thread'):
                if self.optimization_thread.isRunning():
                    self.optimization_thread.quit()
                    self.optimization_thread.wait()
                self.optimization_thread.deleteLater()

            if hasattr(self, 'optimizer'):
                self.optimizer.deleteLater()

            if hasattr(self, 'optimization_map_initialized'):
                delattr(self, 'optimization_map_initialized')

            self.optimizer = SimulatedAnnealingOptimizer(
                self.delivery_drivers,
                self.snapped_delivery_points,
                self.G
            )

            self.optimization_thread = QtCore.QThread(self)
            self.optimizer.moveToThread(self.optimization_thread)

            self.optimization_thread.started.connect(self.optimizer.optimize)
            self.optimizer.update_visualization.connect(self.update_route_visualization)
            self.optimizer.finished.connect(self.on_optimization_finished)
            self.optimizer.finished.connect(self.optimization_thread.quit)
            self.optimization_thread.finished.connect(self.cleanup_optimization)

            self.start_time = time.time()

            self.optimization_thread.start()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Route planning error: {str(e)}")
            print(f"Detailed error: {e}")

    def cleanup_optimization(self):
        """Clean up optimization resources"""
        try:
            if hasattr(self, 'optimization_thread'):
                self.optimization_thread.deleteLater()
                delattr(self, 'optimization_thread')

            if hasattr(self, 'optimizer'):
                self.optimizer.deleteLater()
                delattr(self, 'optimizer')

        except Exception as e:
            print(f"Error in cleanup: {e}")

    def update_route_visualization(self, current_solution, unassigned_deliveries):
        try:
            city_center, zoom = get_city_coordinates(self.current_city or "Kaunas, Lithuania")

            if not hasattr(self, 'optimization_map_initialized'):
                self.init_map(city_center, zoom)
                self.optimization_map_initialized = True
                self.map_zoom = zoom

            self.map = folium.Map(
                location=self.map.location,
                zoom_start=self.map_zoom
            )

            self.delivery_layer = folium.FeatureGroup(name='delivery_points')
            self.routes_layer = folium.FeatureGroup(name='routes')
            self.map.add_child(self.delivery_layer)
            self.map.add_child(self.routes_layer)

            colors = self.get_folium_colors()
            unassigned_set = set(unassigned_deliveries)

            delivery_to_driver = {}
            conflicting_deliveries = set()

            for driver_idx, assignment in enumerate(current_solution):
                for delivery_idx in assignment.delivery_indices:
                    if delivery_idx in delivery_to_driver:
                        conflicting_deliveries.add(delivery_idx)
                    else:
                        delivery_to_driver[delivery_idx] = (driver_idx, assignment.driver_id)

            for driver_idx, assignment in enumerate(current_solution):
                if not assignment.delivery_indices:
                    continue

                color = colors[driver_idx % len(colors)]
                driver_points = []

                for delivery_idx in assignment.delivery_indices:
                    if (delivery_idx in delivery_to_driver and
                            delivery_to_driver[delivery_idx][0] == driver_idx and
                            delivery_idx not in unassigned_set and
                            delivery_idx not in conflicting_deliveries):
                        lat, lon, weight, volume = self.snapped_delivery_points[delivery_idx]
                        driver_points.append({
                            'idx': delivery_idx,
                            'coords': (lat, lon),
                            'weight': weight,
                            'volume': volume
                        })

                if driver_points:
                    for point in driver_points:
                        folium.CircleMarker(
                            location=point['coords'],
                            radius=6,
                            color=color,
                            fill=True,
                            fill_opacity=0.7,
                            popup=(f'Delivery Point {point["idx"] + 1}<br>'
                                   f'Assigned to Driver {assignment.driver_id}<br>'
                                   f'Weight: {point["weight"]}kg<br>'
                                   f'Volume: {point["volume"]}m³')
                        ).add_to(self.delivery_layer)

                    if len(driver_points) > 1:
                        detailed_route = []
                        for i in range(len(driver_points) - 1):
                            start = driver_points[i]['coords']
                            end = driver_points[i + 1]['coords']

                            try:
                                start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
                                end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])

                                path = nx.shortest_path(self.G, start_node, end_node, weight='length')

                                path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                                               for node in path]

                                detailed_route.extend(path_coords)
                            except nx.NetworkXNoPath:
                                detailed_route.extend([start, end])

                        folium.PolyLine(
                            locations=detailed_route,
                            color=color,
                            weight=3,
                            opacity=0.7,
                            popup=f'Driver {assignment.driver_id} Route'
                        ).add_to(self.routes_layer)
            for delivery_idx in conflicting_deliveries:
                lat, lon, weight, volume = self.snapped_delivery_points[delivery_idx]
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=6,
                    color='red',
                    fill=True,
                    fill_opacity=0.7,
                    popup=(f'Delivery Point {delivery_idx + 1}<br>'
                           f'ERROR: Multiple Assignments<br>'
                           f'Weight: {weight}kg<br>'
                           f'Volume: {volume}m³')
                ).add_to(self.delivery_layer)

            for delivery_idx in unassigned_deliveries:
                if delivery_idx not in delivery_to_driver and delivery_idx not in conflicting_deliveries:
                    lat, lon, weight, volume = self.snapped_delivery_points[delivery_idx]
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=6,
                        color='black',
                        fill=True,
                        fill_opacity=0.7,
                        popup=(f'Delivery Point {delivery_idx + 1}<br>'
                               f'UNASSIGNED<br>'
                               f'Weight: {weight}kg<br>'
                               f'Volume: {volume}m³')
                    ).add_to(self.delivery_layer)

            self.load_map()
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            print(f"Error in visualization update: {e}")

    def on_optimization_finished(self, final_solution, unassigned):
        """Handle optimization completion"""
        try:
            city_center, _ = get_city_coordinates(self.current_city or "Kaunas, Lithuania")
            total_distance = 0
            total_time = 0

            for assignment in final_solution:
                if assignment.delivery_indices:
                    delivery_coords = [self.snapped_delivery_points[i][0:2]
                                       for i in assignment.delivery_indices]
                    delivery_coords = [city_center] + delivery_coords + [city_center]
                    route_length = self.optimizer.calculate_route_distance(delivery_coords)
                    total_distance += route_length
                    total_time += route_length / 10

            if self.time_label:
                self.time_label.setText(f"Routes computed in {time.time() - self.start_time:.2f} s")
            if self.travel_time_label:
                self.travel_time_label.setText(f"Total travel time: {total_time / 60:.2f} min")
            if self.distance_label:
                self.distance_label.setText(f"Total distance: {total_distance / 1000:.2f} km")

            if hasattr(self, 'optimizer'):
                self.optimizer.deleteLater()
                delattr(self, 'optimizer')

        except Exception as e:
            print(f"Error in optimization finished handler: {e}")

    def generate_delivery_points(self, num_points):
        """
        Generates delivery points within the graph area, ensuring they're all accessible.
        Uses GeolocationService for point generation and snaps points to the road network.
        """
        if self.G is None:
            QtWidgets.QMessageBox.warning(self, "Graph Not Loaded",
                                          "Please load the graph data first.")
            return

        try:
            node_coords = [(data['y'], data['x'])
                           for _, data in self.G.nodes(data=True)]
            min_lat = min(lat for lat, _ in node_coords)
            max_lat = max(lat for lat, _ in node_coords)
            min_lon = min(lon for _, lon in node_coords)
            max_lon = max(lon for _, lon in node_coords)

            bounds = (min_lat, max_lat, min_lon, max_lon)
            delivery_points = GeolocationService.generate_delivery_points(bounds, num_points)

            self.snapped_delivery_points = []
            center, zoom = get_city_coordinates(self.current_city or "Kaunas, Lithuania")
            self.init_map(center, zoom)

            successful_points = 0
            skipped_points = 0

            for point in delivery_points:
                try:
                    lat, lon = point.coordinates
                    node_id, (snapped_lat, snapped_lon) = self.find_accessible_node(lat, lon)

                    self.snapped_delivery_points.append(
                        (snapped_lat, snapped_lon, point.weight, point.volume)
                    )

                    folium.CircleMarker(
                        location=(snapped_lat, snapped_lon),
                        radius=6,
                        color='orange',
                        fill=True,
                        fill_color='orange',
                        fill_opacity=0.7,
                        popup=f'Delivery Point {successful_points + 1}<br>'
                              f'Weight: {point.weight} kg<br>'
                              f'Volume: {point.volume} m³'
                    ).add_to(self.map)

                    successful_points += 1

                except ValueError as ve:
                    print(f"Skipping inaccessible point ({lat:.6f}, {lon:.6f})")
                    skipped_points += 1
                except Exception as e:
                    print(f"Error processing point ({lat:.6f}, {lon:.6f}): {str(e)}")
                    skipped_points += 1

            if skipped_points > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Delivery Points Generated",
                    f"Successfully placed {successful_points} delivery points.\n"
                    f"Skipped {skipped_points} inaccessible points."
                )

            self.load_map()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error generating deliveries: {str(e)}"
            )
            print(f"Detailed error: {e}")

    def generate_delivery_drivers(self, num_drivers):
        """Generate delivery drivers and display their capacities in a scrollable list."""
        try:
            self.delivery_drivers = GeolocationService.generate_delivery_drivers(num_drivers)

            if self.time_label:
                stats_layout = self.time_label.parent().layout()

                for i in reversed(range(stats_layout.count())):
                    widget = stats_layout.itemAt(i).widget()
                    if isinstance(widget, QtWidgets.QScrollArea) or (
                            isinstance(widget, QtWidgets.QLabel) and widget.text() == "Delivery Drivers:"):
                        widget.setParent(None)

                header_label = QtWidgets.QLabel("Delivery Drivers:")
                header_label.setStyleSheet("font-weight: bold; padding: 5px;")
                stats_layout.addWidget(header_label)

                scroll_area = QtWidgets.QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setMinimumHeight(150)
                scroll_area.setMaximumHeight(200)
                scroll_area.setStyleSheet("""
                    QScrollArea {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        background: white;
                    }
                """)

                driver_container = QtWidgets.QWidget()
                driver_layout = QtWidgets.QVBoxLayout(driver_container)
                driver_layout.setSpacing(5)
                driver_layout.setContentsMargins(5, 5, 5, 5)

                for driver in self.delivery_drivers:
                    driver_label = QtWidgets.QLabel(
                        f"Driver {driver.id}: Capacity {driver.weight_capacity}kg, {driver.volume_capacity}m³"
                    )
                    driver_label.setStyleSheet("padding: 3px;")
                    driver_layout.addWidget(driver_label)

                driver_layout.addStretch()

                scroll_area.setWidget(driver_container)

                stats_layout.addWidget(scroll_area)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error generating drivers: {e}")
