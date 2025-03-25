import json
import os

from PyQt5 import QtCore, QtWebEngineWidgets, QtWebChannel

from config.paths import MAP_HTML
from config.paths import RESOURCES_DIR


class LeafletMapWidget(QtWebEngineWidgets.QWebEngineView):
    load_completed = QtCore.pyqtSignal(bool, str)
    map_ready = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.G = None
        self.current_city = None

        self.channel = QtWebChannel.QWebChannel()
        self.page().setWebChannel(self.channel)

        self.map_handler = MapHandler(self)
        self.channel.registerObject("mapHandler", self.map_handler)

        self.layers = {
            "warehouse": [],
            "deliveries": [],
            "routes": [],
            "unassigned": [],
            "drivers": []
        }

        self.loadFinished.connect(self._on_load_finished)

    def init_map(self, center=None, zoom=None):
        """Initialize the Leaflet map with center and zoom level"""
        from utils.geo_utils import get_city_coordinates

        if center is None or zoom is None:
            center, zoom = get_city_coordinates("Kaunas, Lithuania")

        template_dir = RESOURCES_DIR / 'templates'

        js_template_path = os.path.join(template_dir, "map.js")
        js_output_path = os.path.join(os.path.dirname(MAP_HTML), "map.js")

        with open(js_template_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        with open(js_output_path, "w", encoding="utf-8") as f:
            f.write(js_content)

        html_template_path = os.path.join(template_dir, "map_template.html")
        with open(html_template_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        html_content = html_content.replace("{{CENTER_LAT}}", str(center[0]))
        html_content = html_content.replace("{{CENTER_LON}}", str(center[1]))
        html_content = html_content.replace("{{ZOOM_LEVEL}}", str(zoom))

        with open(MAP_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.load_map()

    def load_map(self):
        """Load the map HTML file into the view"""
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(MAP_HTML))
        self.setUrl(url)

    def _on_load_finished(self, ok):
        """Called when the HTML page has finished loading"""
        if ok:
            # Wait a moment for JavaScript to initialize
            QtCore.QTimer.singleShot(500, lambda: self.map_ready.emit())
            self.load_completed.emit(True, "Map loaded successfully")
        else:
            self.load_completed.emit(False, "Failed to load map")

    def execute_js(self, code, callback=None):
        """Execute JavaScript code in the page"""
        if callback:
            self.page().runJavaScript(code, callback)
        else:
            self.page().runJavaScript(code)

    def update_layer(self, layer_name, data):
        """Update a specific layer with new data"""
        if layer_name not in self.layers:
            return False

        self.layers[layer_name] = data
        js_code = f"if (typeof updateLayer === 'function') {{ updateLayer('{layer_name}', {json.dumps(data)}); }}"
        self.execute_js(js_code)
        return True

    def clear_layer(self, layer_name):
        """Clear a specific layer"""
        if layer_name not in self.layers:
            return False

        self.layers[layer_name] = []
        js_code = f"if (typeof clearLayer === 'function') {{ clearLayer('{layer_name}'); }}"
        self.execute_js(js_code)
        return True

    def clear_all_layers(self):
        """Clear all map layers"""
        for layer_name in self.layers:
            self.clear_layer(layer_name)

    def add_warehouse(self, lat, lon):
        """Add warehouse marker to the map"""
        data = {
            "lat": lat,
            "lng": lon,
            "popup": "Warehouse (Start/End Point)"
        }
        self.update_layer("warehouse", [data])

    def add_delivery_points(self, points):
        """Add delivery points to the map"""
        data = []
        for i, point in enumerate(points):
            lat, lon, weight, volume = point
            data.append({
                "lat": lat,
                "lng": lon,
                "weight": weight,
                "volume": volume,
                "index": i,
                "popup": f"Delivery Point {i + 1}<br>Weight: {weight} kg<br>Volume: {volume} m³"
            })
        self.update_layer("deliveries", data)

    def add_route(self, route_id, driver_id, points, style, popup=None):
        """Add a route to the map"""
        route_data = {
            "id": route_id,
            "driverId": driver_id,
            "path": points,
            "style": style,
            "popup": popup or f"Route {route_id} (Driver {driver_id})"
        }

        routes = self.layers["routes"]
        for i, route in enumerate(routes):
            if route["id"] == route_id:
                routes[i] = route_data
                break
        else:
            routes.append(route_data)

        self.update_layer("routes", routes)

    def start_simulation(self, simulation_data):
        """Start a simulation of the delivery process"""
        js_code = f"if (typeof startSimulation === 'function') {{ startSimulation({json.dumps(simulation_data)}); }}"
        self.execute_js(js_code)

    def set_simulation_speed(self, speed_factor):
        """Set the simulation speed factor (1.0 = real-time)"""
        js_code = f"if (typeof setSimulationSpeed === 'function') {{ setSimulationSpeed({speed_factor}); }}"
        self.execute_js(js_code)


class MapHandler(QtCore.QObject):
    """Object to handle calls from JavaScript to Python"""
    map_initialized = QtCore.pyqtSignal()

    @QtCore.pyqtSlot(str, result=str)
    def handleEvent(self, message):
        """Handle an event from JavaScript"""
        if message == "map_initialized":
            self.map_initialized.emit()

        return "Received in Python"
