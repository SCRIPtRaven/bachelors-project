import json
import os

from PyQt5 import QtCore, QtWebEngineWidgets, QtWebChannel

from config.config import PathsConfig


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
        from utils.geo_utils import get_city_coordinates

        if center is None or zoom is None:
            center, zoom = get_city_coordinates("Kaunas, Lithuania")

        template_dir = PathsConfig.RESOURCES_DIR / 'templates'

        js_files = ["map.js", "disruptions.js", "simulation_actions.js"]
        output_dir = os.path.dirname(PathsConfig.MAP_HTML)

        for js_file in js_files:
            template_path = os.path.join(template_dir, js_file)
            output_path = os.path.join(output_dir, js_file)

            with open(template_path, "r", encoding="utf-8") as f:
                js_content = f.read()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(js_content)

        html_template_path = os.path.join(template_dir, "map_template.html")
        with open(html_template_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        html_content = html_content.replace("{{CENTER_LAT}}", str(center[0]))
        html_content = html_content.replace("{{CENTER_LON}}", str(center[1]))
        html_content = html_content.replace("{{ZOOM_LEVEL}}", str(zoom))

        with open(PathsConfig.MAP_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.load_map()

    def load_map(self):
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(PathsConfig.MAP_HTML))
        self.setUrl(url)

    def _on_load_finished(self, ok):
        if ok:
            QtCore.QTimer.singleShot(500, lambda: self.map_ready.emit())
            self.load_completed.emit(True, "Map loaded successfully")
        else:
            self.load_completed.emit(False, "Failed to load map")

    def execute_js(self, code, callback=None):
        if callback:
            self.page().runJavaScript(code, callback)
        else:
            self.page().runJavaScript(code)

    def update_layer(self, layer_name, data):
        if layer_name not in self.layers:
            return False

        self.layers[layer_name] = data
        js_code = f"if (typeof updateLayer === 'function') {{ updateLayer('{layer_name}', {json.dumps(data)}); }}"
        self.execute_js(js_code)
        return True

    def load_disruptions(self, disruptions):
        disruption_data = []

        for disruption in disruptions:
            if isinstance(disruption, dict):
                data = disruption
            else:
                data = {
                    'id': disruption.id,
                    'type': disruption.type.value,
                    'location': {
                        'lat': disruption.location[0],
                        'lng': disruption.location[1]
                    },
                    'radius': disruption.affected_area_radius,
                    'severity': disruption.severity,
                    'description': disruption.metadata.get('description',
                                                           f"{disruption.type.value.replace('_', ' ').title()}")
                }
            disruption_data.append(data)

        js_code = f"if (typeof loadDisruptions === 'function') {{ loadDisruptions({json.dumps(disruption_data)}); }}"
        self.execute_js(js_code)

    def clear_layer(self, layer_name):
        if layer_name not in self.layers:
            return False

        self.layers[layer_name] = []
        js_code = f"if (typeof clearLayer === 'function') {{ clearLayer('{layer_name}'); }}"
        self.execute_js(js_code)
        return True

    def clear_all_layers(self):
        for layer_name in self.layers:
            self.clear_layer(layer_name)

    def add_warehouse(self, lat, lon):
        data = {
            "lat": lat,
            "lng": lon,
            "popup": "Warehouse (Start/End Point)"
        }
        self.update_layer("warehouse", [data])

    def add_delivery_points(self, points):
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
        js_code = f"if (typeof startSimulation === 'function') {{ startSimulation({json.dumps(simulation_data)}); }}"
        self.execute_js(js_code)

    def set_simulation_speed(self, speed_factor):
        js_code = f"if (typeof setSimulationSpeed === 'function') {{ setSimulationSpeed({speed_factor}); }}"
        self.execute_js(js_code)


class MapHandler(QtCore.QObject):
    map_initialized = QtCore.pyqtSignal()
    manual_disruptions_placed = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

    @QtCore.pyqtSlot(str, result=str)
    def handleEvent(self, message):
        if message == "map_initialized":
            self.map_initialized.emit()
            return "Received in Python"
        
        try:
            import json
            data = json.loads(message)
            
            if data.get('type') == 'manual_disruptions_placed':
                disruptions = data.get('data', {}).get('disruptions', [])
                print(f"Received {len(disruptions)} manually placed disruptions")
                self.manual_disruptions_placed.emit(disruptions)
                return "Manual disruptions received"
                
        except (json.JSONDecodeError, KeyError):
            pass

        return "Received in Python"
