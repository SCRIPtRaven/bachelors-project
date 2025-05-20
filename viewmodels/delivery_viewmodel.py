import json
import os
from datetime import datetime

import osmnx as ox
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

from models.entities.delivery import Delivery
from models.services.geolocation_service import GeolocationService
from utils.geo_utils import find_accessible_node
from viewmodels.viewmodel_messenger import MessageType

SAVED_CONFIGS_DIR = "saved_configurations"


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
        if 'graph' in data:
            self.set_graph(data['graph'])

    def validate_and_generate_points(self, num_deliveries_text):
        if not num_deliveries_text.isdigit():
            self.request_show_message.emit("Invalid Input",
                                           "Please enter a valid number of delivery points.",
                                           "warning")
            return False, "Please enter a valid number of delivery points."

        num_deliveries = int(num_deliveries_text)
        self.generate_points(num_deliveries)
        return True, ""

    def save_deliveries_config(self):
        if not self.snapped_delivery_points:
            self.request_show_message.emit("No Deliveries",
                                           "No deliveries have been generated to save.", "warning")
            return

        if self._graph is None or not hasattr(self._graph,
                                              'graph') or 'name' not in self._graph.graph:
            self.request_show_message.emit("Graph Error",
                                           "Map/Graph data is not fully loaded or identifiable. Cannot determine city name.",
                                           "warning")
            return

        city_name = self._graph.graph.get('name', 'UnknownCity').split(',')[0].replace(" ", "_")

        if not os.path.exists(SAVED_CONFIGS_DIR):
            os.makedirs(SAVED_CONFIGS_DIR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVED_CONFIGS_DIR,
                                f"deliveries_{len(self.snapped_delivery_points)}_{city_name}_{timestamp}.json")

        deliveries_data = {
            "city": city_name,
            "points": []
        }

        for lat, lon, weight, volume in self.snapped_delivery_points:
            deliveries_data["points"].append({
                "latitude": lat,
                "longitude": lon,
                "weight": weight,
                "volume": volume
            })

        try:
            with open(filename, 'w') as f:
                json.dump(deliveries_data, f, indent=4)
            self.request_show_message.emit("Save Successful",
                                           f"Delivery configuration saved to {filename}",
                                           "information")
        except Exception as e:
            self.request_show_message.emit("Save Failed",
                                           f"Error saving delivery configuration: {e}", "critical")

    def load_deliveries_config(self):
        if self._graph is None:
            self.request_show_message.emit("Graph Not Loaded",
                                           "Please load the graph data first before loading deliveries.",
                                           "warning")
            return

        if not os.path.exists(SAVED_CONFIGS_DIR):
            os.makedirs(SAVED_CONFIGS_DIR)

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(None, "Load Delivery Configuration",
                                                  SAVED_CONFIGS_DIR,
                                                  "JSON Files (*.json);;All Files (*)",
                                                  options=options)

        if not filename:
            return

        if not filename.lower().endswith('.json') or not os.path.basename(filename).startswith(
                "deliveries_"):
            self.request_show_message.emit("Load Failed",
                                           "Invalid file selected. Please select a valid delivery configuration file (deliveries_*.json).",
                                           "warning")
            return

        try:
            with open(filename, 'r') as f:
                deliveries_data = json.load(f)

            if 'city' not in deliveries_data or 'points' not in deliveries_data:
                raise ValueError("Delivery data is missing 'city' or 'points' field.")

            current_city_name_full = self._graph.graph.get('name', 'UnknownCity')
            current_city_name_simple = current_city_name_full.split(',')[0].replace(" ",
                                                                                    "_").lower()
            saved_city_name_simple = deliveries_data['city'].lower()

            if saved_city_name_simple != current_city_name_simple:
                self.request_show_message.emit("City Mismatch",
                                               f"These deliveries are for '{deliveries_data['city']}'.\n"
                                               f"The current map is for '{current_city_name_full.split(',')[0]}'.\n"
                                               f"Please load the correct map or delivery file.",
                                               "warning")
                return

            loaded_delivery_entities = []
            for point_data in deliveries_data["points"]:
                if not all(k in point_data for k in ["latitude", "longitude", "weight", "volume"]):
                    raise ValueError("Delivery point data is missing required fields.")

                delivery = Delivery(
                    coordinates=(point_data["latitude"], point_data["longitude"]),
                    weight=point_data["weight"],
                    volume=point_data["volume"]
                )
                loaded_delivery_entities.append(delivery)

            if not loaded_delivery_entities:
                self.request_show_message.emit("Load Failed",
                                               "No valid delivery points found in the file.",
                                               "warning")
                return

            self._process_delivery_points(loaded_delivery_entities)

            if self.messenger:
                self.messenger.send(MessageType.DELIVERY_POINTS_UPDATED,
                                    {'points': self.snapped_delivery_points})

            self.request_show_message.emit("Load Successful",
                                           f"Delivery configuration loaded from {filename}",
                                           "information")

        except json.JSONDecodeError:
            self.request_show_message.emit("Load Failed",
                                           "Invalid JSON file. Could not decode the file content.",
                                           "critical")
        except ValueError as ve:
            self.request_show_message.emit("Load Failed", f"Invalid data format in file: {ve}",
                                           "critical")
        except Exception as e:
            self.request_show_message.emit("Load Failed",
                                           f"Error loading delivery configuration: {e}", "critical")
            import traceback
            traceback.print_exc()
