import json
import os
import time
from datetime import datetime

from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

from models.entities.driver import Driver
from models.services.geolocation_service import GeolocationService
from viewmodels.viewmodel_messenger import MessageType

SAVED_CONFIGS_DIR = "saved_configurations"


class DriverViewModel(QtCore.QObject):
    driver_list_changed = QtCore.pyqtSignal(list)
    driver_stats_updated = QtCore.pyqtSignal(dict)
    driver_selected = QtCore.pyqtSignal(int)
    request_show_message = QtCore.pyqtSignal(str, str, str)

    def __init__(self, messenger=None):
        super().__init__()
        self.delivery_drivers = []
        self.selected_driver_id = None
        self.visualization_queue = None
        self.messenger = messenger

        if self.messenger:
            self.messenger.subscribe(MessageType.ROUTE_CALCULATED, self.handle_route_calculated)

    def set_visualization_queue(self, queue):
        self.visualization_queue = queue

    def generate_drivers(self, num_drivers):
        try:
            self.delivery_drivers = GeolocationService.generate_delivery_drivers(num_drivers)
            self.driver_list_changed.emit(self.delivery_drivers)
            if self.messenger:
                self.messenger.send(MessageType.DRIVER_UPDATED, self.delivery_drivers)
        except Exception as e:
            self.request_show_message.emit("Error", f"Error generating drivers: {e}", "critical")

    def update_driver_stats(self, solution_data):
        if not solution_data or not self.delivery_drivers:
            return

        formatted_stats = {}

        for driver_id, stats in solution_data.items():
            driver = next((d for d in self.delivery_drivers if d.id == driver_id), None)
            if not driver:
                continue

            hours = int(stats['travel_time'] // 3600)
            minutes = int((stats['travel_time'] % 3600) // 60)
            seconds = int(stats['travel_time'] % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            weight_pct = (stats['weight'] / driver.weight_capacity) * 100
            volume_pct = (stats['volume'] / driver.volume_capacity) * 100

            formatted_stats[driver_id] = {
                'text': (
                    f"Driver {driver_id}: {stats['deliveries']} deliveries\n"
                    f"Time: {time_str}\n"
                    f"Distance: {stats['distance']:.2f} km\n"
                    f"Weight: {stats['weight']:.1f}/{driver.weight_capacity:.1f} kg ({weight_pct:.1f}%)\n"
                    f"Volume: {stats['volume']:.3f}/{driver.volume_capacity:.3f} m³ ({volume_pct:.1f}%)"
                ),
                'stats': stats,
                'capacity': {
                    'weight': driver.weight_capacity,
                    'volume': driver.volume_capacity
                }
            }

        self.driver_stats_updated.emit(formatted_stats)

    def on_driver_double_clicked(self, driver_id):
        self.selected_driver_id = None if self.selected_driver_id == driver_id else driver_id

        self.driver_selected.emit(self.selected_driver_id)

        if self.messenger:
            self.messenger.send(MessageType.DRIVER_SELECTED, {
                'driver_id': self.selected_driver_id,
                'timestamp': time.time()
            })

    def handle_route_calculated(self, data):
        if 'driver_stats' in data:
            self.update_driver_stats(data['driver_stats'])

    def validate_and_generate_drivers(self, num_drivers_text):
        if not num_drivers_text.isdigit():
            self.request_show_message.emit("Invalid Input",
                                           "Please enter a valid number of drivers.", "warning")
            return False, "Please enter a valid number of drivers."

        num_drivers = int(num_drivers_text)
        self.generate_drivers(num_drivers)
        return True, ""

    def save_drivers_config(self):
        if not self.delivery_drivers:
            self.request_show_message.emit("No Drivers", "No drivers have been generated to save.",
                                           "warning")
            return

        if not os.path.exists(SAVED_CONFIGS_DIR):
            os.makedirs(SAVED_CONFIGS_DIR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVED_CONFIGS_DIR,
                                f"drivers_{len(self.delivery_drivers)}_{timestamp}.json")

        drivers_data = []
        for driver in self.delivery_drivers:
            drivers_data.append({
                "id": driver.id,
                "weight_capacity": driver.weight_capacity,
                "volume_capacity": driver.volume_capacity
            })

        try:
            with open(filename, 'w') as f:
                json.dump(drivers_data, f, indent=4)
            self.request_show_message.emit("Save Successful",
                                           f"Driver configuration saved to {filename}",
                                           "information")
        except Exception as e:
            self.request_show_message.emit("Save Failed", f"Error saving driver configuration: {e}",
                                           "critical")

    def load_drivers_config(self):
        if not os.path.exists(SAVED_CONFIGS_DIR):
            os.makedirs(SAVED_CONFIGS_DIR)

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(None, "Load Driver Configuration",
                                                  SAVED_CONFIGS_DIR,
                                                  "JSON Files (*.json);;All Files (*)",
                                                  options=options)

        if not filename:
            return

        if not filename.lower().endswith('.json') or not os.path.basename(filename).startswith(
                "drivers_"):
            self.request_show_message.emit("Load Failed",
                                           "Invalid file selected. Please select a valid driver configuration file (drivers_*.json).",
                                           "warning")
            return

        try:
            with open(filename, 'r') as f:
                drivers_data = json.load(f)

            loaded_drivers = []
            for driver_data in drivers_data:
                if not all(k in driver_data for k in ["id", "weight_capacity", "volume_capacity"]):
                    raise ValueError(
                        "Driver data is missing one or more required fields (id, weight_capacity, volume_capacity).")

                driver = Driver(
                    id=driver_data["id"],
                    weight_capacity=driver_data["weight_capacity"],
                    volume_capacity=driver_data["volume_capacity"]
                )
                loaded_drivers.append(driver)

            if not loaded_drivers:
                self.request_show_message.emit("Load Failed",
                                               "No valid driver data found in the file.", "warning")
                return

            self.delivery_drivers = loaded_drivers
            self.driver_list_changed.emit(self.delivery_drivers)
            if self.messenger:
                self.messenger.send(MessageType.DRIVER_UPDATED, self.delivery_drivers)
            self.request_show_message.emit("Load Successful",
                                           f"Driver configuration loaded from {filename}",
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
                                           f"Error loading driver configuration: {e}", "critical")
