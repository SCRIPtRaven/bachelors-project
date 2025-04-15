import time

from PyQt5 import QtCore

from models.services.geolocation import GeolocationService
from viewmodels.viewmodel_messenger import MessageType


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
        """Process and emit driver statistics for UI update"""
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
        """Handle driver selection"""
        self.selected_driver_id = None if self.selected_driver_id == driver_id else driver_id

        self.driver_selected.emit(self.selected_driver_id)

        if self.messenger:
            self.messenger.send(MessageType.DRIVER_SELECTED, {
                'driver_id': self.selected_driver_id,
                'timestamp': time.time()
            })

    def handle_route_calculated(self, data):
        """Handle route calculation messages from other ViewModels"""
        if 'driver_stats' in data:
            self.update_driver_stats(data['driver_stats'])

    def validate_and_generate_drivers(self, num_drivers_text):
        """Validate input and generate drivers"""
        if not num_drivers_text.isdigit():
            return False, "Please enter a valid number of drivers."

        num_drivers = int(num_drivers_text)
        self.generate_drivers(num_drivers)
        return True, ""
