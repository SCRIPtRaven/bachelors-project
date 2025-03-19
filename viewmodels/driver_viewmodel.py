from PyQt5 import QtCore, QtWidgets

from core.viewmodel import ViewModel
from models import Driver
from models.driver_model import DriverModel
from services.geolocation_service import GeolocationService


class DriverViewModel(ViewModel):
    """
    ViewModel for managing delivery drivers and their operations.
    """
    # Signals
    drivers_changed = QtCore.pyqtSignal(list)  # Emitted when drivers change
    driver_selected = QtCore.pyqtSignal(int)  # Emitted when a driver is selected
    generation_completed = QtCore.pyqtSignal(bool, str)  # Emitted when generation is complete

    def __init__(self, driver_model=None):
        super().__init__()
        self._model = driver_model or DriverModel()
        self._is_generating = False
        self._generation_status = ""
        self._selected_driver_id = None

    @property
    def is_generating(self):
        return self._is_generating

    @property
    def generation_status(self):
        return self._generation_status

    @property
    def selected_driver_id(self):
        return self._selected_driver_id

    @selected_driver_id.setter
    def selected_driver_id(self, driver_id):
        if self._selected_driver_id != driver_id:
            self._selected_driver_id = driver_id
            self.driver_selected.emit(driver_id)
            self.property_changed.emit('_selected_driver_id')

    @property
    def drivers(self):
        return self._model.drivers

    def generate_drivers(self, num_drivers):
        """
        Ultra-conservative driver generation that avoids complex object interactions.
        """
        # Set state to generating
        self.set_property('_is_generating', True)
        self.set_property('_generation_status', f"Generating {num_drivers} drivers...")

        # Create a progress dialog to show generation progress
        progress = QtWidgets.QProgressDialog(f"Generating {num_drivers} drivers...",
                                             "Cancel", 0, 100)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        try:
            # Initial progress update
            progress.setValue(10)
            QtCore.QCoreApplication.processEvents()

            # Step 1: Generate driver properties (simple data, not objects)
            driver_properties = []
            for i in range(num_drivers):
                if progress.wasCanceled():
                    break

                # Update progress periodically
                if i % 2 == 0:
                    progress.setValue(10 + int(40 * i / num_drivers))
                    QtCore.QCoreApplication.processEvents()

                # Generate just the properties, not the Driver objects yet
                weight_capacity, volume_capacity = GeolocationService.generate_random_driver_properties()
                driver_properties.append((i + 1, weight_capacity, volume_capacity))

            # Step 2: Clear the model
            progress.setValue(60)
            QtCore.QCoreApplication.processEvents()

            self._model.clear_drivers()

            progress.setValue(70)
            QtCore.QCoreApplication.processEvents()

            # Step 3: Create driver objects and add to model one by one with process events
            for i, (driver_id, weight, volume) in enumerate(driver_properties):
                if progress.wasCanceled():
                    break

                if i % 2 == 0:
                    progress.setValue(70 + int(20 * i / len(driver_properties)))
                    QtCore.QCoreApplication.processEvents()

                # Create and add the driver
                driver = Driver(id=driver_id, weight_capacity=weight, volume_capacity=volume)
                self._model.add_driver(driver)

            # Final progress update
            progress.setValue(90)
            QtCore.QCoreApplication.processEvents()

            # Create a status message
            status = f"Generated {len(self._model.drivers)} drivers."
            self.set_property('_generation_status', status)

            # Make a simple copy of driver data for emission
            driver_data = []
            for driver in self._model.drivers:
                driver_data.append(driver)

            progress.setValue(95)
            QtCore.QCoreApplication.processEvents()

            # Reset generating state BEFORE emitting signals
            self.set_property('_is_generating', False)

            # Emit signals with simple references to the already-created objects
            progress.setValue(97)
            QtCore.QCoreApplication.processEvents()

            # This is likely where the crash happens - make the steps super explicit
            # to find the exact failure point
            print("About to emit drivers_changed signal")
            self.drivers_changed.emit(self._model.drivers)
            print("Emitted drivers_changed signal")

            print("About to emit generation_completed signal")
            self.generation_completed.emit(True, status)
            print("Emitted generation_completed signal")

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error generating drivers: {str(e)}"
            self.set_property('_generation_status', error_msg)
            self.set_property('_is_generating', False)
            self.generation_completed.emit(False, error_msg)
        finally:
            # Ensure generating state is reset
            self.set_property('_is_generating', False)
            progress.close()

    def _handle_generation_result(self, result):
        """
        Handles the result of driver generation in a thread-safe manner.

        This function receives results from the background thread and safely
        transfers ownership of data to the main thread through deep copying
        and delayed signal emission.

        Args:
            result: Dictionary containing generation results
        """
        # Update state (now on main thread)
        self.set_property('_is_generating', False)

        if result['success']:
            # Create a deep copy of the drivers to ensure thread safety
            import copy
            drivers_copy = copy.deepcopy(result['drivers'])

            # Update model with new drivers
            self._model.clear_drivers()
            for driver in drivers_copy:
                self._model.add_driver(driver)

            # Prepare status message
            status = f"Generated {len(drivers_copy)} drivers."
            self.set_property('_generation_status', status)

            # Use a short delay to ensure memory stability before emitting signals
            # This prevents access to potentially unstable memory across thread boundaries
            QtCore.QTimer.singleShot(10, lambda: self._emit_data_changed(drivers_copy, status))
        else:
            # Handle error case
            error_msg = f"Error generating drivers: {result.get('error', 'Unknown error')}"
            self.set_property('_generation_status', error_msg)
            self.generation_completed.emit(False, error_msg)

    def _emit_data_changed(self, drivers, status):
        """
        Safely emit signals with the copied driver data.

        This function runs on the main thread after a short delay to ensure
        memory stability, and uses drivers that have already been safely copied.

        Args:
            drivers: The driver objects to emit (already deep-copied)
            status: Status message to emit with completion signal
        """
        # The drivers are already copies and have been added to the model
        # We're emitting a reference to the model's drivers, which is safe
        # since they're now owned by the main thread
        self.drivers_changed.emit(self._model.drivers)
        self.generation_completed.emit(True, status)

    def _handle_generation_error(self, error):
        """Handles errors during driver generation."""
        self.set_property('_is_generating', False)
        error_msg = f"Error generating drivers: {str(error)}"
        self.set_property('_generation_status', error_msg)
        self.generation_completed.emit(False, error_msg)

    def update_driver_statistics(self, statistics_data):
        """
        Updates statistics for multiple drivers.

        Args:
            statistics_data: Dictionary mapping driver_id to statistics
        """
        for driver_id, stats in statistics_data.items():
            self._model.update_driver_stats(driver_id, stats)

        # Notify that driver data has changed
        self.drivers_changed.emit(self._model.drivers)

    def select_driver(self, driver_id):
        """
        Selects a driver by ID or deselects if the same driver is selected again.

        Args:
            driver_id: The ID of the driver to select
        """
        if self._selected_driver_id == driver_id:
            self.selected_driver_id = None
        else:
            self.selected_driver_id = driver_id

    def get_driver_display_data(self):
        """
        Get driver data formatted for display in the UI.

        Returns:
            List of dictionaries with driver display information
        """
        display_data = []

        for driver in self.drivers:
            # Basic driver info
            driver_data = {
                'id': driver.id,
                'weight_capacity': driver.weight_capacity,
                'volume_capacity': driver.volume_capacity,
                'selected': driver.id == self._selected_driver_id
            }

            # Add statistics if available
            if hasattr(driver, 'stats'):
                driver_data['stats'] = driver.stats

            display_data.append(driver_data)

        return display_data

    def format_driver_label_text(self, driver):
        """
        Generate formatted text for a driver label.

        Args:
            driver: Driver object

        Returns:
            Formatted text string for display
        """
        if hasattr(driver, 'stats') and driver.stats:
            stats = driver.stats
            # Format with statistics
            hours = int(stats.get('travel_time', 0) // 3600)
            minutes = int((stats.get('travel_time', 0) % 3600) // 60)
            seconds = int(stats.get('travel_time', 0) % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            weight_pct = (stats.get('weight', 0) / driver.weight_capacity) * 100 if driver.weight_capacity > 0 else 0
            volume_pct = (stats.get('volume', 0) / driver.volume_capacity) * 100 if driver.volume_capacity > 0 else 0

            return (
                f"Driver {driver.id}: {stats.get('deliveries', 0)} deliveries\n"
                f"Time: {time_str}\n"
                f"Distance: {stats.get('distance', 0):.2f} km\n"
                f"Weight: {stats.get('weight', 0):.1f}/{driver.weight_capacity:.1f} kg ({weight_pct:.1f}%)\n"
                f"Volume: {stats.get('volume', 0):.3f}/{driver.volume_capacity:.3f} m³ ({volume_pct:.1f}%)"
            )
        else:
            # Basic format without statistics
            return f"Driver {driver.id}: Capacity {driver.weight_capacity}kg, {driver.volume_capacity}m³"
