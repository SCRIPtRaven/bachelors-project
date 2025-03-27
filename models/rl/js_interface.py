import json

from PyQt5 import QtCore


class SimulationJsInterface(QtCore.QObject):
    """
    Provides an interface for JavaScript to interact with the Python simulation controller.
    Handles message passing between the JavaScript simulation and Python backend.
    """

    # Signals emitted to Python
    disruption_activated = QtCore.pyqtSignal(int)  # disruption_id
    disruption_resolved = QtCore.pyqtSignal(int)  # disruption_id
    driver_position_updated = QtCore.pyqtSignal(int, float, float)  # driver_id, lat, lon
    delivery_completed = QtCore.pyqtSignal(int, int)  # driver_id, delivery_index
    delivery_failed = QtCore.pyqtSignal(int, int)  # driver_id, delivery_index
    simulation_time_updated = QtCore.pyqtSignal(float)  # current_time
    simulation_paused = QtCore.pyqtSignal()
    simulation_resumed = QtCore.pyqtSignal()
    simulation_finished = QtCore.pyqtSignal()
    action_required = QtCore.pyqtSignal(str)  # action_type

    def __init__(self, simulation_controller=None):
        super().__init__()
        self.simulation_controller = simulation_controller

    def set_simulation_controller(self, controller):
        """Set the simulation controller this interface works with"""
        self.simulation_controller = controller

    @QtCore.pyqtSlot(result=bool)
    def isSimulationControllerAvailable(self):
        """Check if the simulation controller is available and properly initialized"""
        if not self.simulation_controller:
            return False

        # Verify that crucial attributes exist
        try:
            # Check for basic required attributes
            has_current_solution = hasattr(self.simulation_controller, 'current_solution')
            has_driver_positions = hasattr(self.simulation_controller, 'driver_positions')

            return has_current_solution and has_driver_positions
        except Exception:
            return False

    @QtCore.pyqtSlot(str, result=str)
    def handleEvent(self, message_str):
        """Handle events from JavaScript"""
        try:
            message = json.loads(message_str)
            event_type = message.get('type')
            data = message.get('data', {})

            if event_type == 'driver_position_updated':
                driver_id = data.get('driver_id')
                lat = data.get('lat')
                lon = data.get('lon')
                self.driver_position_updated.emit(driver_id, lat, lon)
                return self._create_response(True)

            if event_type == 'disruption_activated':
                disruption_id = data.get('disruption_id')
                print(f"JS Interface: Disruption {disruption_id} activated")  # Add debug log
                self.disruption_activated.emit(disruption_id)
                return self._create_response(True)

            elif event_type == 'disruption_resolved':
                disruption_id = data.get('disruption_id')
                self.disruption_resolved.emit(disruption_id)
                return self._create_response(True)

            elif event_type == 'driver_position_updated':
                driver_id = data.get('driver_id')
                lat = data.get('lat')
                lon = data.get('lon')
                self.driver_position_updated.emit(driver_id, lat, lon)
                return self._create_response(True)

            elif event_type == 'delivery_completed':
                driver_id = data.get('driver_id')
                delivery_index = data.get('delivery_index')
                self.delivery_completed.emit(driver_id, delivery_index)
                return self._create_response(True)

            elif event_type == 'delivery_failed':
                driver_id = data.get('driver_id')
                delivery_index = data.get('delivery_index')
                self.delivery_failed.emit(driver_id, delivery_index)
                return self._create_response(True)

            elif event_type == 'simulation_time_updated':
                current_time = data.get('current_time')
                self.simulation_time_updated.emit(current_time)
                return self._create_response(True)

            elif event_type == 'simulation_paused':
                self.simulation_paused.emit()
                return self._create_response(True)

            elif event_type == 'simulation_resumed':
                self.simulation_resumed.emit()
                return self._create_response(True)

            elif event_type == 'simulation_finished':
                self.simulation_finished.emit()
                return self._create_response(True)

            elif event_type == 'action_required':
                action_type = data.get('action_type')
                self.action_required.emit(action_type)
                return self._create_response(True)

            elif event_type == 'get_actions':
                # This is used to retrieve any pending actions for a driver
                driver_id = data.get('driver_id')
                actions = self._get_pending_actions(driver_id)
                return self._create_response(True, {'actions': actions})

            else:
                return self._create_response(False, {'error': f'Unknown event type: {event_type}'})

        except Exception as e:
            return self._create_response(False, {'error': str(e)})

    def _create_response(self, success, data=None):
        """Create a JSON response string"""
        response = {
            'success': success
        }

        if data:
            response['data'] = data

        return json.dumps(response)

    def _get_pending_actions(self, driver_id):
        """Get any pending actions for a driver"""
        if not self.simulation_controller:
            print(f"No simulation controller available for driver {driver_id}")
            return []

        # Get actions for this driver from the simulation controller
        try:
            if hasattr(self.simulation_controller, 'get_pending_actions_for_driver'):
                actions = self.simulation_controller.get_pending_actions_for_driver(driver_id)
                if actions:
                    print(f"Found {len(actions)} pending actions for driver {driver_id}")
                return actions
        except Exception as e:
            print(f"Error getting pending actions: {e}")

        return []

    @QtCore.pyqtSlot(str)
    def updateDriverRoute(self, route_data_str):
        """
        Update a driver's route from JavaScript

        Args:
            route_data_str: JSON string with route data
        """
        try:
            route_data = json.loads(route_data_str)
            driver_id = route_data.get('driver_id')
            new_route = route_data.get('route')

            if self.simulation_controller:
                self.simulation_controller.update_driver_route(driver_id, new_route)

        except Exception as e:
            print(f"Error updating driver route: {e}")

    @QtCore.pyqtSlot(float, float, float)
    def addDisruption(self, lat, lon, radius):
        """
        Add a disruption from JavaScript (for testing)

        Args:
            lat: Latitude
            lon: Longitude
            radius: Radius in meters
        """
        print(f"Adding disruption at ({lat}, {lon}) with radius {radius}")
        # This would typically call into the disruption service
