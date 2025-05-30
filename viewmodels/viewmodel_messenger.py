from PyQt5 import QtCore


class MessageType:
    DELIVERY_FAILED = "delivery_failed"
    DELIVERY_COMPLETED = "delivery_completed"
    GRAPH_LOADED = "graph_loaded"
    GRAPH_UPDATED = "graph_updated"
    WAREHOUSE_LOCATION_UPDATED = "warehouse_location_updated"

    DRIVER_UPDATED = "driver_updated"
    DELIVERY_POINTS_UPDATED = "delivery_points_updated"

    DRIVER_SELECTED = "driver_selected"

    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_PROGRESS = "optimization_progress"
    ROUTE_CALCULATED = "route_calculated"
    SIMULATION_STARTED = "simulation_started"

    VISUALIZATION_NEEDED = "visualization_needed"
    STATS_UPDATED = "stats_updated"

    DISRUPTION_GENERATED = "disruption_generated"
    DISRUPTION_ACTIVATED = "disruption_activated"
    DISRUPTION_RESOLVED = "disruption_resolved"
    DISRUPTION_VISUALIZATION = "disruption_visualization"
    SIMULATION_TIME_UPDATED = "simulation_time_updated"


class Messenger(QtCore.QObject):
    message_sent = QtCore.pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._subscribers = {}

    def send(self, message_type, data=None):
        message_hash = hash(str(message_type) + str(data))

        if hasattr(self, '_last_messages'):
            if message_hash in self._last_messages:
                return
        else:
            self._last_messages = set()

        self._last_messages.add(message_hash)
        if len(self._last_messages) > 50:
            self._last_messages = set(list(self._last_messages)[-50:])

        #print(f"Messenger: Sending message {message_type}")
        self.message_sent.emit(message_type, data)

    def subscribe(self, message_type, callback):
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []

        if callback not in self._subscribers[message_type]:
            self._subscribers[message_type].append(callback)

            if len(self._subscribers[message_type]) == 1:
                self.message_sent.connect(self._dispatch_message)

    def unsubscribe(self, message_type, callback):
        if message_type in self._subscribers and callback in self._subscribers[message_type]:
            self._subscribers[message_type].remove(callback)

            if not self._subscribers[message_type] and self.message_sent.receivers() > 0:
                self.message_sent.disconnect(self._dispatch_message)
                print(f"Messenger: Removed last subscriber from {message_type}")

    def _dispatch_message(self, message_type, data):
        if message_type in self._subscribers:
            for callback in self._subscribers[message_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in message handler for {message_type}: {str(e)}")
