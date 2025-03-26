from PyQt5 import QtCore

from models.services.disruption.disruption_service import DisruptionService
from viewmodels.viewmodel_messenger import MessageType


class DisruptionViewModel(QtCore.QObject):
    disruption_generated = QtCore.pyqtSignal(list)
    disruption_activated = QtCore.pyqtSignal(object)
    disruption_resolved = QtCore.pyqtSignal(int)
    request_show_message = QtCore.pyqtSignal(str, str, str)

    def __init__(self, messenger=None):
        super().__init__()
        self.messenger = messenger
        self.disruption_service = None
        self.disruptions = []
        self.active_disruptions = []
        self.G = None
        self.warehouse_location = None
        self.delivery_points = None

        if self.messenger:
            self.messenger.subscribe(MessageType.GRAPH_LOADED, self.handle_graph_loaded)
            self.messenger.subscribe(MessageType.WAREHOUSE_LOCATION_UPDATED, self.handle_warehouse_location)
            self.messenger.subscribe(MessageType.DELIVERY_POINTS_UPDATED, self.handle_delivery_points_updated)
            self.messenger.subscribe(MessageType.SIMULATION_STARTED, self.handle_simulation_started)

    def initialize_disruption_service(self):
        """
        Initialize the disruption service if prerequisites are met and it doesn't exist.
        Returns True if the service exists (or was successfully created), False otherwise.
        """
        if self.disruption_service is None:
            if self.G and self.warehouse_location and self.delivery_points:
                print("Initializing DisruptionService...") # Log initialization
                self.disruption_service = DisruptionService(
                    self.G,
                    self.warehouse_location,
                    self.delivery_points
                )
                return True
            else:
                print("Cannot initialize DisruptionService: Missing G, warehouse, or delivery points.")
                return False
        return True

    def generate_disruptions(self, simulation_duration, num_drivers):
        """Generate disruptions for the simulation"""
        if not self.disruption_service:
            self.request_show_message.emit(
                "Error",
                "Cannot generate disruptions: Service not ready.",
                "warning"
            )
            return []

        # Generate new disruptions without replacing existing ones
        new_disruptions = self.disruption_service.generate_disruptions(
            simulation_duration,
            num_drivers
        )

        # Add to existing disruptions
        self.disruptions.extend(new_disruptions)

        # Make sure IDs are unique
        used_ids = set()
        for i, d in enumerate(self.disruptions):
            if d.id in used_ids:
                self.disruptions[i] = d.copy(update={"id": max(used_ids) + 1})
            used_ids.add(d.id)

        self.disruption_generated.emit(self.disruptions)

        if self.messenger:
            self.messenger.send(MessageType.DISRUPTION_GENERATED, {
                'disruptions': self.disruptions,
            })

        return self.disruptions

    def get_active_disruptions(self, simulation_time):
        """Get disruptions active at the given simulation time"""
        if not self.disruption_service:
            return []

        self.active_disruptions = self.disruption_service.get_active_disruptions(simulation_time)

        return self.active_disruptions

    def check_path_disruptions(self, path, simulation_time):
        """Check for disruptions along a path"""
        if not self.disruption_service:
            return []

        return self.disruption_service.get_path_disruptions(path, simulation_time)

    def calculate_delay_factor(self, point, simulation_time):
        """Calculate delay factor for a point"""
        if not self.disruption_service:
            return 1.0

        return self.disruption_service.calculate_delay_factor(point, simulation_time)

    def handle_graph_loaded(self, data):
        """Handle graph loaded from other ViewModels"""
        if 'graph' in data:
            self.G = data['graph']
            self.initialize_disruption_service()

    def handle_warehouse_location(self, data):
        """Handle warehouse location updates"""
        if 'location' in data:
            self.warehouse_location = data['location']
            self.initialize_disruption_service()

    def handle_delivery_points_updated(self, data):
        """Handle delivery points updates"""
        if 'points' in data:
            self.delivery_points = data['points']
            self.initialize_disruption_service()

    def handle_simulation_started(self, data):
        """Handle simulation start event"""
        if 'total_expected_time' in data:
            if not self.disruption_service:
                if not self.initialize_disruption_service():
                    print("Warning: Disruption service not initialized in handle_simulation_started.")
                    pass

            simulation_duration = data.get('total_expected_time', 7200)
            num_drivers = len(data.get('drivers', []))
            solution = data.get('solution', None)
            all_detailed_route_points = data.get('all_detailed_route_points', [])

            print(
                f"DisruptionViewModel received {len(all_detailed_route_points)} detailed route points.")

            if solution and self.disruption_service:
                self.disruption_service.set_solution(solution)

                self.disruption_service.set_detailed_route_points(
                    all_detailed_route_points)

                self.generate_disruptions(simulation_duration, num_drivers)

                self.send_disruptions_to_map()
            elif not self.disruption_service:
                self.request_show_message.emit(
                    "Warning",
                    "Could not generate disruptions: Disruption service not ready.",
                    "warning"
                )
            elif not solution:
                self.request_show_message.emit(
                    "Warning",
                    "Could not generate disruptions: Missing solution data.",
                    "warning"
                )

    def resolve_disruption(self, disruption_id):
        """Mark a disruption as resolved"""
        if not self.disruption_service:
            return False

        success = self.disruption_service.resolve_disruption(disruption_id)
        if success:
            self.disruption_resolved.emit(disruption_id)

            if self.messenger:
                self.messenger.send(MessageType.DISRUPTION_RESOLVED, {
                    'disruption_id': disruption_id,
                })

        return success

    def send_disruptions_to_map(self):
        """Send disruption data to the map for visualization"""
        if self.messenger:
            disruption_data = []

            for disruption in self.disruptions:
                data = {
                    'id': disruption.id,
                    'type': disruption.type.value,
                    'location': {
                        'lat': disruption.location[0],
                        'lng': disruption.location[1]
                    },
                    'radius': disruption.affected_area_radius,
                    'start_time': disruption.start_time,
                    'duration': disruption.duration,
                    'severity': disruption.severity,
                    'description': disruption.metadata.get('description',
                                                           f"{disruption.type.value.replace('_', ' ').title()}")
                }
                disruption_data.append(data)

            self.messenger.send(MessageType.DISRUPTION_VISUALIZATION, {
                'disruptions': disruption_data,
            })
