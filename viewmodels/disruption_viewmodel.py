import os

from PyQt5 import QtCore

from models.rl.rule_based_resolver import RuleBasedResolver
from models.rl.simulation_controller import SimulationController
from models.services.disruption.disruption_service import DisruptionService
from viewmodels.viewmodel_messenger import MessageType


class DisruptionViewModel(QtCore.QObject):
    disruption_generated = QtCore.pyqtSignal(list)
    disruption_activated = QtCore.pyqtSignal(object)
    disruption_resolved = QtCore.pyqtSignal(int)
    request_show_message = QtCore.pyqtSignal(str, str, str)
    action_log_updated = QtCore.pyqtSignal(str)

    def __init__(self, messenger=None):
        super().__init__()
        self.messenger = messenger
        self.disruption_service = None
        self.disruptions = []
        self.active_disruptions = []
        self.G = None
        self.warehouse_location = None
        self.delivery_points = None
        self.simulation_controller = None
        self.resolver = None
        self.drivers = None
        self.current_solution = None

        if self.messenger:
            self.messenger.subscribe(MessageType.GRAPH_LOADED, self.handle_graph_loaded)
            self.messenger.subscribe(MessageType.WAREHOUSE_LOCATION_UPDATED, self.handle_warehouse_location)
            self.messenger.subscribe(MessageType.DELIVERY_POINTS_UPDATED, self.handle_delivery_points_updated)
            self.messenger.subscribe(MessageType.SIMULATION_STARTED, self.handle_simulation_started)
            self.messenger.subscribe(MessageType.DRIVER_UPDATED, self.handle_drivers_updated)
            self.messenger.subscribe(MessageType.DISRUPTION_ACTIVATED, self.handle_disruption_activated)
            self.messenger.subscribe(MessageType.DISRUPTION_RESOLVED, self.handle_disruption_resolved)

    def initialize_disruption_service(self):
        """
        Initialize the disruption service if prerequisites are met and it doesn't exist.
        Returns True if the service exists (or was successfully created), False otherwise.
        """
        if self.disruption_service is not None:
            return True

        if self.G and self.warehouse_location and self.delivery_points:
            print("Initializing DisruptionService...")
            self.disruption_service = DisruptionService(
                self.G,
                self.warehouse_location,
                self.delivery_points
            )
            return True
        else:
            # Log which specific components are missing (only once per set of conditions)
            missing = []
            if not self.G:
                missing.append("G")
            if not self.warehouse_location:
                missing.append("warehouse_location")
            if not self.delivery_points:
                missing.append("delivery_points")
            print(f"Cannot initialize DisruptionService: Missing {', '.join(missing)}.")
            return False

    def initialize_simulation_controller(self):
        """Initialize the simulation controller if prerequisites are met"""
        if self.simulation_controller is None:
            if (self.G and self.warehouse_location and self.delivery_points and
                    self.drivers and self.current_solution and self.disruption_service):

                print("Initializing SimulationController...")
                self.simulation_controller = SimulationController(
                    graph=self.G,
                    warehouse_location=self.warehouse_location,
                    delivery_points=self.delivery_points,
                    drivers=self.drivers,
                    solution=self.current_solution,
                    disruption_service=self.disruption_service
                )

                # Connect signals
                self.simulation_controller.action_log.connect(self.handle_action_log)

                # Initialize the resolver if not already done
                if self.resolver is None:
                    self.initialize_resolver()

                # Set the resolver in the controller
                if self.resolver:
                    self.simulation_controller.set_resolver(self.resolver)

                return True
            else:
                print("Cannot initialize SimulationController: Missing required components.")
                return False
        return True

    def initialize_resolver(self):
        """Initialize the disruption resolver"""
        if self.resolver is None:
            if self.G and self.warehouse_location:
                # Check if trained model exists
                model_path = "dqn_model.h5"

                if os.path.exists(model_path):
                    print("Initializing RL-Based Resolver...")
                    try:
                        from models.rl.rl_resolver import RLResolver
                        self.resolver = RLResolver(
                            model_path=model_path,
                            graph=self.G,
                            warehouse_location=self.warehouse_location
                        )
                        return True
                    except (ImportError, Exception) as e:
                        print(f"Failed to initialize RL resolver: {e}")
                        # Fall back to rule-based resolver

                print("Initializing Rule-Based Resolver...")
                self.resolver = RuleBasedResolver(
                    graph=self.G,
                    warehouse_location=self.warehouse_location
                )
                return True
            else:
                print("Cannot initialize resolver: Missing graph or warehouse location.")
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

        # Update the action log with active disruptions if we have any
        if self.active_disruptions and hasattr(self, 'action_log_updated'):
            active_list = ", ".join([f"{d.type.value} (ID: {d.id})" for d in self.active_disruptions])
            self.action_log_updated.emit(f"Active disruptions: {active_list}")

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

    def handle_warehouse_location(self, data):
        """Handle warehouse location updates"""
        if 'location' in data:
            self.warehouse_location = data['location']

    def handle_delivery_points_updated(self, data):
        """Handle delivery points updates"""
        if 'points' in data:
            self.delivery_points = data['points']
            self._check_all_components_ready()

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

            if solution:
                self.current_solution = solution

            print(
                f"DisruptionViewModel received {len(all_detailed_route_points)} detailed route points.")

            if solution and self.disruption_service:
                self.disruption_service.set_solution(solution)

                self.disruption_service.set_detailed_route_points(
                    all_detailed_route_points)

                self.generate_disruptions(simulation_duration, num_drivers)

                self.initialize_simulation_controller()

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

    def _check_all_components_ready(self):
        """Check if all components are ready and initialize if so"""
        # Only attempt initialization if service doesn't already exist
        if self.disruption_service is None:
            # Try to initialize the service
            success = self.initialize_disruption_service()

            # Only initialize resolver if service initialization succeeded
            if success:
                self.initialize_resolver()

    def handle_drivers_updated(self, data):
        """Handle driver updates"""
        self.drivers = data

    def handle_action_log(self, message):
        """Handle action log messages from the simulation controller"""
        print(f"Action: {message}")
        self.action_log_updated.emit(message)

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

    def handle_disruption_activated(self, data):
        """Handle disruption activation event"""
        disruption_id = data.get('disruption_id')

        # Find the disruption object
        disruption = None
        for d in self.disruptions:
            if d.id == disruption_id:
                disruption = d
                break

        if disruption:
            print(f"DisruptionViewModel: Disruption activated: {disruption.type.value} (ID: {disruption.id})")
            self.action_log_updated.emit(f"Disruption activated: {disruption.type.value} (ID: {disruption.id})")

            # Handle the disruption if we have a simulation controller
            if self.simulation_controller and self.resolver:
                print(f"DisruptionViewModel: Handling disruption with resolver")
                # Ensure resolver is properly set
                if self.simulation_controller.resolver is None:
                    self.simulation_controller.set_resolver(self.resolver)

                actions = self.simulation_controller.handle_disruption(disruption)

                if actions:
                    action_str = ", ".join([a.action_type.name for a in actions])
                    print(f"DisruptionViewModel: Took actions: {action_str}")
                    self.action_log_updated.emit(f"Took actions: {action_str}")
                else:
                    print(f"DisruptionViewModel: No actions taken for disruption {disruption.id}")
                    self.action_log_updated.emit(f"No actions taken for disruption {disruption.id}")

            # Emit signal for UI update
            self.disruption_activated.emit(disruption)

    def handle_disruption_resolved(self, data):
        """Handle disruption resolution event"""
        disruption_id = data.get('disruption_id')
        self.resolve_disruption(disruption_id)
