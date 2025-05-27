import pickle
import queue
from pathlib import Path

from PyQt5 import QtCore

from models.resolvers.ml_classifier_resolver import MLClassifierResolver
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.simulation_controller import SimulationController
from models.services.disruption.disruption_service import DisruptionService
from viewmodels.viewmodel_messenger import MessageType
from workers.disruption_resolution_worker import DisruptionResolutionWorker
from workers.disruption_generation_worker import DisruptionGenerationWorker


class DisruptionViewModel(QtCore.QObject):
    disruption_generated = QtCore.pyqtSignal(list)
    disruption_activated = QtCore.pyqtSignal(object)
    disruption_resolved = QtCore.pyqtSignal(int)
    request_show_message = QtCore.pyqtSignal(str, str, str)
    active_disruptions_changed = QtCore.pyqtSignal(list)
    request_map_route_update = QtCore.pyqtSignal(int)
    request_show_loading = QtCore.pyqtSignal()
    request_hide_loading = QtCore.pyqtSignal()


    def __init__(self, messenger=None):
        super().__init__()
        self._last_solution_id = None
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
        self.last_generated_solution = None

        self.resolver_thread = QtCore.QThread()
        self.resolver_worker = None
        self.processing_disruptions = set()

        self.generation_thread = QtCore.QThread()
        self.generation_worker = None

        self._pending_driver_updates = set()
        self._processing_timer = QtCore.QTimer(self)
        self._processing_timer.setSingleShot(True)
        self._processing_timer.setInterval(0)
        self._processing_timer.timeout.connect(self._process_queued_route_updates)

        self._route_update_queue = queue.Queue()
        self._route_data_cache = {}
        self._route_cache_lock = QtCore.QMutex()

        self.ml_model_type = 'random_forest'
        self._manual_disruptions = []

        self._load_model_configuration()

        if self.messenger:
            self.messenger.subscribe(MessageType.GRAPH_LOADED, self.handle_graph_loaded)
            self.messenger.subscribe(MessageType.WAREHOUSE_LOCATION_UPDATED,
                                     self.handle_warehouse_location)
            self.messenger.subscribe(MessageType.DELIVERY_POINTS_UPDATED,
                                     self.handle_delivery_points_updated)
            self.messenger.subscribe(MessageType.SIMULATION_STARTED, self.handle_simulation_started)
            self.messenger.subscribe(MessageType.DRIVER_UPDATED, self.handle_drivers_updated)
            self.messenger.subscribe(MessageType.ROUTE_CALCULATED, self.handle_route_calculated)
            self.messenger.subscribe(MessageType.DISRUPTION_ACTIVATED,
                                     self.handle_disruption_activated)
            self.messenger.subscribe(MessageType.DISRUPTION_RESOLVED,
                                     self.handle_disruption_resolved)

    @QtCore.pyqtSlot()
    def _process_queued_route_updates(self):
        print("DisruptionVM: Timer timeout - processing queued route updates.")
        updated_driver_ids_in_batch = set()
        try:
            while not self._route_update_queue.empty():
                try:
                    driver_id, route_string = self._route_update_queue.get_nowait()
                    print(f"DisruptionVM: Dequeued update for driver {driver_id}")
                    with QtCore.QMutexLocker(self._route_cache_lock):
                        self._route_data_cache[driver_id] = route_string
                    updated_driver_ids_in_batch.add(driver_id)
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"DisruptionVM: Error processing queue item: {e}")

            if updated_driver_ids_in_batch:
                print(f"DisruptionVM: Processed updates for drivers: {updated_driver_ids_in_batch}")
                for driver_id in updated_driver_ids_in_batch:
                    print(f"DisruptionVM: Emitting request_map_route_update for driver {driver_id}")
                    self.request_map_route_update.emit(driver_id)
            else:
                print("DisruptionVM: Queue was empty or processing failed.")

        except Exception as e:
            print(f"DisruptionVM: Error in _process_queued_route_updates: {e}")

    @QtCore.pyqtSlot()
    def _handle_controller_update_available(self):
        print("DisruptionVM: Received route_update_available. Scheduling processing.")
        if not self._processing_timer.isActive():
            self._processing_timer.start()

    def initialize_disruption_service(self):
        if self.disruption_service is not None:
            return True

        if self.G and self.warehouse_location and self.delivery_points:
            self.disruption_service = DisruptionService(
                self.G,
                self.warehouse_location,
                self.delivery_points
            )
            return True
        else:
            missing = []
            if not self.G:
                missing.append("G")
            if not self.warehouse_location:
                missing.append("warehouse_location")
            if not self.delivery_points:
                missing.append("delivery_points")
            print(f"Cannot initialize DisruptionService: Missing {', '.join(missing)}.")
            return False

    def handle_route_calculated(self, data):
        if 'solution' in data:
            solution_id = id(data['solution'])
            if not hasattr(self, '_last_solution_id') or self._last_solution_id != solution_id:
                self._last_solution_id = solution_id
                self.current_solution = data['solution']
                if self.disruption_service:
                    self.disruption_service.set_solution(data['solution'])

    def handle_disruption_activated_signal(self, disruption):
        if isinstance(disruption, dict):
            disruption_id = disruption.get('disruption_id')
            if disruption_id is not None:
                self.handle_disruption_activated(disruption)
            else:
                print(f"Warning: Received disruption dict without disruption_id: {disruption}")
        else:
            try:
                self.handle_disruption_activated({'disruption_id': disruption.id})
            except AttributeError:
                print(f"Error: Disruption object has no id attribute: {type(disruption).__name__}")

    def initialize_simulation_controller(self):
        if self.simulation_controller is None:
            if (self.G and self.warehouse_location and self.delivery_points and
                    self.drivers and self.current_solution and self.disruption_service):

                try:
                    self.simulation_controller = SimulationController(
                        graph=self.G,
                        warehouse_location=self.warehouse_location,
                        delivery_points=self.delivery_points,
                        drivers=self.drivers,
                        solution=self.current_solution,
                        disruption_service=self.disruption_service,
                        resolver=self.resolver,
                        route_update_queue=self._route_update_queue
                    )

                    self.simulation_controller.route_update_available.connect(
                        self._handle_controller_update_available,
                        QtCore.Qt.ConnectionType.QueuedConnection
                    )

                    self.simulation_controller.disruption_activated.connect(
                        self.handle_disruption_activated_signal)

                    if self.resolver is None:
                        self.initialize_resolver()

                    if self.resolver:
                        self.simulation_controller.set_resolver(self.resolver)

                    print(
                        f"Simulation controller initialized with solution containing {len(self.current_solution)} assignments")
                    return True

                except Exception as e:
                    print(f"Error initializing simulation controller: {e}")
                    import traceback
                    traceback.print_exc()
                    self.simulation_controller = None
                    return False
            else:
                missing = []
                if not self.G: missing.append("graph")
                if not self.warehouse_location: missing.append("warehouse location")
                if not self.delivery_points: missing.append("delivery points")
                if not self.drivers: missing.append("drivers")
                if not self.current_solution: missing.append("solution")
                if not self.disruption_service: missing.append("disruption service")

                print(f"Cannot initialize SimulationController: Missing {', '.join(missing)}.")
                return False

        self.simulation_controller.initialize_simulation()
        return True

    def get_cached_route_update(self, driver_id):
        with QtCore.QMutexLocker(self._route_cache_lock):
            return self._route_data_cache.pop(driver_id, None)

    def initialize_resolver(self, ml_model_type=None):
        if ml_model_type is not None:
            self.ml_model_type = ml_model_type

        if self.resolver is None:
            if self.G and self.warehouse_location:
                try:
                    print(
                        f"Initializing ML Classifier Resolver with model type: {self.ml_model_type}...")
                    ml_resolver = MLClassifierResolver(
                        graph=self.G,
                        warehouse_location=self.warehouse_location,
                        model_type=self.ml_model_type
                    )

                    if ml_resolver.has_classifier():
                        self.resolver = ml_resolver
                        print(f"ML Classifier model ({self.ml_model_type}) loaded successfully.")
                    else:
                        print(
                            f"ML Classifier model ({self.ml_model_type}) not found, falling back to Rule-Based Resolver...")
                        self.resolver = RuleBasedResolver(
                            graph=self.G,
                            warehouse_location=self.warehouse_location
                        )
                except Exception as e:
                    print(
                        f"Error initializing ML resolver: {e}, falling back to Rule-Based Resolver...")
                    import traceback
                    traceback.print_exc()

                    self.resolver = RuleBasedResolver(
                        graph=self.G,
                        warehouse_location=self.warehouse_location
                    )

                if self.simulation_controller:
                    self.simulation_controller.set_resolver(self.resolver)
                return True
            else:
                print("Cannot initialize resolver: Missing graph or warehouse location.")
                return False
        return True

    def set_ml_model_type(self, model_type):
        if model_type not in ['random_forest', 'neural_network']:
            print(
                f"Warning: Invalid model type '{model_type}'. Must be 'random_forest' or 'neural_network'.")
            return False

        self.ml_model_type = model_type

        if self.resolver is not None:
            print(f"Reinitializing resolver with model type: {model_type}")
            prev_resolver_type = type(self.resolver).__name__

            self.resolver = None

            success = self.initialize_resolver(model_type)

            if success and isinstance(self.resolver, MLClassifierResolver):
                print(f"Successfully switched to {model_type} model.")

                if self.simulation_controller:
                    self.simulation_controller.set_resolver(self.resolver)

                return True
            else:
                print(
                    f"Failed to initialize {model_type} model. Current resolver: {type(self.resolver).__name__}")
                return False

        return True

    def set_manual_disruptions(self, disruptions):
        self._manual_disruptions = disruptions
        print(f"DisruptionViewModel: Set {len(disruptions)} manual disruptions")
        
        if not disruptions and hasattr(self, '_disruptions_generated'):
            self._disruptions_generated = False
            self.disruptions = []
            print("DisruptionViewModel: Reset generation flag and cleared disruptions - auto-generation re-enabled")

    def generate_disruptions(self, num_drivers):
        if self._manual_disruptions:
            print(f"Using {len(self._manual_disruptions)} manually placed disruptions instead of generating new ones")
            print("Auto-generation disabled due to manual disruptions")
            self.handle_generation_complete(self._manual_disruptions)
            return self.disruptions

        if hasattr(self, '_disruptions_generated') and self._disruptions_generated:
            return self.disruptions

        self._disruptions_generated = True

        if not self.disruption_service:
            self.request_show_message.emit(
                "Error",
                "Cannot generate disruptions: Service not ready.",
                "warning"
            )
            return []

        if hasattr(self, 'generation_thread') and self.generation_thread.isRunning():
            self.generation_thread.quit()
            if not self.generation_thread.wait(500):
                self.generation_thread.terminate()
                self.generation_thread.wait()

        try:
            if hasattr(self, 'generation_worker') and self.generation_worker:
                self.generation_worker.generation_complete.disconnect()
                self.generation_worker.generation_failed.disconnect()
        except TypeError:
            pass

        self.generation_thread = QtCore.QThread()
        self.generation_worker = DisruptionGenerationWorker(
            self.disruption_service, 
            num_drivers
        )

        self.generation_worker.moveToThread(self.generation_thread)

        self.generation_worker.generation_complete.connect(
            self.handle_generation_complete,
            QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.generation_worker.generation_failed.connect(
            self.handle_generation_failed,
            QtCore.Qt.ConnectionType.QueuedConnection
        )

        self.generation_thread.started.connect(
            self.generation_worker.generate_disruptions,
            QtCore.Qt.ConnectionType.QueuedConnection
        )

        print(f"DisruptionViewModel: Starting generation thread for {num_drivers} drivers")
        self.generation_thread.start()

        return []

    @QtCore.pyqtSlot(list)
    def handle_generation_complete(self, new_disruptions):
        try:
            print(f"DisruptionViewModel: Generation complete, received {len(new_disruptions)} disruptions")
            
            processed_disruptions = []
            for disruption in new_disruptions:
                if isinstance(disruption, dict):
                    from models.entities.disruption import Disruption, DisruptionType
                    
                    if isinstance(disruption['location'], dict):
                        location = (disruption['location']['lat'], disruption['location']['lng'])
                    else:
                        location = (disruption['location'][0], disruption['location'][1])
                    
                    tripwire_location = None
                    if 'tripwire_location' in disruption:
                        if isinstance(disruption['tripwire_location'], dict):
                            tripwire_location = (disruption['tripwire_location']['lat'], disruption['tripwire_location']['lng'])
                        else:
                            tripwire_location = (disruption['tripwire_location'][0], disruption['tripwire_location'][1])
                    
                    disruption_obj = Disruption(
                        id=disruption['id'],
                        type=DisruptionType(disruption['type']),
                        location=location,
                        affected_area_radius=disruption.get('affected_area_radius', disruption.get('radius', 100)),
                        duration=int(disruption['duration']),
                        severity=disruption['severity'],
                        activation_distance=disruption.get('activation_distance', 200),
                        metadata={
                            'manually_placed': disruption.get('manually_placed', False),
                            'description': disruption.get('metadata', {}).get('description', f"Manually placed {disruption['type']}")
                        }
                    )
                    
                    if tripwire_location:
                        disruption_obj.tripwire_location = tripwire_location
                    if 'owning_driver_id' in disruption:
                        disruption_obj.owning_driver_id = disruption['owning_driver_id']
                    processed_disruptions.append(disruption_obj)
                    print(f"Converted manual disruption with ID {disruption['id']} -> {disruption_obj.id}")
                else:
                    processed_disruptions.append(disruption)
                    print(f"Added existing disruption with ID {disruption.id}")
            
            print(f"Processing {len(processed_disruptions)} disruptions with IDs: {[d.id for d in processed_disruptions]}")
            self.disruptions.extend(processed_disruptions)

            if not self._manual_disruptions:
                used_ids = set()
                for i, d in enumerate(self.disruptions):
                    if d.id in used_ids:
                        self.disruptions[i] = d.copy(update={"id": max(used_ids) + 1})
                    used_ids.add(d.id)
            else:
                print(f"Preserving original IDs for {len(processed_disruptions)} manual disruptions")

            self.disruption_generated.emit(self.disruptions)

            if self.messenger:
                self.messenger.send(MessageType.DISRUPTION_GENERATED, {
                    'disruptions': self.disruptions,
                })

            self.send_disruptions_to_map()

            print(f"DisruptionViewModel: Successfully processed {len(self.disruptions)} total disruptions")

        except Exception as e:
            print(f"DisruptionViewModel: Error in handle_generation_complete: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'generation_thread') and self.generation_thread.isRunning():
                self.generation_thread.quit()
                self.generation_thread.wait(500)

    @QtCore.pyqtSlot(str)
    def handle_generation_failed(self, error_message):
        try:
            print(f"DisruptionViewModel: Generation failed: {error_message}")
            
            self.request_show_message.emit(
                "Error",
                f"Failed to generate disruptions: {error_message}",
                "critical"
            )

        except Exception as e:
            print(f"DisruptionViewModel: Error in handle_generation_failed: {e}")
        finally:
            if hasattr(self, 'generation_thread') and self.generation_thread.isRunning():
                self.generation_thread.quit()
                self.generation_thread.wait(500)

    def get_active_disruptions(self):
        if not self.disruption_service:
            return []

        self.active_disruptions = self.disruption_service.get_active_disruptions()

        self.active_disruptions_changed.emit(self.active_disruptions)

        return self.active_disruptions

    def check_path_disruptions(self, path, simulation_time):
        if not self.disruption_service:
            return []

        return self.disruption_service.get_path_disruptions(path, simulation_time)

    def calculate_delay_factor(self, point, simulation_time):
        if not self.disruption_service:
            return 1.0

        return self.disruption_service.calculate_delay_factor(point, simulation_time)

    def handle_graph_loaded(self, data):
        if 'graph' in data:
            self.G = data['graph']

    def handle_warehouse_location(self, data):
        if 'location' in data:
            self.warehouse_location = data['location']

    def handle_delivery_points_updated(self, data):
        if 'points' in data:
            self.delivery_points = data['points']
            self._check_all_components_ready()

    def handle_simulation_started(self, data):
        if hasattr(self, '_simulation_already_started') and self._simulation_already_started:
            return
        self._simulation_already_started = True

        if 'total_expected_time' in data:
            if not self.disruption_service:
                if not self.initialize_disruption_service():
                    print(
                        "Warning: Disruption service not initialized in handle_simulation_started.")
                    pass

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

                self.generate_disruptions(num_drivers)

                self.initialize_simulation_controller()
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

    def resolve_disruption(self, disruption_id, send_message=True):
        if not self.disruption_service:
            return False

        success = self.disruption_service.resolve_disruption(disruption_id)
        if success:
            self.disruption_resolved.emit(disruption_id)

            if send_message and self.messenger:
                self.messenger.send(MessageType.DISRUPTION_RESOLVED, {
                    'disruption_id': disruption_id,
                })

        return success

    def _check_all_components_ready(self):
        if self.disruption_service is None:
            success = self.initialize_disruption_service()

            if success:
                self.initialize_resolver()

    def handle_drivers_updated(self, data):
        self.drivers = data

    def send_disruptions_to_map(self):
        if self.messenger:
            if hasattr(self, '_last_visualization_sent') and self._last_visualization_sent == id(
                    self.disruptions):
                return
            self._last_visualization_sent = id(self.disruptions)

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
                    'severity': disruption.severity,
                    'description': disruption.metadata.get('description',
                                                           f"{disruption.type.value.replace('_', ' ').title()}"),
                    'activation_distance': disruption.activation_distance,
                    'is_active': disruption.is_active,
                    'duration': disruption.duration
                }
                
                if hasattr(disruption, 'tripwire_location') and disruption.tripwire_location:
                    data['tripwire_location'] = disruption.tripwire_location
                    
                if hasattr(disruption, 'owning_driver_id') and disruption.owning_driver_id is not None:
                    data['owning_driver_id'] = disruption.owning_driver_id
                    
                disruption_data.append(data)

            self.messenger.send(MessageType.DISRUPTION_VISUALIZATION, {
                'disruptions': disruption_data,
            })

    def handle_disruption_activated(self, data):
        try:
            disruption_id = data.get('disruption_id')
            print(f"Attempting to activate disruption {disruption_id}")
            
            if disruption_id in self.processing_disruptions:
                print(f"Disruption {disruption_id} already being processed")
                return

            available_ids = [d.id for d in self.disruptions]
            print(f"Available disruption IDs: {available_ids}")
            
            disruption = next((d for d in self.disruptions if d.id == disruption_id), None)
            if not disruption:
                print(f"Could not find disruption with ID {disruption_id}")
                print(f"Total disruptions available: {len(self.disruptions)}")
                return

            self.processing_disruptions.add(disruption_id)

            if not (self.simulation_controller and self.resolver):
                print(f"Missing controller or resolver, cannot process disruption {disruption_id}")
                if disruption_id in self.processing_disruptions:
                    self.processing_disruptions.remove(disruption_id)
                return

            if hasattr(self, 'resolver_thread') and self.resolver_thread:
                if self.resolver_thread.isRunning():
                    print(
                        f"Stopping existing resolver thread before processing disruption {disruption_id}")
                    self.resolver_thread.quit()

                    if not self.resolver_thread.wait(500):
                        print("Thread did not quit gracefully, terminating")
                        self.resolver_thread.terminate()
                        self.resolver_thread.wait()

                try:
                    if hasattr(self, 'resolver_worker') and self.resolver_worker:
                        self.resolver_worker.resolution_complete.disconnect()
                        self.resolver_worker.log_message.disconnect()
                except TypeError:
                    pass

            self.resolver_thread = QtCore.QThread()

            self.resolver_worker = DisruptionResolutionWorker(
                self.simulation_controller,
                self.resolver,
                disruption.copy() if hasattr(disruption, 'copy') else disruption
            )

            self.resolver_worker.moveToThread(self.resolver_thread)

            self.resolver_worker.resolution_complete.connect(
                self.handle_resolution_complete,
                QtCore.Qt.ConnectionType.QueuedConnection
            )

            self.resolver_thread.started.connect(
                self.resolver_worker.resolve_disruption,
                QtCore.Qt.ConnectionType.QueuedConnection
            )

            print(f"Starting resolution thread for disruption {disruption_id}")
            self.resolver_thread.start()

        except Exception as e:
            print(f"Error in disruption activation handler: {e}")
            import traceback
            traceback.print_exc()

            if 'disruption_id' in locals() and disruption_id in self.processing_disruptions:
                self.processing_disruptions.remove(disruption_id)

    @QtCore.pyqtSlot(list)
    def handle_resolution_complete(self, actions):
        disruption_id = None
        try:
            if hasattr(self, 'resolver_worker') and hasattr(self.resolver_worker, 'disruption'):
                disruption_id = self.resolver_worker.disruption.id
        except Exception:
            pass

        try:
            if actions and hasattr(self, 'simulation_controller') and self.simulation_controller:
                for action in actions:
                    print(f"Action type: {type(action).__name__}")
                    print(f"Action properties: driver_id={getattr(action, 'driver_id', 'N/A')}")
                    try:
                        success = action.execute(self.simulation_controller)
                    except Exception as e:
                        print(f"Error executing action: {e}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"Error in resolution handler: {e}")
        finally:
            if hasattr(self, 'resolver_thread') and self.resolver_thread.isRunning():
                self.resolver_thread.quit()
                self.resolver_thread.wait(500)

            if disruption_id is not None and disruption_id in self.processing_disruptions:
                self.processing_disruptions.remove(disruption_id)
                print(f"Removed disruption {disruption_id} from processing set")

    def handle_disruption_resolved(self, data):
        disruption_id = data.get('disruption_id')
        self.resolve_disruption(disruption_id, send_message=False)

    def _load_model_configuration(self):
        config_file = Path('config') / 'ml_model_config.pkl'

        if config_file.exists():
            try:
                with open(config_file, 'rb') as f:
                    config = pickle.load(f)

                if 'model_type' in config:
                    model_type = config['model_type']
                    if model_type in ['random_forest', 'neural_network']:
                        self.ml_model_type = model_type
                        print(f"Loaded model configuration: using {self.ml_model_type} model")
                    else:
                        print(f"Invalid model type in configuration: {model_type}")
            except Exception as e:
                print(f"Error loading model configuration: {e}")
