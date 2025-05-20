from PyQt5 import QtCore, QtWidgets

from models.resolvers.js_interface import SimulationJsInterface
from utils.geo_utils import get_city_coordinates
from utils.visualization import VisualizationQueue
from viewmodels.delivery_viewmodel import DeliveryViewModel
from viewmodels.disruption_viewmodel import DisruptionViewModel
from viewmodels.driver_viewmodel import DriverViewModel
from viewmodels.optimization_viewmodel import OptimizationViewModel
from viewmodels.viewmodel_messenger import Messenger, MessageType
from views.components.labels.clickable_label import ClickableLabel
from views.components.map.leaflet_map import LeafletMapWidget
from workers.graph_loader import GraphLoadWorker


class MapWidget(LeafletMapWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.messenger = Messenger()

        self.js_interface = SimulationJsInterface()
        self.channel.registerObject("simInterface", self.js_interface)

        self.delivery_viewmodel = DeliveryViewModel(messenger=self.messenger)
        self.driver_viewmodel = DriverViewModel(messenger=self.messenger)
        self.disruption_viewmodel = DisruptionViewModel(messenger=self.messenger)
        self.optimization_viewmodel = OptimizationViewModel(
            delivery_viewmodel=self.delivery_viewmodel,
            driver_viewmodel=self.driver_viewmodel,
            messenger=self.messenger,
            disruption_viewmodel=self.disruption_viewmodel,
        )

        self.delivery_viewmodel.delivery_points_processed.connect(
            self.handle_delivery_points_processed)
        self.delivery_viewmodel.request_show_message.connect(self.show_message)
        self.delivery_viewmodel.warehouse_location_changed.connect(
            self.handle_warehouse_location_changed)

        self.driver_viewmodel.driver_list_changed.connect(self.update_driver_list)
        self.driver_viewmodel.driver_stats_updated.connect(self.update_driver_stats_display)
        self.driver_viewmodel.driver_selected.connect(self.handle_driver_selection)
        self.driver_viewmodel.request_show_message.connect(self.show_message)

        self.optimization_viewmodel.request_show_loading.connect(self.show_loading)
        self.optimization_viewmodel.request_hide_loading.connect(self.hide_loading)
        self.optimization_viewmodel.request_clear_layers.connect(self.clear_all_layers)
        self.optimization_viewmodel.request_add_warehouse.connect(self.add_warehouse)
        self.optimization_viewmodel.request_add_delivery_points.connect(self.add_delivery_points)
        self.optimization_viewmodel.request_update_visualization.connect(self.update_visualization)
        self.optimization_viewmodel.request_update_stats.connect(self.update_stats)
        self.optimization_viewmodel.request_start_simulation.connect(self.start_simulation)
        self.optimization_viewmodel.request_set_simulation_speed.connect(self.set_simulation_speed)
        self.optimization_viewmodel.request_show_message.connect(self.show_message)
        self.optimization_viewmodel.request_enable_ui.connect(self.setEnabled)
        self.optimization_viewmodel.visualization_data_ready.connect(
            self.updateVisualizationFromBackground)
        self.optimization_viewmodel.enable_simulation_button.connect(self.enable_simulation_button)
        self.optimization_viewmodel.request_load_disruptions.connect(self.load_disruptions)

        self.disruption_viewmodel.disruption_generated.connect(self.handle_disruptions_generated)
        self.disruption_viewmodel.disruption_activated.connect(self.handle_disruption_activated)
        self.disruption_viewmodel.disruption_resolved.connect(self.handle_disruption_resolved)
        self.disruption_viewmodel.request_show_message.connect(self.show_message)
        self.disruption_viewmodel.request_map_route_update.connect(
            self.handle_viewmodel_route_update_request
        )

        self.visualization_queue = VisualizationQueue(self.optimization_viewmodel)
        self.driver_viewmodel.set_visualization_queue(self.visualization_queue)

        self.js_interface.disruption_activated.connect(self.handle_js_disruption_activated)
        self.js_interface.disruption_resolved.connect(self.handle_js_disruption_resolved)
        self.js_interface.driver_position_updated.connect(self.handle_js_driver_position)
        self.js_interface.delivery_completed.connect(self.handle_js_delivery_completed)
        self.js_interface.delivery_failed.connect(self.handle_js_delivery_failed)
        self.js_interface.simulation_time_updated.connect(self.handle_js_simulation_time)
        self.js_interface.driver_position_updated.connect(self.handle_js_driver_position)
        self.js_interface.disruption_activated.connect(self.handle_js_disruption_activated)

        self.driver_labels = {}
        self._selected_driver_id = None

        self.is_loading = False

    @QtCore.pyqtSlot(int)
    def handle_viewmodel_route_update_request(self, driver_id):
        print(f"MapWidget: Received route update *request* for driver {driver_id}")
        route_string = self.disruption_viewmodel.get_cached_route_update(driver_id)
        if route_string is not None:
            print(
                f"MapWidget: Retrieved route string for driver {driver_id} (length {len(route_string)})")
            self._execute_js_route_update(driver_id, route_string)
        else:
            print(
                f"MapWidget: Warning - No cached route string found for driver {driver_id} (could be race condition or error)")

    def _execute_js_route_update(self, driver_id, route_string):
        try:
            js_safe_route_string = route_string.replace('\\', '\\\\').replace("'", "\\'")

            js_code = f"""
            (function() {{ 
                console.log('JS CALL from MapWidget for driver {driver_id}');

                const driver = simulationDrivers.find(d => d.id === {driver_id});
                if (!driver) {{ console.error('Driver {driver_id} not found'); return false; }}
                
                let routeData;
                try {{
                    routeData = JSON.parse('{js_safe_route_string}'); 
                }} catch(e) {{ console.error('Error parsing route data:', e); return false; }}
                
                let routePoints = [];
                if (routeData.points) {{
                    routePoints = routeData.points.split(';').map(pair => {{ 
                        const coords = pair.split(','); 
                        return [parseFloat(coords[0]), parseFloat(coords[1])]; 
                    }}).filter(p => p[0] && p[1]);
                }}
                
                if (routePoints.length < 2) {{
                    console.error('Not enough route points');
                    return false;
                }}
                
                const action = {{
                    action_type: 'REROUTE_BASIC',
                    driver_id: {driver_id},
                    new_route: routePoints,
                    times: routeData.times || [], 
                    delivery_indices: routeData.delivery_indices || [],
                    rerouted_segment_start: routeData.rerouted_segment_start,
                    rerouted_segment_end: routeData.rerouted_segment_end
                }};
                
                if (typeof handleRerouteAction === 'function') {{ 
                    handleRerouteAction(map, driver, action);
                    return true;
                }} else {{ 
                    console.error('handleRerouteAction function is not available');
                    return false;
                }}
            }})();
            """

            if js_code:
                print(
                    f"MapWidget: Executing JS route update (Full Route) for driver {driver_id}...")
                self.execute_js(js_code)
            else:
                print(f"MapWidget: No JS code generated for route update driver {driver_id}")

        except Exception as e:
            print(f"MapWidget: Error processing route update string: {e}")
            print(f"Route string was: {route_string}")
            import traceback
            traceback.print_exc()

    def handle_js_driver_position(self, driver_id, lat, lon):
        if hasattr(self, 'disruption_viewmodel') and self.disruption_viewmodel:
            if hasattr(self.disruption_viewmodel, 'simulation_controller') and \
                    self.disruption_viewmodel.simulation_controller:
                self.disruption_viewmodel.simulation_controller.update_driver_position(
                    driver_id, (lat, lon))

    def handle_js_disruption_activated(self, disruption_id):
        print(f"Python received disruption activation for ID: {disruption_id}")
        if self.messenger:
            self.messenger.send(MessageType.DISRUPTION_ACTIVATED, {
                'disruption_id': disruption_id
            })

    def handle_delivery_points_processed(self, points, successful, skipped):
        self.add_delivery_points(points)

        if skipped > 0:
            self.show_message(
                "Delivery Points Generated",
                f"Successfully placed {successful} delivery points.\n"
                f"Skipped {skipped} inaccessible points.",
                "information"
            )

    def handle_js_disruption_resolved(self, disruption_id):
        if self.messenger:
            self.messenger.send(MessageType.DISRUPTION_RESOLVED, {
                'disruption_id': disruption_id
            })

    def handle_js_delivery_completed(self, driver_id, delivery_index):
        if self.messenger:
            self.messenger.send(MessageType.DELIVERY_COMPLETED, {
                'driver_id': driver_id,
                'delivery_index': delivery_index
            })

        if hasattr(self, 'disruption_viewmodel') and self.disruption_viewmodel:
            if hasattr(self.disruption_viewmodel,
                       'simulation_controller') and self.disruption_viewmodel.simulation_controller:
                self.disruption_viewmodel.simulation_controller.completed_deliveries.add(
                    delivery_index)

    def handle_js_delivery_failed(self, driver_id, delivery_index):
        if self.messenger:
            self.messenger.send(MessageType.DELIVERY_FAILED, {
                'driver_id': driver_id,
                'delivery_index': delivery_index
            })

        if hasattr(self, 'disruption_viewmodel') and self.disruption_viewmodel:
            if hasattr(self.disruption_viewmodel,
                       'simulation_controller') and self.disruption_viewmodel.simulation_controller:
                self.disruption_viewmodel.simulation_controller.skipped_deliveries.add(
                    delivery_index)

    def handle_js_simulation_time(self, simulation_time):
        if self.messenger:
            self.messenger.send(MessageType.SIMULATION_TIME_UPDATED, {
                'simulation_time': simulation_time
            })

        if hasattr(self, 'disruption_viewmodel') and self.disruption_viewmodel:
            if hasattr(self.disruption_viewmodel,
                       'simulation_controller') and self.disruption_viewmodel.simulation_controller:
                self.disruption_viewmodel.simulation_controller.update_simulation_time(
                    simulation_time)

    def update_visualization(self, solution, unassigned_deliveries=None):
        if isinstance(solution, tuple):
            solution, unassigned = solution
        else:
            unassigned = unassigned_deliveries

        self.addToVisualizationQueue((solution, unassigned))

    def handle_disruptions_generated(self, disruptions):
        pass

    def handle_disruption_activated(self, disruption):
        print(f"Disruption {disruption.id} activated: {disruption.type.value}")

    def handle_disruption_resolved(self, disruption_id):
        print(f"Disruption {disruption_id} resolved")

    def load_disruptions(self, disruptions):
        super().load_disruptions(disruptions)

    def handle_warehouse_location_changed(self, location):
        self.add_warehouse(location[0], location[1])

        self.optimization_viewmodel.set_warehouse_location(location)

    def update_driver_list(self, drivers):
        layout = self.time_label.parent().layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QtWidgets.QScrollArea) or (
                    isinstance(widget, QtWidgets.QLabel) and widget.text() == "Delivery Drivers:"):
                widget.setParent(None)

        header_label = QtWidgets.QLabel("Delivery Drivers:")
        header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header_label)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        scroll_area.setMaximumHeight(300)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background: white;
            }
        """)

        driver_container = QtWidgets.QWidget()
        driver_layout = QtWidgets.QVBoxLayout(driver_container)
        driver_layout.setSpacing(5)
        driver_layout.setContentsMargins(5, 5, 5, 5)

        self.driver_labels = {}
        for driver in drivers:
            label = ClickableLabel(
                f"Driver {driver.id}: Capacity {driver.weight_capacity}kg, {driver.volume_capacity}m³",
                driver_id=driver.id
            )
            label.doubleClicked.connect(self.driver_viewmodel.on_driver_double_clicked)
            self.driver_labels[driver.id] = label
            driver_layout.addWidget(label)

        driver_layout.addStretch()
        scroll_area.setWidget(driver_container)
        layout.addWidget(scroll_area)

    def update_driver_stats_display(self, formatted_stats):
        if not self.driver_labels:
            return

        for driver_id, stats in formatted_stats.items():
            if driver_id in self.driver_labels:
                self.driver_labels[driver_id].setText(stats['text'])
                self.driver_labels[driver_id].setWordWrap(True)
                self.driver_labels[driver_id].setMinimumHeight(80)

    def handle_driver_selection(self, driver_id):
        self._selected_driver_id = driver_id

        for d_id, label in self.driver_labels.items():
            label.setSelected(d_id == driver_id)

    def show_loading(self):
        self.execute_js(
            "if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

    def hide_loading(self):
        self.execute_js(
            "if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

    def enable_simulation_button(self, enabled):
        main_window = self.get_main_window()
        if main_window and hasattr(main_window, 'btn_simulate'):
            main_window.btn_simulate.setEnabled(enabled)

    def update_stats(self, computation_time, travel_time, distance):
        if hasattr(self, 'time_label'):
            self.time_label.setText(f"Routes computed in {computation_time:.2f} seconds")

        if hasattr(self, 'travel_time_label'):
            total_minutes = travel_time / 60
            self.travel_time_label.setText(f"Total travel time: {total_minutes:.2f} minutes")

        if hasattr(self, 'distance_label'):
            total_kilometers = distance / 1000
            self.distance_label.setText(f"Total distance: {total_kilometers:.2f} km")

    def show_message(self, title, message, message_type):
        if message_type == "information":
            QtWidgets.QMessageBox.information(self, title, message)
        elif message_type == "warning":
            QtWidgets.QMessageBox.warning(self, title, message)
        elif message_type == "critical":
            QtWidgets.QMessageBox.critical(self, title, message)

    def set_stats_labels(self, time_label, travel_time_label, distance_label):
        self.time_label = time_label
        self.travel_time_label = travel_time_label
        self.distance_label = distance_label

    def load_graph_data(self, city_name):
        if self.is_loading:
            return

        self.is_loading = True
        self.setEnabled(False)

        self.load_worker = GraphLoadWorker(city_name)
        self.load_worker.finished.connect(self.on_graph_loaded)
        self.load_worker.start()

    def on_graph_loaded(self, success, message, graph, city_name):
        self.is_loading = False
        self.setEnabled(True)

        if success:
            self.G = graph
            self.current_city = city_name

            self.messenger.send(MessageType.GRAPH_LOADED, {
                'graph': graph,
                'city_name': city_name
            })

            center, zoom = get_city_coordinates(city_name)
            self.init_map(center, zoom)
            self.load_map()

            QtWidgets.QMessageBox.information(self, "Success", message)
            self.load_completed.emit(True, message)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            self.load_completed.emit(False, message)

        self.load_worker.deleteLater()

    def find_shortest_route(self):
        if not self._validate_optimization_prerequisites():
            return

        try:
            if hasattr(self, 'optimization_thread'):
                if self.optimization_thread.isRunning():
                    self.optimization_thread.quit()
                    self.optimization_thread.wait()
                self.optimization_thread.deleteLater()

            self.optimization_thread = QtCore.QThread()

            self.optimization_viewmodel.prepare_optimization(
                self.delivery_drivers,
                self.delivery_viewmodel.snapped_delivery_points,
                self.G
            )

            self.optimization_viewmodel.moveToThread(self.optimization_thread)
            self.optimization_thread.started.connect(
                self.optimization_viewmodel.run_optimization
            )
            self.setEnabled(False)
            self.optimization_thread.start()

        except Exception as e:
            self.setEnabled(True)
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Route planning error: {str(e)}"
            )
            print(f"Detailed error: {e}")

    def generate_delivery_points(self, num_points):
        self.delivery_viewmodel.generate_points(num_points)

    def generate_delivery_drivers(self, num_drivers):
        self.driver_viewmodel.generate_drivers(num_drivers)

    def _validate_optimization_prerequisites(self):
        valid, message = self.optimization_viewmodel.validate_optimization_prerequisites()
        if not valid:
            QtWidgets.QMessageBox.warning(self, "Missing Data", message)
        return valid

    def get_main_window(self):
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QWidget) and hasattr(parent, 'btn_simulate'):
                return parent
            parent = parent.parent()
        return None

    @QtCore.pyqtSlot(object)
    def updateVisualizationFromBackground(self, data):
        try:
            self.clear_layer("routes")
            self.clear_layer("deliveries")

            self.update_layer("deliveries", data["delivery_points"])

            for route in data["routes"]:
                self.add_route(
                    route["id"],
                    route["driver_id"],
                    route["path"],
                    route["style"],
                    route["popup"]
                )

        except Exception as e:
            print(f"Error updating visualization from background: {e}")
            import traceback
            traceback.print_exc()

    def get_warehouse_location(self):
        return self.delivery_viewmodel.get_warehouse_location()

    def run_simulation(self):
        if not hasattr(self.optimization_viewmodel, 'run_simulation'):
            print("Error: OptimizationViewModel does not have run_simulation method.")
            self.show_message("Error", "Simulation functionality not available.", "critical")
            return

        if not hasattr(self.optimization_viewmodel,
                       'current_solution') or not self.optimization_viewmodel.current_solution:
            self.show_message("Error",
                              "Cannot run simulation: No route solution available. Please find routes first.",
                              "critical")
            return

        self.disruption_viewmodel.current_solution = self.optimization_viewmodel.current_solution

        if hasattr(self.disruption_viewmodel, 'initialize_simulation_controller'):
            success = self.disruption_viewmodel.initialize_simulation_controller()
            if success and self.disruption_viewmodel.simulation_controller:
                self.js_interface.set_simulation_controller(
                    self.disruption_viewmodel.simulation_controller)
            else:
                self.show_message("Error",
                                  "Cannot start simulation: Simulation controller initialization failed.",
                                  "critical")
                return
        self.optimization_viewmodel.run_simulation()

    @property
    def snapped_delivery_points(self):
        return self.delivery_viewmodel.snapped_delivery_points

    @property
    def delivery_drivers(self):
        return self.driver_viewmodel.delivery_drivers

    @property
    def current_solution(self):
        return getattr(self.optimization_viewmodel, 'current_solution', None)

    @property
    def unassigned_deliveries(self):
        return getattr(self.optimization_viewmodel, 'unassigned_deliveries', None)

    @QtCore.pyqtSlot(object)
    def addToVisualizationQueue(self, data):
        self.visualization_queue.append(data)

    @property
    def selected_driver_id(self):
        if hasattr(self, '_selected_driver_id'):
            return self._selected_driver_id
        return self.driver_viewmodel.selected_driver_id

    @selected_driver_id.setter
    def selected_driver_id(self, value):
        self._selected_driver_id = value
