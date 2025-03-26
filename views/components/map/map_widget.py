from PyQt5 import QtCore, QtWidgets

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

        self.delivery_viewmodel = DeliveryViewModel(messenger=self.messenger)
        self.driver_viewmodel = DriverViewModel(messenger=self.messenger)
        self.disruption_viewmodel = DisruptionViewModel(messenger=self.messenger)
        self.optimization_viewmodel = OptimizationViewModel(
            delivery_viewmodel=self.delivery_viewmodel,
            driver_viewmodel=self.driver_viewmodel,
            messenger=self.messenger,
            disruption_viewmodel=self.disruption_viewmodel,
        )

        self.delivery_viewmodel.delivery_points_processed.connect(self.handle_delivery_points_processed)
        self.delivery_viewmodel.request_show_message.connect(self.show_message)
        self.delivery_viewmodel.warehouse_location_changed.connect(self.handle_warehouse_location_changed)

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
        self.optimization_viewmodel.visualization_data_ready.connect(self.updateVisualizationFromBackground)
        self.optimization_viewmodel.solution_switch_available.connect(self.handle_solution_switch_available)
        self.optimization_viewmodel.enable_simulation_button.connect(self.enable_simulation_button)
        self.optimization_viewmodel.request_load_disruptions.connect(self.load_disruptions)

        self.disruption_viewmodel.disruption_generated.connect(self.handle_disruptions_generated)
        self.disruption_viewmodel.disruption_activated.connect(self.handle_disruption_activated)
        self.disruption_viewmodel.disruption_resolved.connect(self.handle_disruption_resolved)
        self.disruption_viewmodel.request_show_message.connect(self.show_message)

        self.visualization_queue = VisualizationQueue(self.optimization_viewmodel)
        self.driver_viewmodel.set_visualization_queue(self.visualization_queue)

        self.driver_labels = {}
        self._selected_driver_id = None

        self.is_loading = False
        self.is_computing = False

    def handle_delivery_points_processed(self, points, successful, skipped):
        """Handle processed delivery points"""
        self.add_delivery_points(points)

        if skipped > 0:
            self.show_message(
                "Delivery Points Generated",
                f"Successfully placed {successful} delivery points.\n"
                f"Skipped {skipped} inaccessible points.",
                "information"
            )

    def update_visualization(self, solution, unassigned_deliveries=None):
        """Handle visualization updates from the OptimizationViewModel"""
        if isinstance(solution, tuple):
            solution, unassigned = solution
        else:
            unassigned = unassigned_deliveries

        self.addToVisualizationQueue((solution, unassigned))

    def handle_disruptions_generated(self, disruptions):
        """Handle generated disruptions"""
        print(f"Generated {len(disruptions)} disruptions for simulation")

    def handle_disruption_activated(self, disruption):
        """Handle disruption activation"""
        print(f"Disruption {disruption.id} activated: {disruption.type.value}")

    def handle_disruption_resolved(self, disruption_id):
        """Handle disruption resolution"""
        print(f"Disruption {disruption_id} resolved")

    def load_disruptions(self, disruptions):
        """Load disruptions onto the map"""
        super().load_disruptions(disruptions)

    def toggle_disruptions(self, enabled):
        """Toggle disruptions on/off"""
        super().toggle_disruptions(enabled)

    def handle_warehouse_location_changed(self, location):
        """Handle warehouse location updates"""
        self.add_warehouse(location[0], location[1])

        self.optimization_viewmodel.set_warehouse_location(location)

    def update_driver_list(self, drivers):
        """Handle driver list updates from the ViewModel"""
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
        """Update driver labels with stats from the ViewModel"""
        if not self.driver_labels:
            return

        for driver_id, stats in formatted_stats.items():
            if driver_id in self.driver_labels:
                self.driver_labels[driver_id].setText(stats['text'])
                self.driver_labels[driver_id].setWordWrap(True)
                self.driver_labels[driver_id].setMinimumHeight(80)

    def handle_driver_selection(self, driver_id):
        """Handle driver selection from the ViewModel"""
        self._selected_driver_id = driver_id

        for d_id, label in self.driver_labels.items():
            label.setSelected(d_id == driver_id)

    def show_loading(self):
        self.execute_js("if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

    def hide_loading(self):
        self.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

    def handle_solution_switch_available(self, available):
        main_window = self.get_main_window()
        if main_window and hasattr(main_window, 'solution_switch'):
            main_window.solution_switch.show()
            main_window.solution_switch.setEnabled(available)

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

            self.delivery_viewmodel.set_graph(graph)

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
        """Get the MainWindow instance"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QWidget) and hasattr(parent, 'solution_switch'):
                return parent
            parent = parent.parent()
        return None

    @QtCore.pyqtSlot(object)
    def updateVisualizationFromBackground(self, data):
        """Update the visualization with data calculated in a background thread"""
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
        """Get warehouse location from ViewModel"""
        return self.delivery_viewmodel.get_warehouse_location()

    def run_simulation(self):
        """Run a simulation of the delivery routes"""
        if not hasattr(self.optimization_viewmodel, 'run_simulation'):
            print("Error: OptimizationViewModel does not have run_simulation method.")
            self.show_message("Error", "Simulation functionality not available.", "critical")
            return

        # Continue with optimization viewmodel's run_simulation
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
        """Safe method to add items to visualization queue from other threads"""
        self.visualization_queue.append(data)

    @property
    def selected_driver_id(self):
        if hasattr(self, '_selected_driver_id'):
            return self._selected_driver_id
        return self.driver_viewmodel.selected_driver_id

    @selected_driver_id.setter
    def selected_driver_id(self, value):
        self._selected_driver_id = value
