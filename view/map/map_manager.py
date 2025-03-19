# ui/map/map_manager.py
from PyQt5 import QtCore, QtWidgets


class MapManager(QtCore.QObject):
    """
    Manager class that connects the map widget with ViewModels.
    Handles map events and visualization updates.
    """

    def __init__(self, map_widget, delivery_vm, driver_vm, visualization_vm, map_vm=None):
        super().__init__()
        self.map_widget = map_widget
        self.delivery_vm = delivery_vm
        self.driver_vm = driver_vm
        self.visualization_vm = visualization_vm
        self.map_vm = map_vm

        if self.map_vm:
            # Connect map viewmodel signals
            self.map_vm.map_initialized.connect(self._on_map_initialized)
            self.map_vm.map_state_changed.connect(self._on_map_state_changed)

        # Connect other signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect signals between ViewModels and map widget."""
        # Delivery ViewModel signals
        self.delivery_vm.deliveries_changed.connect(self._on_deliveries_changed)

        # Driver ViewModel signals
        self.driver_vm.drivers_changed.connect(self._on_drivers_changed)
        self.driver_vm.driver_selected.connect(self._on_driver_selected)

        # Visualization ViewModel signals
        self.visualization_vm.visualization_updated.connect(self._on_visualization_updated)
        self.visualization_vm.optimization_started.connect(self._on_optimization_started)
        self.visualization_vm.optimization_completed.connect(self._on_optimization_completed)
        self.visualization_vm.simulation_started.connect(self._on_simulation_started)

    def _on_deliveries_changed(self, delivery_points):
        """Handle delivery points change."""
        # Update map with delivery points
        self.map_widget.add_delivery_points(delivery_points)

        # If we have an optimization in progress, make sure the visualization
        # ViewModel has the updated delivery points
        if hasattr(self.visualization_vm, 'set_delivery_points'):
            self.visualization_vm.set_delivery_points(delivery_points)

    def _on_drivers_changed(self, drivers):
        """Handle drivers changed event from ViewModel."""
        try:
            # Use a very simple approach - just print a message
            print(f"MapManager received drivers_changed signal with {len(drivers)} drivers")

            # Don't do any complex processing here - just acknowledge receipt
            # We'll handle this differently in a moment

            # Queue an update for later using a timer to break the direct connection
            QtCore.QTimer.singleShot(10, lambda: self._update_drivers_display(drivers))
        except Exception as e:
            print(f"Error in _on_drivers_changed: {e}")
            import traceback
            traceback.print_exc()

    def _update_drivers_display(self, drivers):
        """Update the drivers display after a brief delay."""
        try:
            # Now we can safely process the drivers
            if drivers and len(drivers) > 0:
                print(f"Updating display with {len(drivers)} drivers")
                # Actual display update logic
        except Exception as e:
            print(f"Error updating drivers display: {e}")
            import traceback
            traceback.print_exc()

    def _on_driver_selected(self, driver_id):
        """Handle driver selection."""
        # Highlight the selected driver's route
        self.map_widget.highlight_driver_route(driver_id)

    def _on_visualization_updated(self, visualization_data):
        """Handle visualization update."""
        # Update the map with visualization data
        if 'delivery_points' in visualization_data:
            self.map_widget.update_layer('deliveries', visualization_data['delivery_points'])

        if 'routes' in visualization_data:
            # Clear existing routes
            self.map_widget.clear_layer('routes')

            # Add new routes
            for route in visualization_data['routes']:
                self.map_widget.add_route(
                    route['id'],
                    route['driver_id'],
                    route['path'],
                    route['style'],
                    route['popup']
                )

    def _on_optimization_started(self):
        """Handle optimization start."""
        # Show loading indicator
        self.map_widget.execute_js("if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

        # Disable map interactions during optimization
        self.map_widget.setEnabled(False)

    def _on_optimization_completed(self, solution, unassigned):
        """Handle optimization completion."""
        # Hide loading indicator
        self.map_widget.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

        # Re-enable map interactions
        self.map_widget.setEnabled(True)

        # Update statistics display
        if hasattr(self.map_widget, 'update_statistics'):
            statistics = self._calculate_solution_statistics(solution)
            self.map_widget.update_statistics(statistics)

    def _on_simulation_started(self, simulation_data):
        """Handle simulation start."""
        # Start the simulation on the map with proper data
        self.map_widget.start_simulation(simulation_data)

        # You may also want to set simulation speed based on application settings
        try:
            from config.app_settings import SIMULATION_SPEED
            speed = SIMULATION_SPEED
        except (ImportError, AttributeError):
            speed = 25.0

        self.map_widget.set_simulation_speed(speed)

    def _calculate_solution_statistics(self, solution):
        """
        Calculate statistics for the solution.

        Args:
            solution: The optimization solution

        Returns:
            Dictionary with statistics
        """
        # This will be implemented in detail later
        return {
            'time': 0,
            'distance': 0,
            'deliveries': 0
        }

    def find_shortest_route(self):
        """Start the optimization process."""
        if not self._validate_optimization_prerequisites():
            return

        # Get the warehouse location
        warehouse_location = self.map_widget.get_warehouse_location()

        # Start the optimization
        self.visualization_vm.start_optimization(
            self.delivery_vm.snapped_delivery_points,
            self.driver_vm.drivers,
            self.map_widget.G
        )

    def _validate_optimization_prerequisites(self):
        """
        Validate that all prerequisites for optimization are met.

        Returns:
            bool: True if prerequisites are met, False otherwise
        """
        if (self.map_widget.G is None or
                not self.delivery_vm.snapped_delivery_points or
                not self.driver_vm.drivers):
            QtWidgets.QMessageBox.warning(
                self.map_widget,
                "Missing Data",
                "Please ensure graph data, delivery points, and drivers are all loaded."
            )
            return False
        return True

    def _on_map_initialized(self, success, message):
        """Handle map initialization."""
        if success:
            # Initialize the map with center and zoom from ViewModel
            if hasattr(self.map_vm, '_center') and hasattr(self.map_vm, '_zoom'):
                self.map_widget.init_map(self.map_vm.center, self.map_vm.zoom)

            # Add warehouse marker
            if self.map_vm.warehouse_location:
                self.map_widget.add_warehouse(
                    self.map_vm.warehouse_location[0],
                    self.map_vm.warehouse_location[1]
                )

    def _on_map_state_changed(self, state):
        """Handle map state changes."""
        # Update the map widget with state changes
        if 'warehouse_location' in state and state['warehouse_location']:
            lat, lon = state['warehouse_location']
            self.map_widget.add_warehouse(lat, lon)

    def load_city(self, city_name):
        """Load a city into the map."""
        if not city_name:
            return

        # Use the MapViewModel to handle the graph loading
        if self.map_vm:
            self.map_vm.load_graph(city_name)
        else:
            # Error handling if MapViewModel is not available
            print("Error: MapViewModel not available for loading city")
            QtWidgets.QMessageBox.warning(
                self.map_widget,
                "Error",
                "Application is not properly initialized. Please restart the application."
            )
