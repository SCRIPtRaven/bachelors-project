import osmnx as ox
from PyQt5 import QtWidgets, QtCore

from utils.geolocation import get_city_coordinates
from utils.route_color_manager import RouteColorManager
from view.map.core.leaflet_map_widget import LeafletMapWidget


class MapWidget(LeafletMapWidget):
    """
    Map widget that displays the road network, delivery points, and routes.
    Works with ViewModels instead of directly manipulating data.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Core properties
        self.route_color_manager = RouteColorManager()
        self.G = None
        self.current_city = None
        self.is_loading = False

        # ViewModel references
        self._delivery_viewmodel = None
        self._driver_viewmodel = None
        self._visualization_viewmodel = None
        self._map_viewmodel = None
        self._map_state = "uninitialized"

    # ViewModel property getters/setters with signal connections
    @property
    def delivery_viewmodel(self):
        return self._delivery_viewmodel

    @delivery_viewmodel.setter
    def delivery_viewmodel(self, viewmodel):
        self._delivery_viewmodel = viewmodel

    @property
    def driver_viewmodel(self):
        return self._driver_viewmodel

    @driver_viewmodel.setter
    def driver_viewmodel(self, viewmodel):
        self._driver_viewmodel = viewmodel

        if viewmodel:
            # Connect to driver ViewModel signals
            viewmodel.driver_selected.connect(self._on_driver_selected)

    @property
    def visualization_viewmodel(self):
        return self._visualization_viewmodel

    @visualization_viewmodel.setter
    def visualization_viewmodel(self, viewmodel):
        self._visualization_viewmodel = viewmodel

        if viewmodel:
            # Connect to visualization ViewModel signals
            viewmodel.visualization_updated.connect(self._on_visualization_updated)
            viewmodel.optimization_started.connect(self._on_optimization_started)
            viewmodel.optimization_completed.connect(self._on_optimization_completed)
            viewmodel.simulation_started.connect(self._on_simulation_started)

    @property
    def map_viewmodel(self):
        return self._map_viewmodel

    @map_viewmodel.setter
    def map_viewmodel(self, viewmodel):
        self._map_viewmodel = viewmodel

        if viewmodel:
            # Connect to map ViewModel signals
            viewmodel.map_initialized.connect(self._on_map_initialized)
            viewmodel.map_state_changed.connect(self._on_map_state_changed)

    # Property redirections to ViewModels
    @property
    def snapped_delivery_points(self):
        """Get delivery points from ViewModel"""
        if self._delivery_viewmodel:
            return self._delivery_viewmodel.snapped_delivery_points
        return []

    @property
    def delivery_drivers(self):
        """Get drivers from ViewModel"""
        if self._driver_viewmodel:
            return self._driver_viewmodel.drivers
        return []

    @property
    def selected_driver_id(self):
        """Get selected driver ID from ViewModel"""
        if self._driver_viewmodel:
            return self._driver_viewmodel.selected_driver_id
        return None

    # Core map functionality
    def add_delivery_points(self, points):
        """Add delivery points to the map, called by ViewModel signal"""
        data = []
        for i, point in enumerate(points):
            lat, lon, weight, volume = point
            data.append({
                "lat": lat,
                "lng": lon,
                "weight": weight,
                "volume": volume,
                "index": i,
                "popup": f"Delivery Point {i + 1}<br>Weight: {weight} kg<br>Volume: {volume} m³"
            })
        self.update_layer("deliveries", data)

    def find_shortest_route(self):
        """Start the optimization process using the ViewModel"""
        if not self._validate_optimization_prerequisites():
            return

        if self._visualization_viewmodel:
            # Get the warehouse location
            warehouse_location = self.get_warehouse_location()

            # Set warehouse location
            self._visualization_viewmodel.set_warehouse_location(warehouse_location)

            # Set delivery points
            self._visualization_viewmodel.set_delivery_points(self.snapped_delivery_points)

            # Start optimization
            self._visualization_viewmodel.start_optimization(
                self.snapped_delivery_points,
                self.delivery_drivers,
                self.G
            )

    def get_warehouse_location(self):
        """Calculate the center of the graph to place the warehouse"""
        if not self.G:
            return None

        lats = []
        lons = []
        for _, data in self.G.nodes(data=True):
            lats.append(data['y'])
            lons.append(data['x'])

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        warehouse_node = ox.nearest_nodes(self.G, X=center_lon, Y=center_lat)
        warehouse_coords = (self.G.nodes[warehouse_node]['y'], self.G.nodes[warehouse_node]['x'])

        return warehouse_coords

    def set_stats_labels(self, time_label, travel_time_label, distance_label):
        """Set references to external statistic labels for updating"""
        self.time_label = time_label
        self.travel_time_label = travel_time_label
        self.distance_label = distance_label

    def update_statistics(self, stats):
        """Update the statistics display with data from ViewModel"""
        if hasattr(self, 'time_label') and hasattr(self, 'travel_time_label') and hasattr(self, 'distance_label'):
            # Format time values
            computation_time = stats.get('computation_time', 0)
            total_travel_time = stats.get('total_time', 0)
            total_distance = stats.get('total_distance', 0)

            # Update labels
            self.time_label.setText(f"Time taken to compute route: {computation_time:.2f} seconds")
            self.travel_time_label.setText(f"Total travel time: {total_travel_time / 60:.2f} minutes")
            self.distance_label.setText(f"Total distance: {total_distance:.2f} km")

    # Signal handlers for ViewModel events
    def _on_drivers_changed(self, drivers):
        """Handle drivers changed event from ViewModel"""
        # We can just show a message for now, as the visualization itself
        # will be handled by the MapManager
        if drivers and len(drivers) > 0:
            print(f"Updated with {len(drivers)} drivers")

    def _on_driver_selected(self, driver_id):
        """Handle driver selection from ViewModel"""
        # This will be handled by the MapManager updating the visualization
        pass

    def _on_visualization_updated(self, visualization_data):
        """Handle visualization update from ViewModel"""
        try:
            self.clear_layer("routes")
            self.clear_layer("deliveries")

            if 'delivery_points' in visualization_data:
                self.update_layer("deliveries", visualization_data['delivery_points'])

            if 'routes' in visualization_data:
                for route in visualization_data['routes']:
                    self.add_route(
                        route["id"],
                        route["driver_id"],
                        route["path"],
                        route["style"],
                        route["popup"]
                    )
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()

    def _on_optimization_started(self):
        """Handle optimization started event from ViewModel"""
        # Show loading indicator
        self.execute_js("if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

        # Disable map interactions during optimization
        self.setEnabled(False)

    def _on_optimization_completed(self, solution, unassigned):
        """Handle optimization completed event from ViewModel"""
        # Hide loading indicator
        self.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

        # Re-enable map interactions
        self.setEnabled(True)

        # Update statistics
        if self._visualization_viewmodel:
            stats = self._visualization_viewmodel.get_statistics()
            self.update_statistics(stats)

    def _on_simulation_started(self, simulation_data):
        """Handle simulation started event from ViewModel"""
        # Start the simulation with the provided data
        self.start_simulation(simulation_data)

        # Set simulation speed from app settings
        try:
            from config.app_settings import SIMULATION_SPEED
            self.set_simulation_speed(SIMULATION_SPEED)
        except (ImportError, AttributeError):
            # Default speed if setting not available
            self.set_simulation_speed(50)

    def _on_map_initialized(self, success, message):
        """Handle map initialization synchronously"""
        if success and self._map_viewmodel:
            # Get data from viewmodel
            self.G = self._map_viewmodel.graph
            self.current_city = self._map_viewmodel.city_name
            center = self._map_viewmodel.center
            zoom = self._map_viewmodel.zoom

            # Initialize map directly (no timers or async)
            if center and zoom:
                self.init_map(center, zoom)

            # Add warehouse directly
            warehouse = self._map_viewmodel.warehouse_location
            if warehouse:
                self.add_warehouse(warehouse[0], warehouse[1])

            # Signal completion
            self.load_completed.emit(True, message)
        else:
            self.load_completed.emit(False, message)

    def _safe_init_map(self, center, zoom):
        """Safely initialize the map with error handling"""
        try:
            # Crucial: Ensure we're on the main thread
            if QtCore.QThread.currentThread() != QtCore.QCoreApplication.instance().thread():
                print("ERROR: Attempting to initialize map from non-main thread!")
                # Schedule it properly on the main thread
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_safe_init_map",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(tuple, center),
                    QtCore.Q_ARG(int, zoom)
                )
                return

            # Continue only if we're on the main thread
            self.init_map(center, zoom)
        except Exception as e:
            print(f"Error initializing map: {e}")
            import traceback
            traceback.print_exc()

    def _begin_map_initialization(self, center, zoom, message):
        """Begin the map initialization sequence with state tracking"""
        try:
            if self._map_state != "loading":
                return

            # Safe initialization with explicit error handling
            try:
                self.init_map(center, zoom)
                # Wait longer before declaring success
                QtCore.QTimer.singleShot(1000, lambda: self._complete_map_initialization(message))
            except Exception as e:
                self._map_state = "error"
                print(f"Error in map initialization: {e}")
                import traceback
                traceback.print_exc()
                self.load_completed.emit(False, f"Map initialization error: {str(e)}")
        except Exception as e:
            print(f"Critical error in _begin_map_initialization: {e}")
            import traceback
            traceback.print_exc()

    def _complete_map_initialization(self, message):
        """Complete the map initialization sequence"""
        try:
            if self._map_state != "loading":
                return

            self._map_state = "initialized"
            # Now it's safe to emit the completion signal
            self.load_completed.emit(True, message)
        except Exception as e:
            print(f"Error completing map initialization: {e}")
            import traceback
            traceback.print_exc()
            self.load_completed.emit(False, f"Error finalizing map: {str(e)}")

    def _on_map_state_changed(self, state):
        """Handle map state changes from MapViewModel"""
        try:
            if 'warehouse_location' in state and state['warehouse_location']:
                lat, lon = state['warehouse_location']
                # Schedule warehouse addition with a slight delay
                QtCore.QTimer.singleShot(300, lambda: self._safe_add_warehouse(lat, lon))
        except Exception as e:
            print(f"Error in _on_map_state_changed: {e}")
            import traceback
            traceback.print_exc()

    def _safe_add_warehouse(self, lat, lon):
        """Safely add warehouse marker with error handling"""
        try:
            self.add_warehouse(lat, lon)
        except Exception as e:
            print(f"Error adding warehouse: {e}")
            import traceback
            traceback.print_exc()

    def _on_graph_loaded_direct(self, success, message, graph, city_name):
        """Fallback handler for graph loading without MapViewModel"""
        self.is_loading = False
        self.setEnabled(True)

        if success:
            self.G = graph
            self.current_city = city_name

            center, zoom = get_city_coordinates(city_name)
            self.init_map(center, zoom)
            self.load_map()

            if hasattr(self.parent(), 'on_graph_loaded'):
                self.parent().on_graph_loaded(graph, city_name)

            QtWidgets.QMessageBox.information(self, "Success", message)
            self.load_completed.emit(True, message)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            self.load_completed.emit(False, message)

        if hasattr(self, 'load_worker'):
            self.load_worker.deleteLater()

    def _validate_optimization_prerequisites(self):
        """Validate that all prerequisites for optimization are met"""
        if self.G is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing Data", "Graph data not loaded. Please load a map first."
            )
            return False

        if not self.snapped_delivery_points:
            QtWidgets.QMessageBox.warning(
                self, "Missing Data", "No delivery points available. Please generate delivery points."
            )
            return False

        if not self.delivery_drivers:
            QtWidgets.QMessageBox.warning(
                self, "Missing Data", "No drivers available. Please generate drivers."
            )
            return False

        return True

    def get_main_window(self):
        """Get the MainWindow instance for coordination"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QtWidgets.QWidget) and hasattr(parent, 'solution_switch'):
                return parent
            parent = parent.parent()
        return None
