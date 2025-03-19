import os

from PyQt5 import QtCore

from core.viewmodel import ViewModel


class MapViewModel(ViewModel):
    """
    ViewModel for managing map state and interactions.
    """
    map_state_changed = QtCore.pyqtSignal(dict)
    map_initialized = QtCore.pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self._graph = None
        self._center = None
        self._zoom = None
        self._city_name = None
        self._is_loading = False
        self._warehouse_location = None

    @property
    def graph(self):
        return self._graph

    @property
    def is_loading(self):
        return self._is_loading

    @property
    def city_name(self):
        return self._city_name

    @property
    def warehouse_location(self):
        return self._warehouse_location

    @property
    def center(self):
        """Get the map center coordinates."""
        return self._center

    @property
    def zoom(self):
        """Get the map zoom level."""
        return self._zoom

    def load_graph(self, city_name):
        """Load the graph for a given city."""
        if self._is_loading:
            return

        self.set_property('_is_loading', True)

        try:
            # Import necessary modules
            import osmnx as ox
            from config.paths import get_graph_file_path, get_travel_times_path
            from data.graph_manager import update_travel_times_from_csv
            from utils.geolocation import get_city_coordinates

            # Load graph
            graph_path = get_graph_file_path(city_name)
            G = ox.load_graphml(graph_path)

            # Apply travel times from CSV if available
            travel_times_path = get_travel_times_path(city_name)
            if os.path.exists(travel_times_path):
                update_travel_times_from_csv(G, travel_times_path)
                print(f"Applied travel times from {travel_times_path}")

            # Calculate parameters
            center, zoom = get_city_coordinates(city_name)

            # Calculate warehouse location
            warehouse_location = self._calculate_warehouse_location(G)

            # Update state and emit signal
            self._graph = G
            self._city_name = city_name
            self._center = center
            self._zoom = zoom
            self._warehouse_location = warehouse_location

            # Emit signals
            self.map_initialized.emit(True, "Graph loaded successfully")

        except Exception as e:
            print(f"ERROR in load_graph: {str(e)}")
            import traceback
            traceback.print_exc()
            self.map_initialized.emit(False, f"Error loading graph: {str(e)}")
        finally:
            self.set_property('_is_loading', False)

    def _calculate_warehouse_location(self, G):
        """Calculate the warehouse location at the center of the graph."""
        warehouse_lat = 0
        warehouse_lon = 0
        node_count = 0

        for _, data in G.nodes(data=True):
            warehouse_lat += data['y']
            warehouse_lon += data['x']
            node_count += 1

        if node_count > 0:
            warehouse_lat /= node_count
            warehouse_lon /= node_count

        import osmnx as ox
        warehouse_node = ox.nearest_nodes(G, X=warehouse_lon, Y=warehouse_lat)
        warehouse_location = (G.nodes[warehouse_node]['y'], G.nodes[warehouse_node]['x'])

        return warehouse_location

    def _handle_load_result(self, result):
        """Handle the result of the graph loading task."""
        self.set_property('_is_loading', False)

        if result['success']:
            self._graph = result['graph']
            self._city_name = result['city_name']
            self._center = result['center']
            self._zoom = result['zoom']
            self._warehouse_location = result['warehouse_location']

            QtCore.QCoreApplication.processEvents()

            self.map_initialized.emit(True, result['message'])

            QtCore.QTimer.singleShot(100, lambda: self.map_state_changed.emit({
                'graph': self._graph,
                'center': self._center,
                'zoom': self._zoom,
                'warehouse_location': self._warehouse_location
            }))
        else:
            self.map_initialized.emit(False, result['message'])

    def _handle_load_error(self, error):
        """Handle errors during graph loading."""
        self.set_property('_is_loading', False)
        error_message = f"Error loading graph: {str(error)}"
        self.map_initialized.emit(False, error_message)
