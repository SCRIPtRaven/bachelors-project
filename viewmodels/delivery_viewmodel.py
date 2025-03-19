from PyQt5 import QtCore, QtWidgets

from core.viewmodel import ViewModel
from models.delivery_model import DeliveryModel
from services.geolocation_service import GeolocationService
from utils.map_utils import find_accessible_node


class DeliveryViewModel(ViewModel):
    """
    ViewModel for managing delivery operations and UI state.
    """
    # Signals
    deliveries_changed = QtCore.pyqtSignal(list)  # Emitted when delivery points change
    generation_completed = QtCore.pyqtSignal(bool, str)  # Emitted when generation is complete

    def __init__(self, delivery_model=None):
        super().__init__()
        self._model = delivery_model or DeliveryModel()
        self._is_generating = False
        self._generation_status = ""

    @property
    def is_generating(self):
        return self._is_generating

    @property
    def generation_status(self):
        return self._generation_status

    @property
    def delivery_points(self):
        return self._model.delivery_points

    @property
    def snapped_delivery_points(self):
        return self._model.snapped_delivery_points

    def set_graph(self, graph):
        """Sets the graph and updates the model."""
        self._model.set_graph(graph)

    def generate_delivery_points(self, num_points):
        """
        Generates delivery points on the main thread.
        Shows a progress dialog to keep UI responsive.
        """
        if not self._model.graph:
            self.generation_completed.emit(False, "Graph not loaded. Please load a graph first.")
            return

        # Set state to generating
        self.set_property('_is_generating', True)
        self.set_property('_generation_status', f"Generating {num_points} delivery points...")

        # Create a progress dialog
        progress = QtWidgets.QProgressDialog(f"Generating {num_points} delivery points...",
                                             "Cancel", 0, 100)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        try:
            # Calculate bounds from graph
            node_coords = [(data['y'], data['x'])
                           for _, data in self._model.graph.nodes(data=True)]
            bounds = (
                min(lat for lat, _ in node_coords),
                max(lat for lat, _ in node_coords),
                min(lon for _, lon in node_coords),
                max(lon for _, lon in node_coords)
            )

            progress.setValue(10)
            QtCore.QCoreApplication.processEvents()

            # Generate points
            points = GeolocationService.generate_delivery_points(bounds, num_points)

            progress.setValue(50)
            QtCore.QCoreApplication.processEvents()

            # Process and snap points
            snapped_points = []
            successful_points = 0
            skipped_points = 0

            for i, point in enumerate(points):
                if progress.wasCanceled():
                    break

                progress.setValue(50 + int(40 * i / len(points)))
                if i % 5 == 0:  # Process events occasionally to keep UI responsive
                    QtCore.QCoreApplication.processEvents()

                try:
                    lat, lon = point.coordinates
                    node_id, (snapped_lat, snapped_lon) = find_accessible_node(
                        self._model.graph, lat, lon
                    )

                    snapped_points.append(
                        (snapped_lat, snapped_lon, point.weight, point.volume)
                    )
                    successful_points += 1

                except ValueError:
                    skipped_points += 1
                except Exception as e:
                    skipped_points += 1

            progress.setValue(95)
            QtCore.QCoreApplication.processEvents()

            # Update model
            self._model.clear_deliveries()
            self._model.snapped_delivery_points = snapped_points

            # Set status
            status = f"Generated {successful_points} delivery points."
            if skipped_points > 0:
                status += f" Skipped {skipped_points} inaccessible points."

            self.set_property('_generation_status', status)

            # Emit signals
            self.deliveries_changed.emit(self._model.snapped_delivery_points)
            self.generation_completed.emit(True, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error generating deliveries: {str(e)}"
            self.set_property('_generation_status', error_msg)
            self.generation_completed.emit(False, error_msg)
        finally:
            self.set_property('_is_generating', False)
            progress.close()

    def _handle_generation_result(self, result):
        """
        Handles the result of delivery point generation in a thread-safe manner.

        This function receives results from the background thread and safely
        transfers ownership of data to the main thread through deep copying
        and delayed signal emission.

        Args:
            result: Dictionary containing generation results
        """
        # Update state (now on main thread)
        self.set_property('_is_generating', False)

        if result['success']:
            # Update model with new points
            self._model.clear_deliveries()

            # Create a deep copy of the points to ensure thread safety
            import copy
            points_copy = copy.deepcopy(result['points'])

            # Update model with the copied data
            self._model.snapped_delivery_points = points_copy

            # Prepare status message
            status = f"Generated {result['successful']} delivery points."
            if result['skipped'] > 0:
                status += f" Skipped {result['skipped']} inaccessible points."

            self.set_property('_generation_status', status)

            # Use a short delay to ensure memory stability before emitting signals
            # This prevents access to potentially unstable memory across thread boundaries
            QtCore.QTimer.singleShot(10, lambda: self._emit_data_changed(points_copy, status))
        else:
            # Handle error case
            error_msg = f"Error generating deliveries: {result.get('error', 'Unknown error')}"
            self.set_property('_generation_status', error_msg)
            self.generation_completed.emit(False, error_msg)

    def _emit_data_changed(self, points, status):
        """
        Safely emit signals with the copied delivery point data.

        This function runs on the main thread after a short delay to ensure
        memory stability, and uses simple data structures in signal emissions.

        Args:
            points: The delivery points to emit (already deep-copied)
            status: Status message to emit with completion signal
        """
        # Convert complex objects to simple tuples if needed
        # (In this case, points are already in tuple form, but this ensures immutability)
        simple_points = [(p[0], p[1], p[2], p[3]) for p in points]

        # Emit signals with the simple data structures
        self.deliveries_changed.emit(simple_points)
        self.generation_completed.emit(True, status)

    def _handle_generation_error(self, error):
        """Handles errors during delivery point generation."""
        self.set_property('_is_generating', False)
        error_msg = f"Error generating deliveries: {str(error)}"
        self.set_property('_generation_status', error_msg)
        self.generation_completed.emit(False, error_msg)
