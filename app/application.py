from PyQt5 import QtCore

from viewmodels.delivery_viewmodel import DeliveryViewModel
from viewmodels.driver_viewmodel import DriverViewModel
from viewmodels.map_viewmodel import MapViewModel
from viewmodels.visualization_viewmodel import VisualizationViewModel


class Application(QtCore.QObject):
    """
    Central coordinator for the application that manages ViewModels
    and their interactions.
    """

    def __init__(self):
        super().__init__()
        self.delivery_viewmodel = DeliveryViewModel()
        self.driver_viewmodel = DriverViewModel()
        self.visualization_viewmodel = VisualizationViewModel()
        self.map_viewmodel = MapViewModel()

        # Connect signals between view models
        self.visualization_viewmodel.optimization_completed.connect(
            self._on_optimization_completed
        )

        # Connect driver selection between view models
        self.driver_viewmodel.driver_selected.connect(
            self._on_driver_selected
        )

        self.map_viewmodel.map_initialized.connect(self._on_map_initialized)

    def initialize(self):
        """Initialize all application components"""
        pass

    def _on_optimization_completed(self, solution, unassigned):
        """Handle optimization completion by updating driver statistics."""
        if solution:
            # Calculate driver statistics from the solution
            driver_stats = self._calculate_driver_statistics(solution)

            # Update driver view model
            self.driver_viewmodel.update_driver_statistics(driver_stats)

    def _on_driver_selected(self, driver_id):
        """Handle driver selection by updating visualization."""
        self.visualization_viewmodel.selected_driver_id = driver_id

    def _on_map_initialized(self, success, message):
        """Handle map initialization."""
        if success:
            # Update other view models with the graph
            self.delivery_viewmodel.set_graph(self.map_viewmodel.graph)

            # Set warehouse location in visualization view model
            self.visualization_viewmodel.set_warehouse_location(
                self.map_viewmodel.warehouse_location
            )

    def _calculate_driver_statistics(self, solution):
        """
        Calculate statistics for each driver based on the optimization solution.

        Args:
            solution: The optimization solution containing driver assignments

        Returns:
            Dictionary mapping driver IDs to their statistics
        """
        driver_stats = {}

        for assignment in solution:
            driver_id = assignment.driver_id

            # Basic statistics
            stats = {
                'deliveries': len(assignment.delivery_indices),
                'weight': assignment.total_weight,
                'volume': assignment.total_volume,
                'travel_time': 0,  # Will be calculated by visualization VM
                'distance': 0  # Will be calculated by visualization VM
            }

            driver_stats[driver_id] = stats

        return driver_stats

    def start_optimization(self, graph, warehouse_location):
        """
        Start the optimization process using data from all view models.

        Args:
            graph: The graph representing the road network
            warehouse_location: The warehouse location (lat, lon)
        """
        # Set up the visualization ViewModel with needed data
        self.visualization_viewmodel.set_graph(graph)
        self.visualization_viewmodel.set_warehouse_location(warehouse_location)
        self.visualization_viewmodel.set_delivery_points(self.delivery_viewmodel.snapped_delivery_points)

        # Start the optimization
        self.visualization_viewmodel.start_optimization(
            self.delivery_viewmodel.snapped_delivery_points,
            self.driver_viewmodel.drivers,
            graph
        )
