from core.model import Model


class DeliveryModel(Model):
    """
    Model for managing delivery data and operations.
    """

    def __init__(self, graph=None):
        super().__init__()
        self.graph = graph
        self.delivery_points = []
        self.snapped_delivery_points = []

    def set_graph(self, graph):
        """Sets the graph used for route planning."""
        self.graph = graph

    def clear_deliveries(self):
        """Clears all delivery points."""
        self.delivery_points = []
        self.snapped_delivery_points = []

    def add_delivery_point(self, delivery):
        """Adds a delivery point to the model."""
        self.delivery_points.append(delivery)

    def add_snapped_delivery_point(self, lat, lon, weight, volume):
        """Adds a snapped delivery point to the model."""
        self.snapped_delivery_points.append((lat, lon, weight, volume))
