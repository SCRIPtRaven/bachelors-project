from typing import List, Dict, Tuple


class DeliverySystemState:
    """
    Represents the complete state of the delivery system.
    Encapsulates drivers, deliveries, disruptions, and simulation state.
    """

    def __init__(self, drivers, deliveries, disruptions, simulation_time, graph, warehouse_location,
                 driver_positions=None, driver_assignments=None, driver_routes=None):
        self.drivers = drivers
        self.deliveries = deliveries
        self.disruptions = disruptions
        self.simulation_time = simulation_time
        self.graph = graph
        self.warehouse_location = warehouse_location

        self.driver_positions = driver_positions if driver_positions is not None else self._get_driver_positions()
        self.driver_assignments = driver_assignments if driver_assignments is not None else self._get_driver_assignments()
        self.driver_capacities = self._get_driver_capacities()
        self.disruption_areas = self._get_disruption_areas()
        self.driver_routes = driver_routes if driver_routes is not None else {}

    def _get_driver_positions(self) -> Dict[int, Tuple[float, float]]:
        """Extract current positions of all drivers"""
        positions = {}
        for driver in self.drivers:
            if hasattr(driver, 'current_position'):
                positions[driver.id] = driver.current_position
        return positions

    def _get_driver_assignments(self) -> Dict[int, List[int]]:
        """Get current delivery assignments for each driver"""
        assignments = {}
        for driver in self.drivers:
            if hasattr(driver, 'assigned_deliveries'):
                assignments[driver.id] = driver.assigned_deliveries
            elif hasattr(driver, 'delivery_indices'):
                assignments[driver.id] = driver.delivery_indices
        return assignments

    def _get_driver_capacities(self) -> Dict[int, Tuple[float, float]]:
        """Get remaining capacity (weight, volume) for each driver"""
        capacities = {}
        for driver in self.drivers:
            if hasattr(driver, 'weight_capacity') and hasattr(driver, 'volume_capacity'):
                remaining_weight = driver.weight_capacity - getattr(driver, 'current_weight', 0)
                remaining_volume = driver.volume_capacity - getattr(driver, 'current_volume', 0)
                capacities[driver.id] = (remaining_weight, remaining_volume)
        return capacities

    def _get_disruption_areas(self) -> List[Dict]:
        """Extract locations and areas affected by disruptions"""
        areas = []
        for disruption in self.disruptions:
            areas.append({
                'id': disruption.id,
                'type': disruption.type.value,
                'location': disruption.location,
                'radius': disruption.affected_area_radius,
                'severity': disruption.severity
            })
        return areas
