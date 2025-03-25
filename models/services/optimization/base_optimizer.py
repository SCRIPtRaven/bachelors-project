import math

import networkx as nx
import osmnx as ox
from PyQt5 import QtCore


class DeliveryOptimizer(QtCore.QObject):
    """
    Base class for delivery optimization algorithms that provides common functionality
    for calculating routes, travel times, and solution evaluation.
    """
    finished = QtCore.pyqtSignal(object, object)

    def __init__(self, delivery_drivers, snapped_delivery_points, G, warehouse_coords):
        super().__init__()
        self.delivery_drivers = delivery_drivers
        self.snapped_delivery_points = snapped_delivery_points
        self.G = G
        self.warehouse_location = warehouse_coords
        self.nearest_nodes_cache = {}
        self.route_cache = {}

    def _format_time_hms(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

    def _calculate_driver_utilization(self, solution):
        if not solution or not self.delivery_drivers:
            return 0.0

        driver_utils = []

        for assignment in solution:
            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            if driver:
                weight_util = (assignment.total_weight / driver.weight_capacity) * 100
                volume_util = (assignment.total_volume / driver.volume_capacity) * 100
                avg_util = (weight_util + volume_util) / 2
                driver_utils.append(avg_util)

        return sum(driver_utils) / len(driver_utils) if driver_utils else 0

    def _calculate_balance_score(self, solution):
        if not solution or not self.delivery_drivers:
            return 100.0

        utilizations = []
        for assignment in solution:
            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            if driver:
                weight_util = assignment.total_weight / driver.weight_capacity
                volume_util = assignment.total_volume / driver.volume_capacity
                avg_util = (weight_util + volume_util) / 2
                utilizations.append(avg_util)

        if not utilizations:
            return 100.0

        mean_util = sum(utilizations) / len(utilizations)
        if mean_util == 0:
            return 100.0

        variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
        std_dev = math.sqrt(variance)

        return (std_dev / mean_util) * 100 if mean_util > 0 else 100.0

    def calculate_total_time(self, solution):
        total_time = 0

        for assignment in solution:
            if not assignment.delivery_indices:
                continue

            route = self._get_route_with_warehouse(assignment)

            for i in range(len(route) - 1):
                start_node = route[i]
                end_node = route[i + 1]

                travel_time = self._calculate_travel_time(start_node, end_node)
                total_time += travel_time

        return total_time

    def _get_route_with_warehouse(self, assignment):
        route = [self.warehouse_location]

        for idx in assignment.delivery_indices:
            lat, lon, _, _ = self.snapped_delivery_points[idx]
            route.append((lat, lon))

        route.append(self.warehouse_location)

        return route

    def _calculate_travel_time(self, start_coords, end_coords):
        cache_key = (start_coords, end_coords)

        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        try:
            start_node = self._get_nearest_node(start_coords)
            end_node = self._get_nearest_node(end_coords)

            path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')

            travel_time = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.G.get_edge_data(u, v, 0)
                travel_time += edge_data.get('travel_time', 60)

            self.route_cache[cache_key] = travel_time

            return travel_time

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dx = end_coords[1] - start_coords[1]
            dy = end_coords[0] - start_coords[0]

            distance_km = ((dx ** 2 + dy ** 2) ** 0.5) * 111
            travel_time = (distance_km / 30) * 3600

            self.route_cache[cache_key] = travel_time

            return travel_time

    def _get_nearest_node(self, coords):
        if coords in self.nearest_nodes_cache:
            return self.nearest_nodes_cache[coords]

        nearest_node = ox.nearest_nodes(self.G, X=coords[1], Y=coords[0])
        self.nearest_nodes_cache[coords] = nearest_node

        return nearest_node

    def _optimize_route_order(self, assignment):
        if not assignment.delivery_indices:
            return

        optimized_indices = []
        remaining_indices = set(assignment.delivery_indices)

        current_location = self.warehouse_location

        while remaining_indices:
            nearest_idx = None
            nearest_distance = float('inf')

            for idx in remaining_indices:
                lat, lon, _, _ = self.snapped_delivery_points[idx]
                delivery_location = (lat, lon)

                travel_time = self._calculate_travel_time(current_location, delivery_location)

                if travel_time < nearest_distance:
                    nearest_distance = travel_time
                    nearest_idx = idx

            optimized_indices.append(nearest_idx)
            remaining_indices.remove(nearest_idx)

            lat, lon, _, _ = self.snapped_delivery_points[nearest_idx]
            current_location = (lat, lon)

        assignment.delivery_indices = optimized_indices

    def _evaluate_solution(self, solution, unassigned=None):
        total_time = self.calculate_total_time(solution)

        unassigned_penalty = len(unassigned) * 3600 if unassigned else 0

        driver_utils = []

        for assignment in solution:
            if not assignment.delivery_indices:
                driver_utils.append(0.0)
                continue

            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            weight_util = assignment.total_weight / driver.weight_capacity
            volume_util = assignment.total_volume / driver.volume_capacity
            avg_util = (weight_util + volume_util) / 2
            driver_utils.append(avg_util)

        if driver_utils:
            avg_util = sum(driver_utils) / len(driver_utils)
            if avg_util > 0:
                util_variance = sum((u - avg_util) ** 2 for u in driver_utils) / len(driver_utils)
                util_std_dev = math.sqrt(util_variance)
                balance_penalty = (util_std_dev / avg_util) * 7200
            else:
                balance_penalty = 0
        else:
            balance_penalty = 0

        empty_routes = sum(1 for assignment in solution if not assignment.delivery_indices)
        empty_route_penalty = empty_routes * 3600

        total_cost = total_time + unassigned_penalty + balance_penalty + empty_route_penalty

        return total_cost

    def optimize(self):
        """
        Abstract method to be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement optimize()")
