import math

import networkx as nx
import osmnx as ox
from PyQt5.QtCore import pyqtSignal, QObject
from tqdm import tqdm

from logic.delivery_optimizer import SimulatedAnnealingOptimizer
from models import DeliveryAssignment, Delivery


class GreedyOptimizer(QObject):
    update_visualization = pyqtSignal(object, object)
    finished = pyqtSignal(object, object)

    def __init__(self, drivers, delivery_tuples, G, map_widget):
        super().__init__()
        self.drivers = drivers
        self.deliveries = [
            Delivery(coordinates=(d[0], d[1]), weight=d[2], volume=d[3])
            for d in delivery_tuples
        ]
        self.G = G
        self.map_widget = map_widget
        self.time_cache = {}
        self.best_solution = None
        self.best_time = float('inf')
        self.best_constraint_score = 0.0
        self.unassigned_deliveries = set()

        self._precompute_travel_time_matrix()

    def _precompute_travel_time_matrix(self):
        """
        Precompute a matrix of travel times (in seconds) between the warehouse and
        all delivery points (and between deliveries themselves).
        """
        warehouse = self.map_widget.get_warehouse_location()
        self.all_points = [warehouse] + [delivery.coordinates for delivery in self.deliveries]
        self.point_to_index = {pt: idx for idx, pt in enumerate(self.all_points)}
        self.all_nodes = [ox.nearest_nodes(self.G, X=pt[1], Y=pt[0]) for pt in self.all_points]

        n = len(self.all_nodes)
        self.travel_time_matrix = {}
        for i in range(n):
            for j in range(i, n):
                try:
                    ttime = nx.shortest_path_length(
                        self.G, self.all_nodes[i], self.all_nodes[j], weight='travel_time'
                    )
                except Exception:
                    try:
                        distance = nx.shortest_path_length(
                            self.G, self.all_nodes[i], self.all_nodes[j], weight='length'
                        )
                        ttime = distance / (20 * 1000 / 3600)
                    except Exception:
                        print("Using fallback time for route (VERY BAD)")
                        ttime = 1800
                self.travel_time_matrix[(i, j)] = ttime
                self.travel_time_matrix[(j, i)] = ttime

    def _euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """Simple Euclidean distance; used only for a sorting heuristic."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def optimize(self):
        """
        Greedily build a delivery assignment by iterating over all deliveries (sorted by proximity
        to the warehouse) and inserting each into the driver’s route at the position that minimizes
        the extra travel time. Capacity constraints are checked before assignment.
        Also updates the tqdm progress bar with the current travel time metric.
        """
        warehouse = self.map_widget.get_warehouse_location()

        solution = []
        for driver in self.drivers:
            solution.append(
                DeliveryAssignment(
                    driver_id=driver.id,
                    delivery_indices=[],
                    total_weight=0,
                    total_volume=0
                )
            )

        deliveries_with_distance = []
        for i, delivery in enumerate(self.deliveries):
            dist = self._euclidean_distance(warehouse, delivery.coordinates)
            deliveries_with_distance.append((i, delivery, dist))
        deliveries_with_distance.sort(key=lambda x: x[2])

        pbar = tqdm(deliveries_with_distance, desc="Assigning Deliveries", ncols=100)
        for i, delivery, _ in pbar:
            best_increase = float('inf')
            best_assignment = None
            best_insertion_pos = None

            for assignment in solution:
                driver = next(d for d in self.drivers if d.id == assignment.driver_id)
                if (assignment.total_weight + delivery.weight > driver.weight_capacity or
                        assignment.total_volume + delivery.volume > driver.volume_capacity):
                    continue

                current_route = (
                        [warehouse] +
                        [self.deliveries[idx].coordinates for idx in assignment.delivery_indices] +
                        [warehouse]
                )

                insertion_best = float('inf')
                insertion_position = None
                for pos in range(1, len(current_route)):
                    prev_point = current_route[pos - 1]
                    next_point = current_route[pos]
                    old_segment = self.get_cached_travel_time(prev_point, next_point)
                    new_segment = (self.get_cached_travel_time(prev_point, delivery.coordinates) +
                                   self.get_cached_travel_time(delivery.coordinates, next_point))
                    increase = new_segment - old_segment
                    if increase < insertion_best:
                        insertion_best = increase
                        insertion_position = pos

                if insertion_best < best_increase:
                    best_increase = insertion_best
                    best_assignment = assignment
                    best_insertion_pos = insertion_position

            if best_assignment is not None and best_insertion_pos is not None:
                best_assignment.delivery_indices.insert(best_insertion_pos - 1, i)
                best_assignment.total_weight += delivery.weight
                best_assignment.total_volume += delivery.volume
            else:
                self.unassigned_deliveries.add(i)

            current_total_time = self.calculate_total_time(solution)
            pbar.set_description(f"Time: {self.format_time(current_total_time)}")

        self.best_solution = solution
        self.best_time = self.calculate_total_time(solution)
        self.best_constraint_score = self.calculate_total_constraint_score(solution)
        self.finished.emit(self.best_solution, self.unassigned_deliveries)

    calculate_route_time = SimulatedAnnealingOptimizer.calculate_route_time
    get_cached_travel_time = SimulatedAnnealingOptimizer.get_cached_travel_time
    calculate_constraint_score = SimulatedAnnealingOptimizer.calculate_constraint_score
    calculate_total_constraint_score = SimulatedAnnealingOptimizer.calculate_total_constraint_score
    calculate_total_time = SimulatedAnnealingOptimizer.calculate_total_time
    format_time = SimulatedAnnealingOptimizer.format_time
