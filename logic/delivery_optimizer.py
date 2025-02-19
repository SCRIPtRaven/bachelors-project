import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import osmnx as ox
from PyQt5.QtCore import QObject, pyqtSignal
from tqdm import tqdm

from config.optimization_settings import OPTIMIZATION_SETTINGS
from models import Delivery, DeliveryAssignment


class SimulatedAnnealingOptimizer(QObject):
    update_visualization = pyqtSignal(object, object)
    finished = pyqtSignal(object, object)
    visualization_complete = pyqtSignal()

    def __init__(self, drivers, delivery_tuples, G):
        super().__init__()
        self.drivers = drivers
        self.deliveries = [
            Delivery(coordinates=(d[0], d[1]), weight=d[2], volume=d[3])
            for d in delivery_tuples
        ]
        self.G = G
        self.distance_cache = {}
        self.best_solution = None
        self.best_fitness = float('inf')
        self.unassigned_deliveries = set()

        self.update_interval = 0.5
        self.last_update = 0

    def get_cached_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Get cached distance between two points or calculate if not cached."""
        cache_key = (start, end)
        if cache_key not in self.distance_cache:
            try:
                start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
                end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])
                path_length = nx.shortest_path_length(
                    self.G, start_node, end_node, weight='length'
                )
                self.distance_cache[cache_key] = path_length
                self.distance_cache[(end, start)] = path_length
            except:
                self.distance_cache[cache_key] = 10000
                self.distance_cache[(end, start)] = 10000
        return self.distance_cache[cache_key]

    def calculate_route_distance(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total route distance between consecutive points."""
        if len(points) < 2:
            return 0

        return sum(self.get_cached_distance(points[i], points[i + 1])
                   for i in range(len(points) - 1))

    def calculate_fitness(self, assignment: DeliveryAssignment) -> float:
        if not assignment.delivery_indices:
            return float('inf')

        driver = next(d for d in self.drivers if d.id == assignment.driver_id)

        if (assignment.total_weight > driver.weight_capacity or
                assignment.total_volume > driver.volume_capacity):
            return float('inf')

        delivery_points = [self.deliveries[i].coordinates for i in assignment.delivery_indices]
        total_distance = self.calculate_route_distance(delivery_points)

        weight_utilization = assignment.total_weight / driver.weight_capacity
        volume_utilization = assignment.total_volume / driver.volume_capacity
        avg_utilization = (weight_utilization + volume_utilization) / 2

        return total_distance * (1 + (1 - avg_utilization))

    def optimize(self):
        try:
            current_solution = self.generate_initial_solution()
            current_fitness = sum(a.fitness for a in current_solution)

            self.best_solution = current_solution
            self.best_fitness = current_fitness
            initial_temperature = OPTIMIZATION_SETTINGS['INITIAL_TEMPERATURE']
            temperature = initial_temperature

            progress_format = "{desc}: {percentage:3.0f}% |{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}]"

            with tqdm(
                    total=initial_temperature,
                    desc="Optimizing Routes",
                    bar_format=progress_format,
                    ncols=100
            ) as progress:
                last_temp = initial_temperature
                iteration_count = 0
                last_improvement = 0
                stagnation_limit = 1000

                while temperature > OPTIMIZATION_SETTINGS['MIN_TEMPERATURE']:
                    improved = False

                    for _ in range(OPTIMIZATION_SETTINGS['ITERATIONS_PER_TEMPERATURE']):
                        iteration_count += 1

                        if OPTIMIZATION_SETTINGS['VISUALIZE_PROCESS']:
                            current_time = time.time()
                            if current_time - self.last_update >= self.update_interval:
                                self.update_visualization.emit(current_solution, self.unassigned_deliveries)
                                self.last_update = current_time

                        neighbor_solution = self.get_neighbor_solution(current_solution)
                        neighbor_fitness = sum(a.fitness for a in neighbor_solution)

                        if neighbor_fitness < current_fitness:
                            current_solution = neighbor_solution
                            current_fitness = neighbor_fitness
                            improved = True
                            last_improvement = iteration_count

                            if current_fitness < self.best_fitness:
                                self.best_solution = current_solution
                                self.best_fitness = current_fitness

                        else:
                            delta = neighbor_fitness - current_fitness
                            if random.random() < math.exp(-delta / temperature):
                                current_solution = neighbor_solution
                                current_fitness = neighbor_fitness

                        progress.set_description(
                            f"T: {temperature:.1f} | Best: {self.best_fitness:.0f} | Iter: {iteration_count}"
                        )

                        if iteration_count - last_improvement > stagnation_limit:
                            temperature = OPTIMIZATION_SETTINGS['MIN_TEMPERATURE']
                            break

                    if improved:
                        temperature *= OPTIMIZATION_SETTINGS['COOLING_RATE']
                    else:
                        temperature *= OPTIMIZATION_SETTINGS['COOLING_RATE'] * 0.5

                    temp_difference = last_temp - temperature
                    progress.update(temp_difference)
                    last_temp = temperature

                progress.update(last_temp - OPTIMIZATION_SETTINGS['MIN_TEMPERATURE'])

            self.update_visualization.emit(self.best_solution, self.unassigned_deliveries)
            self.finished.emit(self.best_solution, self.unassigned_deliveries)

        except Exception as e:
            print(f"Error in optimization: {e}")
            self.finished.emit(None, set())

    def generate_initial_solution(self) -> List[DeliveryAssignment]:
        """Generate initial solution using geographic sectors"""
        assignments = []
        used_deliveries = set()

        all_lats = [d.coordinates[0] for d in self.deliveries]
        all_lons = [d.coordinates[1] for d in self.deliveries]
        lat_mid = (max(all_lats) + min(all_lats)) / 2
        lon_mid = (max(all_lons) + min(all_lons)) / 2

        quadrants = {
            'NE': [], 'NW': [], 'SE': [], 'SW': []
        }

        for idx, delivery in enumerate(self.deliveries):
            lat, lon = delivery.coordinates
            quad = (
                       'N' if lat > lat_mid else 'S'
                   ) + (
                       'E' if lon > lon_mid else 'W'
                   )
            quadrants[quad].append(idx)

        drivers_per_quad = {quad: [] for quad in quadrants}
        for idx, driver in enumerate(self.drivers):
            quad = list(quadrants.keys())[idx % 4]
            drivers_per_quad[quad].append(driver)

        for quad, drivers in drivers_per_quad.items():
            if not drivers:
                continue

            quad_deliveries = quadrants[quad]
            deliveries_per_driver = len(quad_deliveries) // len(drivers)

            for driver in drivers:
                delivery_indices = []
                total_weight = 0
                total_volume = 0

                quad_center = (
                    lat_mid + (0.5 if 'N' in quad else -0.5),
                    lon_mid + (0.5 if 'E' in quad else -0.5)
                )

                sorted_deliveries = sorted(
                    [i for i in quad_deliveries if i not in used_deliveries],
                    key=lambda idx: (
                            (self.deliveries[idx].coordinates[0] - quad_center[0]) ** 2 +
                            (self.deliveries[idx].coordinates[1] - quad_center[1]) ** 2
                    )
                )

                for idx in sorted_deliveries[:deliveries_per_driver]:
                    delivery = self.deliveries[idx]
                    if (total_weight + delivery.weight <= driver.weight_capacity and
                            total_volume + delivery.volume <= driver.volume_capacity):
                        delivery_indices.append(idx)
                        total_weight += delivery.weight
                        total_volume += delivery.volume
                        used_deliveries.add(idx)

                assignment = DeliveryAssignment(
                    driver_id=driver.id,
                    delivery_indices=delivery_indices,
                    total_weight=total_weight,
                    total_volume=total_volume
                )
                assignment.fitness = self.calculate_fitness(assignment)
                assignments.append(assignment)

        self.unassigned_deliveries = set(range(len(self.deliveries))) - used_deliveries
        return assignments

    def get_neighbor_solution(self, current: List[DeliveryAssignment]) -> List[DeliveryAssignment]:
        """Generate neighbor solution by various moves while ensuring unique delivery assignments."""
        new_solution = [
            DeliveryAssignment(
                driver_id=a.driver_id,
                delivery_indices=a.delivery_indices.copy(),
                total_weight=a.total_weight,
                total_volume=a.total_volume,
                fitness=a.fitness
            ) for a in current
        ]

        move_type = random.choice(['swap', 'transfer', 'reverse'])

        if move_type == 'swap':
            if len(new_solution) < 2:
                return new_solution

            drivers_with_deliveries = [
                (idx, assignment) for idx, assignment in enumerate(new_solution)
                if assignment.delivery_indices
            ]

            if len(drivers_with_deliveries) < 2:
                return new_solution

            (idx1, a1), (idx2, a2) = random.sample(drivers_with_deliveries, 2)

            idx1_delivery = random.choice(a1.delivery_indices)
            idx2_delivery = random.choice(a2.delivery_indices)

            d1 = self.deliveries[idx1_delivery]
            d2 = self.deliveries[idx2_delivery]

            new_weight1 = a1.total_weight - d1.weight + d2.weight
            new_volume1 = a1.total_volume - d1.volume + d2.volume
            new_weight2 = a2.total_weight - d2.weight + d1.weight
            new_volume2 = a2.total_volume - d2.volume + d1.volume

            driver1 = next(d for d in self.drivers if d.id == a1.driver_id)
            driver2 = next(d for d in self.drivers if d.id == a2.driver_id)

            if (new_weight1 <= driver1.weight_capacity and
                    new_volume1 <= driver1.volume_capacity and
                    new_weight2 <= driver2.weight_capacity and
                    new_volume2 <= driver2.volume_capacity):
                a1.delivery_indices[a1.delivery_indices.index(idx1_delivery)] = idx2_delivery
                a2.delivery_indices[a2.delivery_indices.index(idx2_delivery)] = idx1_delivery

                a1.total_weight = new_weight1
                a1.total_volume = new_volume1
                a2.total_weight = new_weight2
                a2.total_volume = new_volume2

                a1.fitness = self.calculate_fitness(a1)
                a2.fitness = self.calculate_fitness(a2)

        elif move_type == 'transfer':
            if len(new_solution) < 2:
                return new_solution

            source_drivers = [(idx, a) for idx, a in enumerate(new_solution) if a.delivery_indices]
            if not source_drivers:
                return new_solution

            source_idx, source = random.choice(source_drivers)
            target_idx = random.choice([i for i in range(len(new_solution)) if i != source_idx])
            target = new_solution[target_idx]

            delivery_idx = random.choice(source.delivery_indices)
            delivery = self.deliveries[delivery_idx]

            new_weight = target.total_weight + delivery.weight
            new_volume = target.total_volume + delivery.volume
            target_driver = next(d for d in self.drivers if d.id == target.driver_id)

            if (new_weight <= target_driver.weight_capacity and
                    new_volume <= target_driver.volume_capacity):
                source.delivery_indices.remove(delivery_idx)
                source.total_weight -= delivery.weight
                source.total_volume -= delivery.volume
                source.fitness = self.calculate_fitness(source)

                target.delivery_indices.append(delivery_idx)
                target.total_weight = new_weight
                target.total_volume = new_volume
                target.fitness = self.calculate_fitness(target)

        elif move_type == 'reverse':
            assignment = random.choice([a for a in new_solution if len(a.delivery_indices) > 2])
            if assignment.delivery_indices:
                start = random.randint(0, len(assignment.delivery_indices) - 2)
                end = random.randint(start + 1, len(assignment.delivery_indices))
                assignment.delivery_indices[start:end] = reversed(assignment.delivery_indices[start:end])
                assignment.fitness = self.calculate_fitness(assignment)

        return new_solution
