import math
import random
import statistics
import time
from copy import deepcopy
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

        self.update_interval = 0.5
        self.last_update = 0

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

        self.travel_time_matrix = {}
        for i in range(len(self.all_nodes)):
            for j in range(i, len(self.all_nodes)):
                try:
                    travel_time = nx.shortest_path_length(
                        self.G, self.all_nodes[i], self.all_nodes[j], weight='travel_time'
                    )
                except Exception:
                    try:
                        distance = nx.shortest_path_length(
                            self.G, self.all_nodes[i], self.all_nodes[j], weight='length'
                        )
                        travel_time = distance / (20 * 1000 / 3600)
                    except Exception:
                        print("Using fallback time for route (VERY BAD)")
                        travel_time = 1800
                self.travel_time_matrix[(i, j)] = travel_time
                self.travel_time_matrix[(j, i)] = travel_time

    def get_cached_travel_time(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Get travel time between two points. If both points were precomputed, use the
        travel_time_matrix; otherwise, fall back to the on-demand computation.
        Returns time in seconds.
        """
        start_idx = self.point_to_index.get(start)
        end_idx = self.point_to_index.get(end)
        if start_idx is not None and end_idx is not None:
            return self.travel_time_matrix[(start_idx, end_idx)]
        cache_key = (start, end)
        if cache_key not in self.time_cache:
            try:
                start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
                end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])
                travel_time = nx.shortest_path_length(
                    self.G, start_node, end_node, weight='travel_time'
                )
                self.time_cache[cache_key] = travel_time
                self.time_cache[(end, start)] = travel_time
            except:
                try:
                    distance = nx.shortest_path_length(
                        self.G, start_node, end_node, weight='length'
                    )
                    travel_time = distance / (20 * 1000 / 3600)
                except:
                    print("Using fallback time for route (VERY BAD)")
                    travel_time = 1800
                self.time_cache[cache_key] = travel_time
                self.time_cache[(end, start)] = travel_time
        return self.time_cache[cache_key]

    def calculate_route_time(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total route time between consecutive points in seconds."""
        if len(points) < 2:
            return 0
        return sum(self.get_cached_travel_time(points[i], points[i + 1])
                   for i in range(len(points) - 1))

    def calculate_total_time(self, solution) -> float:
        """Calculate total travel time for all routes in the solution in seconds."""
        total_time = 0
        warehouse_coords = self.map_widget.get_warehouse_location()

        for assignment in solution:
            if assignment.delivery_indices:
                delivery_points = [self.deliveries[i].coordinates for i in assignment.delivery_indices]
                route_points = [warehouse_coords] + delivery_points + [warehouse_coords]
                total_time += self.calculate_route_time(route_points)

        return total_time

    def format_time(self, seconds: float) -> str:
        """Convert seconds into a human-readable HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def calculate_constraint_score(self, assignment: DeliveryAssignment) -> float:
        """
        Calculate how well this assignment uses capacity vs. the "ideal" usage
        of total system capacity. The closer to 1.0, the better.
        """
        if not assignment.delivery_indices:
            return 0.0

        driver = next(d for d in self.drivers if d.id == assignment.driver_id)

        total_weight_capacity = sum(d.weight_capacity for d in self.drivers)
        total_volume_capacity = sum(d.volume_capacity for d in self.drivers)
        total_deliveries_weight = sum(d.weight for d in self.deliveries)
        total_deliveries_volume = sum(d.volume for d in self.deliveries)

        target_weight_util = min(1.0,
                                 total_deliveries_weight / total_weight_capacity) if total_weight_capacity > 0 else 1.0
        target_volume_util = min(1.0,
                                 total_deliveries_volume / total_volume_capacity) if total_volume_capacity > 0 else 1.0

        weight_util = 0.0
        volume_util = 0.0
        if driver.weight_capacity > 0:
            weight_util = (assignment.total_weight / driver.weight_capacity) / target_weight_util
        if driver.volume_capacity > 0:
            volume_util = (assignment.total_volume / driver.volume_capacity) / target_volume_util

        weight_score = min(1.0, weight_util)
        volume_score = min(1.0, volume_util)

        return (weight_score + volume_score) / 2.0

    def calculate_total_constraint_score(self, solution: List[DeliveryAssignment]) -> float:
        if not solution:
            return 0.0

        individual_scores = [self.calculate_constraint_score(a) for a in solution if a.delivery_indices]

        if not individual_scores:
            return 0.0

        avg_score = sum(individual_scores) / len(individual_scores)

        if len(individual_scores) > 1:
            imbalance = statistics.stdev(individual_scores)
            balance_penalty = 0.1 * imbalance
            return max(0, avg_score - balance_penalty)
        return avg_score

    def count_total_assigned_deliveries(self, solution: List[DeliveryAssignment]) -> int:
        """
        Counts how many deliveries are assigned in the given solution.
        Useful if you want to refine how you penalize unassigned deliveries,
        or if you want to compare assigned vs. unassigned quickly.
        """
        return sum(len(a.delivery_indices) for a in solution)

    def optimize(self):
        """
        Modified to:
          1) Use a "meltdown" approach if we stagnate too long (re-randomize).
          2) Provide higher iteration counts and a slightly different temperature schedule.
        """
        try:
            current_solution = self.generate_initial_solution()
            initial_time = self.calculate_total_time(current_solution)

            print(f"\nInitial Solution:")
            print(f"Cumulative Travel Time: {self.format_time(initial_time)} "
                  f"({initial_time / 3600:.2f} hours)")

            current_time = initial_time
            current_constraint_score = self.calculate_total_constraint_score(current_solution)

            self.best_solution = deepcopy(current_solution)
            self.best_time = current_time
            self.best_constraint_score = current_constraint_score

            initial_temperature = OPTIMIZATION_SETTINGS['INITIAL_TEMPERATURE']
            temperature = initial_temperature
            min_temperature = OPTIMIZATION_SETTINGS['MIN_TEMPERATURE']
            cooling_rate = OPTIMIZATION_SETTINGS['COOLING_RATE']
            iterations_per_temp = OPTIMIZATION_SETTINGS['ITERATIONS_PER_TEMPERATURE']
            stagnation_limit = 1000

            progress_format = "{desc} {percentage:3.0f}% |{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}]"

            with tqdm(
                    total=initial_temperature,
                    desc="Optimizing Routes",
                    bar_format=progress_format,
                    ncols=100
            ) as progress:
                last_temp = initial_temperature
                iteration_count = 0
                last_improvement = 0

                while temperature > min_temperature:
                    improved = False

                    for _ in range(iterations_per_temp):
                        neighbor_solution = self.get_neighbor_solution(
                            current_solution,
                            temperature,
                            initial_temperature
                        )
                        neighbor_time = self.calculate_total_time(neighbor_solution)
                        neighbor_constraint_score = self.calculate_total_constraint_score(neighbor_solution)

                        # Hard acceptance if strictly better
                        if neighbor_time < current_time and neighbor_constraint_score >= current_constraint_score:
                            current_solution = neighbor_solution
                            current_constraint_score = neighbor_constraint_score
                            improved = True
                            last_improvement = iteration_count

                            if (neighbor_time < self.best_time and
                                    neighbor_constraint_score >= self.best_constraint_score):
                                self.best_solution = deepcopy(current_solution)
                                self.best_time = neighbor_time
                                self.best_constraint_score = neighbor_constraint_score

                        else:
                            # Soft acceptance
                            current_obj = self.calculate_objective_value(
                                current_time, current_constraint_score)
                            neighbor_obj = self.calculate_objective_value(
                                neighbor_time, neighbor_constraint_score)
                            delta = neighbor_obj - current_obj

                            if delta < 0 or random.random() < math.exp(-delta / temperature):
                                current_solution = neighbor_solution
                                current_constraint_score = neighbor_constraint_score
                                improved = True
                                last_improvement = iteration_count

                                if neighbor_time < self.best_time:
                                    self.best_solution = deepcopy(current_solution)
                                    self.best_time = neighbor_time
                                    self.best_constraint_score = neighbor_constraint_score

                        iteration_count += 1

                        current_time = time.time()
                        if OPTIMIZATION_SETTINGS['VISUALIZE_PROCESS'] and (
                                current_time - self.last_update) >= self.update_interval:
                            self.update_visualization.emit(deepcopy(self.best_solution), self.unassigned_deliveries)
                            self.last_update = current_time

                        progress.set_description(
                            f"Time: {self.format_time(self.best_time)} | "
                            f"Constraints: {self.best_constraint_score:.3f} | "
                            f"Iter: {iteration_count}"
                        )

                        # ---------------------------
                        # STAGNATION / MELTDOWN LOGIC
                        # ---------------------------
                        if iteration_count - last_improvement > stagnation_limit:
                            # Perform a meltdown: randomize current solution heavily
                            # but keep the best_solution as our global best
                            self.perform_random_meltdown(current_solution)
                            current_time = self.calculate_total_time(current_solution)
                            current_constraint_score = self.calculate_total_constraint_score(current_solution)
                            last_improvement = iteration_count

                    if improved:
                        temperature *= cooling_rate
                    else:
                        temperature *= (cooling_rate * 0.5)

                    temp_difference = last_temp - temperature
                    progress.update(temp_difference)
                    last_temp = temperature

                progress.update(last_temp - min_temperature)

            print(f"\nFinal Optimized Solution:")
            print(f"Cumulative Travel Time: {self.format_time(self.best_time)} "
                  f"({self.best_time / 3600:.2f} hours)")
            time_improvement = (initial_time - self.best_time) / initial_time * 100 if initial_time > 0 else 0
            print(f"\nTime Improvement: {time_improvement:.2f}%")

            self.update_visualization.emit(self.best_solution, self.unassigned_deliveries)
            self.finished.emit(self.best_solution, self.unassigned_deliveries)

        except Exception as e:
            print(f"Error in optimization: {e}")
            self.finished.emit(None, set())

    def calculate_objective_value(self, time_seconds: float, constraint_score: float) -> float:
        """
        Modified to add a penalty for unassigned deliveries so that
        solutions ignoring feasible deliveries are not favored.
        """
        time_in_hours = time_seconds / 3600

        unassigned_count = len(self.unassigned_deliveries)

        penalty_for_unassigned = 2 * unassigned_count

        return time_in_hours * (2.0 - constraint_score) + penalty_for_unassigned

    def perform_random_meltdown(self, current_solution: List[DeliveryAssignment]) -> None:
        """
        A new helper method to handle "meltdown" when the solver stagnates.
        This heavily randomizes the current solution (but doesn't touch best_solution).
        """

        self.massive_redistribution(current_solution)

    def generate_initial_solution(self) -> List[DeliveryAssignment]:
        """
        Generate a more randomized initial solution with controlled randomness

        Key Principles:
        1. Distribute deliveries across drivers randomly
        2. Respect individual driver capacity constraints
        3. Ensure most deliveries are assigned
        """
        assignments = []
        used_deliveries = set()

        randomized_delivery_indices = list(range(len(self.deliveries)))
        random.shuffle(randomized_delivery_indices)

        driver_weights = [0] * len(self.drivers)
        driver_volumes = [0] * len(self.drivers)

        for delivery_idx in randomized_delivery_indices:
            delivery = self.deliveries[delivery_idx]

            eligible_driver_indices = []
            for i, driver in enumerate(self.drivers):
                new_weight = driver_weights[i] + delivery.weight
                new_volume = driver_volumes[i] + delivery.volume
                if new_weight <= driver.weight_capacity and new_volume <= driver.volume_capacity:
                    eligible_driver_indices.append(i)

            if not eligible_driver_indices:
                self.unassigned_deliveries.add(delivery_idx)
                continue

            chosen_driver_index = random.choice(eligible_driver_indices)

            driver_weights[chosen_driver_index] += delivery.weight
            driver_volumes[chosen_driver_index] += delivery.volume

            chosen_driver = self.drivers[chosen_driver_index]
            existing_assignment = next(
                (a for a in assignments if a.driver_id == chosen_driver.id),
                None
            )

            if existing_assignment:
                existing_assignment.delivery_indices.append(delivery_idx)
                existing_assignment.total_weight += delivery.weight
                existing_assignment.total_volume += delivery.volume
            else:
                new_assignment = DeliveryAssignment(
                    driver_id=chosen_driver.id,
                    delivery_indices=[delivery_idx],
                    total_weight=delivery.weight,
                    total_volume=delivery.volume
                )
                assignments.append(new_assignment)

            used_deliveries.add(delivery_idx)

        self.unassigned_deliveries = set(range(len(self.deliveries))) - used_deliveries

        return assignments

    def _route_time_with_warehouse(self, assignment: DeliveryAssignment) -> float:
        """
        A private helper to quickly calculate time for a single route,
        including travel from and back to the warehouse.
        """
        warehouse_coords = self.map_widget.get_warehouse_location()
        if not assignment.delivery_indices:
            return 0.0

        points = [warehouse_coords] + [self.deliveries[i].coordinates
                                       for i in assignment.delivery_indices] \
                 + [warehouse_coords]
        return self.calculate_route_time(points)

    # ------------------------------------------------------------
    #                   NEIGHBOR GENERATION
    # ------------------------------------------------------------
    def get_neighbor_solution(self, current: List[DeliveryAssignment],
                              temperature: float,
                              initial_temperature: float) -> List[DeliveryAssignment]:
        """
        Generate a neighbor solution using a random move strategy.
        """

        new_solution = deepcopy(current)

        move_strategies = [
            # High temperature => more disruptive moves
            {
                'moves': [
                    'massive_redistribution',
                    'driver_swap_all',
                    'random_scramble',
                    'cross_exchange'
                ],
                'weight': 0.5
            },
            # Medium temperature
            {
                'moves': [
                    'swap',
                    'transfer',
                    'two_opt'
                ],
                'weight': 0.35
            },
            # Low temperature => small/fine moves
            {
                'moves': [
                    'reverse',
                    'transfer',
                    'two_opt'
                ],
                'weight': 0.15
            }
        ]

        # 3 temperature “tiers”: 0..(T/3), (T/3)..(2T/3), (2T/3)..T
        tier = min(2, int(3 * temperature / initial_temperature))
        current_move_set = move_strategies[tier]['moves']

        chosen_move = random.choice(current_move_set)

        if chosen_move == 'massive_redistribution':
            self.massive_redistribution(new_solution)
        elif chosen_move == 'driver_swap_all':
            self.driver_swap_all(new_solution)
        elif chosen_move == 'random_scramble':
            self.random_scramble(new_solution)
        elif chosen_move == 'swap':
            self.swap_deliveries(new_solution)
        elif chosen_move == 'transfer':
            self.transfer_delivery(new_solution)
        elif chosen_move == 'reverse':
            self.reverse_route(new_solution)
        elif chosen_move == 'cross_exchange':
            self.cross_exchange_deliveries(new_solution)
        elif chosen_move == 'two_opt':
            self.two_opt_intra_route(new_solution)

        return new_solution

    # -------------------------
    # Move Implementations
    # -------------------------

    def massive_redistribution(self, solution: List[DeliveryAssignment]):
        """
        Reassign *all* currently assigned deliveries among the drivers.
        Only leave a delivery unassigned if no driver can fit it.
        """
        all_deliveries = []
        for a in solution:
            for d_idx in a.delivery_indices:
                all_deliveries.append(d_idx)

        if not all_deliveries:
            return

        for a in solution:
            a.delivery_indices.clear()
            a.total_weight = 0
            a.total_volume = 0

        random.shuffle(all_deliveries)

        for d_idx in all_deliveries:
            delivery = self.deliveries[d_idx]
            feasible_assignments = []
            for a in solution:
                driver = next(d for d in self.drivers if d.id == a.driver_id)
                if (a.total_weight + delivery.weight <= driver.weight_capacity and
                        a.total_volume + delivery.volume <= driver.volume_capacity):
                    feasible_assignments.append(a)

            if feasible_assignments:
                chosen_a = random.choice(feasible_assignments)
                chosen_a.delivery_indices.append(d_idx)
                chosen_a.total_weight += delivery.weight
                chosen_a.total_volume += delivery.volume
            else:
                self.unassigned_deliveries.add(d_idx)

    def driver_swap_all(self, solution: List[DeliveryAssignment]):
        """
        Pick two different drivers and swap their entire sets of deliveries.
        """

        if len(solution) < 2:
            return

        a1, a2 = random.sample(solution, 2)

        d1 = next(d for d in self.drivers if d.id == a1.driver_id)
        d2 = next(d for d in self.drivers if d.id == a2.driver_id)

        new_a1_weight = sum(self.deliveries[idx].weight for idx in a2.delivery_indices)
        new_a1_volume = sum(self.deliveries[idx].volume for idx in a2.delivery_indices)

        new_a2_weight = sum(self.deliveries[idx].weight for idx in a1.delivery_indices)
        new_a2_volume = sum(self.deliveries[idx].volume for idx in a1.delivery_indices)

        if (new_a1_weight <= d1.weight_capacity and
                new_a1_volume <= d1.volume_capacity and
                new_a2_weight <= d2.weight_capacity and
                new_a2_volume <= d2.volume_capacity):
            old_a1_indices = a1.delivery_indices
            a1.delivery_indices = a2.delivery_indices
            a2.delivery_indices = old_a1_indices

            a1.total_weight = new_a1_weight
            a1.total_volume = new_a1_volume
            a2.total_weight = new_a2_weight
            a2.total_volume = new_a2_volume

    def random_scramble(self, solution: List[DeliveryAssignment]):
        """
        Pick one driver and shuffle the order of their deliveries.
        This can change route time.
        """
        if not solution:
            return

        a = random.choice(solution)
        if len(a.delivery_indices) > 1:
            random.shuffle(a.delivery_indices)

    def swap_deliveries(self, solution: List[DeliveryAssignment]):
        """
        Swap a single delivery between two drivers (if feasible).
        """

        if len(solution) < 2:
            return

        a1, a2 = random.sample(solution, 2)
        if not a1.delivery_indices or not a2.delivery_indices:
            return

        idx1 = random.choice(a1.delivery_indices)
        idx2 = random.choice(a2.delivery_indices)

        delivery1 = self.deliveries[idx1]
        delivery2 = self.deliveries[idx2]

        d1 = next(d for d in self.drivers if d.id == a1.driver_id)
        d2 = next(d for d in self.drivers if d.id == a2.driver_id)

        new_a1_weight = a1.total_weight - delivery1.weight + delivery2.weight
        new_a1_volume = a1.total_volume - delivery1.volume + delivery2.volume
        new_a2_weight = a2.total_weight - delivery2.weight + delivery1.weight
        new_a2_volume = a2.total_volume - delivery2.volume + delivery1.volume

        if (new_a1_weight <= d1.weight_capacity and
                new_a1_volume <= d1.volume_capacity and
                new_a2_weight <= d2.weight_capacity and
                new_a2_volume <= d2.volume_capacity):
            a1.delivery_indices.remove(idx1)
            a1.delivery_indices.append(idx2)
            a2.delivery_indices.remove(idx2)
            a2.delivery_indices.append(idx1)

            a1.total_weight = new_a1_weight
            a1.total_volume = new_a1_volume
            a2.total_weight = new_a2_weight
            a2.total_volume = new_a2_volume

    def transfer_delivery(self, solution: List[DeliveryAssignment]):
        """
        Move exactly one delivery from one driver to another, if capacity allows.
        """

        if len(solution) < 2:
            return

        source_candidates = [a for a in solution if a.delivery_indices]
        if not source_candidates:
            return

        source = random.choice(source_candidates)
        delivery_idx = random.choice(source.delivery_indices)
        delivery = self.deliveries[delivery_idx]

        target_candidates = [a for a in solution if a.driver_id != source.driver_id]
        if not target_candidates:
            return

        target = random.choice(target_candidates)
        target_driver = next(d for d in self.drivers if d.id == target.driver_id)

        new_target_weight = target.total_weight + delivery.weight
        new_target_volume = target.total_volume + delivery.volume

        if new_target_weight <= target_driver.weight_capacity and new_target_volume <= target_driver.volume_capacity:
            source.delivery_indices.remove(delivery_idx)
            source.total_weight -= delivery.weight
            source.total_volume -= delivery.volume

            target.delivery_indices.append(delivery_idx)
            target.total_weight += delivery.weight
            target.total_volume += delivery.volume

    def reverse_route(self, solution: List[DeliveryAssignment]):
        """
        Reverse the order of deliveries for one driver, as a small local change.
        """
        if not solution:
            return
        a = random.choice(solution)
        if len(a.delivery_indices) > 1:
            a.delivery_indices.reverse()

    def two_opt_intra_route(self, solution: List[DeliveryAssignment]) -> None:
        """
        Performs a single 2-Opt improvement on one driver's route, if possible.

        We:
          1) Pick a random driver that has at least 4 deliveries (since 2-Opt
             re-links edges).
          2) Pick two edges to 'flip'.
        """
        candidates = [a for a in solution if len(a.delivery_indices) >= 4]
        if not candidates:
            return

        assignment = random.choice(candidates)
        route = assignment.delivery_indices.copy()

        if len(route) < 4:
            return

        i, k = sorted(random.sample(range(len(route)), 2))
        if k - i < 2:
            return

        assignment.delivery_indices = route[:i] + route[i:k + 1][::-1] + route[k + 1:]

    def cross_exchange_deliveries(self, solution: List[DeliveryAssignment]) -> None:
        """
        Attempt a 2-delivery cross-exchange between two different drivers/routes.
        We'll pick two deliveries from route A and two from route B and swap them,
        if capacity allows.
        """
        if len(solution) < 2:
            return

        a1, a2 = random.sample(solution, 2)
        if len(a1.delivery_indices) < 2 or len(a2.delivery_indices) < 2:
            return

        d_indices_a1 = random.sample(a1.delivery_indices, 2)
        d_indices_a2 = random.sample(a2.delivery_indices, 2)

        for d_idx in d_indices_a1:
            a1.delivery_indices.remove(d_idx)
            a1.total_weight -= self.deliveries[d_idx].weight
            a1.total_volume -= self.deliveries[d_idx].volume

        for d_idx in d_indices_a2:
            a2.delivery_indices.remove(d_idx)
            a2.total_weight -= self.deliveries[d_idx].weight
            a2.total_volume -= self.deliveries[d_idx].volume

        driver1 = next(d for d in self.drivers if d.id == a1.driver_id)
        driver2 = next(d for d in self.drivers if d.id == a2.driver_id)

        new_a1_weight = a1.total_weight + sum(self.deliveries[idx].weight for idx in d_indices_a2)
        new_a1_volume = a1.total_volume + sum(self.deliveries[idx].volume for idx in d_indices_a2)
        new_a2_weight = a2.total_weight + sum(self.deliveries[idx].weight for idx in d_indices_a1)
        new_a2_volume = a2.total_volume + sum(self.deliveries[idx].volume for idx in d_indices_a1)

        if (new_a1_weight <= driver1.weight_capacity and
                new_a1_volume <= driver1.volume_capacity and
                new_a2_weight <= driver2.weight_capacity and
                new_a2_volume <= driver2.volume_capacity):

            a1.delivery_indices.extend(d_indices_a2)
            a1.total_weight = new_a1_weight
            a1.total_volume = new_a1_volume

            a2.delivery_indices.extend(d_indices_a1)
            a2.total_weight = new_a2_weight
            a2.total_volume = new_a2_volume
        else:
            for d_idx in d_indices_a1:
                a1.delivery_indices.append(d_idx)
                a1.total_weight += self.deliveries[d_idx].weight
                a1.total_volume += self.deliveries[d_idx].volume

            for d_idx in d_indices_a2:
                a2.delivery_indices.append(d_idx)
                a2.total_weight += self.deliveries[d_idx].weight
                a2.total_volume += self.deliveries[d_idx].volume
