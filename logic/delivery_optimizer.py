import math
import random
import time
from copy import deepcopy

from PyQt5 import QtCore
from tqdm import tqdm

from config import OPTIMIZATION_SETTINGS
from logic.base_delivery_optimizer import DeliveryOptimizer
from models import DeliveryAssignment


class SimulatedAnnealingOptimizer(DeliveryOptimizer):
    """
    Optimizer that uses simulated annealing algorithm to find optimal delivery assignments
    and route ordering by allowing worse solutions early in the optimization process.
    """
    update_visualization = QtCore.pyqtSignal(object, object)

    def __init__(self, delivery_drivers, snapped_delivery_points, G, map_widget):
        super().__init__(delivery_drivers, snapped_delivery_points, G, map_widget)

    def optimize(self):
        """
        Optimizes delivery assignments using simulated annealing with adaptive cooling.
        Dynamically adjusts cooling rate based on improvement progress, slowing cooling
        when finding promising solutions and accelerating when exploration stagnates.
        """
        try:
            initial_solution, unassigned = self._generate_initial_solution()
            current_solution = deepcopy(initial_solution)
            current_unassigned = set(unassigned)
            best_solution = deepcopy(current_solution)
            best_unassigned = set(current_unassigned)

            current_cost = self._evaluate_solution(current_solution, current_unassigned)
            best_cost = current_cost

            initial_time = self.calculate_total_time(initial_solution)
            best_time = initial_time

            print("\nInitial Solution Statistics:")
            print(f"Total Travel Time: {self._format_time_hms(initial_time)} ({initial_time / 60:.2f} minutes)")
            print("-" * 50)

            temperature = OPTIMIZATION_SETTINGS['INITIAL_TEMPERATURE']
            base_cooling_rate = OPTIMIZATION_SETTINGS['COOLING_RATE']
            cooling_rate = base_cooling_rate
            min_temperature = OPTIMIZATION_SETTINGS['MIN_TEMPERATURE']
            iterations_per_temperature = OPTIMIZATION_SETTINGS['ITERATIONS_PER_TEMPERATURE']

            min_cooling_rate = 0.90
            max_cooling_rate = 0.98
            no_improvement_threshold = 3
            improvement_threshold = 0.01

            no_improvement_count = 0

            estimated_iterations = int(
                math.log(min_temperature / temperature) / math.log(base_cooling_rate)) * iterations_per_temperature

            progress_bar = tqdm(total=estimated_iterations, desc="Optimizing routes")
            last_update_time = time.time()
            iteration_count = 0

            while temperature > min_temperature:
                temp_start_cost = best_cost

                for i in range(iterations_per_temperature):
                    neighbor_solution, neighbor_unassigned = self._generate_neighbor_solution(
                        current_solution, current_unassigned
                    )

                    neighbor_cost = self._evaluate_solution(neighbor_solution, neighbor_unassigned)
                    cost_diff = neighbor_cost - current_cost

                    if (cost_diff < 0) or (random.random() < math.exp(-cost_diff / temperature)):
                        current_solution = deepcopy(neighbor_solution)
                        current_unassigned = set(neighbor_unassigned)
                        current_cost = neighbor_cost

                        if current_cost < best_cost:
                            best_solution = deepcopy(current_solution)
                            best_unassigned = set(current_unassigned)
                            best_cost = current_cost
                            best_time = self.calculate_total_time(best_solution)

                    progress_bar.update(1)
                    iteration_count += 1

                    total_time = self.calculate_total_time(current_solution)
                    driver_utilization = self._calculate_driver_utilization(current_solution)
                    balance_score = self._calculate_balance_score(current_solution)

                    progress_bar.set_postfix({
                        'temp': f"{temperature:.2f}",
                        'cost': f"{current_cost:.2f}",
                        'best': f"{best_cost:.2f}",
                        'time': f"{total_time / 60:.2f}min",
                        'best_time': f"{best_time / 60:.2f}min",
                        'util': f"{driver_utilization:.2f}%",
                        'balance': f"{balance_score:.2f}",
                        'unassigned': len(current_unassigned),
                        'cooling': f"{cooling_rate:.4f}"
                    })

                    current_time = time.time()
                    if (OPTIMIZATION_SETTINGS['VISUALIZE_PROCESS'] and
                            (current_time - last_update_time > 1.0)):
                        self.update_visualization.emit(best_solution, best_unassigned)
                        last_update_time = current_time

                rel_improvement = (temp_start_cost - best_cost) / temp_start_cost if temp_start_cost > 0 else 0

                if rel_improvement > improvement_threshold:
                    cooling_rate = min(max_cooling_rate, cooling_rate + 0.01)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= no_improvement_threshold:
                    cooling_rate = max(min_cooling_rate, cooling_rate - 0.02)
                    no_improvement_count = 0

                    if random.random() < 0.2:
                        temperature = min(temperature * 1.2, OPTIMIZATION_SETTINGS['INITIAL_TEMPERATURE'] / 2)

                temperature *= cooling_rate

                remaining_temp_levels = math.log(min_temperature / temperature) / math.log(cooling_rate)
                remaining_iterations = int(remaining_temp_levels * iterations_per_temperature)
                progress_bar.total = iteration_count + remaining_iterations

            progress_bar.close()

            self.update_visualization.emit(best_solution, best_unassigned)

            final_time = self.calculate_total_time(best_solution)
            time_improvement = (initial_time - final_time) / initial_time * 100 if initial_time > 0 else 0

            print("\nFinal Solution Statistics:")
            print(f"Total Travel Time: {self._format_time_hms(final_time)} ({final_time / 60:.2f} minutes)")
            print(f"Solution Cost: {best_cost:.2f}")
            print(f"Time Improvement: {time_improvement:.2f}%")
            print(f"Final Cooling Rate: {cooling_rate:.4f}")
            print("=" * 50)

            self.finished.emit(best_solution, best_unassigned)
            return best_solution, best_unassigned

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None)
            return None, None

    def _generate_initial_solution(self):
        """
        Generate an initial balanced solution by distributing deliveries to drivers
        using a round-robin approach with capacity constraints.
        """
        delivery_indices = list(range(len(self.snapped_delivery_points)))
        delivery_indices.sort(
            key=lambda idx: -(self.snapped_delivery_points[idx][2] / self.snapped_delivery_points[idx][3]))

        solution = []
        for driver in self.delivery_drivers:
            assignment = DeliveryAssignment(
                driver_id=driver.id,
                delivery_indices=[],
                total_weight=0.0,
                total_volume=0.0
            )
            solution.append(assignment)

        unassigned = set()

        driver_idx = 0
        for idx in delivery_indices:
            _, _, weight, volume = self.snapped_delivery_points[idx]

            assignment = solution[driver_idx]
            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)

            if (assignment.total_weight + weight <= driver.weight_capacity and
                    assignment.total_volume + volume <= driver.volume_capacity):

                assignment.delivery_indices.append(idx)
                assignment.total_weight += weight
                assignment.total_volume += volume
            else:
                found_driver = False

                for j in range(1, len(solution)):
                    alt_driver_idx = (driver_idx + j) % len(solution)
                    alt_assignment = solution[alt_driver_idx]
                    alt_driver = next((d for d in self.delivery_drivers if d.id == alt_assignment.driver_id), None)

                    if (alt_assignment.total_weight + weight <= alt_driver.weight_capacity and
                            alt_assignment.total_volume + volume <= alt_driver.volume_capacity):
                        alt_assignment.delivery_indices.append(idx)
                        alt_assignment.total_weight += weight
                        alt_assignment.total_volume += volume
                        found_driver = True
                        break

                if not found_driver:
                    unassigned.add(idx)

            driver_idx = (driver_idx + 1) % len(solution)

        self._balance_initial_solution(solution)

        for assignment in solution:
            if len(assignment.delivery_indices) > 1:
                self._optimize_route_order(assignment)

        return solution, unassigned

    def _balance_initial_solution(self, solution):
        driver_utils = []
        for i, assignment in enumerate(solution):
            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            weight_util = assignment.total_weight / driver.weight_capacity
            volume_util = assignment.total_volume / driver.volume_capacity
            avg_util = (weight_util + volume_util) / 2
            driver_utils.append((i, avg_util))

        driver_utils.sort(key=lambda x: -x[1])

        for j in range(5):
            for high_idx, _ in driver_utils[:len(driver_utils) // 2]:
                high_assignment = solution[high_idx]

                if not high_assignment.delivery_indices:
                    continue

                for low_idx, _ in reversed(driver_utils[len(driver_utils) // 2:]):
                    low_assignment = solution[low_idx]

                    for delivery_idx in list(high_assignment.delivery_indices):
                        _, _, weight, volume = self.snapped_delivery_points[delivery_idx]

                        low_driver = next((d for d in self.delivery_drivers if d.id == low_assignment.driver_id), None)

                        if (low_assignment.total_weight + weight <= low_driver.weight_capacity and
                                low_assignment.total_volume + volume <= low_driver.volume_capacity):
                            high_assignment.delivery_indices.remove(delivery_idx)
                            high_assignment.total_weight -= weight
                            high_assignment.total_volume -= volume

                            low_assignment.delivery_indices.append(delivery_idx)
                            low_assignment.total_weight += weight
                            low_assignment.total_volume += volume

                            break

    def _generate_neighbor_solution(self, current_solution, current_unassigned):
        """
        Generate a neighboring solution by applying a random move.
        """
        neighbor_solution = deepcopy(current_solution)
        neighbor_unassigned = set(current_unassigned)

        move_type = random.choices(
            ["swap", "move", "reorder", "assign", "unassign", "balance"],
            weights=[0.20, 0.20, 0.15, 0.15, 0.05, 0.25],
            k=1
        )[0]

        if move_type == "swap" and len(neighbor_solution) >= 2:
            # Swap deliveries between two drivers
            self._apply_swap_move(neighbor_solution)

        elif move_type == "move" and len(neighbor_solution) >= 2:
            # Move a delivery from one driver to another
            self._apply_move_move(neighbor_solution)

        elif move_type == "reorder":
            # Reorder deliveries for a driver
            self._apply_reorder_move(neighbor_solution)

        elif move_type == "assign" and neighbor_unassigned:
            # Assign an unassigned delivery
            self._apply_assign_move(neighbor_solution, neighbor_unassigned)

        elif move_type == "unassign":
            # Unassign a delivery
            self._apply_unassign_move(neighbor_solution, neighbor_unassigned)

        elif move_type == "balance" and len(neighbor_solution) >= 2:
            # Balance workload between drivers
            self._apply_balance_move(neighbor_solution)

        self._recalculate_assignment_totals(neighbor_solution)

        return neighbor_solution, neighbor_unassigned

    def _apply_balance_move(self, solution):
        """
        Apply a move specifically designed to balance workload between drivers.
        Moves deliveries from the most utilized driver to the least utilized.
        """
        driver_utils = []
        for i, assignment in enumerate(solution):
            if not assignment.delivery_indices:
                driver_utils.append((i, 0.0))
                continue

            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            weight_util = assignment.total_weight / driver.weight_capacity
            volume_util = assignment.total_volume / driver.volume_capacity
            avg_util = (weight_util + volume_util) / 2
            driver_utils.append((i, avg_util))

        driver_utils.sort(key=lambda x: -x[1])

        if len(driver_utils) < 2:
            return

        high_idx, high_util = driver_utils[0]
        low_idx, low_util = driver_utils[-1]

        if high_util - low_util < 0.1:  # Less than 10% difference
            return

        high_assignment = solution[high_idx]
        low_assignment = solution[low_idx]

        if not high_assignment.delivery_indices:
            return

        for delivery_idx in list(high_assignment.delivery_indices):
            _, _, weight, volume = self.snapped_delivery_points[delivery_idx]

            low_driver = next((d for d in self.delivery_drivers if d.id == low_assignment.driver_id), None)

            if (low_assignment.total_weight + weight <= low_driver.weight_capacity and
                    low_assignment.total_volume + volume <= low_driver.volume_capacity):
                high_assignment.delivery_indices.remove(delivery_idx)
                high_assignment.total_weight -= weight
                high_assignment.total_volume -= volume

                low_assignment.delivery_indices.append(delivery_idx)
                low_assignment.total_weight += weight
                low_assignment.total_volume += volume

                self._optimize_route_order(high_assignment)
                self._optimize_route_order(low_assignment)

                break

    def _apply_swap_move(self, solution):
        """
        Swap deliveries between two random drivers.
        """
        valid_assignments = [a for a in solution if a.delivery_indices]
        if len(valid_assignments) < 2:
            return

        idx1, idx2 = random.sample(range(len(valid_assignments)), 2)
        assignment1 = valid_assignments[idx1]
        assignment2 = valid_assignments[idx2]

        if not assignment1.delivery_indices or not assignment2.delivery_indices:
            return

        delivery_idx1 = random.choice(assignment1.delivery_indices)
        delivery_idx2 = random.choice(assignment2.delivery_indices)

        _, _, weight1, volume1 = self.snapped_delivery_points[delivery_idx1]
        _, _, weight2, volume2 = self.snapped_delivery_points[delivery_idx2]

        driver1 = next((d for d in self.delivery_drivers if d.id == assignment1.driver_id), None)
        driver2 = next((d for d in self.delivery_drivers if d.id == assignment2.driver_id), None)

        if (assignment1.total_weight - weight1 + weight2 <= driver1.weight_capacity and
                assignment1.total_volume - volume1 + volume2 <= driver1.volume_capacity and
                assignment2.total_weight - weight2 + weight1 <= driver2.weight_capacity and
                assignment2.total_volume - volume2 + volume1 <= driver2.volume_capacity):
            assignment1.delivery_indices.remove(delivery_idx1)
            assignment2.delivery_indices.remove(delivery_idx2)

            assignment1.delivery_indices.append(delivery_idx2)
            assignment2.delivery_indices.append(delivery_idx1)

            self._optimize_route_order(assignment1)
            self._optimize_route_order(assignment2)

    def _apply_move_move(self, solution):
        """
        Move a delivery from one driver to another.
        """
        valid_assignments = [a for a in solution if a.delivery_indices]
        if not valid_assignments:
            return

        from_idx = random.randrange(len(valid_assignments))
        from_assignment = valid_assignments[from_idx]

        if not from_assignment.delivery_indices:
            return

        delivery_idx = random.choice(from_assignment.delivery_indices)
        _, _, weight, volume = self.snapped_delivery_points[delivery_idx]

        to_drivers = [a for a in solution if a.driver_id != from_assignment.driver_id]
        if not to_drivers:
            return

        to_assignment = random.choice(to_drivers)

        to_driver = next((d for d in self.delivery_drivers if d.id == to_assignment.driver_id), None)

        if (to_assignment.total_weight + weight <= to_driver.weight_capacity and
                to_assignment.total_volume + volume <= to_driver.volume_capacity):
            from_assignment.delivery_indices.remove(delivery_idx)
            to_assignment.delivery_indices.append(delivery_idx)

            self._optimize_route_order(from_assignment)
            self._optimize_route_order(to_assignment)

    def _apply_reorder_move(self, solution):
        """
        Reorder deliveries for a random driver using 2-opt.
        """
        valid_assignments = [a for a in solution if len(a.delivery_indices) >= 3]
        if not valid_assignments:
            return

        assignment = random.choice(valid_assignments)

        if len(assignment.delivery_indices) >= 3:
            pos1 = random.randrange(len(assignment.delivery_indices) - 1)
            pos2 = pos1 + 1

            assignment.delivery_indices[pos1], assignment.delivery_indices[pos2] = \
                assignment.delivery_indices[pos2], assignment.delivery_indices[pos1]

    def _apply_assign_move(self, solution, unassigned):
        """
        Try to assign an unassigned delivery to a driver.
        """
        if not unassigned:
            return

        delivery_idx = random.choice(list(unassigned))
        _, _, weight, volume = self.snapped_delivery_points[delivery_idx]

        capable_assignments = []
        for assignment in solution:
            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)

            if (assignment.total_weight + weight <= driver.weight_capacity and
                    assignment.total_volume + volume <= driver.volume_capacity):
                capable_assignments.append(assignment)

        if capable_assignments:
            capable_assignments.sort(key=lambda a: (
                    a.total_weight / next(d for d in self.delivery_drivers if d.id == a.driver_id).weight_capacity +
                    a.total_volume / next(d for d in self.delivery_drivers if d.id == a.driver_id).volume_capacity
            ))

            assignment = capable_assignments[0]

            assignment.delivery_indices.append(delivery_idx)
            unassigned.remove(delivery_idx)

            self._optimize_route_order(assignment)

    def _apply_unassign_move(self, solution, unassigned):
        """
        Unassign a delivery from a driver.
        """
        valid_assignments = [a for a in solution if a.delivery_indices]
        if not valid_assignments:
            return

        assignment = random.choice(valid_assignments)

        if not assignment.delivery_indices:
            return

        delivery_idx = random.choice(assignment.delivery_indices)

        assignment.delivery_indices.remove(delivery_idx)
        unassigned.add(delivery_idx)

    def _recalculate_assignment_totals(self, solution):
        for assignment in solution:
            total_weight = 0.0
            total_volume = 0.0

            for idx in assignment.delivery_indices:
                _, _, weight, volume = self.snapped_delivery_points[idx]
                total_weight += weight
                total_volume += volume

            assignment.total_weight = total_weight
            assignment.total_volume = total_volume

    def _evaluate_solution(self, solution, unassigned=None):
        total_time = self.calculate_total_time(solution)

        unassigned_penalty = len(unassigned) * 3600 if unassigned else 0

        empty_routes = sum(1 for assignment in solution if not assignment.delivery_indices)
        empty_route_penalty = empty_routes * 3600

        driver_utils = []
        delivery_counts = []

        for assignment in solution:
            if not assignment.delivery_indices:
                driver_utils.append(0.0)
                delivery_counts.append(0)
                continue

            driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)
            weight_util = assignment.total_weight / driver.weight_capacity
            volume_util = assignment.total_volume / driver.volume_capacity
            avg_util = (weight_util + volume_util) / 2
            driver_utils.append(avg_util)
            delivery_counts.append(len(assignment.delivery_indices))

        if driver_utils:
            avg_util = sum(driver_utils) / len(driver_utils)
            util_variance = sum((u - avg_util) ** 2 for u in driver_utils) / len(driver_utils)
            util_std_dev = math.sqrt(util_variance)

            if delivery_counts:
                avg_count = sum(delivery_counts) / len(delivery_counts)
                if avg_count > 0:
                    count_variance = sum((c - avg_count) ** 2 for c in delivery_counts) / len(delivery_counts)
                    count_std_dev = math.sqrt(count_variance)
                    count_cv = count_std_dev / avg_count if avg_count > 0 else 0
                else:
                    count_cv = 0
            else:
                count_cv = 0

            balance_penalty = (util_std_dev / avg_util if avg_util > 0 else 0) * 7200
            count_penalty = count_cv * 3600
        else:
            balance_penalty = 0
            count_penalty = 0

        low_count_penalty = sum(3600 / (count + 1) for count in delivery_counts if count < 3)

        total_cost = (
                total_time +
                unassigned_penalty +
                empty_route_penalty +
                balance_penalty +
                count_penalty +
                low_count_penalty
        )

        return total_cost
