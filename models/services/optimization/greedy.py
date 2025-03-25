import time

from tqdm import tqdm

from models.entities.delivery import DeliveryAssignment
from models.services.optimization.base_optimizer import DeliveryOptimizer


class GreedyOptimizer(DeliveryOptimizer):
    """
    A pure greedy optimization algorithm for vehicle routing problems that makes locally
    optimal choices at each step without any subsequent improvement phases.
    """

    def optimize(self):
        try:
            start_time = time.time()
            progress_bar = tqdm(total=2, desc="Greedy optimization")

            # Phase 1: Construct balanced initial solution
            progress_bar.set_description("Phase 1: Balanced delivery assignment")
            solution, unassigned = self._construct_initial_solution()

            progress_bar.update(1)
            progress_bar.set_description("Phase 2: Route optimization")

            # Phase 2: Optimize routes using nearest neighbor
            for assignment in solution:
                if len(assignment.delivery_indices) > 1:
                    self._optimize_route_order(assignment)

            final_cost = self._evaluate_solution(solution, unassigned)
            total_time = self.calculate_total_time(solution)
            driver_utilization = self._calculate_driver_utilization(solution)
            balance_score = self._calculate_balance_score(solution)

            progress_bar.update(1)
            progress_bar.set_postfix({
                'cost': f"{final_cost:.2f}",
                'time': f"{total_time / 60:.2f}min",
                'util': f"{driver_utilization:.2f}%",
                'balance': f"{balance_score:.2f}",
                'unassigned': len(unassigned)
            })

            print("\nGreedy Optimization Results:")
            print(f"Total Travel Time: {self._format_time_hms(total_time)} ({total_time / 60:.2f} minutes)")
            print(f"Driver Utilization: {driver_utilization:.2f}%")
            print(f"Balance Score: {balance_score:.2f}")
            print(f"Unassigned Deliveries: {len(unassigned)}")
            print(f"Solution Cost: {final_cost:.2f}")
            print(f"Computation Time: {time.time() - start_time:.2f} seconds")
            print("=" * 50)

            progress_bar.close()

            self.finished.emit(solution, unassigned)

            return solution, unassigned

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None)
            return None, None

    def _construct_initial_solution(self):
        """
        Construct a solution using a balanced greedy assignment approach.
        Focuses on creating balanced workloads from the start by assigning
        each delivery to the least utilized driver that can handle it.
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

        for idx in delivery_indices:
            _, _, weight, volume = self.snapped_delivery_points[idx]

            driver_utils = []
            for i, assignment in enumerate(solution):
                driver = next((d for d in self.delivery_drivers if d.id == assignment.driver_id), None)

                if (assignment.total_weight + weight <= driver.weight_capacity and
                        assignment.total_volume + volume <= driver.volume_capacity):
                    weight_util = assignment.total_weight / driver.weight_capacity
                    volume_util = assignment.total_volume / driver.volume_capacity
                    avg_util = (weight_util + volume_util) / 2

                    driver_utils.append((i, avg_util))

            if driver_utils:
                driver_utils.sort(key=lambda x: x[1])
                best_driver_idx = driver_utils[0][0]

                assignment = solution[best_driver_idx]
                assignment.delivery_indices.append(idx)
                assignment.total_weight += weight
                assignment.total_volume += volume
            else:
                unassigned.add(idx)

        return solution, unassigned
