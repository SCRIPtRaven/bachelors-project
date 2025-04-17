from typing import Dict, Any, Tuple, Optional

import gym
import numpy as np
from gym import spaces

from models.entities.disruption import Disruption, DisruptionType
from models.rl.actions import ActionType, DisruptionAction
from models.rl.simulation_controller import SimulationController
from models.rl.state import DeliverySystemState


class DeliveryEnvironment(gym.Env):
    """
    OpenAI Gym environment for training reinforcement learning agents
    on disruption resolution in a delivery routing system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment with configuration parameters

        Args:
            config: Dictionary containing configuration parameters
              - graph: NetworkX graph of the road network
              - warehouse_location: (lat, lon) of warehouse
              - delivery_points: List of delivery points
              - drivers: List of driver objects
              - disruption_generator: Function to generate disruptions
              - simulation_steps: Number of steps to run the simulation
              - step_size: Simulation time increment per step (seconds)
        """
        self.config = config
        self.graph = config["graph"]
        self.warehouse_location = config["warehouse_location"]
        self.delivery_points = config["delivery_points"]
        self.drivers = config["drivers"]
        self.disruption_generator = config["disruption_generator"]
        self.simulation_steps = config.get("simulation_steps", 100)
        self.step_size = config.get("step_size", 300)

        # Simplified action space:
        # 0: No action
        # 1-10: Reroute driver 1-10
        # 11-20: Reassign from driver 1-10 to nearest available driver
        # 21-30: Wait driver 1-10
        # 31-40: Skip delivery for driver 1-10
        max_drivers = 10
        max_disruptions = 5
        strategies_per_driver = 5
        self.action_space = spaces.Discrete(1 + max_drivers * 4)
        self.action_space = spaces.Discrete(1 + max_drivers * strategies_per_driver)

        # Observation space based on DeliverySystemState encoding
        # This should match the size of the vector returned by encode_for_rl()
        # Global features + driver features + disruption features
        global_features = 3
        driver_features = 8 * max_drivers  # 8 features per driver
        disruption_features = 10 * max_disruptions

        state_size = global_features + driver_features + disruption_features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_size,), dtype=np.float32
        )

        self.simulation_controller = None

        self.current_state = None
        self.current_step = 0
        self.latest_disruption = None
        self.active_disruptions = []
        self.episode_return = 0.0
        self.last_action_time = 0

        self.stats = {
            "disruptions_handled": 0,
            "actions_taken": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "time_saved": 0.0,
            "original_completion_time": 0.0,
            "final_completion_time": 0.0
        }

    def check_solution_feasibility(self):
        """Check if the current scenario is feasibly solvable"""
        if not self.active_disruptions:
            return "No active disruptions"

        # Check if disruptions affect any drivers
        affected_drivers = set()
        for d in self.active_disruptions:
            affected_drivers.update(d.affected_driver_ids)

        if not affected_drivers:
            return "No drivers affected by disruptions"

        # Check if there are valid actions available
        from models.rl.rule_based_resolver import RuleBasedResolver
        resolver = RuleBasedResolver(self.graph, self.warehouse_location)
        for d in self.active_disruptions:
            actions = resolver.resolve_disruptions(self.current_state, [d])
            if actions:
                return f"Solvable: {len(actions)} possible actions for disruption {d.id}"

        return "No valid actions found for any disruption"

    def reset(self):
        """
        Reset the environment to its initial state

        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.latest_disruption = None
        self.active_disruptions = []
        self.episode_return = 0.0
        self.last_action_time = 0

        self.stats = {
            "disruptions_handled": 0,
            "actions_taken": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "time_saved": 0.0,
            "original_completion_time": 0.0,
            "final_completion_time": 0.0
        }

        initial_solution = self._generate_initial_solution()

        self.simulation_controller = SimulationController(
            graph=self.graph,
            warehouse_location=self.warehouse_location,
            delivery_points=self.delivery_points,
            drivers=self.drivers,
            solution=initial_solution,
            disruption_service=None,
            resolver=None
        )

        self.simulation_controller.initialize_simulation()

        self.stats["original_completion_time"] = self.simulation_controller.original_estimated_time

        self.current_state = self._get_current_state()

        return self.current_state.encode_for_rl()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by selecting and executing an action

        Args:
            action_id: Index of the action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        time_before = self.simulation_controller.simulation_time

        action = self._decode_action(action_id)
        success = False

        if action:
            success = action.execute(self.simulation_controller)

            if success:
                self.stats["actions_taken"] += 1

        self._advance_simulation(self.step_size)

        self._check_disruptions()

        self.current_state = self._get_current_state()

        reward = self._calculate_reward(action, success, time_before)
        self.episode_return += reward

        self.current_step += 1
        done = self._is_episode_done()

        if done:
            self.stats["final_completion_time"] = self.simulation_controller.current_estimated_time
            self.stats["time_saved"] = max(0,
                                           self.stats["original_completion_time"] - self.stats["final_completion_time"])
            self.stats["successful_deliveries"] = len(self.simulation_controller.completed_deliveries)
            self.stats["failed_deliveries"] = len(self.simulation_controller.skipped_deliveries)

        info = {
            "step": self.current_step,
            "current_time": self.simulation_controller.simulation_time,
            "disruption_count": self.stats["disruptions_handled"],
            "action_count": self.stats["actions_taken"],
            "episode_return": self.episode_return,
            "action_success": success
        }

        if done:
            info["final_stats"] = self.stats

        return self.current_state.encode_for_rl(), reward, done, info

    def _generate_initial_solution(self):
        """
        Generate an initial solution for the delivery problem

        Returns:
            List of delivery assignments
        """
        from models.services.optimization.greedy import GreedyOptimizer

        optimizer = GreedyOptimizer(
            self.drivers,
            self.delivery_points,
            self.graph,
            self.warehouse_location
        )

        solution, _ = optimizer.optimize()
        return solution

    def _get_current_state(self) -> DeliverySystemState:
        """
        Get the current state of the delivery system

        Returns:
            DeliverySystemState object
        """
        return DeliverySystemState(
            drivers=self.drivers,
            deliveries=self.delivery_points,
            disruptions=self.active_disruptions,
            simulation_time=self.simulation_controller.simulation_time,
            graph=self.graph,
            warehouse_location=self.warehouse_location
        )

    def _decode_action(self, action_id: int) -> Optional[DisruptionAction]:
        """Convert action_id to a DisruptionAction with better rerouting strategy"""
        if action_id == 0:
            return None

        max_drivers = 10
        strategies_per_driver = 5

        driver_idx = (action_id - 1) // strategies_per_driver
        strategy_idx = (action_id - 1) % strategies_per_driver
        driver_id = driver_idx + 1

        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        if driver_id not in self.simulation_controller.driver_positions:
            return None

        disruption = self.latest_disruption
        if not disruption:
            if self.active_disruptions:
                disruption = self.active_disruptions[0]
            else:
                return None

        reroute_action = self._create_parameterized_reroute_action(
            driver_id,
            disruption,
            self.current_state,
            strategy_idx
        )

        return reroute_action

    def _create_parameterized_reroute_action(self, driver_id, disruption, state, strategy_idx):
        """Create a reroute action with different strategy parameters based on index"""
        from models.rl.rule_based_resolver import RuleBasedResolver

        strategies = [
            # Conservative - wide avoidance
            {
                "weight_multiplier": 5.0,
                "search_radius_factor": 2.0,
                "perpendicular_offset": 1.5
            },
            # Balanced - moderate avoidance
            {
                "weight_multiplier": 10.0,
                "search_radius_factor": 1.5,
                "perpendicular_offset": 1.0
            },
            # Default - similar to rule-based
            {
                "weight_multiplier": 15.0,
                "search_radius_factor": 1.2,
                "perpendicular_offset": 0.8
            },
            # Aggressive - tight avoidance
            {
                "weight_multiplier": 20.0,
                "search_radius_factor": 1.0,
                "perpendicular_offset": 0.5
            },
            # Very aggressive - minimal avoidance
            {
                "weight_multiplier": 30.0,
                "search_radius_factor": 0.8,
                "perpendicular_offset": 0.3
            }
        ]

        resolver = RuleBasedResolver(state.graph, state.warehouse_location)

        strategy = strategies[strategy_idx]

        original_find_path = resolver._find_path_avoiding_disruption

        def modified_find_path(graph, start_point, end_point, disruption):
            resolver.weight_multiplier = strategy["weight_multiplier"]
            resolver.search_radius_factor = strategy["search_radius_factor"]
            resolver.perpendicular_offset = strategy["perpendicular_offset"]
            return original_find_path(graph, start_point, end_point, disruption)

        resolver._find_path_avoiding_disruption = modified_find_path

        reroute_action = resolver._create_reroute_action(driver_id, disruption, state)

        resolver._find_path_avoiding_disruption = original_find_path

        return reroute_action

    def _advance_simulation(self, time_delta: float):
        """
        Advance the simulation by the given time delta

        Args:
            time_delta: Time to advance in seconds
        """
        self.simulation_controller.simulation_time += time_delta

    def _check_disruptions(self):
        """
        Check for newly active disruptions and update active_disruptions list
        """
        current_time = self.simulation_controller.simulation_time

        if (self.current_step + 1) % 10 == 0:
            disruption = self._generate_disruption(current_time)

            if disruption:
                self.active_disruptions.append(disruption)
                self.latest_disruption = disruption
                self.last_action_time = current_time
                self.stats["disruptions_handled"] += 1

        updated_active = []
        for disruption in self.active_disruptions:
            updated_active.append(disruption)

        self.active_disruptions = updated_active

    def _generate_disruption(self, current_time: float) -> Optional[Disruption]:
        """
        Generate a new disruption at the current time

        Args:
            current_time: Current simulation time

        Returns:
            Disruption object or None
        """
        if self.disruption_generator:
            return self.disruption_generator(current_time)

        from models.entities.disruption import Disruption, DisruptionType
        import random

        disruption_type = random.choice(list(DisruptionType))

        if random.random() < 0.7 and self.delivery_points:
            point_idx = random.randint(0, len(self.delivery_points) - 1)
            location = self.delivery_points[point_idx][0:2]
        else:
            if self.simulation_controller.driver_positions:
                driver_id = random.choice(list(self.simulation_controller.driver_positions.keys()))
                location = self.simulation_controller.driver_positions[driver_id]
            else:
                if self.delivery_points:
                    point_idx = random.randint(0, len(self.delivery_points) - 1)
                    location = self.delivery_points[point_idx][0:2]
                else:
                    location = self.warehouse_location

        radius = random.uniform(100, 500)
        duration = random.randint(900, 3600)
        severity = random.uniform(0.3, 1.0)

        disruption_id = len(self.active_disruptions) + 1

        affected_driver_ids = set()

        return Disruption(
            id=disruption_id,
            type=disruption_type,
            location=location,
            affected_area_radius=radius,
            duration=duration,
            severity=severity,
            affected_driver_ids=affected_driver_ids
        )

    def _calculate_reward(self, action, success, time_before) -> float:
        """Enhanced reward function focused on rerouting quality"""
        if not action:
            return -0.1

        if not success:
            return -2.0

        reward = 5.0

        if action.action_type == ActionType.REROUTE:
            if hasattr(action, 'driver_id') and action.driver_id in self.simulation_controller.driver_routes:
                route_info = self.simulation_controller.driver_routes[action.driver_id]

                old_route_time = getattr(self.simulation_controller, '_old_route_times', {}).get(action.driver_id, 0)
                new_route_time = sum(route_info.get('times', []))

                time_delta = old_route_time - new_route_time
                time_delta_reward = min(time_delta / 300, 5.0)

                reward += time_delta_reward

                if hasattr(action, 'affected_disruption_id') and action.affected_disruption_id:
                    disruption = next((d for d in self.active_disruptions
                                       if d.id == action.affected_disruption_id), None)

                    if disruption and disruption.type == DisruptionType.ROAD_CLOSURE:
                        reward += 2.0
                    elif disruption and disruption.type == DisruptionType.TRAFFIC_JAM:
                        reward += disruption.severity * 1.5

        time_after = self.simulation_controller.simulation_time
        simulation_progress = (time_after - time_before) / self.step_size

        progress_reward = simulation_progress * 0.5
        reward += progress_reward

        current_completed = len(self.simulation_controller.completed_deliveries)
        previous_completed = getattr(self, '_previous_completed_count', 0)
        new_completions = current_completed - previous_completed
        self._previous_completed_count = current_completed

        reward += new_completions * 3.0

        return reward

    def _is_episode_done(self) -> bool:
        """
        Check if the episode is done

        Returns:
            Boolean indicating if the episode is done
        """
        if self.current_step >= self.simulation_steps:
            return True

        total_deliveries = sum(len(a.delivery_indices) for a in self.simulation_controller.current_solution)
        deliveries_completed = len(self.simulation_controller.completed_deliveries)
        deliveries_skipped = len(self.simulation_controller.skipped_deliveries)

        return (deliveries_completed + deliveries_skipped) >= total_deliveries

    def render(self, mode='human'):
        """
        Render the environment

        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Time: {self._format_time(self.simulation_controller.simulation_time)}")
            print(f"Active Disruptions: {len(self.active_disruptions)}")
            print(f"Total Reward: {self.episode_return:.2f}")
            print(f"Deliveries Completed: {len(self.simulation_controller.completed_deliveries)}")
            print(f"Deliveries Skipped: {len(self.simulation_controller.skipped_deliveries)}")
            print(f"Actions Taken: {self.stats['actions_taken']}")
            print("-" * 50)

    def _format_time(self, seconds):
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
