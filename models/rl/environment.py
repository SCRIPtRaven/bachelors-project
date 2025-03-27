from typing import Dict, Any, Tuple, Optional

import gym
import numpy as np
from gym import spaces

from models.entities.disruption import Disruption
from models.rl.actions import ActionType, DisruptionAction, RerouteAction, ReassignDeliveriesAction
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
        self.action_space = spaces.Discrete(1 + max_drivers * 4)

        # Observation space based on DeliverySystemState encoding
        # This should match the size of the vector returned by encode_for_rl()
        # Global features + driver features + disruption features
        global_features = 3
        driver_features = 4 * max_drivers
        disruption_features = 7 * 5

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
        """
        Convert action_id to a DisruptionAction object

        Args:
            action_id: Index of the action in the action space

        Returns:
            DisruptionAction object or None
        """
        max_drivers = 10

        if action_id == 0:
            return None

        if 1 <= action_id <= max_drivers:
            driver_id = action_id
            return self._create_reroute_action(driver_id)

        if max_drivers + 1 <= action_id <= max_drivers * 2:
            driver_id = action_id - max_drivers
            return self._create_reassign_action(driver_id)

        if max_drivers * 2 + 1 <= action_id <= max_drivers * 3:
            driver_id = action_id - (max_drivers * 2)

            if self.latest_disruption and (self.simulation_controller.simulation_time - self.last_action_time) <= 600:
                wait_time = min(1800, self.latest_disruption.duration)
                return self._create_wait_action(driver_id, wait_time)
            return None

        if max_drivers * 3 + 1 <= action_id <= max_drivers * 4:
            driver_id = action_id - (max_drivers * 3)
            return self._create_skip_action(driver_id)

        return None

    def _create_reroute_action(self, driver_id: int) -> Optional[RerouteAction]:
        """
        Create a reroute action for a driver

        Args:
            driver_id: ID of the driver to reroute

        Returns:
            RerouteAction object or None
        """
        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        if driver_id not in self.simulation_controller.driver_positions:
            return None

        from models.rl.rule_based_resolver import RuleBasedResolver

        resolver = RuleBasedResolver(self.graph, self.warehouse_location)
        disruption = self.latest_disruption

        if not disruption:
            if self.active_disruptions:
                disruption = self.active_disruptions[0]
            else:
                return None

        reroute_action = resolver._create_reroute_action(
            driver_id,
            disruption,
            self.current_state
        )

        return reroute_action

    def _create_reassign_action(self, from_driver_id: int) -> Optional[ReassignDeliveriesAction]:
        """
        Create a reassign action for deliveries from a driver

        Args:
            from_driver_id: ID of the driver to reassign from

        Returns:
            ReassignDeliveriesAction object or None
        """
        from_assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == from_driver_id:
                from_assignment = a
                break

        if not from_assignment or not from_assignment.delivery_indices:
            return None

        from models.rl.rule_based_resolver import RuleBasedResolver

        resolver = RuleBasedResolver(self.graph, self.warehouse_location)
        reassignment_actions = resolver._create_reassignment_actions(
            from_driver_id,
            self.current_state
        )

        if reassignment_actions:
            return reassignment_actions[0]
        return None

    def _create_wait_action(self, driver_id: int, wait_time: float) -> Optional[DisruptionAction]:
        """
        Create a wait action for a driver

        Args:
            driver_id: ID of the driver
            wait_time: Time to wait in seconds

        Returns:
            WaitAction object or None
        """
        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        from models.rl.actions import WaitAction

        return WaitAction(
            driver_id=driver_id,
            wait_time=wait_time,
            disruption_id=self.latest_disruption.id if self.latest_disruption else 0
        )

    def _create_skip_action(self, driver_id: int) -> Optional[DisruptionAction]:
        """
        Create a skip action for a driver's current delivery

        Args:
            driver_id: ID of the driver

        Returns:
            SkipDeliveryAction object or None
        """
        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        delivery_idx = assignment.delivery_indices[0]

        from models.rl.actions import SkipDeliveryAction

        return SkipDeliveryAction(
            driver_id=driver_id,
            delivery_index=delivery_idx
        )

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
        """
        Calculate the reward for the current step

        Args:
            action: The action that was taken
            success: Whether the action was successful
            time_before: Simulation time before the action

        Returns:
            Reward value
        """
        reward = 0.0

        if action and not success:
            reward -= 2.0

        if action and success:
            reward += 1.0

            if action.action_type == ActionType.REROUTE:
                reward += 2.0
            elif action.action_type == ActionType.REASSIGN_DELIVERIES:
                reward += 3.0
            elif action.action_type == ActionType.WAIT:
                reward += 0.5
            elif action.action_type == ActionType.SKIP_DELIVERY:
                reward += 0.0

        time_now = self.simulation_controller.simulation_time
        time_estimate_before = self.simulation_controller.current_estimated_time
        time_estimate_after = self.simulation_controller.current_estimated_time

        time_saved = time_estimate_before - time_estimate_after
        normalized_time_saved = time_saved / 3600

        if action and success:
            reward += normalized_time_saved * 5.0

        deliveries_completed = len(self.simulation_controller.completed_deliveries)
        deliveries_skipped = len(self.simulation_controller.skipped_deliveries)

        total_deliveries = sum(len(a.delivery_indices) for a in self.simulation_controller.current_solution)
        progress = (deliveries_completed + deliveries_skipped) / max(1, total_deliveries)

        completion_weight = 0.8
        skip_weight = 0.2

        weighted_progress = (completion_weight * deliveries_completed + skip_weight * deliveries_skipped) / max(1,
                                                                                                                total_deliveries)

        progress_reward = weighted_progress * 10.0

        step_progress = progress_reward / self.simulation_steps
        reward += step_progress

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
