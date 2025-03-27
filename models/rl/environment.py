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
        self.step_size = config.get("step_size", 300)  # 5 minutes

        # Set up action and observation spaces
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
        disruption_features = 7 * 5  # 5 max disruptions with 7 features each

        state_size = global_features + driver_features + disruption_features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_size,), dtype=np.float32
        )

        # Initialize simulation controller
        self.simulation_controller = None

        # Current state tracking
        self.current_state = None
        self.current_step = 0
        self.latest_disruption = None
        self.active_disruptions = []
        self.episode_return = 0.0
        self.last_action_time = 0

        # Statistics
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

        # Reset statistics
        self.stats = {
            "disruptions_handled": 0,
            "actions_taken": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "time_saved": 0.0,
            "original_completion_time": 0.0,
            "final_completion_time": 0.0
        }

        # Initialize simulation controller with fresh solution
        initial_solution = self._generate_initial_solution()

        self.simulation_controller = SimulationController(
            graph=self.graph,
            warehouse_location=self.warehouse_location,
            delivery_points=self.delivery_points,
            drivers=self.drivers,
            solution=initial_solution,
            disruption_service=None,  # We'll handle disruptions directly
            resolver=None  # No resolver, we are the resolver
        )

        self.simulation_controller.initialize_simulation()

        # Record original estimated completion time
        self.stats["original_completion_time"] = self.simulation_controller.original_estimated_time

        # Get initial state
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
        # Track current time before action
        time_before = self.simulation_controller.simulation_time

        # Execute action in environment
        action = self._decode_action(action_id)
        success = False

        if action:
            success = action.execute(self.simulation_controller)

            if success:
                self.stats["actions_taken"] += 1

        # Advance simulation time
        self._advance_simulation(self.step_size)

        # Check for newly active disruptions
        self._check_disruptions()

        # Get new state
        self.current_state = self._get_current_state()

        # Calculate reward
        reward = self._calculate_reward(action, success, time_before)
        self.episode_return += reward

        # Check if episode is done
        self.current_step += 1
        done = self._is_episode_done()

        if done:
            # Record final statistics
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
        # This would be the same method used in the application
        # For simplicity, we'll assume it's available as a helper function
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

        # 0 = No action
        if action_id == 0:
            return None

        # 1-10 = Reroute driver 1-10
        if 1 <= action_id <= max_drivers:
            driver_id = action_id
            return self._create_reroute_action(driver_id)

        # 11-20 = Reassign from driver 1-10 to nearest available driver
        if max_drivers + 1 <= action_id <= max_drivers * 2:
            driver_id = action_id - max_drivers
            return self._create_reassign_action(driver_id)

        # 21-30 = Wait driver 1-10
        if max_drivers * 2 + 1 <= action_id <= max_drivers * 3:
            driver_id = action_id - (max_drivers * 2)

            # Only create wait action if there's a recent disruption
            if self.latest_disruption and (self.simulation_controller.simulation_time - self.last_action_time) <= 600:
                wait_time = min(1800, self.latest_disruption.duration)  # Wait up to 30 minutes
                return self._create_wait_action(driver_id, wait_time)
            return None

        # 31-40 = Skip delivery for driver 1-10
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
        # Find the driver's assignment
        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        # Get the driver's current position
        if driver_id not in self.simulation_controller.driver_positions:
            return None

        # Create a new route that avoids active disruptions
        from models.rl.rule_based_resolver import RuleBasedResolver

        resolver = RuleBasedResolver(self.graph, self.warehouse_location)
        disruption = self.latest_disruption

        if not disruption:
            # Use the first active disruption if no latest disruption
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
        # Find the driver's assignment
        from_assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == from_driver_id:
                from_assignment = a
                break

        if not from_assignment or not from_assignment.delivery_indices:
            return None

        # Create a reassignment using a rule-based approach
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
        # Check if the driver exists and has assignments
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
        # Check if the driver exists and has assignments
        assignment = None
        for a in self.simulation_controller.current_solution:
            if a.driver_id == driver_id:
                assignment = a
                break

        if not assignment or not assignment.delivery_indices:
            return None

        # Get the first delivery index
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
        # In a real environment, this would use the disruption service
        # For training, we'll generate disruptions based on time
        current_time = self.simulation_controller.simulation_time

        # Generate a new disruption with some probability
        if (self.current_step + 1) % 10 == 0:  # Every 10 steps
            disruption = self._generate_disruption(current_time)

            if disruption:
                self.active_disruptions.append(disruption)
                self.latest_disruption = disruption
                self.last_action_time = current_time
                self.stats["disruptions_handled"] += 1

        # Update active disruptions list
        updated_active = []
        for disruption in self.active_disruptions:
            if disruption.start_time <= current_time <= (disruption.start_time + disruption.duration):
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

        # Default implementation if no generator provided
        from models.entities.disruption import Disruption, DisruptionType
        import random

        # Random disruption type
        disruption_type = random.choice(list(DisruptionType))

        # Random location (near a delivery point or driver)
        location = None
        if random.random() < 0.7 and self.delivery_points:
            # 70% chance to be near a delivery point
            point_idx = random.randint(0, len(self.delivery_points) - 1)
            location = self.delivery_points[point_idx][0:2]
        else:
            # 30% chance to be near a driver
            if self.simulation_controller.driver_positions:
                driver_id = random.choice(list(self.simulation_controller.driver_positions.keys()))
                location = self.simulation_controller.driver_positions[driver_id]
            else:
                # Fall back to a random delivery point
                if self.delivery_points:
                    point_idx = random.randint(0, len(self.delivery_points) - 1)
                    location = self.delivery_points[point_idx][0:2]
                else:
                    # Fall back to warehouse location
                    location = self.warehouse_location

        # Random parameters
        radius = random.uniform(100, 500)  # 100m to 500m
        duration = random.randint(900, 3600)  # 15min to 60min
        severity = random.uniform(0.3, 1.0)

        disruption_id = len(self.active_disruptions) + 1

        # For vehicle breakdowns, assign to a random driver
        affected_driver_ids = set()
        if disruption_type == DisruptionType.VEHICLE_BREAKDOWN:
            driver_ids = [a.driver_id for a in self.simulation_controller.current_solution]
            if driver_ids:
                affected_driver_ids.add(random.choice(driver_ids))

        return Disruption(
            id=disruption_id,
            type=disruption_type,
            location=location,
            affected_area_radius=radius,
            start_time=current_time,
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

        # Penalty for unsuccessful actions
        if action and not success:
            reward -= 2.0

        # Reward for successful actions based on type
        if action and success:
            # Base reward for taking a successful action
            reward += 1.0

            # Additional reward based on action type
            if action.action_type == ActionType.REROUTE:
                reward += 2.0
            elif action.action_type == ActionType.REASSIGN_DELIVERIES:
                reward += 3.0
            elif action.action_type == ActionType.WAIT:
                reward += 0.5
            elif action.action_type == ActionType.SKIP_DELIVERY:
                reward += 0.0  # Neutral for skip (cost balanced by potential benefit)

        # Reward for time efficiency
        time_now = self.simulation_controller.simulation_time
        time_estimate_before = self.simulation_controller.current_estimated_time
        time_estimate_after = self.simulation_controller.current_estimated_time

        time_saved = time_estimate_before - time_estimate_after
        normalized_time_saved = time_saved / 3600  # Normalize by an hour

        # Only add time reward if an action was taken
        if action and success:
            reward += normalized_time_saved * 5.0

        # Reward for completing deliveries during this step
        deliveries_completed = len(self.simulation_controller.completed_deliveries)
        deliveries_skipped = len(self.simulation_controller.skipped_deliveries)

        total_deliveries = sum(len(a.delivery_indices) for a in self.simulation_controller.current_solution)
        progress = (deliveries_completed + deliveries_skipped) / max(1, total_deliveries)

        # Weighted progress reward
        completion_weight = 0.8  # Favor completed over skipped
        skip_weight = 0.2

        weighted_progress = (completion_weight * deliveries_completed + skip_weight * deliveries_skipped) / max(1,
                                                                                                                total_deliveries)

        # Continuously reward progress
        progress_reward = weighted_progress * 10.0

        # Only add a bit of progress reward each step
        step_progress = progress_reward / self.simulation_steps
        reward += step_progress

        return reward

    def _is_episode_done(self) -> bool:
        """
        Check if the episode is done

        Returns:
            Boolean indicating if the episode is done
        """
        # Episode ends when we've reached the maximum steps
        if self.current_step >= self.simulation_steps:
            return True

        # Or when all deliveries are completed or skipped
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
            # Print current state to console
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
