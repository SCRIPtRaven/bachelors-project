import math
from typing import List

import numpy as np

from models.entities.disruption import Disruption
from models.rl.actions import DisruptionAction
from models.rl.environment import DeliveryEnvironment
from models.rl.resolver import DisruptionResolver
from models.rl.rule_based_resolver import RuleBasedResolver
from models.rl.state import DeliverySystemState


class RLResolver(DisruptionResolver):
    """
    RL-based implementation of DisruptionResolver that uses a trained model
    to determine the best actions to take.
    """

    def __init__(self, model_path, graph, warehouse_location):
        """
        Initialize the RL resolver with a trained model

        Args:
            model_path: Path to the Keras model file
            graph: NetworkX graph of the road network
            warehouse_location: (lat, lon) of warehouse
        """
        self.graph = graph
        self.warehouse_location = warehouse_location
        self.action_count = 0
        self.successful_actions = 0

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded RL model from {model_path}")

            self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)

            env_config = {
                "graph": graph,
                "warehouse_location": warehouse_location,
                "delivery_points": [],
                "drivers": [],
                "disruption_generator": None
            }
            self.dummy_env = DeliveryEnvironment(env_config)

        except Exception as e:
            print(f"Error loading RL model: {e}")
            print("Falling back to rule-based resolver completely")
            self.model = None
            self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost

        Uses the model to predict if taking an action is likely to be beneficial
        """
        if self.model:
            try:
                state_vector = state.encode_for_rl()
                action_values = self.model.predict(state_vector.reshape(1, -1), verbose=0)[0]

                no_action_value = action_values[0]
                best_action_value = np.max(action_values)

                threshold = 1.0
                return best_action_value - no_action_value > threshold
            except Exception as e:
                print(f"Error in RL decision: {e}, falling back to rule-based")
                return self.rule_based_resolver.should_recalculate(state, disruption)
        else:
            return self.rule_based_resolver.should_recalculate(state, disruption)

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: list) -> list:
        """
        Determine actions to take in response to disruptions using RL model
        """
        if not active_disruptions:
            return []

        if self.model:
            try:
                state_vector = state.encode_for_rl()
                action_values = self.model.predict(state_vector.reshape(1, -1), verbose=0)[0]

                best_actions = np.argsort(action_values)[-3:][::-1]  # Top 3 actions
                viable_actions = []

                self.dummy_env.current_state = state
                self.dummy_env.active_disruptions = active_disruptions
                self.dummy_env.latest_disruption = active_disruptions[0]

                for action_id in best_actions:
                    if action_id == 0:
                        continue

                    action = self.dummy_env._decode_action(action_id)
                    if action:
                        affected_driver_ids = self._get_affected_driver_ids(state, active_disruptions)

                        if not hasattr(action, 'driver_id') or action.driver_id in affected_driver_ids:
                            viable_actions.append(action)

                if viable_actions:
                    self.action_count += len(viable_actions)
                    return viable_actions

                print("No viable RL actions found, falling back to rule-based")
                return self.rule_based_resolver.resolve_disruptions(state, active_disruptions)

            except Exception as e:
                print(f"Error in RL action selection: {e}, falling back to rule-based")
                return self.rule_based_resolver.resolve_disruptions(state, active_disruptions)
        else:
            return self.rule_based_resolver.resolve_disruptions(state, active_disruptions)

    def _get_affected_driver_ids(self, state, disruptions):
        """Get IDs of drivers affected by the disruptions"""
        affected_ids = set()

        for disruption in disruptions:
            for driver_id, position in state.driver_positions.items():
                distance = self._calculate_distance(position, disruption.location)
                if distance <= disruption.affected_area_radius * 2:
                    affected_ids.add(driver_id)

        return affected_ids

    def _calculate_distance(self, point1, point2):
        """Calculate Haversine distance between points"""
        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000

        return c * r

    def on_disruption_detected(self, disruption: Disruption, state: DeliverySystemState) -> List[DisruptionAction]:
        """Specialized handler for newly detected disruptions"""
        if self.should_recalculate(state, disruption):
            return self.resolve_disruptions(state, [disruption])
        return []
