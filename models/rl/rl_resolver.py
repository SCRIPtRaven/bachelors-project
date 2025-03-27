import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from models.rl.resolver import DisruptionResolver
from models.rl.state import DeliverySystemState
from models.entities.disruption import Disruption
from models.rl.actions import ActionType, DisruptionAction
from models.rl.environment import DeliveryEnvironment


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
        self.model = load_model(model_path)
        self.graph = graph
        self.warehouse_location = warehouse_location

        env_config = {
            "graph": graph,
            "warehouse_location": warehouse_location,
            "delivery_points": [],
            "drivers": [],
            "disruption_generator": None
        }
        self.dummy_env = DeliveryEnvironment(env_config)

        self.action_count = 0
        self.successful_actions = 0

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost

        Uses the model to predict if taking an action is likely to be beneficial
        """
        state_vector = state.encode_for_rl()

        action_values = self.model.predict(state_vector.reshape(1, -1))[0]

        no_action_value = action_values[0]
        best_action_value = np.max(action_values)

        threshold = 2.0

        return best_action_value - no_action_value > threshold

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: list) -> list:
        """
        Determine actions to take in response to disruptions

        Uses the trained model to select the best action
        """
        state_vector = state.encode_for_rl()

        action_values = self.model.predict(state_vector.reshape(1, -1))[0]

        action_id = np.argmax(action_values)

        if action_id == 0:
            return []

        self.dummy_env.current_state = state
        self.dummy_env.active_disruptions = active_disruptions
        self.dummy_env.latest_disruption = active_disruptions[0] if active_disruptions else None

        action = self.dummy_env._decode_action(action_id)

        if action:
            self.action_count += 1
            return [action]
        else:
            return []

    def on_disruption_detected(self, disruption: Disruption, state: DeliverySystemState) -> list:
        """
        Called when a new disruption is detected

        Args:
            disruption: The newly detected disruption
            state: Current system state

        Returns:
            List of actions to take in response
        """
        modified_state = DeliverySystemState(
            drivers=state.drivers,
            deliveries=state.deliveries,
            disruptions=[disruption],
            simulation_time=state.simulation_time,
            graph=state.graph,
            warehouse_location=state.warehouse_location
        )

        if self.should_recalculate(modified_state, disruption):
            return self.resolve_disruptions(modified_state, [disruption])
        return []