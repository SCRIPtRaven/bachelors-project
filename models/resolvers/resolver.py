from abc import ABC, abstractmethod
from typing import List

from models.entities.disruption import Disruption
from models.resolvers.actions import DisruptionAction
from models.resolvers.state import DeliverySystemState


class DisruptionResolver(ABC):
    """
    Abstract base class for all disruption resolver implementations.
    Defines the interface that resolvers must implement.
    """

    @abstractmethod
    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """
        Determine actions to take in response to disruptions

        Args:
            state: Current system state
            active_disruptions: Currently active disruptions

        Returns:
            List of actions to take
        """
        pass

    @abstractmethod
    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost

        Args:
            state: Current system state
            disruption: The disruption being considered

        Returns:
            Boolean indicating if recalculation should proceed
        """
        pass

    def on_disruption_detected(self, disruption: Disruption, state: DeliverySystemState) -> List[DisruptionAction]:
        """
        Called when a new disruption is detected.

        Args:
            disruption: The newly detected disruption
            state: Current system state

        Returns:
            List of actions to take in response
        """
        if self.should_recalculate(state, disruption):
            return self.resolve_disruptions(state, [disruption])
        return []

    def on_simulation_update(self, state: DeliverySystemState) -> List[DisruptionAction]:
        """
        Called periodically during simulation to allow for proactive interventions.

        Args:
            state: Current system state

        Returns:
            List of actions to take
        """
        return []
