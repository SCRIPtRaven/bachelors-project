from abc import ABC, abstractmethod
from typing import List

from models.entities.disruption import Disruption
from models.resolvers.actions import DisruptionAction
from models.resolvers.state import DeliverySystemState


class DisruptionResolver(ABC):
    @abstractmethod
    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        pass

    @abstractmethod
    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        pass

    def on_disruption_detected(self, disruption: Disruption, state: DeliverySystemState) -> List[DisruptionAction]:
        if self.should_recalculate(state, disruption):
            return self.resolve_disruptions(state, [disruption])
        return []

    def on_simulation_update(self, state: DeliverySystemState) -> List[DisruptionAction]:
        return []
