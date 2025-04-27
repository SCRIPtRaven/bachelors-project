from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple


class ActionType(Enum):
    """Types of actions the disruption resolver can take"""
    RECIPIENT_UNAVAILABLE = auto()
    REROUTE_BASIC = auto()


class DisruptionAction:
    """Base class for all disruption resolution actions"""
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    def execute(self, controller):
        """Execute this action in the simulation"""
        raise NotImplementedError("Subclasses must implement execute()")

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to a dictionary format for storage/transmission"""
        return {
            'action_type': self.action_type.name
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DisruptionAction':
        """Create an action instance from dictionary data"""
        action_type = ActionType[data['action_type']]

        if action_type == ActionType.REROUTE_BASIC:
            return RerouteBasicAction.from_dict(data)
        elif action_type == ActionType.RECIPIENT_UNAVAILABLE:
            return RecipientUnavailableAction.from_dict(data)
        raise ValueError(f"Unknown action type: {action_type}")


class RerouteBasicAction(DisruptionAction):
    """Action to change a driver's route to avoid a disruption"""

    def __init__(self, driver_id: int, new_route: List[Tuple[float, float]],
                 affected_disruption_id: Optional[int] = None,
                 rerouted_segment_start: Optional[int] = None,
                 rerouted_segment_end: Optional[int] = None,
                 next_delivery_index: Optional[int] = None,
                 delivery_indices=None):
        super().__init__(ActionType.REROUTE_BASIC)
        self.driver_id = driver_id
        self.new_route = new_route
        self.affected_disruption_id = affected_disruption_id
        self.rerouted_segment_start = rerouted_segment_start
        self.rerouted_segment_end = rerouted_segment_end
        self.next_delivery_index = next_delivery_index
        self.delivery_indices = delivery_indices or []

    def execute(self, controller):
        """Update the driver's route with metadata about the rerouted segment"""
        print(f"REROUTE ACTION EXECUTE: driver_id={self.driver_id}, route_length={len(self.new_route)}, "
              f"delivery_indices={self.delivery_indices}, "
              f"segment={self.rerouted_segment_start}-{self.rerouted_segment_end}")

        result = controller.update_driver_route(
            driver_id=self.driver_id,
            new_route=self.new_route,
            rerouted_segment_start=self.rerouted_segment_start,
            rerouted_segment_end=self.rerouted_segment_end,
            next_delivery_index=self.next_delivery_index
        )

        if result and hasattr(controller, 'pending_actions'):
            if self.driver_id not in controller.pending_actions:
                controller.pending_actions[self.driver_id] = []
            controller.pending_actions[self.driver_id].append(self)

        return result

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'driver_id': self.driver_id,
            'new_route': self.new_route,
            'affected_disruption_id': self.affected_disruption_id,
            'rerouted_segment_start': self.rerouted_segment_start,
            'rerouted_segment_end': self.rerouted_segment_end,
            'next_delivery_index': self.next_delivery_index,
            'delivery_indices': self.delivery_indices
        })
        return data

    @staticmethod
    def from_dict(data):
        return RerouteBasicAction(
            driver_id=data['driver_id'],
            new_route=data['new_route'],
            affected_disruption_id=data.get('affected_disruption_id'),
            rerouted_segment_start=data.get('rerouted_segment_start'),
            rerouted_segment_end=data.get('rerouted_segment_end'),
            next_delivery_index=data.get('next_delivery_index'),
            delivery_indices=data.get('delivery_indices', [])
        )


class RecipientUnavailableAction(DisruptionAction):
    """Action to handle unavailable recipient by queueing for later"""

    def __init__(self, driver_id: int, delivery_index: int, disruption_id: int, duration: int):
        super().__init__(ActionType.RECIPIENT_UNAVAILABLE)
        self.driver_id = driver_id
        self.delivery_index = delivery_index
        self.disruption_id = disruption_id
        self.duration = duration

    def execute(self, controller):
        """Add to pending deliveries and recalculate route"""
        success = controller.handle_recipient_unavailable(
            self.driver_id,
            self.delivery_index,
            self.disruption_id,
            self.duration
        )

        if success:
            end_time = controller.simulation_time + self.duration
            controller.action_log.emit(
                f"Recipient unavailable for delivery {self.delivery_index}. "
                f"Will check again at {controller._format_time(end_time)}"
            )

        return success

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'driver_id': self.driver_id,
            'delivery_index': self.delivery_index,
            'disruption_id': self.disruption_id,
            'duration': self.duration
        })
        return data

    @staticmethod
    def from_dict(data):
        return RecipientUnavailableAction(
            driver_id=data['driver_id'],
            delivery_index=data['delivery_index'],
            disruption_id=data['disruption_id'],
            duration=data['duration']
        )
