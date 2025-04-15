from enum import Enum, auto
from typing import List, Dict, Any, Optional


class ActionType(Enum):
    """Types of actions the disruption resolver can take"""
    RECIPIENT_UNAVAILABLE = auto()
    REROUTE = auto()
    REASSIGN_DELIVERIES = auto()
    WAIT = auto()
    SKIP_DELIVERY = auto()
    PRIORITIZE_DELIVERY = auto()
    NO_ACTION = auto()


class DisruptionAction:
    """Base class for all disruption resolution actions"""

    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    def execute(self, controller):
        """Execute this action in the simulation"""
        raise NotImplementedError("Subclasses must implement execute()")

    def estimate_cost(self, state):
        """Estimate the computational cost of executing this action"""
        raise NotImplementedError("Subclasses must implement estimate_cost()")

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to a dictionary format for storage/transmission"""
        return {
            'action_type': self.action_type.name
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DisruptionAction':
        """Create an action instance from dictionary data"""
        action_type = ActionType[data['action_type']]

        if action_type == ActionType.REROUTE:
            return RerouteAction.from_dict(data)
        elif action_type == ActionType.REASSIGN_DELIVERIES:
            return ReassignDeliveriesAction.from_dict(data)
        elif action_type == ActionType.WAIT:
            return WaitAction.from_dict(data)
        elif action_type == ActionType.SKIP_DELIVERY:
            return SkipDeliveryAction.from_dict(data)
        elif action_type == ActionType.RECIPIENT_UNAVAILABLE:
            return RecipientUnavailableAction.from_dict(data)
        else:
            return NoAction()


class RerouteAction(DisruptionAction):
    """Action to change a driver's route to avoid a disruption"""

    def __init__(self, driver_id: int, new_route: List[tuple], affected_disruption_id: Optional[int] = None,
                 rerouted_segment_start: Optional[int] = None, rerouted_segment_end: Optional[int] = None,
                 next_delivery_index: Optional[int] = None, delivery_indices=None):
        super().__init__(ActionType.REROUTE)
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
            self.driver_id,
            self.new_route,
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
        return RerouteAction(
            driver_id=data['driver_id'],
            new_route=data['new_route'],
            affected_disruption_id=data.get('affected_disruption_id'),
            rerouted_segment_start=data.get('rerouted_segment_start'),
            rerouted_segment_end=data.get('rerouted_segment_end'),
            next_delivery_index=data.get('next_delivery_index'),
            delivery_indices=data.get('delivery_indices', [])
        )


class ReassignDeliveriesAction(DisruptionAction):
    """Action to transfer deliveries between drivers"""

    def __init__(self, from_driver_id: int, to_driver_id: int, delivery_indices: List[int]):
        super().__init__(ActionType.REASSIGN_DELIVERIES)
        self.from_driver_id = from_driver_id
        self.to_driver_id = to_driver_id
        self.delivery_indices = delivery_indices

    def execute(self, controller):
        """Transfer deliveries from one driver to another"""
        return controller.reassign_deliveries(
            self.from_driver_id,
            self.to_driver_id,
            self.delivery_indices
        )

    def estimate_cost(self, state):
        """
        Estimate computational cost based on number of deliveries
        and need to recalculate routes for both drivers
        """
        num_deliveries = len(self.delivery_indices)
        return 1.0 + (0.1 * num_deliveries)  # Base cost + per-delivery cost

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'from_driver_id': self.from_driver_id,
            'to_driver_id': self.to_driver_id,
            'delivery_indices': self.delivery_indices
        })
        return data

    @staticmethod
    def from_dict(data):
        return ReassignDeliveriesAction(
            from_driver_id=data['from_driver_id'],
            to_driver_id=data['to_driver_id'],
            delivery_indices=data['delivery_indices']
        )


class WaitAction(DisruptionAction):
    """Action to instruct a driver to wait for a disruption to clear"""

    def __init__(self, driver_id: int, wait_time: int, disruption_id: int):
        super().__init__(ActionType.WAIT)
        self.driver_id = driver_id
        self.wait_time = wait_time  # Time to wait in seconds
        self.disruption_id = disruption_id

    def execute(self, controller):
        """Add wait time to the driver's current activity"""
        return controller.add_driver_wait_time(self.driver_id, self.wait_time)

    def estimate_cost(self, state):
        """Wait actions have minimal computational cost"""
        return 0.1

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'driver_id': self.driver_id,
            'wait_time': self.wait_time,
            'disruption_id': self.disruption_id
        })
        return data

    @staticmethod
    def from_dict(data):
        return WaitAction(
            driver_id=data['driver_id'],
            wait_time=data['wait_time'],
            disruption_id=data['disruption_id']
        )


class SkipDeliveryAction(DisruptionAction):
    """Action to skip a delivery due to recipient unavailability"""

    def __init__(self, driver_id: int, delivery_index: int):
        super().__init__(ActionType.SKIP_DELIVERY)
        self.driver_id = driver_id
        self.delivery_index = delivery_index

    def execute(self, controller):
        """Mark delivery as skipped and update route"""
        return controller.skip_delivery(self.driver_id, self.delivery_index)

    def estimate_cost(self, state):
        """Skip actions have low computational cost"""
        return 0.2

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'driver_id': self.driver_id,
            'delivery_index': self.delivery_index
        })
        return data

    @staticmethod
    def from_dict(data):
        return SkipDeliveryAction(
            driver_id=data['driver_id'],
            delivery_index=data['delivery_index']
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
            # Register the end handler for when recipient becomes available
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
