from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple


class ActionType(Enum):
    REROUTE_BASIC = auto()
    NO_ACTION = auto()
    REROUTE_TIGHT_AVOIDANCE = auto()
    REROUTE_WIDE_AVOIDANCE = auto()

    @property
    def display_name(self) -> str:
        return self.name.lower()


class DisruptionAction:
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    def execute(self, controller):
        raise NotImplementedError("Subclasses must implement execute()")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.name
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DisruptionAction':
        action_type = ActionType[data['action_type']]

        if action_type == ActionType.REROUTE_BASIC:
            return RerouteBasicAction.from_dict(data)
        elif action_type == ActionType.NO_ACTION:
            return NoAction.from_dict(data)
        elif action_type == ActionType.REROUTE_TIGHT_AVOIDANCE:
            return RerouteTightAvoidanceAction.from_dict(data)
        elif action_type == ActionType.REROUTE_WIDE_AVOIDANCE:
            return RerouteWideAvoidanceAction.from_dict(data)
        raise ValueError(f"Unknown action type: {action_type}")


class RerouteBasicAction(DisruptionAction):
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
        print(
            f"REROUTE ACTION EXECUTE: driver_id={self.driver_id}, route_length={len(self.new_route)}, "
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


class NoAction(DisruptionAction):
    def __init__(self, driver_id: int, affected_disruption_id: Optional[int] = None):
        super().__init__(ActionType.NO_ACTION)
        self.driver_id = driver_id
        self.affected_disruption_id = affected_disruption_id

    def execute(self, controller):
        print(f"NO ACTION EXECUTE: driver_id={self.driver_id}")

        if hasattr(controller, 'pending_actions'):
            if self.driver_id not in controller.pending_actions:
                controller.pending_actions[self.driver_id] = []
            controller.pending_actions[self.driver_id].append(self)

        return True

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'driver_id': self.driver_id,
            'affected_disruption_id': self.affected_disruption_id
        })
        return data

    @staticmethod
    def from_dict(data):
        return NoAction(
            driver_id=data['driver_id'],
            affected_disruption_id=data.get('affected_disruption_id')
        )


class RerouteTightAvoidanceAction(RerouteBasicAction):
    def __init__(self, driver_id: int, new_route: List[Tuple[float, float]],
                 affected_disruption_id: Optional[int] = None,
                 rerouted_segment_start: Optional[int] = None,
                 rerouted_segment_end: Optional[int] = None,
                 next_delivery_index: Optional[int] = None,
                 delivery_indices=None):
        super().__init__(
            driver_id=driver_id,
            new_route=new_route,
            affected_disruption_id=affected_disruption_id,
            rerouted_segment_start=rerouted_segment_start,
            rerouted_segment_end=rerouted_segment_end,
            next_delivery_index=next_delivery_index,
            delivery_indices=delivery_indices
        )
        self.action_type = ActionType.REROUTE_TIGHT_AVOIDANCE

    def execute(self, controller):
        print(f"REROUTE TIGHT AVOIDANCE ACTION EXECUTE: driver_id={self.driver_id}, "
              f"route_length={len(self.new_route)}, "
              f"delivery_indices={self.delivery_indices}, "
              f"segment={self.rerouted_segment_start}-{self.rerouted_segment_end}")

        return super().execute(controller)

    @staticmethod
    def from_dict(data):
        return RerouteTightAvoidanceAction(
            driver_id=data['driver_id'],
            new_route=data['new_route'],
            affected_disruption_id=data.get('affected_disruption_id'),
            rerouted_segment_start=data.get('rerouted_segment_start'),
            rerouted_segment_end=data.get('rerouted_segment_end'),
            next_delivery_index=data.get('next_delivery_index'),
            delivery_indices=data.get('delivery_indices', [])
        )


class RerouteWideAvoidanceAction(RerouteBasicAction):
    def __init__(self, driver_id: int, new_route: List[Tuple[float, float]],
                 affected_disruption_id: Optional[int] = None,
                 rerouted_segment_start: Optional[int] = None,
                 rerouted_segment_end: Optional[int] = None,
                 next_delivery_index: Optional[int] = None,
                 delivery_indices=None):
        super().__init__(
            driver_id=driver_id,
            new_route=new_route,
            affected_disruption_id=affected_disruption_id,
            rerouted_segment_start=rerouted_segment_start,
            rerouted_segment_end=rerouted_segment_end,
            next_delivery_index=next_delivery_index,
            delivery_indices=delivery_indices
        )
        self.action_type = ActionType.REROUTE_WIDE_AVOIDANCE

    def execute(self, controller):
        print(f"REROUTE WIDE AVOIDANCE ACTION EXECUTE: driver_id={self.driver_id}, "
              f"route_length={len(self.new_route)}, "
              f"delivery_indices={self.delivery_indices}, "
              f"segment={self.rerouted_segment_start}-{self.rerouted_segment_end}")

        return super().execute(controller)

    @staticmethod
    def from_dict(data):
        return RerouteWideAvoidanceAction(
            driver_id=data['driver_id'],
            new_route=data['new_route'],
            affected_disruption_id=data.get('affected_disruption_id'),
            rerouted_segment_start=data.get('rerouted_segment_start'),
            rerouted_segment_end=data.get('rerouted_segment_end'),
            next_delivery_index=data.get('next_delivery_index'),
            delivery_indices=data.get('delivery_indices', [])
        )
