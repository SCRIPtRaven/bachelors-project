from enum import Enum
from typing import Tuple, Dict, Any, Set

from pydantic import BaseModel, Field


class DisruptionType(Enum):
    TRAFFIC_JAM = "traffic_jam"
    ROAD_CLOSURE = "road_closure"


class Disruption(BaseModel):
    id: int
    type: DisruptionType
    location: Tuple[float, float]
    affected_area_radius: float
    duration: int
    severity: float = Field(..., ge=0.0, le=1.0)
    resolved: bool = False
    affected_driver_ids: Set[int] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    activation_distance: float = 300.0
    is_active: bool = False

    def activate(self):
        if not self.is_active:
            self.is_active = True
            return True
        return False
