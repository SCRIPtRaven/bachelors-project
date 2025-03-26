from enum import Enum
from typing import Tuple, Dict, Any, Set

from pydantic import BaseModel, Field


class DisruptionType(Enum):
    TRAFFIC_JAM = "traffic_jam"
    RECIPIENT_UNAVAILABLE = "recipient_unavailable"
    ROAD_CLOSURE = "road_closure"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"


class Disruption(BaseModel):
    id: int
    type: DisruptionType
    location: Tuple[float, float]  # (lat, lon)
    affected_area_radius: float  # in meters
    start_time: int  # simulation time in seconds
    duration: int  # duration in seconds
    severity: float = Field(..., ge=0.0, le=1.0)  # 0.0 to 1.0, affects impact
    resolved: bool = False
    affected_driver_ids: Set[int] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
