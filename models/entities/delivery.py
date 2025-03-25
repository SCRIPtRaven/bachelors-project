from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator


class Delivery(BaseModel):
    coordinates: Tuple[float, float] = Field(..., description="Latitude and longitude")
    weight: float = Field(..., ge=0.0, le=30, description="Weight in kg")
    volume: float = Field(..., ge=0.0, le=0.125, description="Volume in cubic meters")


class DeliveryAssignment(BaseModel):
    driver_id: int
    delivery_indices: List[int]
    total_weight: float = Field(0.0)
    total_volume: float = Field(0.0)

    @field_validator('total_weight', 'total_volume')
    def validate_non_negative(cls, v: float) -> float:
        if abs(v) < 1e-10:
            return 0.0
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v
