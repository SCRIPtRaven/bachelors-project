from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator


class DeliveryPoint(BaseModel):
    coordinates: Tuple[float, float] = Field(..., description="Latitude and longitude")
    weight: float = Field(..., ge=2, le=25, description="Weight in kg")
    volume: float = Field(..., ge=0.01, le=0.5, description="Volume in cubic meters")


class Delivery(BaseModel):
    coordinates: Tuple[float, float]
    weight: float = Field(..., ge=2, le=25)
    volume: float = Field(..., ge=0.01, le=0.5)


class DeliveryAssignment(BaseModel):
    driver_id: int
    delivery_indices: List[int]
    total_weight: float = Field(0.0)
    total_volume: float = Field(0.0)
    fitness: float = Field(float('inf'))

    @field_validator('total_weight', 'total_volume')
    def validate_non_negative(cls, v: float) -> float:
        if abs(v) < 1e-10:
            return 0.0
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v
