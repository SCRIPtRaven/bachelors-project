from pydantic import BaseModel, Field

from config.delivery_settings import DRIVER_CONSTRAINTS


class Driver(BaseModel):
    id: int = Field(..., ge=1)
    weight_capacity: float = Field(
        ...,
        ge=DRIVER_CONSTRAINTS['weight_capacity']['min'],
        le=DRIVER_CONSTRAINTS['weight_capacity']['max']
    )
    volume_capacity: float = Field(
        ...,
        ge=DRIVER_CONSTRAINTS['volume_capacity']['min'],
        le=DRIVER_CONSTRAINTS['volume_capacity']['max']
    )
