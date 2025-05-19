from pydantic import BaseModel, Field

from config.config import DeliveryConfig


class Driver(BaseModel):
    id: int = Field(..., ge=1)
    weight_capacity: float = Field(
        ...,
        ge=DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['min'],
        le=DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['max']
    )
    volume_capacity: float = Field(
        ...,
        ge=DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['min'],
        le=DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['max']
    )
