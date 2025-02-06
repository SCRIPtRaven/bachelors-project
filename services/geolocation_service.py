import math
import random
from dataclasses import dataclass
from typing import Tuple

from config.settings import INNER_POINTS_RATIO, PACKAGE_CONSTRAINTS, DRIVER_CONSTRAINTS


# TODO : Implement centralized data type handling and definitions
@dataclass
class DeliveryPoint:
    coordinates: Tuple[float, float]
    weight: float
    volume: float


@dataclass
class DeliveryDriver:
    id: int
    weight_capacity: float  # in kg
    volume_capacity: float  # in cubic meters


class GeolocationService:
    @staticmethod
    def generate_random_package_properties():
        """Generate random weight and volume within defined constraints."""
        weight_step = 0.05  # 50 gram steps
        volume_step = 0.001  # 1 liter steps

        weight_steps = int((PACKAGE_CONSTRAINTS['weight']['max'] -
                            PACKAGE_CONSTRAINTS['weight']['min']) / weight_step)
        volume_steps = int((PACKAGE_CONSTRAINTS['volume']['max'] -
                            PACKAGE_CONSTRAINTS['volume']['min']) / volume_step)

        weight_step_count = random.randint(0, weight_steps)
        volume_step_count = random.randint(0, volume_steps)

        weight = PACKAGE_CONSTRAINTS['weight']['min'] + (weight_step_count * weight_step)
        volume = PACKAGE_CONSTRAINTS['volume']['min'] + (volume_step_count * volume_step)

        weight = round(weight, 2)
        volume = round(volume, 3)

        return weight, volume

    @staticmethod
    def generate_grid_points(min_lat, max_lat, min_lon, max_lon, num_points):
        try:
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            grid_size = math.ceil(math.sqrt(num_points))

            step_lat = lat_range / grid_size
            step_lon = lon_range / grid_size

            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) >= num_points:
                        break
                    lat = min_lat + i * step_lat + random.uniform(0, step_lat)
                    lon = min_lon + j * step_lon + random.uniform(0, step_lon)
                    weight, volume = GeolocationService.generate_random_package_properties()
                    points.append(DeliveryPoint(
                        coordinates=(lat, lon),
                        weight=weight,
                        volume=volume
                    ))
                if len(points) >= num_points:
                    break
            return points
        except Exception as e:
            print(f"Error generating grid points: {e}")
            return []

    @staticmethod
    def generate_delivery_points(bounds, num_points):
        min_lat, max_lat, min_lon, max_lon = bounds

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        inner_lat_range = lat_range * 0.5
        inner_lon_range = lon_range * 0.5

        lat_mid = (min_lat + max_lat) / 2
        lon_mid = (min_lon + max_lon) / 2

        inner_bounds = (
            lat_mid - inner_lat_range / 2,
            lat_mid + inner_lat_range / 2,
            lon_mid - inner_lon_range / 2,
            lon_mid + inner_lon_range / 2
        )

        inner_count = int(num_points * INNER_POINTS_RATIO)
        outer_count = num_points - inner_count

        inner_points = GeolocationService.generate_grid_points(
            inner_bounds[0], inner_bounds[1], inner_bounds[2], inner_bounds[3], inner_count
        )

        outer_points = []
        while len(outer_points) < outer_count:
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
            if not (inner_bounds[0] <= lat <= inner_bounds[1] and
                    inner_bounds[2] <= lon <= inner_bounds[3]):
                weight, volume = GeolocationService.generate_random_package_properties()
                outer_points.append(DeliveryPoint(
                    coordinates=(lat, lon),
                    weight=weight,
                    volume=volume
                ))

        return inner_points + outer_points

    @staticmethod
    def generate_random_driver_properties():
        """Generate random weight and volume capacities within defined constraints."""
        weight_step = 5.0  # 5 kg steps
        volume_step = 0.1  # 100 liter steps

        weight_steps = int((DRIVER_CONSTRAINTS['weight_capacity']['max'] -
                            DRIVER_CONSTRAINTS['weight_capacity']['min']) / weight_step)
        volume_steps = int((DRIVER_CONSTRAINTS['volume_capacity']['max'] -
                            DRIVER_CONSTRAINTS['volume_capacity']['min']) / volume_step)

        weight_step_count = random.randint(0, weight_steps)
        volume_step_count = random.randint(0, volume_steps)

        weight_capacity = (DRIVER_CONSTRAINTS['weight_capacity']['min'] +
                           (weight_step_count * weight_step))
        volume_capacity = (DRIVER_CONSTRAINTS['volume_capacity']['min'] +
                           (volume_step_count * volume_step))

        weight_capacity = round(weight_capacity, 1)
        volume_capacity = round(volume_capacity, 2)

        return weight_capacity, volume_capacity

    @staticmethod
    def generate_delivery_drivers(num_drivers):
        """Generate the specified number of delivery drivers with random capacities."""
        drivers = []
        for i in range(num_drivers):
            weight_capacity, volume_capacity = GeolocationService.generate_random_driver_properties()
            drivers.append(DeliveryDriver(
                id=i + 1,
                weight_capacity=weight_capacity,
                volume_capacity=volume_capacity
            ))
        return drivers
