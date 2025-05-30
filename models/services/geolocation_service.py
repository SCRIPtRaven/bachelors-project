import math
import random

from config.config import DeliveryConfig
from models.entities.delivery import Delivery
from models.entities.driver import Driver


class GeolocationService:
    @staticmethod
    def generate_random_package_properties():
        weight_steps = int((DeliveryConfig.PACKAGE_CONSTRAINTS['weight']['max'] -
                            DeliveryConfig.PACKAGE_CONSTRAINTS['weight']['min']) /
                           DeliveryConfig.PACKAGE_CONSTRAINTS['weight']['step'])
        volume_steps = int((DeliveryConfig.PACKAGE_CONSTRAINTS['volume']['max'] -
                            DeliveryConfig.PACKAGE_CONSTRAINTS['volume']['min']) /
                           DeliveryConfig.PACKAGE_CONSTRAINTS['volume']['step'])

        weight_step_count = random.randint(0, weight_steps)
        volume_step_count = random.randint(0, volume_steps)

        weight = DeliveryConfig.PACKAGE_CONSTRAINTS['weight']['min'] + (
                weight_step_count * DeliveryConfig.PACKAGE_CONSTRAINTS['weight']['step'])
        volume = DeliveryConfig.PACKAGE_CONSTRAINTS['volume']['min'] + (
                volume_step_count * DeliveryConfig.PACKAGE_CONSTRAINTS['volume']['step'])

        weight = round(weight, 2)
        volume = round(volume, 3)

        return weight, volume

    @staticmethod
    def generate_delivery_points(bounds, num_points):
        min_lat, max_lat, min_lon, max_lon = bounds

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        min_distance = min(lat_range, lon_range) * 0.015

        points = []
        attempts = 0
        max_attempts = num_points * 10

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            if random.random() < DeliveryConfig.INNER_POINTS_RATIO:
                margin = 0.15
                lat = random.uniform(
                    min_lat + lat_range * margin,
                    max_lat - lat_range * margin
                )
                lon = random.uniform(
                    min_lon + lon_range * margin,
                    max_lon - lon_range * margin
                )
            else:
                lat = random.uniform(min_lat, max_lat)
                lon = random.uniform(min_lon, max_lon)

            too_close = False
            for existing_point in points:
                ex_lat, ex_lon = existing_point.coordinates
                dist = math.sqrt(
                    (lat - ex_lat) ** 2 +
                    (lon - ex_lon) ** 2
                )
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                weight, volume = GeolocationService.generate_random_package_properties()
                points.append(Delivery(
                    coordinates=(lat, lon),
                    weight=weight,
                    volume=volume
                ))

        return points

    @staticmethod
    def generate_random_driver_properties():
        weight_steps = int((DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['max'] -
                            DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['min']) /
                           DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity'][
                               'step'])
        volume_steps = int((DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['max'] -
                            DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['min']) /
                           DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity'][
                               'step'])

        weight_step_count = random.randint(0, weight_steps)
        volume_step_count = random.randint(0, volume_steps)

        weight_capacity = (DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['min'] +
                           (weight_step_count * DeliveryConfig.DRIVER_CONSTRAINTS['weight_capacity']['step']))
        volume_capacity = (DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['min'] +
                           (volume_step_count * DeliveryConfig.DRIVER_CONSTRAINTS['volume_capacity']['step']))

        weight_capacity = round(weight_capacity, 1)
        volume_capacity = round(volume_capacity, 2)

        return weight_capacity, volume_capacity

    @staticmethod
    def generate_delivery_drivers(num_drivers):
        drivers = []
        for i in range(num_drivers):
            weight_capacity, volume_capacity = GeolocationService.generate_random_driver_properties()
            drivers.append(Driver(
                id=i + 1,
                weight_capacity=weight_capacity,
                volume_capacity=volume_capacity
            ))
        return drivers
