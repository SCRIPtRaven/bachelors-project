import math
import random

from config.settings import INNER_POINTS_RATIO


class GeolocationService:
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
                    points.append((lat, lon))
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
                outer_points.append((lat, lon))

        return inner_points + outer_points
