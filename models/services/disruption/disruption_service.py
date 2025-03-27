import math
import random

from config.app_settings import ENABLED_DISRUPTION_TYPES
from models.entities.disruption import Disruption, DisruptionType


class DisruptionService:
    def __init__(self, G, warehouse_location, snapped_delivery_points):
        self.G = G
        self.warehouse_location = warehouse_location
        self.snapped_delivery_points = snapped_delivery_points
        self.disruptions = []
        self.solution = None
        self.next_disruption_id = 1
        self.disruption_probability = 0.8
        self.location_cache = {}
        self.detailed_route_points = []

    def set_solution(self, solution):
        """Update the route solution after optimization"""
        self.solution = solution

    def set_detailed_route_points(self, points):
        """Store the detailed list of (lat, lon) points from all calculated routes."""
        self.detailed_route_points = points

    def generate_disruptions(self, num_drivers):
        """Generate a set of random disruptions for the simulation"""
        self.disruptions = []

        num_delivery_points = len(self.snapped_delivery_points) if self.snapped_delivery_points else 0

        base_disruptions = num_drivers + (num_delivery_points * 0.1)

        min_disruptions = 5
        max_disruptions = 15

        num_disruptions = min(max_disruptions,
                              max(min_disruptions,
                                  int(base_disruptions + random.randint(-1, 2))))

        print(f"Attempting to generate {num_disruptions} disruptions")

        generated_count = 0

        for disruption_type in [t for t in DisruptionType if t.value in ENABLED_DISRUPTION_TYPES]:
            print(f"Generating a {disruption_type.value} disruption")
            disruption = self._create_specific_disruption(disruption_type)
            if disruption:
                self.disruptions.append(disruption)
                generated_count += 1
                self.next_disruption_id += 1

        remaining = num_disruptions - generated_count

        for _ in range(remaining):
            disruption = self._create_random_disruption()
            if disruption:
                self.disruptions.append(disruption)
                generated_count += 1
                self.next_disruption_id += 1

        print(f"Successfully generated {len(self.disruptions)} disruptions:")
        type_counts = {}
        for d in self.disruptions:
            type_counts[d.type.value] = type_counts.get(d.type.value, 0) + 1
        for t, count in type_counts.items():
            print(f"- {t}: {count}")

        return self.disruptions

    def _create_specific_disruption(self, disruption_type):
        """Create a disruption of a specific type"""
        location = None
        radius = 0
        duration = 0
        severity = 0.0
        metadata = {}
        affected_driver_ids = set()

        if disruption_type == DisruptionType.RECIPIENT_UNAVAILABLE:
            if not self.snapped_delivery_points:
                print("Warning: Cannot create RECIPIENT_UNAVAILABLE disruption, no delivery points available.")
                return None

            point_idx = random.randint(0, len(self.snapped_delivery_points) - 1)
            location = self.snapped_delivery_points[point_idx][:2]

            radius = 5
            duration = random.randint(1800, 7200)
            severity = 1.0
            metadata = {
                "delivery_point_index": point_idx,
                "description": f"Recipient unavailable at Delivery {point_idx + 1}"
            }

        elif disruption_type == DisruptionType.TRAFFIC_JAM:
            if not self.detailed_route_points:
                print("Warning: Cannot create TRAFFIC_JAM disruption, no detailed route points available.")
                return None

            location = random.choice(self.detailed_route_points)
            radius = random.uniform(100, 500)
            duration = random.randint(1800, 5400)
            severity = random.uniform(0.3, 0.9)
            metadata = {
                "description": "Heavy traffic congestion"
            }

        elif disruption_type == DisruptionType.ROAD_CLOSURE:
            if not self.detailed_route_points:
                print("Warning: Cannot create ROAD_CLOSURE disruption, no detailed route points available.")
                return None

            location = random.choice(self.detailed_route_points)
            radius = random.uniform(50, 250)
            duration = random.randint(3600, 14400)
            severity = 1.0
            metadata = {
                "description": "Road closed due to incident"
            }

        if location is None:
            return None

        disruption = Disruption(
            id=self.next_disruption_id,
            type=disruption_type,
            location=location,
            affected_area_radius=radius,
            duration=max(60, duration),
            severity=severity,
            metadata=metadata,
            affected_driver_ids=affected_driver_ids,
            is_active=False,
        )

        return disruption

    def _create_random_disruption(self):
        """Create a random disruption based on type and location constraints"""
        available_types = [t for t in DisruptionType if t.value in ENABLED_DISRUPTION_TYPES]

        if not available_types:
            return None
        disruption_type = random.choice(available_types)

        location = None
        radius = 0
        duration = 0
        severity = 0.0
        metadata = {}
        affected_driver_ids = set()

        if disruption_type == DisruptionType.RECIPIENT_UNAVAILABLE:
            if not self.snapped_delivery_points:
                print("Warning: Cannot create RECIPIENT_UNAVAILABLE disruption, no delivery points available.")
                return None

            point_idx = random.randint(0, len(self.snapped_delivery_points) - 1)
            location = self.snapped_delivery_points[point_idx][:2]

            radius = 5
            duration = random.randint(360, 720)
            severity = 1.0
            metadata = {
                "delivery_point_index": point_idx,
                "description": f"Recipient unavailable at Delivery {point_idx + 1}"
            }

        elif disruption_type in [DisruptionType.TRAFFIC_JAM, DisruptionType.ROAD_CLOSURE]:
            if not self.detailed_route_points:
                print(f"Warning: Cannot create {disruption_type.value} disruption, no detailed route points available.")
                return None

            location = random.choice(self.detailed_route_points)

            if disruption_type == DisruptionType.TRAFFIC_JAM:
                radius = random.uniform(100, 500)
                duration = random.randint(1800, 5400)
                severity = random.uniform(0.3, 0.9)
                metadata = {
                    "description": "Heavy traffic congestion"
                }
            elif disruption_type == DisruptionType.ROAD_CLOSURE:
                radius = random.uniform(50, 250)
                duration = random.randint(3600, 14400)
                severity = 1.0
                metadata = {
                    "description": "Road closed due to incident"
                }
        if location is None:
            return None

        duration = max(60, duration)

        disruption = Disruption(
            id=self.next_disruption_id,
            type=disruption_type,
            location=location,
            affected_area_radius=radius,
            duration=duration,
            severity=severity,
            metadata=metadata,
            affected_driver_ids=affected_driver_ids,
            is_active=False,
        )

        self.next_disruption_id += 1
        return disruption

    def get_active_disruptions(self, simulation_time):
        """Get all disruptions active at the given simulation time"""
        active = []
        for disruption in self.disruptions:
            if disruption.is_active and not disruption.resolved:
                active.append(disruption)
        return active

    def resolve_disruption(self, disruption_id):
        """Mark a disruption as resolved"""
        for disruption in self.disruptions:
            if disruption.id == disruption_id:
                disruption.resolved = True
                return True
        return False

    def is_point_affected(self, point, simulation_time):
        """Check if a point is affected by any active disruption"""
        cache_key = (point, simulation_time)
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]

        active_disruptions = self.get_active_disruptions(simulation_time)
        for disruption in active_disruptions:
            distance = self._haversine_distance(point, disruption.location)
            if distance <= disruption.affected_area_radius:
                self.location_cache[cache_key] = disruption
                return disruption

        self.location_cache[cache_key] = None
        return None

    def get_path_disruptions(self, path, simulation_time):
        """Find disruptions affecting a path"""
        disruptions = []
        for point in path:
            disruption = self.is_point_affected(point, simulation_time)
            if disruption and disruption not in disruptions:
                disruptions.append(disruption)
        return disruptions

    def calculate_delay_factor(self, point, simulation_time):
        """Calculate delay factor for a point at the given time"""
        disruption = self.is_point_affected(point, simulation_time)
        if not disruption:
            return 1.0

        if disruption.type == DisruptionType.ROAD_CLOSURE:
            return 0.1
        else:
            return max(0.5, 1.0 - disruption.severity)

    def _haversine_distance(self, point1, point2):
        """Calculate distance between two points in meters using Haversine formula"""
        lat1 = math.radians(point1[0])
        lon1 = math.radians(point1[1])
        lat2 = math.radians(point2[0])
        lon2 = math.radians(point2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000
        return c * r

    def check_drivers_near_disruptions(self, driver_positions, driver_routes=None):
        """
        Check if any drivers are close to inactive disruptions and activate them.
        """
        newly_activated = []

        eligible_disruptions = [
            d for d in self.disruptions
            if not d.is_active and not d.resolved
        ]

        if not eligible_disruptions:
            return []

        for disruption in eligible_disruptions:
            for driver_id, position in driver_positions.items():
                distance = self._haversine_distance(position, disruption.location)

                if distance <= disruption.activation_distance:
                    if self._driver_route_affected(driver_id, disruption, driver_routes):
                        if disruption.activate():
                            print(
                                f"Driver {driver_id} triggered disruption {disruption.id} at distance {distance:.1f}m")
                            disruption.affected_driver_ids.add(driver_id)
                            newly_activated.append(disruption)
                            break

        return newly_activated

    def _driver_route_affected(self, driver_id, disruption, driver_routes=None):
        """
        Check if a driver's route passes through a disruption area.
        """
        if not driver_routes or driver_id not in driver_routes:
            return True

        route = driver_routes[driver_id]
        if not route or not isinstance(route, dict) or 'points' not in route:
            return True

        route_points = route['points']

        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]

            if self._segment_near_disruption(start_point, end_point, disruption):
                return True

        return False

    def _segment_near_disruption(self, start, end, disruption):
        """Check if a route segment passes near a disruption"""
        if (self._haversine_distance(start, disruption.location) <= disruption.affected_area_radius or
                self._haversine_distance(end, disruption.location) <= disruption.affected_area_radius):
            return True

        closest_distance = self._point_to_segment_distance(
            disruption.location, start, end)

        return closest_distance <= disruption.affected_area_radius

    def _point_to_segment_distance(self, point, segment_start, segment_end):
        """Calculate the shortest distance from a point to a line segment"""
        p_lat, p_lon = point
        s_lat, s_lon = segment_start
        e_lat, e_lon = segment_end

        dx = e_lat - s_lat
        dy = e_lon - s_lon

        segment_length_sq = dx * dx + dy * dy

        if segment_length_sq < 1e-10:
            return self._haversine_distance(point, segment_start)

        t = max(0, min(1, ((p_lat - s_lat) * dx + (p_lon - s_lon) * dy) / segment_length_sq))

        closest_lat = s_lat + t * dx
        closest_lon = s_lon + t * dy

        return self._haversine_distance(point, (closest_lat, closest_lon))
