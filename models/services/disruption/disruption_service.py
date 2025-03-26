import math
import random

from models.entities.disruption import Disruption, DisruptionType


class DisruptionService:
    def __init__(self, G, warehouse_location, snapped_delivery_points):
        self.G = G
        self.warehouse_location = warehouse_location
        self.snapped_delivery_points = snapped_delivery_points
        self.disruptions = []
        self.solution = None
        self.next_disruption_id = 1
        self.disruption_probability = 0.3  # 30% chance of disruption per hour
        self.location_cache = {}
        self.detailed_route_points = []

    def set_solution(self, solution):
        """Update the route solution after optimization"""
        self.solution = solution

    def set_detailed_route_points(self, points):
        """Store the detailed list of (lat, lon) points from all calculated routes."""
        self.detailed_route_points = points
        print(f"DisruptionService stored {len(self.detailed_route_points)} detailed points.")  # Debug print

    def generate_disruptions(self, simulation_duration, num_drivers):
        """Generate a set of random disruptions for the simulation duration"""
        # Clear previous disruptions
        self.disruptions = []

        # Calculate number of disruptions based on simulation duration
        # Assuming duration is in seconds
        hours = simulation_duration / 3600
        expected_disruptions = int(hours * self.disruption_probability * num_drivers)
        num_disruptions = random.randint(
            max(0, expected_disruptions - 2),
            max(1, expected_disruptions + 2)
        )

        generated_count = 0
        for _ in range(num_disruptions):
            disruption = self._create_random_disruption(simulation_duration)
            if disruption:
                self.disruptions.append(disruption)
                generated_count += 1

        print(
            f"Attempted to generate {num_disruptions}, successfully generated {generated_count} disruptions.")

        # Sort disruptions by start time
        self.disruptions.sort(key=lambda d: d.start_time)
        return self.disruptions

    def _create_random_disruption(self, max_time):
        """Create a random disruption based on type and location constraints"""
        disruption_type = random.choice(list(DisruptionType))

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
                # "retry_delay": random.randint(1800, 3600), # Maybe handle retry logic elsewhere
                "description": f"Recipient unavailable at Delivery {point_idx + 1}"
            }

        elif disruption_type in [DisruptionType.TRAFFIC_JAM, DisruptionType.ROAD_CLOSURE,
                                 DisruptionType.VEHICLE_BREAKDOWN]:
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
            else:
                radius = 10
                duration = random.randint(1800, 4800)
                severity = 1.0

                affected_driver_id = None
                if self.solution:
                    driver_ids_with_routes = [a.driver_id for a in self.solution if a.delivery_indices]
                    if driver_ids_with_routes:
                        affected_driver_id = random.choice(driver_ids_with_routes)
                        affected_driver_ids.add(affected_driver_id)
                        metadata = {
                            "description": f"Vehicle breakdown (Driver {affected_driver_id})"
                        }
                    else:
                        metadata = {"description": "Vehicle breakdown (Driver unknown)"}
                else:
                    metadata = {"description": "Vehicle breakdown (Driver unknown)"}

        if location is None:
            return None

        current_time = 8 * 3600
        effective_simulation_duration = max_time
        start_offset = random.randint(60, max(300,
                                              int(effective_simulation_duration * 0.8)))
        start_time = current_time + start_offset

        duration = max(60, duration)

        disruption = Disruption(
            id=self.next_disruption_id,
            type=disruption_type,
            location=location,  # Should be (lat, lon)
            affected_area_radius=radius,
            start_time=start_time,
            duration=duration,
            severity=severity,
            metadata=metadata,
            affected_driver_ids=affected_driver_ids if disruption_type == DisruptionType.VEHICLE_BREAKDOWN else set()
        )

        self.next_disruption_id += 1
        return disruption

    def get_active_disruptions(self, simulation_time):
        """Get all disruptions active at the given simulation time"""
        active = []
        for disruption in self.disruptions:
            if (disruption.start_time <= simulation_time and
                    disruption.start_time + disruption.duration >= simulation_time and
                    not disruption.resolved):
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
            return 0.1  # Almost blocked, but allow very slow movement
        elif disruption.type == DisruptionType.VEHICLE_BREAKDOWN:
            return 0.0  # Complete stop
        else:
            return max(0.5, 1.0 - disruption.severity)

    def _haversine_distance(self, point1, point2):
        """Calculate distance between two points in meters using Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(point1[0])
        lon1 = math.radians(point1[1])
        lat2 = math.radians(point2[0])
        lon2 = math.radians(point2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
