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
            location=location,
            affected_area_radius=radius,
            start_time=start_time,
            duration=duration,
            severity=severity,
            metadata=metadata,
            affected_driver_ids=affected_driver_ids,
            is_active=False,  # <-- Explicitly make it inactive
            is_proximity_based=True
        )

        self.next_disruption_id += 1
        return disruption

    def get_active_disruptions(self, simulation_time):
        """Get all disruptions active at the given simulation time"""
        active = []
        for disruption in self.disruptions:
            # ADD ACTIVATION CHECK - must be explicitly activated
            if (disruption.start_time <= simulation_time and
                    disruption.start_time + disruption.duration >= simulation_time and
                    disruption.is_active and  # <-- This is the key change
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

    def check_drivers_near_disruptions(self, driver_positions, driver_routes=None):
        """
        Check if any drivers are close to inactive disruptions and activate them.
        """
        newly_activated = []

        # Only check inactive, proximity-based, non-resolved disruptions
        eligible_disruptions = [
            d for d in self.disruptions
            if not d.is_active and d.is_proximity_based and not d.resolved
        ]

        if not eligible_disruptions:
            return []

        for disruption in eligible_disruptions:
            # Check each driver's position against this disruption
            for driver_id, position in driver_positions.items():
                # Calculate distance between driver and disruption
                distance = self._haversine_distance(position, disruption.location)

                # Check if driver is within activation distance
                if distance <= disruption.activation_distance:
                    # Check if this driver's route passes through the disruption area
                    if self._driver_route_affected(driver_id, disruption, driver_routes):
                        # Activate the disruption
                        if disruption.activate():
                            print(
                                f"Driver {driver_id} triggered disruption {disruption.id} at distance {distance:.1f}m")
                            disruption.affected_driver_ids.add(driver_id)
                            newly_activated.append(disruption)
                            break  # Move to next disruption after one is activated

        return newly_activated

    def _driver_route_affected(self, driver_id, disruption, driver_routes=None):
        """
        Check if a driver's route passes through a disruption area.
        """
        # If we don't have route information, assume all drivers could be affected
        if not driver_routes or driver_id not in driver_routes:
            return True

        # Get the driver's route points
        route = driver_routes[driver_id]
        if not route or not isinstance(route, dict) or 'points' not in route:
            return True

        route_points = route['points']

        # Check if any route segment passes near the disruption
        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]

            # Check if this segment is close to the disruption
            if self._segment_near_disruption(start_point, end_point, disruption):
                return True

        return False

    def _segment_near_disruption(self, start, end, disruption):
        """Check if a route segment passes near a disruption"""
        # Check if either endpoint is within the disruption radius
        if (self._haversine_distance(start, disruption.location) <= disruption.affected_area_radius or
                self._haversine_distance(end, disruption.location) <= disruption.affected_area_radius):
            return True

        # Check if the line segment passes close to the disruption
        closest_distance = self._point_to_segment_distance(
            disruption.location, start, end)

        return closest_distance <= disruption.affected_area_radius

    def _point_to_segment_distance(self, point, segment_start, segment_end):
        """Calculate the shortest distance from a point to a line segment"""
        # Convert coordinates to a simpler system for calculations
        # This is an approximation that works for relatively small distances
        p_lat, p_lon = point
        s_lat, s_lon = segment_start
        e_lat, e_lon = segment_end

        # Vector from start to end
        dx = e_lat - s_lat
        dy = e_lon - s_lon

        # Squared length of segment
        segment_length_sq = dx * dx + dy * dy

        # If segment is essentially a point, return distance to that point
        if segment_length_sq < 1e-10:
            return self._haversine_distance(point, segment_start)

        # Calculate projection of point onto segment line
        t = max(0, min(1, ((p_lat - s_lat) * dx + (p_lon - s_lon) * dy) / segment_length_sq))

        # Calculate closest point on segment
        closest_lat = s_lat + t * dx
        closest_lon = s_lon + t * dy

        # Return distance to closest point
        return self._haversine_distance(point, (closest_lat, closest_lon))
