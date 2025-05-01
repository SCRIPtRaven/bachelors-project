import random

from config.config import DisruptionConfig
from models.entities.disruption import Disruption, DisruptionType
from utils.geo_utils import calculate_haversine_distance


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
        self.solution = solution

    def set_detailed_route_points(self, points):
        self.detailed_route_points = points

    def generate_disruptions(self, num_drivers):
        self.disruptions = []

        num_delivery_points = len(self.snapped_delivery_points) if self.snapped_delivery_points else 0

        base_disruptions = num_drivers + (num_delivery_points * 0.1)

        min_disruptions = 20
        max_disruptions = 22

        num_disruptions = min(max_disruptions,
                              max(min_disruptions,
                                  int(base_disruptions + random.randint(-1, 2))))

        print(f"Attempting to generate {num_disruptions} disruptions")

        generated_count = 0

        for disruption_type in [t for t in DisruptionType if t.value in DisruptionConfig.ENABLED_TYPES]:
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
        location = None
        radius = 0
        duration = 0
        severity = 0.0
        metadata = {}
        affected_driver_ids = set()

        max_attempts = 30
        min_buffer_distance = 50

        if disruption_type == DisruptionType.RECIPIENT_UNAVAILABLE:
            if not self.snapped_delivery_points:
                print("Warning: Cannot create RECIPIENT_UNAVAILABLE disruption, no delivery points available.")
                return None

            point_idx = random.randint(0, len(self.snapped_delivery_points) - 1)
            location = self.snapped_delivery_points[point_idx][:2]

            radius = 5
            duration = random.randint(720, 1800)
            severity = 1.0
            metadata = {
                "delivery_point_index": point_idx,
                "description": f"Recipient unavailable at Delivery {point_idx + 1}"
            }


        elif disruption_type in [DisruptionType.TRAFFIC_JAM, DisruptionType.ROAD_CLOSURE]:
            if not self.detailed_route_points:
                print(f"Warning: Cannot create {disruption_type.value} disruption, no detailed route points available.")
                return None

            if disruption_type == DisruptionType.TRAFFIC_JAM:
                min_radius, max_radius = 80, 300
                min_fallback_radius = 40
                duration_range = (1800, 5400)
                severity_range = (0.3, 0.9)
                description = "Heavy traffic congestion"

            else:
                min_radius, max_radius = 30, 150
                min_fallback_radius = 20
                duration_range = (3600, 14400)
                severity_range = (0.7, 1.0)
                description = "Road closed due to incident"

            for attempt in range(max_attempts):
                candidate_location = random.choice(self.detailed_route_points)
                candidate_radius = random.uniform(min_radius, max_radius)
                if (not self._disruption_covers_critical_point(candidate_location, candidate_radius) and
                        not self._disruption_overlaps_existing(candidate_location, candidate_radius,
                                                               min_buffer_distance)):
                    location = candidate_location
                    radius = candidate_radius
                    break

            if location is None:
                print(f"Warning: Initial attempts failed for {disruption_type.value}, trying with smaller radius")
                for attempt in range(max_attempts):
                    candidate_location = random.choice(self.detailed_route_points)
                    candidate_radius = random.uniform(min_fallback_radius, min_radius)
                    if (not self._disruption_covers_critical_point(candidate_location, candidate_radius) and
                            not self._disruption_overlaps_existing(candidate_location, candidate_radius,
                                                                   min_buffer_distance / 2)):
                        location = candidate_location
                        radius = candidate_radius
                        break

            if location is None:
                print(f"Warning: Using last resort approach for {disruption_type.value}")
                for attempt in range(max_attempts):
                    candidate_location = random.choice(self.detailed_route_points)
                    candidate_radius = min_fallback_radius
                    if not self._disruption_covers_critical_point(candidate_location, candidate_radius):
                        location = candidate_location
                        radius = candidate_radius
                        print(f"Created {disruption_type.value} that may overlap with other disruptions")
                        break

            if location is None:
                print(f"Failed to create {disruption_type.value} after {max_attempts * 3} attempts - skipping")
                return None

            duration = random.randint(*duration_range)
            severity = random.uniform(*severity_range)
            metadata = {"description": description}

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

    def _disruption_covers_critical_point(self, location, radius, buffer_distance=20):
        effective_radius = radius + buffer_distance

        if self.warehouse_location:
            distance = calculate_haversine_distance(location, self.warehouse_location)
            warehouse_buffer = buffer_distance * 2
            if distance <= (radius + warehouse_buffer):
                return True

        if self.snapped_delivery_points:
            for point in self.snapped_delivery_points:
                delivery_location = point[:2]
                distance = calculate_haversine_distance(location, delivery_location)

                if distance <= effective_radius:
                    return True

        return False

    def _disruption_overlaps_existing(self, location, radius, buffer_distance=50):
        for existing in self.disruptions:
            if existing.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                continue

            min_center_distance = existing.affected_area_radius + radius + buffer_distance

            actual_distance = calculate_haversine_distance(location, existing.location)

            if actual_distance < min_center_distance:
                return True

        return False

    def _create_random_disruption(self):
        available_types = [t for t in DisruptionType if t.value in DisruptionConfig.ENABLED_TYPES]

        if not available_types:
            return None
        disruption_type = random.choice(available_types)

        return self._create_specific_disruption(disruption_type)

    def get_active_disruptions(self, simulation_time):
        active = []
        for disruption in self.disruptions:
            if disruption.is_active and not disruption.resolved:
                active.append(disruption)
        return active

    def resolve_disruption(self, disruption_id):
        for disruption in self.disruptions:
            if disruption.id == disruption_id:
                disruption.resolved = True
                return True
        return False

    def is_point_affected(self, point, simulation_time):
        cache_key = (point, simulation_time)
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]

        active_disruptions = self.get_active_disruptions(simulation_time)
        for disruption in active_disruptions:
            distance = calculate_haversine_distance(point, disruption.location)
            if distance <= disruption.affected_area_radius:
                self.location_cache[cache_key] = disruption
                return disruption

        self.location_cache[cache_key] = None
        return None

    def get_path_disruptions(self, path, simulation_time):
        disruptions = []
        for point in path:
            disruption = self.is_point_affected(point, simulation_time)
            if disruption and disruption not in disruptions:
                disruptions.append(disruption)
        return disruptions

    def calculate_delay_factor(self, point, simulation_time):
        disruption = self.is_point_affected(point, simulation_time)
        if not disruption:
            return 1.0

        if disruption.type == DisruptionType.ROAD_CLOSURE:
            return 0.1
        else:
            return max(0.5, 1.0 - disruption.severity)

    def check_drivers_near_disruptions(self, driver_positions, driver_routes=None):
        newly_activated = []

        eligible_disruptions = [
            d for d in self.disruptions
            if not d.is_active and not d.resolved
        ]

        if not eligible_disruptions:
            return []

        for disruption in eligible_disruptions:
            for driver_id, position in driver_positions.items():
                distance = calculate_haversine_distance(position, disruption.location)

                if distance <= disruption.activation_distance:
                    if self._driver_route_affected(driver_id, disruption, driver_routes):
                        if disruption.activate():
                            print(
                                f"Driver {driver_id} triggered disruption {disruption.id} at distance {distance:.1f}m")
                            disruption.affected_driver_ids.add(driver_id)
                            
                            if not hasattr(disruption, 'metadata') or disruption.metadata is None:
                                disruption.metadata = {}
                            
                            disruption.metadata['triggered_by_driver'] = driver_id
                            
                            newly_activated.append(disruption)
                            break

        return newly_activated

    def _driver_route_affected(self, driver_id, disruption, driver_routes=None):
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
        if (calculate_haversine_distance(start, disruption.location) <= disruption.affected_area_radius or
                calculate_haversine_distance(end, disruption.location) <= disruption.affected_area_radius):
            return True

        closest_distance = self._point_to_segment_distance(
            disruption.location, start, end)

        return closest_distance <= disruption.affected_area_radius

    def _point_to_segment_distance(self, point, segment_start, segment_end):
        p_lat, p_lon = point
        s_lat, s_lon = segment_start
        e_lat, e_lon = segment_end

        dx = e_lat - s_lat
        dy = e_lon - s_lon

        segment_length_sq = dx * dx + dy * dy

        if segment_length_sq < 1e-10:
            return calculate_haversine_distance(point, segment_start)

        t = max(0, min(1, ((p_lat - s_lat) * dx + (p_lon - s_lon) * dy) / segment_length_sq))

        closest_lat = s_lat + t * dx
        closest_lon = s_lon + t * dy

        return calculate_haversine_distance(point, (closest_lat, closest_lon))
