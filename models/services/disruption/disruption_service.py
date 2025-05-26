import random

from config.config import DisruptionConfig
from models.entities.disruption import Disruption, DisruptionType
from utils.geo_utils import calculate_haversine_distance
import networkx as nx
import osmnx as ox


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
        self._cached_driver_routes = None

    def set_detailed_route_points(self, points):
        self.detailed_route_points = points

    def generate_disruptions(self, num_drivers):
        self.disruptions = []
        
        self._cached_driver_routes = None

        num_delivery_points = len(
            self.snapped_delivery_points) if self.snapped_delivery_points else 0

        base_disruptions = num_drivers + (num_delivery_points * 0.1)

        min_disruptions = DisruptionConfig.MIN_DISRUPTIONS
        max_disruptions = DisruptionConfig.MAX_DISRUPTIONS

        num_disruptions = min(max_disruptions,
                              max(min_disruptions,
                                  int(base_disruptions + random.randint(-1, 2))))

        print(f"Attempting to generate {num_disruptions} disruptions")

        generated_count = 0

        for disruption_type in [t for t in DisruptionType if
                                t.value in DisruptionConfig.ENABLED_TYPES]:
            print(f"Generating a {disruption_type.value} disruption")
            disruption = self._create_specific_disruption(disruption_type)
            if disruption:
                self.disruptions.append(disruption)
                generated_count += 1
                self.next_disruption_id += 1

        remaining = num_disruptions - generated_count
        max_attempts_per_disruption = 50

        for attempt in range(remaining):
            disruption = None
            for _ in range(max_attempts_per_disruption):
                disruption = self._create_random_disruption()
                if disruption:
                    break
            
            if disruption:
                self.disruptions.append(disruption)
                generated_count += 1
                self.next_disruption_id += 1
            else:
                print(f"Warning: Failed to generate disruption {attempt + 1} after {max_attempts_per_disruption} attempts")

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

        if disruption_type in [DisruptionType.TRAFFIC_JAM, DisruptionType.ROAD_CLOSURE]:
            if not self.detailed_route_points:
                print(
                    f"Warning: Cannot create {disruption_type.value} disruption, no detailed route points available.")
                return None

            if disruption_type == DisruptionType.TRAFFIC_JAM:
                min_radius, max_radius = 80, 200
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
                if (not self._disruption_covers_critical_point(candidate_location,
                                                               candidate_radius) and
                        not self._disruption_overlaps_existing(candidate_location, candidate_radius,
                                                               min_buffer_distance) and
                        not self._disruption_intersects_other_routes(candidate_location, candidate_radius, 
                                                                    self.detailed_route_points)):
                    location = candidate_location
                    radius = candidate_radius
                    break

            if location is None:
                print(
                    f"Warning: Initial attempts failed for {disruption_type.value}, trying with smaller radius")
                for attempt in range(max_attempts):
                    candidate_location = random.choice(self.detailed_route_points)
                    candidate_radius = random.uniform(min_fallback_radius, min_radius)
                    if (not self._disruption_covers_critical_point(candidate_location,
                                                                   candidate_radius) and
                            not self._disruption_overlaps_existing(candidate_location,
                                                                   candidate_radius,
                                                                   min_buffer_distance / 2) and
                            not self._disruption_intersects_other_routes(candidate_location, candidate_radius,
                                                                        self.detailed_route_points)):
                        location = candidate_location
                        radius = candidate_radius
                        break

            if location is None:
                print(f"Warning: Using last resort approach for {disruption_type.value}")
                for attempt in range(max_attempts):
                    candidate_location = random.choice(self.detailed_route_points)
                    candidate_radius = min_fallback_radius
                    if (not self._disruption_covers_critical_point(candidate_location,
                                                                  candidate_radius) and
                            not self._disruption_intersects_other_routes(candidate_location, candidate_radius,
                                                                        self.detailed_route_points)):
                        location = candidate_location
                        radius = candidate_radius
                        print(
                            f"Created {disruption_type.value} that may overlap with other disruptions but doesn't affect other drivers")
                        break

            if location is None:
                print(
                    f"Failed to create {disruption_type.value} after {max_attempts * 3} attempts - skipping")
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

        self._setup_disruption_tripwire(disruption)

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

    def get_active_disruptions(self):
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

        active_disruptions = self.get_active_disruptions()
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
            if disruption.owning_driver_id is not None:
                driver_id = disruption.owning_driver_id
                if driver_id not in driver_positions:
                    continue
                    
                position = driver_positions[driver_id]
                
                if self._driver_crossed_tripwire(driver_id, position, disruption, driver_routes):
                    if disruption.activate():
                        print(f"Driver {driver_id} crossed tripwire for disruption {disruption.id}")
                        disruption.affected_driver_ids.add(driver_id)

                        if not hasattr(disruption, 'metadata') or disruption.metadata is None:
                            disruption.metadata = {}

                        disruption.metadata['triggered_by_driver'] = driver_id
                        newly_activated.append(disruption)
            else:
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

    def _driver_crossed_tripwire(self, driver_id, current_position, disruption, driver_routes=None):
        if not disruption.tripwire_location:
            return False
            
        if not hasattr(disruption, '_driver_positions'):
            disruption._driver_positions = {}
            
        previous_position = disruption._driver_positions.get(driver_id)
        disruption._driver_positions[driver_id] = current_position
        
        if previous_position is None:
            return False
            
        if isinstance(disruption.tripwire_location, dict):
            tripwire_start = disruption.tripwire_location['start']
            tripwire_end = disruption.tripwire_location['end']
            
            return self._line_segments_intersect(
                previous_position, current_position,
                tripwire_start, tripwire_end
            )
        else:
            if not driver_routes or driver_id not in driver_routes:
                return False
                
            route = driver_routes[driver_id]
            if not route or not isinstance(route, dict) or 'points' not in route:
                return False
                
            route_points = route['points']
            if not route_points or len(route_points) < 2:
                return False
                
            tripwire_segment = self._find_tripwire_segment_on_route(disruption.tripwire_location, route_points)
            if tripwire_segment is None:
                return False
                
            segment_start, segment_end = tripwire_segment
            
            return self._line_segments_intersect(
                previous_position, current_position,
                segment_start, segment_end
            )

    def _find_tripwire_segment_on_route(self, tripwire_location, route_points, threshold=50):
        min_distance = float('inf')
        closest_segment = None
        
        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]
            
            distance = self._point_to_segment_distance(tripwire_location, start_point, end_point)
            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_segment = (start_point, end_point)
                
        return closest_segment

    def _line_segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_tripwire_location(self, disruption_location, route_points, activation_distance):
        if not route_points or len(route_points) < 2:
            return None
            
        closest_segment_idx = -1
        min_distance = float('inf')
        closest_point_on_segment = None
        closest_t = 0
        
        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]
            
            p_lat, p_lon = disruption_location
            s_lat, s_lon = start_point
            e_lat, e_lon = end_point
            
            dx = e_lat - s_lat
            dy = e_lon - s_lon
            segment_length_sq = dx * dx + dy * dy
            
            if segment_length_sq > 1e-10:
                t = max(0, min(1, ((p_lat - s_lat) * dx + (p_lon - s_lon) * dy) / segment_length_sq))
                closest_lat = s_lat + t * dx
                closest_lon = s_lon + t * dy
                closest_point = (closest_lat, closest_lon)
            else:
                closest_point = start_point
                t = 0
            
            distance = calculate_haversine_distance(disruption_location, closest_point)
            if distance < min_distance:
                min_distance = distance
                closest_segment_idx = i
                closest_point_on_segment = closest_point
                closest_t = t
                    
        if closest_segment_idx == -1 or closest_point_on_segment is None:
            return None
            
        start_point = route_points[closest_segment_idx]
        end_point = route_points[closest_segment_idx + 1]
        
        route_dx = end_point[0] - start_point[0]
        route_dy = end_point[1] - start_point[1]
        
        route_length = (route_dx**2 + route_dy**2)**0.5
        if route_length == 0:
            return None
            
        route_dx_norm = route_dx / route_length
        route_dy_norm = route_dy / route_length
        
        perp_dx = -route_dy_norm
        perp_dy = route_dx_norm
        
        tripwire_center = None
        remaining_distance = activation_distance
        
        current_segment_idx = closest_segment_idx
        current_t = closest_t
        
        while remaining_distance > 0 and current_segment_idx >= 0:
            segment_start = route_points[current_segment_idx]
            segment_end = route_points[current_segment_idx + 1]
            
            if current_t > 0:
                current_pos_lat = segment_start[0] + current_t * (segment_end[0] - segment_start[0])
                current_pos_lon = segment_start[1] + current_t * (segment_end[1] - segment_start[1])
                current_pos = (current_pos_lat, current_pos_lon)
                distance_to_start = calculate_haversine_distance(current_pos, segment_start)
            else:
                distance_to_start = 0
                current_pos = segment_start
            
            if remaining_distance <= distance_to_start:
                segment_dx = segment_end[0] - segment_start[0]
                segment_dy = segment_end[1] - segment_start[1]
                segment_length = calculate_haversine_distance(segment_start, segment_end)
                
                if segment_length > 0:
                    ratio = (distance_to_start - remaining_distance) / segment_length
                    tripwire_center = (
                        segment_start[0] + ratio * segment_dx,
                        segment_start[1] + ratio * segment_dy
                    )
                else:
                    tripwire_center = segment_start
                break
            else:
                remaining_distance -= distance_to_start
                current_segment_idx -= 1
                current_t = 1.0
        
        if tripwire_center is None:
            if route_points:
                tripwire_center = route_points[0]
            else:
                return None
        
        center_to_route_distance = self._point_to_segment_distance_to_route(tripwire_center, route_points)
        if center_to_route_distance > 50:
            print(f"Warning: Tripwire center is {center_to_route_distance:.1f}m from route, may be off-route")
        
        return tripwire_center

    def _setup_disruption_tripwire(self, disruption):
        if not self.solution or not self.G:
            return
            
        if not hasattr(self, '_cached_driver_routes') or not self._cached_driver_routes:
            self._cached_driver_routes = self._calculate_all_driver_routes()
            
        all_driver_routes = self._cached_driver_routes
        if not all_driver_routes:
            return
            
        owning_driver_id = None
        owning_route_points = None
        
        for driver_id, route_points in all_driver_routes.items():
            if self._point_belongs_to_route(disruption.location, route_points):
                owning_driver_id = driver_id
                owning_route_points = route_points
                break
                
        if owning_driver_id is not None and owning_route_points:
            disruption.owning_driver_id = owning_driver_id
            
            tripwire_location = self._calculate_tripwire_location(
                disruption.location, 
                owning_route_points, 
                disruption.activation_distance
            )
            
            if tripwire_location:
                disruption.tripwire_location = tripwire_location
                print(f"Disruption {disruption.id} assigned to driver {owning_driver_id}")
                print(f"  Disruption location: {disruption.location}")
                print(f"  Tripwire point: {tripwire_location}")
                
                center_distance = self._point_to_segment_distance_to_route(tripwire_location, owning_route_points)
                print(f"  Tripwire distance to route: {center_distance:.1f}m")
            else:
                print(f"Warning: Could not calculate tripwire location for disruption {disruption.id}")
        else:
            print(f"Warning: Could not determine owning driver for disruption {disruption.id} at {disruption.location}")

    def _segment_near_disruption(self, start, end, disruption):
        if (calculate_haversine_distance(start,
                                         disruption.location) <= disruption.affected_area_radius or
                calculate_haversine_distance(end,
                                             disruption.location) <= disruption.affected_area_radius):
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

    def _point_to_segment_distance_to_route(self, point, route_points):
        if not route_points or len(route_points) < 2:
            return float('inf')
        
        min_distance = float('inf')
        for i in range(len(route_points) - 1):
            distance = self._point_to_segment_distance(point, route_points[i], route_points[i + 1])
            min_distance = min(min_distance, distance)
        
        return min_distance

    def _disruption_intersects_other_routes(self, location, radius, spawning_driver_route_points):
        if not self.solution or not self.G:
            return False
        
        if not hasattr(self, '_cached_driver_routes') or not self._cached_driver_routes:
            self._cached_driver_routes = self._calculate_all_driver_routes()
        
        all_driver_routes = self._cached_driver_routes
        
        if not all_driver_routes:
            return False
        
        spawning_driver_id = None
        for driver_id, route_points in all_driver_routes.items():
            if self._point_belongs_to_route(location, route_points):
                spawning_driver_id = driver_id
                break
        
        if spawning_driver_id is None:
            print(f"Warning: Could not determine spawning driver for disruption at {location}")
            return False
        
        intersecting_drivers = []
        for driver_id, route_points in all_driver_routes.items():
            if driver_id == spawning_driver_id:
                continue
            
            if self._route_intersects_disruption(route_points, location, radius):
                intersecting_drivers.append(driver_id)
        
        if intersecting_drivers:
            print(f"Disruption at {location} with radius {radius:.1f}m would affect drivers {intersecting_drivers} in addition to spawning driver {spawning_driver_id} - rejecting")
            return True
        
        return False

    def _calculate_all_driver_routes(self):
        all_routes = {}
        
        if not self.solution:
            return all_routes
        
        print(f"Calculating routes for {len(self.solution)} drivers")
        
        for assignment in self.solution:
            driver_id = assignment.driver_id
            
            if not assignment.delivery_indices:
                all_routes[driver_id] = [self.warehouse_location, self.warehouse_location]
                print(f"Driver {driver_id}: No deliveries, warehouse-to-warehouse route")
                continue
            
            route_points = [self.warehouse_location]
            
            for delivery_idx in assignment.delivery_indices:
                if delivery_idx < len(self.snapped_delivery_points):
                    lat, lon = self.snapped_delivery_points[delivery_idx][:2]
                    route_points.append((lat, lon))
            
            route_points.append(self.warehouse_location)
            
            detailed_route = self._calculate_detailed_route(route_points)
            all_routes[driver_id] = detailed_route
            print(f"Driver {driver_id}: Route with {len(assignment.delivery_indices)} deliveries, {len(detailed_route)} detailed points")
        
        return all_routes

    def _calculate_detailed_route(self, route_points):
        if len(route_points) < 2:
            return route_points
        
        detailed_points = []
        
        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]
            
            try:
                start_node = ox.nearest_nodes(self.G, X=start_point[1], Y=start_point[0])
                end_node = ox.nearest_nodes(self.G, X=end_point[1], Y=end_point[0])
                
                path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
                
                segment_points = []
                for node in path:
                    if 'y' in self.G.nodes[node] and 'x' in self.G.nodes[node]:
                        lat, lon = self.G.nodes[node]['y'], self.G.nodes[node]['x']
                        segment_points.append((lat, lon))
                
                if i == 0:
                    detailed_points.extend(segment_points)
                else:
                    detailed_points.extend(segment_points[1:])
                    
            except Exception as e:
                if i == 0:
                    detailed_points.extend([start_point, end_point])
                else:
                    detailed_points.append(end_point)
        
        return detailed_points

    def _point_belongs_to_route(self, point, route_points, threshold=100):
        if not route_points or len(route_points) < 2:
            return False
        
        for i in range(len(route_points) - 1):
            start = route_points[i]
            end = route_points[i + 1]
            
            distance = self._point_to_segment_distance(point, start, end)
            if distance <= threshold:
                return True
        
        return False

    def _route_intersects_disruption(self, route_points, disruption_location, disruption_radius):
        if not route_points or len(route_points) < 2:
            return False
        
        for i in range(len(route_points) - 1):
            start = route_points[i]
            end = route_points[i + 1]
            
            if self._segment_near_disruption(start, end, type('obj', (object,), {
                'location': disruption_location,
                'affected_area_radius': disruption_radius
            })()):
                return True
        
        return False
