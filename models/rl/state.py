import math
from typing import List, Dict, Tuple

import numpy as np


class DeliverySystemState:
    """
    Represents the complete state of the delivery system for RL algorithms.
    Encapsulates drivers, deliveries, disruptions, and simulation state.
    """

    def __init__(self, drivers, deliveries, disruptions, simulation_time, graph, warehouse_location,
                 driver_positions=None, driver_assignments=None, driver_routes=None):
        self.drivers = drivers
        self.deliveries = deliveries
        self.disruptions = disruptions
        self.simulation_time = simulation_time
        self.graph = graph
        self.warehouse_location = warehouse_location

        self.driver_positions = driver_positions if driver_positions is not None else self._get_driver_positions()
        self.driver_assignments = driver_assignments if driver_assignments is not None else self._get_driver_assignments()
        self.driver_capacities = self._get_driver_capacities()
        self.disruption_areas = self._get_disruption_areas()
        self.driver_routes = driver_routes if driver_routes is not None else {}

    def _get_driver_positions(self) -> Dict[int, Tuple[float, float]]:
        """Extract current positions of all drivers"""
        positions = {}
        for driver in self.drivers:
            if hasattr(driver, 'current_position'):
                positions[driver.id] = driver.current_position
        return positions

    def _get_driver_assignments(self) -> Dict[int, List[int]]:
        """Get current delivery assignments for each driver"""
        assignments = {}
        for driver in self.drivers:
            if hasattr(driver, 'assigned_deliveries'):
                assignments[driver.id] = driver.assigned_deliveries
            elif hasattr(driver, 'delivery_indices'):
                assignments[driver.id] = driver.delivery_indices
        return assignments

    def _get_driver_capacities(self) -> Dict[int, Tuple[float, float]]:
        """Get remaining capacity (weight, volume) for each driver"""
        capacities = {}
        for driver in self.drivers:
            if hasattr(driver, 'weight_capacity') and hasattr(driver, 'volume_capacity'):
                remaining_weight = driver.weight_capacity - getattr(driver, 'current_weight', 0)
                remaining_volume = driver.volume_capacity - getattr(driver, 'current_volume', 0)
                capacities[driver.id] = (remaining_weight, remaining_volume)
        return capacities

    def _get_disruption_areas(self) -> List[Dict]:
        """Extract locations and areas affected by disruptions"""
        areas = []
        for disruption in self.disruptions:
            areas.append({
                'id': disruption.id,
                'type': disruption.type.value,
                'location': disruption.location,
                'radius': disruption.affected_area_radius,
                'severity': disruption.severity
            })
        return areas

    def encode_for_rl(self) -> np.ndarray:
        """
        Transform the complex state into a fixed-size vector for RL algorithms.
        Enhanced representation with richer features for better learning.
        """
        try:
            # Fixed sizes for consistent dimensions
            MAX_DRIVERS = 10
            MAX_DISRUPTIONS = 5

            # Global state features
            normalized_time = (self.simulation_time - 8 * 3600) / (8 * 3600)  # 8am to 4pm
            global_features = [
                normalized_time,
                len(self.disruptions) / MAX_DISRUPTIONS,
                len(self.deliveries) / 100
            ]

            # Enhanced driver features (8 per driver)
            driver_features = []
            for i in range(MAX_DRIVERS):
                if i < len(self.drivers):
                    driver = self.drivers[i]
                    driver_id = driver.id

                    # Position and distances
                    position = self.driver_positions.get(driver_id, self.warehouse_location)
                    warehouse_distance = self._calculate_distance(position, self.warehouse_location)
                    norm_warehouse_distance = min(warehouse_distance / 10000, 1.0)

                    # Next delivery distance
                    next_delivery_distance = 1.0
                    next_delivery_id = None
                    assigned_deliveries = self.driver_assignments.get(driver_id, [])
                    if assigned_deliveries:
                        next_delivery_id = assigned_deliveries[0]
                        if next_delivery_id < len(self.deliveries):
                            delivery_position = self.deliveries[next_delivery_id][:2]
                            next_delivery_distance = self._calculate_distance(position, delivery_position)
                            next_delivery_distance = min(next_delivery_distance / 10000, 1.0)

                    # Capacity utilization
                    weight_utilization = 0.0
                    volume_utilization = 0.0
                    if driver_id in self.driver_capacities:
                        try:
                            weight_capacity, volume_capacity = self.driver_capacities[driver_id]
                            if hasattr(driver, 'weight_capacity') and driver.weight_capacity > 0:
                                weight_utilization = 1.0 - (weight_capacity / driver.weight_capacity)
                            if hasattr(driver, 'volume_capacity') and driver.volume_capacity > 0:
                                volume_utilization = 1.0 - (volume_capacity / driver.volume_capacity)
                        except:
                            pass

                    # Route properties
                    route_density = 0.0
                    route_complexity = 0.0
                    if driver_id in self.driver_routes and 'points' in self.driver_routes[driver_id]:
                        route_points = self.driver_routes[driver_id]['points']
                        if len(route_points) > 2:
                            try:
                                total_route_distance = sum(
                                    self._calculate_distance(route_points[i], route_points[i + 1])
                                    for i in range(len(route_points) - 1))
                                route_density = min(len(route_points) / (total_route_distance / 1000 + 0.001), 1.0)

                                if len(route_points) > 3:
                                    # Direction changes as proxy for complexity
                                    direction_changes = 0
                                    for j in range(1, len(route_points) - 1):
                                        v1 = (route_points[j][0] - route_points[j - 1][0],
                                              route_points[j][1] - route_points[j - 1][1])
                                        v2 = (route_points[j + 1][0] - route_points[j][0],
                                              route_points[j + 1][1] - route_points[j][1])

                                        # Simple check for significant direction change
                                        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                                        magnitude1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
                                        magnitude2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

                                        if magnitude1 > 0 and magnitude2 > 0:
                                            cos_angle = dot_product / (magnitude1 * magnitude2)
                                            if cos_angle < 0.7:  # Roughly 45 degrees
                                                direction_changes += 1

                                    route_complexity = min(direction_changes / 10, 1.0)
                            except:
                                pass

                    driver_features.extend([
                        norm_warehouse_distance,  # Distance to warehouse
                        next_delivery_distance,  # Distance to next delivery
                        len(assigned_deliveries) / 20,  # Normalized delivery count
                        weight_utilization,  # Weight capacity utilization
                        volume_utilization,  # Volume capacity utilization
                        route_density,  # Route density metric
                        route_complexity,  # Route complexity metric
                        1.0 if next_delivery_id is not None else 0.0  # Has assignment flag
                    ])
                else:
                    driver_features.extend([0.0] * 8)  # Padding for non-existent drivers

            # Enhanced disruption features (10 per disruption)
            disruption_features = []
            for i in range(MAX_DISRUPTIONS):
                if i < len(self.disruptions):
                    disruption = self.disruptions[i]

                    # One-hot encoding of type
                    is_traffic = 0.0
                    is_road = 0.0
                    is_recipient = 0.0

                    try:
                        disruption_type = disruption.type.value
                        is_traffic = 1.0 if disruption_type == 'traffic_jam' else 0.0
                        is_road = 1.0 if disruption_type == 'road_closure' else 0.0
                        is_recipient = 1.0 if disruption_type == 'recipient_unavailable' else 0.0
                    except:
                        pass

                    # Basic disruption properties
                    severity = getattr(disruption, 'severity', 0.5)
                    radius = getattr(disruption, 'affected_area_radius', 0.0) / 1000

                    # Time-related features
                    normalized_duration = min(getattr(disruption, 'duration', 3600) / 7200,
                                              1.0)  # Normalized to 2 hours
                    time_active = 0.0
                    time_remaining = 1.0
                    if hasattr(disruption, '_activationTime') and disruption._activationTime is not None:
                        duration = getattr(disruption, 'duration', 3600)
                        if duration > 0:
                            elapsed = self.simulation_time - disruption._activationTime
                            time_active = min(elapsed / duration, 1.0)
                            time_remaining = max(0, 1.0 - time_active)

                    # Closest drivers and deliveries
                    min_distance_to_driver = 1.0
                    for driver_id, pos in self.driver_positions.items():
                        distance = self._calculate_distance(pos, disruption.location)
                        normalized_distance = min(distance / (disruption.affected_area_radius * 3), 1.0)
                        min_distance_to_driver = min(min_distance_to_driver, normalized_distance)

                    min_distance_to_delivery = 1.0
                    for delivery in self.deliveries:
                        delivery_position = delivery[:2]
                        distance = self._calculate_distance(delivery_position, disruption.location)
                        normalized_distance = min(distance / (disruption.affected_area_radius * 3), 1.0)
                        min_distance_to_delivery = min(min_distance_to_delivery, normalized_distance)

                    # Road network density (approximation)
                    road_density = 0.5  # Default if can't calculate
                    try:
                        if self.graph:
                            nodes_within_radius = 0
                            for node, data in self.graph.nodes(data=True):
                                if 'y' in data and 'x' in data:
                                    node_location = (data['y'], data['x'])
                                    distance = self._calculate_distance(disruption.location, node_location)
                                    if distance <= disruption.affected_area_radius:
                                        nodes_within_radius += 1

                            area = 3.14159 * (disruption.affected_area_radius / 1000) ** 2  # km²
                            density = min(nodes_within_radius / max(area, 0.01), 100) / 100  # Normalize 0-1
                            road_density = density
                    except:
                        pass

                    disruption_features.extend([
                        is_traffic,  # Traffic jam flag
                        is_road,  # Road closure flag
                        is_recipient,  # Recipient unavailable flag
                        severity,  # Disruption severity
                        radius,  # Normalized radius in km
                        normalized_duration,  # Normalized duration
                        time_active,  # Fraction of time active
                        time_remaining,  # Fraction of time remaining
                        min_distance_to_driver,  # Normalized distance to closest driver
                        min_distance_to_delivery,  # Normalized distance to closest delivery
                    ])
                else:
                    disruption_features.extend([0.0] * 10)  # Padding for non-existent disruptions

            # Combine all features
            state_vector = np.array(global_features + driver_features + disruption_features, dtype=np.float32)

            # Calculate expected size for verification
            expected_size = (
                    len(global_features) +  # 3 global features
                    MAX_DRIVERS * 8 +  # 8 features per driver
                    MAX_DISRUPTIONS * 10  # 10 features per disruption
            )
            # 3 + 80 + 50 = 133

            # Debug output for state size mismatches
            if len(state_vector) != expected_size:
                print(f"Warning: State vector size {len(state_vector)} doesn't match expected size {expected_size}")
                print(
                    f"Global: {len(global_features)}, Driver: {len(driver_features)}, Disruption: {len(disruption_features)}")

            return state_vector

        except Exception as e:
            print(f"Error in encode_for_rl: {e}")
            import traceback
            traceback.print_exc()
            # Return zero vector of expected size as fallback
            expected_size = 3 + (10 * 8) + (5 * 10)  # 133
            return np.zeros(expected_size, dtype=np.float32)

    def _calculate_route_complexity(self, route_points):
        """Calculate route complexity based on direction changes"""
        if len(route_points) < 3:
            return 0.0

        direction_changes = 0
        for i in range(1, len(route_points) - 1):
            v1 = (route_points[i][0] - route_points[i - 1][0], route_points[i][1] - route_points[i - 1][1])
            v2 = (route_points[i + 1][0] - route_points[i][0], route_points[i + 1][1] - route_points[i][1])

            # Normalize vectors
            v1_mag = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
            v2_mag = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

            if v1_mag > 0 and v2_mag > 0:
                v1_norm = (v1[0] / v1_mag, v1[1] / v1_mag)
                v2_norm = (v2[0] / v2_mag, v2[1] / v2_mag)

                # Dot product to get cosine of angle
                dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]

                # If angle is significant (cos < 0.9), count as direction change
                if dot_product < 0.9:
                    direction_changes += 1

        return direction_changes

    def _calculate_road_density(self, location, radius):
        """Calculate road density around a location"""
        if not self.graph:
            return 0.5

        try:
            nodes_within_radius = 0
            edges_within_radius = 0

            for node, data in self.graph.nodes(data=True):
                if 'y' in data and 'x' in data:
                    node_location = (data['y'], data['x'])
                    distance = self._calculate_distance(location, node_location)

                    if distance <= radius:
                        nodes_within_radius += 1

                        for neighbor in self.graph.neighbors(node):
                            if (node, neighbor) not in edges_within_radius:
                                edges_within_radius += 1

            area = math.pi * (radius / 1000) ** 2
            density = edges_within_radius / max(area, 0.01)

            return min(density / 100, 1.0)
        except Exception:
            return 0.5

    def _calculate_distance(self, pos1, pos2):
        """Calculate Haversine distance between two lat/lon points in meters"""
        import math

        lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
        lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000

        return c * r
