from typing import List, Dict, Tuple

import numpy as np


class DeliverySystemState:
    """
    Represents the complete state of the delivery system for RL algorithms.
    Encapsulates drivers, deliveries, disruptions, and simulation state.
    """

    def __init__(self, drivers, deliveries, disruptions, simulation_time, graph, warehouse_location,
                 driver_positions=None, driver_assignments=None):
        self.drivers = drivers  # List of driver objects
        self.deliveries = deliveries  # List of delivery objects or indices
        self.disruptions = disruptions  # List of active disruptions
        self.simulation_time = simulation_time  # Current simulation time in seconds
        self.graph = graph  # Road network graph
        self.warehouse_location = warehouse_location  # Location of the warehouse (lat, lon)

        # Derived data
        self.driver_positions = driver_positions if driver_positions is not None else self._get_driver_positions()
        self.driver_assignments = driver_assignments if driver_assignments is not None else self._get_driver_assignments()
        self.driver_capacities = self._get_driver_capacities()
        self.disruption_areas = self._get_disruption_areas()

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
        Returns a normalized numpy array representing the state.
        """
        try:
            # We'll create a fixed-size representation with:
            # 1. Global state features (simulation time, # of disruptions)
            # 2. Features for each driver (up to a max number)
            # 3. Features for each disruption (up to a max number)

            MAX_DRIVERS = 10
            MAX_DISRUPTIONS = 5

            # Normalize time within a workday (assuming 8-hour workday)
            normalized_time = (self.simulation_time - 8 * 3600) / (8 * 3600)  # 8am to 4pm

            # Global features
            global_features = [
                normalized_time,
                len(self.disruptions) / MAX_DISRUPTIONS,
                len(self.deliveries) / 100  # Normalize by assuming max 100 deliveries
            ]

            # Driver features - with error checking
            driver_features = []

            try:
                for i in range(MAX_DRIVERS):
                    if i < len(self.drivers):
                        driver = self.drivers[i]
                        driver_id = driver.id

                        # Position relative to warehouse
                        if driver_id in self.driver_positions:
                            pos = self.driver_positions[driver_id]
                            try:
                                warehouse_distance = self._calculate_distance(pos, self.warehouse_location)
                                # Normalize by assuming max distance of 10 km
                                normalized_distance = min(warehouse_distance / 10000, 1.0)
                            except:
                                normalized_distance = 0.5  # Default if calculation fails
                        else:
                            normalized_distance = 0.0

                        # Remaining assignments
                        remaining_deliveries = len(self.driver_assignments.get(driver_id, [])) / 20

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
                                pass  # Use defaults if calculation fails

                        driver_features.extend([
                            normalized_distance,
                            remaining_deliveries,
                            weight_utilization,
                            volume_utilization
                        ])
                    else:
                        # Padding for non-existent drivers
                        driver_features.extend([0.0, 0.0, 0.0, 0.0])
            except Exception as e:
                print(f"Error encoding driver features: {e}")
                # Fill with zeros if error occurs
                driver_features = [0.0] * (MAX_DRIVERS * 4)

            # Disruption features - with error checking
            disruption_features = []

            try:
                for i in range(MAX_DISRUPTIONS):
                    if i < len(self.disruptions):
                        disruption = self.disruptions[i]

                        # Normalized disruption time remaining
                        time_remaining = (disruption.start_time + disruption.duration - self.simulation_time)
                        max_duration = 4 * 3600  # Assume max duration of 4 hours
                        normalized_time = max(0, min(time_remaining / max_duration, 1.0))

                        # Safely check disruption type
                        is_traffic = 0.0
                        is_road = 0.0
                        is_vehicle = 0.0
                        is_recipient = 0.0

                        try:
                            disruption_type = disruption.type.value
                            is_traffic = 1.0 if disruption_type == 'traffic_jam' else 0.0
                            is_road = 1.0 if disruption_type == 'road_closure' else 0.0
                            is_vehicle = 1.0 if disruption_type == 'vehicle_breakdown' else 0.0
                            is_recipient = 1.0 if disruption_type == 'recipient_unavailable' else 0.0
                        except:
                            pass  # Use defaults if type access fails

                        disruption_features.extend([
                            is_traffic,
                            is_road,
                            is_vehicle,
                            is_recipient,
                            getattr(disruption, 'severity', 0.5),
                            normalized_time,
                            getattr(disruption, 'affected_area_radius', 0.0) / 1000
                            # Normalize by assuming max radius of 1 km
                        ])
                    else:
                        # Padding for non-existent disruptions
                        disruption_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            except Exception as e:
                print(f"Error encoding disruption features: {e}")
                # Fill with zeros if error occurs
                disruption_features = [0.0] * (MAX_DISRUPTIONS * 7)

            # Combine all features
            return np.array(global_features + driver_features + disruption_features, dtype=np.float32)

        except Exception as e:
            print(f"Error in encode_for_rl: {e}")
            # Return a zero-filled array of the expected size if error occurs
            expected_size = 3 + (10 * 4) + (5 * 7)  # globals + drivers + disruptions
            return np.zeros(expected_size, dtype=np.float32)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Haversine distance between two lat/lon points in meters"""
        import math

        # Convert to radians
        lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
        lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters

        return c * r
