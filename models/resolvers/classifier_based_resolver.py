import math
import os
import time
from typing import List, Dict, Set, Optional, Tuple, Any

import joblib
import networkx as nx
import osmnx as ox
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import DisruptionAction, RerouteAction, RecipientUnavailableAction, ActionType
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.state import DeliverySystemState


class FeatureExtractor:
    """Extracts features from state and disruption for the classifier model."""

    def extract_features(self, state: DeliverySystemState, disruption: Disruption) -> np.ndarray:
        """
        Convert a state and disruption into a feature vector for the classifier.

        Features include:
        - Disruption type (one-hot encoded)
        - Disruption severity and radius
        - Time of day features (hour, is_rush_hour)
        - Driver metrics (count, proximity, capacity utilization)
        - Delivery metrics (count, affected deliveries)
        - Road network metrics (network density around disruption)
        """
        features = []

        # Disruption type (one-hot encoded)
        disruption_type_features = [0, 0, 0]  # traffic_jam, recipient_unavailable, road_closure
        if disruption.type == DisruptionType.TRAFFIC_JAM:
            disruption_type_features[0] = 1
        elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            disruption_type_features[1] = 1
        elif disruption.type == DisruptionType.ROAD_CLOSURE:
            disruption_type_features[2] = 1
        features.extend(disruption_type_features)

        # Disruption properties
        features.append(disruption.severity)
        features.append(disruption.affected_area_radius)
        features.append(disruption.duration / 3600.0)  # Normalized to hours

        # Time of day features
        hours = (state.simulation_time % (24 * 3600)) / 3600
        features.append(hours / 24.0)  # Normalized hour of day (0-1)

        # Peak hour indicator (7-9 AM and 4-6 PM)
        is_morning_rush = 7 <= hours <= 9
        is_evening_rush = 16 <= hours <= 18
        features.append(1.0 if is_morning_rush or is_evening_rush else 0.0)

        # Driver metrics
        affected_drivers = self._get_affected_drivers(disruption, state)
        features.append(len(affected_drivers) / max(len(state.drivers), 1))  # Ratio of affected drivers

        # Get closest driver metrics
        closest_driver_id, closest_distance = self._get_closest_driver(disruption, state)
        if closest_driver_id:
            # Distance to disruption (normalized by radius)
            features.append(closest_distance / max(disruption.affected_area_radius, 1.0))

            # Driver capacity metrics
            driver_capacity = self._get_driver_capacity_metrics(closest_driver_id, state)
            features.extend(driver_capacity)

            # Driver workload
            remaining_deliveries = len(state.driver_assignments.get(closest_driver_id, []))
            features.append(remaining_deliveries / max(len(state.deliveries), 1))  # Normalized workload
        else:
            # Default values if no closest driver
            features.extend([1.0, 0.5, 0.5, 0.0])

        # Delivery metrics
        total_deliveries = len(state.deliveries)
        features.append(total_deliveries / 50.0)  # Normalized by expected max

        # Count of deliveries affected by this disruption
        affected_deliveries = self._count_affected_deliveries(disruption, state)
        features.append(affected_deliveries / max(total_deliveries, 1))  # Ratio of affected deliveries

        # Unique values for each type of disruption
        if disruption.type == DisruptionType.TRAFFIC_JAM:
            # For traffic jams, add road network density around disruption
            features.append(self._estimate_network_density(disruption, state))
            features.extend([0, 0])  # Placeholder for other disruption types
        elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            # For recipient unavailable, add recipient priority
            features.append(0)  # Placeholder for traffic jam
            features.append(self._estimate_delivery_priority(disruption, state))
            features.append(0)  # Placeholder for road closure
        elif disruption.type == DisruptionType.ROAD_CLOSURE:
            # For road closures, add criticality metric
            features.append(0)  # Placeholder for traffic jam
            features.append(0)  # Placeholder for recipient unavailable
            features.append(self._estimate_road_criticality(disruption, state))
        else:
            features.extend([0, 0, 0])  # Default placeholders

        return np.array(features, dtype=np.float32)

    def _get_affected_drivers(self, disruption: Disruption, state: DeliverySystemState) -> Set[int]:
        """Identify drivers affected by the disruption."""
        affected_drivers = set()

        # For recipient unavailable, find driver assigned to that delivery
        if disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            if "delivery_point_index" in disruption.metadata:
                delivery_idx = disruption.metadata["delivery_point_index"]
                for driver_id, assignments in state.driver_assignments.items():
                    if delivery_idx in assignments:
                        affected_drivers.add(driver_id)

        # Check for drivers whose position or route is near the disruption
        for driver_id, position in state.driver_positions.items():
            distance = self._calculate_distance(position, disruption.location)
            if distance <= disruption.affected_area_radius * 2:  # Wider radius for potential effect
                affected_drivers.add(driver_id)

        # Check for drivers whose deliveries are near the disruption
        for driver_id, assignments in state.driver_assignments.items():
            for delivery_idx in assignments:
                if delivery_idx < len(state.deliveries):
                    delivery = state.deliveries[delivery_idx]
                    # Use coordinates attribute instead of indexing
                    lat, lon = delivery.coordinates

                    distance = self._calculate_distance((lat, lon), disruption.location)
                    if distance <= disruption.affected_area_radius * 2:
                        affected_drivers.add(driver_id)
                        break

        return affected_drivers

    def _get_closest_driver(self, disruption: Disruption, state: DeliverySystemState) -> Tuple[Optional[int], float]:
        """Find the closest driver to the disruption."""
        closest_distance = float('inf')
        closest_driver_id = None

        for driver_id, position in state.driver_positions.items():
            distance = self._calculate_distance(position, disruption.location)
            if distance < closest_distance:
                closest_distance = distance
                closest_driver_id = driver_id

        return closest_driver_id, closest_distance if closest_driver_id is not None else 0

    def _get_driver_capacity_metrics(self, driver_id: int, state: DeliverySystemState) -> List[float]:
        """Get capacity utilization metrics for a driver."""
        if driver_id not in state.driver_capacities:
            return [0.5, 0.5]  # Default values

        weight_remaining, volume_remaining = state.driver_capacities[driver_id]

        # Find the driver object to get total capacity
        for driver in state.drivers:
            if driver.id == driver_id:
                weight_ratio = weight_remaining / max(driver.weight_capacity, 0.1)
                volume_ratio = volume_remaining / max(driver.volume_capacity, 0.001)
                return [weight_ratio, volume_ratio]

        return [0.5, 0.5]  # Default if driver not found

    def _count_affected_deliveries(self, disruption: Disruption, state: DeliverySystemState) -> int:
        """Count deliveries affected by the disruption."""
        affected_count = 0

        for idx, delivery in enumerate(state.deliveries):
            # Use coordinates attribute instead of indexing
            lat, lon = delivery.coordinates
            distance = self._calculate_distance((lat, lon), disruption.location)

            if distance <= disruption.affected_area_radius:
                affected_count += 1

        return affected_count

    def _estimate_network_density(self, disruption: Disruption, state: DeliverySystemState) -> float:
        """Estimate the road network density around the disruption."""
        if not state.graph:
            return 0.5  # Default value

        # Simple implementation - could be improved with actual network analysis
        try:
            # Count nearby nodes as proxy for network density
            nearby_nodes = 0
            total_nodes = len(state.graph.nodes)

            for node, data in state.graph.nodes(data=True):
                if 'y' in data and 'x' in data:
                    node_pos = (data['y'], data['x'])
                    distance = self._calculate_distance(node_pos, disruption.location)

                    if distance <= disruption.affected_area_radius * 3:  # Check a wider area
                        nearby_nodes += 1

            # Normalize by expected maximum (arbitrary value)
            return min(nearby_nodes / 100.0, 1.0)
        except Exception:
            return 0.5  # Default on error

    def _estimate_delivery_priority(self, disruption: Disruption, state: DeliverySystemState) -> float:
        """Estimate the priority of an affected delivery."""
        # For this implementation, we use a simple heuristic
        # In a real system, you'd have actual delivery priorities
        if "delivery_point_index" in disruption.metadata:
            delivery_idx = disruption.metadata["delivery_point_index"]

            # Simple priority heuristic - deliveries scheduled earlier are higher priority
            for driver_id, assignments in state.driver_assignments.items():
                if delivery_idx in assignments:
                    position = assignments.index(delivery_idx)
                    return 1.0 - (position / max(len(assignments), 1))

        return 0.5  # Default priority

    def _estimate_road_criticality(self, disruption: Disruption, state: DeliverySystemState) -> float:
        """Estimate how critical a road is based on number of planned routes through it."""
        # Count how many drivers have routes near this disruption
        affected_routes = 0

        for driver_id, route_data in state.driver_routes.items():
            if 'points' in route_data:
                route_points = route_data['points']

                for point in route_points:
                    distance = self._calculate_distance(point, disruption.location)
                    if distance <= disruption.affected_area_radius:
                        affected_routes += 1
                        break

        # Normalize by number of drivers
        return min(affected_routes / max(len(state.drivers), 1), 1.0)

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points in meters."""
        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters

        return c * r


class ClassifierModel:
    """Machine learning model for predicting optimal disruption resolution actions."""

    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.is_trained = False

        # Initialize model with better hyperparameters
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',  # Important for imbalanced classes
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, features, labels):
        """Train the classifier model with comprehensive logging."""
        from sklearn.model_selection import train_test_split

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")

        # Count class distribution
        from collections import Counter
        train_counts = Counter(y_train)
        val_counts = Counter(y_val)

        print("Training class distribution:")
        for label, count in sorted(train_counts.items()):
            action_name = "No Action" if label == -1 else f"Action Type {label}"
            print(f"  {action_name}: {count} ({count / len(y_train) * 100:.1f}%)")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        training_time = time.time() - start_time

        # Evaluate on validation set
        from sklearn.metrics import classification_report, accuracy_score
        val_preds = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)

        print(f"Model trained in {training_time:.2f} seconds")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("\nDetailed validation performance:")
        print(classification_report(y_val, val_preds))

        # Print feature importances for Random Forest
        if self.model_type == "random_forest":
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print("\nTop 10 most important features:")
            for i, idx in enumerate(indices[:10]):
                print(f"{i + 1}. Feature {idx}: {importances[idx]:.4f}")

    def predict(self, features):
        """Predict action type from features."""
        if not self.is_trained:
            raise ValueError("Model has not been trained")

        return self.model.predict(features)

    def predict_proba(self, features):
        """Predict action probabilities from features."""
        if not self.is_trained:
            raise ValueError("Model has not been trained")

        return self.model.predict_proba(features)

    def save(self, file_path: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path: str):
        """Load a trained model from disk."""
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"Model loaded from {file_path}")


class ActionGenerator:
    """Generates concrete actions based on classifier predictions."""

    def __init__(self, rule_based_resolver: RuleBasedResolver):
        self.rule_based_resolver = rule_based_resolver

    def generate_action(self, action_type: int, state: DeliverySystemState, disruption: Disruption) -> Optional[
        DisruptionAction]:
        """Generate a concrete action based on the predicted action type."""
        if action_type == -1:  # No action needed
            return None

        if action_type == ActionType.REROUTE.value:
            return self._generate_reroute_action(state, disruption)
        elif action_type == ActionType.RECIPIENT_UNAVAILABLE.value:
            return self._generate_recipient_unavailable_action(state, disruption)
        else:
            # Unknown action type - fall back to rule-based resolver
            actions = self.rule_based_resolver.on_disruption_detected(disruption, state)
            return actions[0] if actions else None

    def _generate_reroute_action(self, state: DeliverySystemState, disruption: Disruption) -> Optional[RerouteAction]:
        """Generate a reroute action for the given state and disruption."""
        affected_drivers = self._get_affected_drivers(disruption, state)
        if not affected_drivers:
            return None

        # Choose the most appropriate driver to reroute
        driver_id = self._select_best_driver_for_reroute(affected_drivers, state, disruption)
        if not driver_id:
            return None

        # Create a focused state with just this driver for detailed route planning
        driver_state = self._create_driver_state(state, driver_id)

        # Use rule-based resolver for the actual route calculation (complex logic)
        actions = self.rule_based_resolver.resolve_disruptions(driver_state, [disruption])

        # Find and return the reroute action
        for action in actions:
            if isinstance(action, RerouteAction) and action.driver_id == driver_id:
                return action

        return None

    def _generate_recipient_unavailable_action(self, state: DeliverySystemState, disruption: Disruption) -> Optional[
        RecipientUnavailableAction]:
        """Generate recipient unavailable action for the given state and disruption."""
        if disruption.type != DisruptionType.RECIPIENT_UNAVAILABLE:
            return None

        # Find the affected delivery point
        delivery_idx = None
        if "delivery_point_index" in disruption.metadata:
            delivery_idx = disruption.metadata["delivery_point_index"]
        else:
            # Try to find a delivery point near the disruption
            for idx, delivery in enumerate(state.deliveries):
                lat, lon = delivery[0:2]
                distance = self._calculate_distance((lat, lon), disruption.location)
                if distance <= disruption.affected_area_radius:
                    delivery_idx = idx
                    break

        if delivery_idx is None:
            return None

        # Find the driver assigned to this delivery
        driver_id = None
        for d_id, assignments in state.driver_assignments.items():
            if delivery_idx in assignments:
                driver_id = d_id
                break

        if driver_id is None:
            return None

        # Create the action
        return RecipientUnavailableAction(
            driver_id=driver_id,
            delivery_index=delivery_idx,
            disruption_id=disruption.id,
            duration=disruption.duration
        )

    def _get_affected_drivers(self, disruption: Disruption, state: DeliverySystemState) -> List[int]:
        """Get drivers affected by the disruption."""
        affected_drivers = []

        if disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
            if "delivery_point_index" in disruption.metadata:
                delivery_idx = disruption.metadata["delivery_point_index"]
                for driver_id, assignments in state.driver_assignments.items():
                    if delivery_idx in assignments:
                        affected_drivers.append(driver_id)
                        break

        # Consider drivers near the disruption
        effective_radius = disruption.affected_area_radius * 2
        for driver_id, position in state.driver_positions.items():
            distance = self._calculate_distance(position, disruption.location)
            if distance <= effective_radius:
                affected_drivers.append(driver_id)
                continue

            # Check if any assigned deliveries are near disruption
            if driver_id in state.driver_assignments:
                for delivery_idx in state.driver_assignments[driver_id]:
                    if delivery_idx < len(state.deliveries):
                        delivery = state.deliveries[delivery_idx]
                        lat, lon = delivery.coordinates

                        distance = self._calculate_distance((lat, lon), disruption.location)
                        if distance <= effective_radius:
                            affected_drivers.append(driver_id)
                            break

        return list(set(affected_drivers))  # Remove duplicates

    def _select_best_driver_for_reroute(self, affected_drivers: List[int],
                                        state: DeliverySystemState,
                                        disruption: Disruption) -> Optional[int]:
        """Select the most appropriate driver to reroute."""
        if not affected_drivers:
            return None

        # Simple implementation: choose the driver closest to the disruption
        min_distance = float('inf')
        closest_driver = None

        for driver_id in affected_drivers:
            if driver_id in state.driver_positions:
                position = state.driver_positions[driver_id]
                distance = self._calculate_distance(position, disruption.location)

                if distance < min_distance:
                    min_distance = distance
                    closest_driver = driver_id

        return closest_driver

    def _create_driver_state(self, state: DeliverySystemState, driver_id: int) -> DeliverySystemState:
        """Create a state focused on a single driver for route planning."""
        # Extract just the driver info
        driver = next((d for d in state.drivers if d.id == driver_id), None)
        if not driver:
            raise ValueError(f"Driver {driver_id} not found in state")

        driver_positions = {driver_id: state.driver_positions.get(driver_id, state.warehouse_location)}
        driver_assignments = {driver_id: state.driver_assignments.get(driver_id, [])}
        driver_routes = {}
        if driver_id in state.driver_routes:
            driver_routes[driver_id] = state.driver_routes[driver_id]

        # Create a new state with just this driver
        return DeliverySystemState(
            drivers=[driver],
            deliveries=state.deliveries,
            disruptions=state.disruptions,
            simulation_time=state.simulation_time,
            graph=state.graph,
            warehouse_location=state.warehouse_location,
            driver_positions=driver_positions,
            driver_assignments=driver_assignments,
            driver_routes=driver_routes
        )

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points in meters."""
        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters

        return c * r


class SuperchargedResolver(DisruptionResolver):
    """
    An advanced resolver that explores many solution variations to find optimal actions.
    Not suitable for real-time use due to high computation cost, but excellent for training data.
    """

    def __init__(self, graph, warehouse_location, computation_budget=5.0):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.computation_budget = computation_budget
        # Base resolver for initial solution
        self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)

    def should_recalculate(self, state, disruption):
        """Always recalculate for training data generation"""
        return True

    def on_disruption_detected(self, disruption, state):
        """Find optimal actions by exploring many variations"""
        # Start with standard rule-based solution
        base_actions = self.rule_based_resolver.on_disruption_detected(disruption, state)

        best_actions = base_actions
        best_score = self._evaluate_solution(best_actions, disruption, state)

        start_time = time.time()
        variations_explored = 0

        # Explore until time budget is exhausted
        while time.time() - start_time < self.computation_budget:
            # Generate variations based on disruption type
            if disruption.type == DisruptionType.ROAD_CLOSURE or disruption.type == DisruptionType.TRAFFIC_JAM:
                candidate_actions = self._explore_reroute_variations(disruption, state)
            elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                candidate_actions = self._explore_recipient_unavailable_variations(disruption, state)
            else:
                break  # Unknown disruption type

            variations_explored += len(candidate_actions)

            # Evaluate each variation
            for actions in candidate_actions:
                score = self._evaluate_solution(actions, disruption, state)
                if score > best_score:
                    best_score = score
                    best_actions = actions

        print(
            f"Explored {variations_explored} variations in {time.time() - start_time:.2f}s, score improvement: {best_score / max(self._evaluate_solution(base_actions, disruption, state), 0.0001):.2f}x")
        return best_actions

    def _explore_reroute_variations(self, disruption, state):
        """Generate multiple rerouting strategies with different parameters"""
        variations = []

        # Find affected drivers
        affected_drivers = self._get_affected_drivers(disruption, state)

        # Try different avoidance radii
        radius_multipliers = [1.2, 1.5, 2.0, 2.5, 3.0]

        # Try different routing weights
        routing_priorities = ['distance', 'time', 'safety']

        for driver_id in affected_drivers:
            # Basic position data
            position = state.driver_positions.get(driver_id)
            if not position:
                continue

            route_data = state.driver_routes.get(driver_id, {})
            original_route = route_data.get('points', [])
            if not original_route or len(original_route) < 2:
                continue

            # Try different avoidance radii
            for radius_mult in radius_multipliers:
                action = self._create_reroute_with_radius(driver_id, disruption, state, radius_mult)
                if action:
                    variations.append([action])

            # Try different path finding strategies
            for priority in routing_priorities:
                for radius_mult in [1.5, 2.5]:
                    action = self._create_reroute_with_priority(driver_id, disruption, state, priority, radius_mult)
                    if action:
                        variations.append([action])

            # Try multi-waypoint approaches
            for num_waypoints in [1, 2, 3]:
                action = self._create_reroute_with_waypoints(driver_id, disruption, state, num_waypoints)
                if action:
                    variations.append([action])

        return variations

    def _explore_recipient_unavailable_variations(self, disruption, state):
        """Generate variations for handling unavailable recipients"""
        variations = []

        # Find the affected delivery
        delivery_idx = None
        if "delivery_point_index" in disruption.metadata:
            delivery_idx = disruption.metadata["delivery_point_index"]
        else:
            return variations

        # Find driver assigned to this delivery
        driver_id = None
        for d_id, assignments in state.driver_assignments.items():
            if delivery_idx in assignments:
                driver_id = d_id
                break

        if driver_id is None:
            return variations

        # Try different wait durations
        duration_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        for factor in duration_factors:
            action = RecipientUnavailableAction(
                driver_id=driver_id,
                delivery_index=delivery_idx,
                disruption_id=disruption.id,
                duration=int(disruption.duration * factor)
            )
            variations.append([action])

        return variations

    def _create_reroute_with_radius(self, driver_id, disruption, state, radius_multiplier):
        """Create a reroute action with a specific avoidance radius"""
        try:
            # Find driver route data
            route_data = state.driver_routes.get(driver_id, {})
            original_route = route_data.get('points', [])
            if not original_route or len(original_route) < 2:
                return None

            # Find segments that are affected by the disruption
            affected_segment_start = -1
            affected_segment_end = -1

            # Apply the custom radius multiplier
            effective_radius = disruption.affected_area_radius * radius_multiplier

            # Find the affected segments of the route
            for i in range(len(original_route) - 1):
                start_point = original_route[i]
                end_point = original_route[i + 1]

                # Check if segment is close to disruption
                if self._segment_near_disruption(start_point, end_point, disruption, effective_radius):
                    affected_segment_start = i

                    # Find where the affected segment ends
                    for j in range(i + 1, len(original_route) - 1):
                        if not self._segment_near_disruption(original_route[j], original_route[j + 1], disruption,
                                                             effective_radius):
                            affected_segment_end = j + 1
                            break

                    if affected_segment_end == -1:
                        affected_segment_end = len(original_route) - 1

                    break

            # If no affected segment found, return None
            if affected_segment_start == -1:
                return None

            # Get start and end points for the reroute
            start_point = original_route[affected_segment_start]
            end_point = original_route[affected_segment_end]

            # Find a path that avoids the disruption
            detour_points = self._find_path_avoiding_disruption(
                state.graph,
                start_point,
                end_point,
                disruption,
                radius_multiplier
            )

            if not detour_points or len(detour_points) < 2:
                return None

            # Construct a new route with the detour
            new_route = []
            new_route.extend(original_route[:affected_segment_start])

            # Add detour, avoiding duplicates at segment boundaries
            if new_route and detour_points and self._points_equal(new_route[-1], detour_points[0]):
                new_route.extend(detour_points[1:])
            else:
                new_route.extend(detour_points)

            # Add the remainder of the original route
            if new_route and affected_segment_end + 1 < len(original_route):
                if self._points_equal(new_route[-1], original_route[affected_segment_end + 1]):
                    new_route.extend(original_route[affected_segment_end + 2:])
                else:
                    new_route.extend(original_route[affected_segment_end + 1:])

            # Calculate delivery indices in the new route
            delivery_indices = []

            # Get assigned deliveries for this driver
            assignments = state.driver_assignments.get(driver_id, [])

            # For each assigned delivery, find its location in the new route
            for delivery_idx in assignments:
                if delivery_idx < len(state.deliveries):
                    delivery_point = state.deliveries[delivery_idx].coordinates

                    min_distance = float('inf')
                    best_idx = -1

                    # Find the closest point in the new route to this delivery
                    for i, point in enumerate(new_route):
                        distance = self._calculate_distance(point, delivery_point)
                        if distance < min_distance:
                            min_distance = distance
                            best_idx = i

                    if best_idx != -1 and min_distance < 50:  # 50m threshold
                        delivery_indices.append(best_idx)

            # Get index of next delivery if possible
            next_delivery_idx = None
            if assignments:
                next_delivery_idx = assignments[0]

            # Create the reroute action
            return RerouteAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_start + len(detour_points) - 1,
                next_delivery_index=next_delivery_idx,
                delivery_indices=delivery_indices
            )
        except Exception as e:
            print(f"Error in create_reroute_with_radius: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_reroute_with_priority(self, driver_id, disruption, state, priority, radius_multiplier):
        """Create reroute optimizing for different priorities (safety, speed, etc.)"""
        try:
            # Find driver route data
            route_data = state.driver_routes.get(driver_id, {})
            original_route = route_data.get('points', [])
            if not original_route or len(original_route) < 2:
                return None

            # Find segments that are affected by the disruption
            affected_segment_start = -1
            affected_segment_end = -1

            # Apply the custom radius multiplier
            effective_radius = disruption.affected_area_radius * radius_multiplier

            # Find the affected segments of the route
            for i in range(len(original_route) - 1):
                start_point = original_route[i]
                end_point = original_route[i + 1]

                # Check if segment is close to disruption
                if self._segment_near_disruption(start_point, end_point, disruption, effective_radius):
                    affected_segment_start = i

                    # Find where the affected segment ends
                    for j in range(i + 1, len(original_route) - 1):
                        if not self._segment_near_disruption(original_route[j], original_route[j + 1], disruption,
                                                             effective_radius):
                            affected_segment_end = j + 1
                            break

                    if affected_segment_end == -1:
                        affected_segment_end = len(original_route) - 1

                    break

            # If no affected segment found, return None
            if affected_segment_start == -1:
                return None

            # Get start and end points for the reroute
            start_point = original_route[affected_segment_start]
            end_point = original_route[affected_segment_end]

            # Find a path that avoids the disruption with the given priority
            detour_points = self._find_path_with_priority(
                state.graph,
                start_point,
                end_point,
                disruption,
                radius_multiplier,
                priority
            )

            if not detour_points or len(detour_points) < 2:
                return None

            # Construct a new route with the detour
            new_route = []
            new_route.extend(original_route[:affected_segment_start])

            # Add detour, avoiding duplicates at segment boundaries
            if new_route and detour_points and self._points_equal(new_route[-1], detour_points[0]):
                new_route.extend(detour_points[1:])
            else:
                new_route.extend(detour_points)

            # Add the remainder of the original route
            if new_route and affected_segment_end + 1 < len(original_route):
                if self._points_equal(new_route[-1], original_route[affected_segment_end + 1]):
                    new_route.extend(original_route[affected_segment_end + 2:])
                else:
                    new_route.extend(original_route[affected_segment_end + 1:])

            # Calculate delivery indices in the new route
            delivery_indices = []

            # Get assigned deliveries for this driver
            assignments = state.driver_assignments.get(driver_id, [])

            # For each assigned delivery, find its location in the new route
            for delivery_idx in assignments:
                if delivery_idx < len(state.deliveries):
                    delivery_point = state.deliveries[delivery_idx].coordinates

                    min_distance = float('inf')
                    best_idx = -1

                    # Find the closest point in the new route to this delivery
                    for i, point in enumerate(new_route):
                        distance = self._calculate_distance(point, delivery_point)
                        if distance < min_distance:
                            min_distance = distance
                            best_idx = i

                    if best_idx != -1 and min_distance < 50:  # 50m threshold
                        delivery_indices.append(best_idx)

            # Get index of next delivery if possible
            next_delivery_idx = None
            if assignments:
                next_delivery_idx = assignments[0]

            # Create the reroute action
            return RerouteAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_start + len(detour_points) - 1,
                next_delivery_index=next_delivery_idx,
                delivery_indices=delivery_indices
            )
        except Exception as e:
            print(f"Error in create_reroute_with_priority: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_path_with_priority(self, graph, start_point, end_point, disruption, radius_multiplier, priority):
        """Find path with specific priority (distance, time, safety)"""
        if not graph:
            return [start_point, end_point]

        try:
            G_mod = graph.copy()

            # Convert coordinates to float
            start_point = (float(start_point[0]), float(start_point[1]))
            end_point = (float(end_point[0]), float(end_point[1]))

            # Get nearest nodes
            start_node = ox.nearest_nodes(G_mod, X=start_point[1], Y=start_point[0])
            end_node = ox.nearest_nodes(G_mod, X=end_point[1], Y=end_point[0])

            # Apply the custom radius multiplier
            effective_radius = disruption.affected_area_radius * radius_multiplier

            # Find nodes near the disruption
            disruption_nodes = []

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = self._calculate_distance(node_point, disruption.location)
                if distance <= effective_radius:
                    disruption_nodes.append(node)

            # Modify edge weights based on priority
            edge_weight = 'travel_time'  # Default weight

            if priority == 'distance':
                edge_weight = 'length'
                # Make edges near disruption heavily weighted
                weight_multiplier = 5.0
            elif priority == 'time':
                edge_weight = 'travel_time'
                # Time-based weight - moderate penalty
                weight_multiplier = 2.0
            elif priority == 'safety':
                edge_weight = 'travel_time'
                # Safety-oriented - severe penalty for disruption proximity
                weight_multiplier = 10.0

            # Apply weights to edges
            for node in disruption_nodes:
                for neighbor in list(G_mod.neighbors(node)):
                    if G_mod.has_edge(node, neighbor):
                        for edge_key in list(G_mod[node][neighbor].keys()):
                            if edge_weight in G_mod[node][neighbor][edge_key]:
                                original_weight = float(G_mod[node][neighbor][edge_key][edge_weight])
                                G_mod[node][neighbor][edge_key][edge_weight] = original_weight * weight_multiplier

            # Special handling for safety priority: add distance-based penalties
            if priority == 'safety':
                # Add extra penalties based on distance from disruption center
                for u, v, k, data in G_mod.edges(data=True, keys=True):
                    u_data = G_mod.nodes[u]
                    v_data = G_mod.nodes[v]

                    if 'y' in u_data and 'x' in u_data and 'y' in v_data and 'x' in v_data:
                        # Get midpoint of edge
                        mid_lat = (float(u_data['y']) + float(v_data['y'])) / 2
                        mid_lon = (float(u_data['x']) + float(v_data['x'])) / 2

                        # Distance from edge midpoint to disruption
                        distance = self._calculate_distance((mid_lat, mid_lon), disruption.location)

                        # Apply inverse-distance penalty (closer = higher penalty)
                        safety_factor = max(0.1, min(1.0, distance / (effective_radius * 2)))

                        if distance <= effective_radius * 3:
                            safety_penalty = 1.0 / safety_factor

                            if edge_weight in data:
                                data[edge_weight] = float(data[edge_weight]) * safety_penalty

            # Find path using modified graph
            try:
                path = nx.shortest_path(G_mod, start_node, end_node, weight=edge_weight)

                # Convert path to coordinates
                route_points = []
                for node in path:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = float(G_mod.nodes[node]['y'])
                        lon = float(G_mod.nodes[node]['x'])
                        route_points.append((lat, lon))

                # Ensure start and end points are exact
                if route_points:
                    route_points[0] = start_point
                    route_points[-1] = end_point

                return route_points

            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"Path finding error in find_path_with_priority: {e}")

                # Try alternative: use a waypoint approach
                try:
                    # Create waypoint perpendicular to disruption direction
                    disruption_lat, disruption_lon = disruption.location

                    # Vector from disruption to midpoint of start and end
                    mid_lat = (start_point[0] + end_point[0]) / 2
                    mid_lon = (start_point[1] + end_point[1]) / 2

                    # Create a perpendicular vector
                    dx = mid_lon - disruption_lon
                    dy = mid_lat - disruption_lat

                    # Normalize and rotate 90 degrees
                    magnitude = (dx ** 2 + dy ** 2) ** 0.5
                    if magnitude > 0:
                        dx, dy = -dy / magnitude, dx / magnitude
                    else:
                        dx, dy = 1, 0

                    # Create waypoint at safe distance
                    safe_distance = effective_radius * 1.5
                    waypoint_lat = disruption_lat + dy * safe_distance * 0.0001
                    waypoint_lon = disruption_lon + dx * safe_distance * 0.0001

                    # Find nearest node to waypoint
                    waypoint_node = ox.nearest_nodes(G_mod, X=waypoint_lon, Y=waypoint_lat)

                    # Find paths to and from waypoint
                    path1 = nx.shortest_path(G_mod, start_node, waypoint_node, weight=edge_weight)
                    path2 = nx.shortest_path(G_mod, waypoint_node, end_node, weight=edge_weight)

                    # Combine paths
                    combined_path = path1[:-1] + path2

                    # Convert to coordinates
                    route_points = []
                    for node in combined_path:
                        if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                            lat = float(G_mod.nodes[node]['y'])
                            lon = float(G_mod.nodes[node]['x'])
                            route_points.append((lat, lon))

                    # Ensure start and end points are exact
                    if route_points:
                        route_points[0] = start_point
                        route_points[-1] = end_point

                    return route_points

                except Exception as e2:
                    print(f"Waypoint routing failed: {e2}")

        except Exception as e:
            print(f"Error in find_path_with_priority: {e}")
            import traceback
            traceback.print_exc()

        # Fallback: direct path with intermediate points to avoid disruption
        mid_lat = (start_point[0] + end_point[0]) / 2
        mid_lon = (start_point[1] + end_point[1]) / 2

        disruption_lat, disruption_lon = disruption.location
        dx = mid_lon - disruption_lon
        dy = mid_lat - disruption_lat

        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        if magnitude > 0:
            dx, dy = dx / magnitude, dy / magnitude
        else:
            dx, dy = 1, 0

        # Create perpendicular offset
        perpendicular_dx, perpendicular_dy = -dy, dx

        # Create detour point
        detour_distance = effective_radius * 1.5 * 0.00001  # Convert to degrees
        detour_lat = mid_lat + perpendicular_dy * detour_distance
        detour_lon = mid_lon + perpendicular_dx * detour_distance

        return [start_point, (detour_lat, detour_lon), end_point]

    def _create_reroute_with_waypoints(self, driver_id, disruption, state, num_waypoints):
        """Create reroute using strategic waypoints to guide path finding"""
        try:
            # Find driver route data
            route_data = state.driver_routes.get(driver_id, {})
            original_route = route_data.get('points', [])
            if not original_route or len(original_route) < 2:
                return None

            # Find segments that are affected by the disruption
            affected_segment_start = -1
            affected_segment_end = -1

            # Use standard radius for detection
            effective_radius = disruption.affected_area_radius * 1.5

            # Find the affected segments of the route
            for i in range(len(original_route) - 1):
                start_point = original_route[i]
                end_point = original_route[i + 1]

                # Check if segment is close to disruption
                if self._segment_near_disruption(start_point, end_point, disruption, effective_radius):
                    affected_segment_start = i

                    # Find where the affected segment ends
                    for j in range(i + 1, len(original_route) - 1):
                        if not self._segment_near_disruption(original_route[j], original_route[j + 1], disruption,
                                                             effective_radius):
                            affected_segment_end = j + 1
                            break

                    if affected_segment_end == -1:
                        affected_segment_end = len(original_route) - 1

                    break

            # If no affected segment found, return None
            if affected_segment_start == -1:
                return None

            # Get start and end points for the reroute
            start_point = original_route[affected_segment_start]
            end_point = original_route[affected_segment_end]

            # Generate waypoints and find path
            waypoints = self._generate_strategic_waypoints(
                start_point,
                end_point,
                disruption,
                num_waypoints
            )

            if not waypoints or len(waypoints) < 2:
                return None

            # Find path through waypoints
            detour_points = self._find_path_through_waypoints(
                state.graph,
                start_point,
                end_point,
                waypoints,
                disruption
            )

            if not detour_points or len(detour_points) < 2:
                return None

            # Construct a new route with the detour
            new_route = []
            new_route.extend(original_route[:affected_segment_start])

            # Add detour, avoiding duplicates at segment boundaries
            if new_route and detour_points and self._points_equal(new_route[-1], detour_points[0]):
                new_route.extend(detour_points[1:])
            else:
                new_route.extend(detour_points)

            # Add the remainder of the original route
            if new_route and affected_segment_end + 1 < len(original_route):
                if self._points_equal(new_route[-1], original_route[affected_segment_end + 1]):
                    new_route.extend(original_route[affected_segment_end + 2:])
                else:
                    new_route.extend(original_route[affected_segment_end + 1:])

            # Calculate delivery indices in the new route
            delivery_indices = []

            # Get assigned deliveries for this driver
            assignments = state.driver_assignments.get(driver_id, [])

            # For each assigned delivery, find its location in the new route
            for delivery_idx in assignments:
                if delivery_idx < len(state.deliveries):
                    delivery_point = state.deliveries[delivery_idx].coordinates

                    min_distance = float('inf')
                    best_idx = -1

                    # Find the closest point in the new route to this delivery
                    for i, point in enumerate(new_route):
                        distance = self._calculate_distance(point, delivery_point)
                        if distance < min_distance:
                            min_distance = distance
                            best_idx = i

                    if best_idx != -1 and min_distance < 50:  # 50m threshold
                        delivery_indices.append(best_idx)

            # Get index of next delivery if possible
            next_delivery_idx = None
            if assignments:
                next_delivery_idx = assignments[0]

            # Create the reroute action
            return RerouteAction(
                driver_id=driver_id,
                new_route=new_route,
                affected_disruption_id=disruption.id,
                rerouted_segment_start=affected_segment_start,
                rerouted_segment_end=affected_segment_start + len(detour_points) - 1,
                next_delivery_index=next_delivery_idx,
                delivery_indices=delivery_indices
            )
        except Exception as e:
            print(f"Error in create_reroute_with_waypoints: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_strategic_waypoints(self, start_point, end_point, disruption, num_waypoints):
        """Generate strategic waypoints for routing around disruption"""
        try:
            waypoints = []

            # Convert coordinates to float
            start_point = (float(start_point[0]), float(start_point[1]))
            end_point = (float(end_point[0]), float(end_point[1]))
            disruption_location = (float(disruption.location[0]), float(disruption.location[1]))

            # Calculate midpoint between start and end
            mid_lat = (start_point[0] + end_point[0]) / 2
            mid_lon = (start_point[1] + end_point[1]) / 2

            # Vector from disruption to midpoint
            dx = mid_lon - disruption_location[1]
            dy = mid_lat - disruption_location[0]

            # Normalize vector
            magnitude = (dx ** 2 + dy ** 2) ** 0.5
            if magnitude > 0:
                dx, dy = dx / magnitude, dy / magnitude
            else:
                dx, dy = 1, 0

            # Create perpendicular vector for wide detours
            perp_dx, perp_dy = -dy, dx

            # Effective radius for waypoint placement
            safe_distance = disruption.affected_area_radius * 2

            if num_waypoints == 1:
                # Single waypoint - place it away from disruption
                waypoint_lat = mid_lat + dy * safe_distance * 0.00001  # Convert to degrees
                waypoint_lon = mid_lon + dx * safe_distance * 0.00001
                waypoints.append((waypoint_lat, waypoint_lon))

            elif num_waypoints == 2:
                # Two waypoints - place them on either side of a line perpendicular to direct path
                waypoint1_lat = mid_lat + perp_dy * safe_distance * 0.00001
                waypoint1_lon = mid_lon + perp_dx * safe_distance * 0.00001

                waypoint2_lat = mid_lat - perp_dy * safe_distance * 0.00001
                waypoint2_lon = mid_lon - perp_dx * safe_distance * 0.00001

                # Check which side is farther from disruption
                dist1 = self._calculate_distance((waypoint1_lat, waypoint1_lon), disruption_location)
                dist2 = self._calculate_distance((waypoint2_lat, waypoint2_lon), disruption_location)

                if dist1 > dist2:
                    waypoints.append((waypoint1_lat, waypoint1_lon))
                else:
                    waypoints.append((waypoint2_lat, waypoint2_lon))

                # Add a second waypoint along the path to/from the first
                factor = 0.3  # Position 30% from start to first waypoint
                second_wp_lat = start_point[0] + (waypoints[0][0] - start_point[0]) * factor
                second_wp_lon = start_point[1] + (waypoints[0][1] - start_point[1]) * factor
                waypoints.insert(0, (second_wp_lat, second_wp_lon))

            elif num_waypoints >= 3:
                # Three or more waypoints - create a curved detour
                # First, decide which side to detour on
                waypoint_test_lat = mid_lat + perp_dy * safe_distance * 0.00001
                waypoint_test_lon = mid_lon + perp_dx * safe_distance * 0.00001

                dist_test = self._calculate_distance((waypoint_test_lat, waypoint_test_lon), disruption_location)

                # If test point is too close, use the opposite side
                if dist_test < safe_distance:
                    perp_dx, perp_dy = -perp_dx, -perp_dy

                # Create a curved path with multiple waypoints
                for i in range(num_waypoints):
                    # Position along the path (0 to 1)
                    t = (i + 1) / (num_waypoints + 1)

                    # Basic linear interpolation
                    base_lat = start_point[0] * (1 - t) + end_point[0] * t
                    base_lon = start_point[1] * (1 - t) + end_point[1] * t

                    # Add perpendicular offset that varies with position (maximum at middle)
                    offset_factor = 4 * t * (1 - t)  # Parabolic function, max at t=0.5

                    wp_lat = base_lat + perp_dy * safe_distance * offset_factor * 0.00001
                    wp_lon = base_lon + perp_dx * safe_distance * offset_factor * 0.00001

                    waypoints.append((wp_lat, wp_lon))

            return waypoints

        except Exception as e:
            print(f"Error generating waypoints: {e}")
            return []

    def _find_path_through_waypoints(self, graph, start_point, end_point, waypoints, disruption):
        """Find a path that goes through all specified waypoints"""
        if not graph or not waypoints:
            return [start_point, end_point]

        try:
            G_mod = graph.copy()

            # Convert coordinates to float
            start_point = (float(start_point[0]), float(start_point[1]))
            end_point = (float(end_point[0]), float(end_point[1]))
            waypoints = [(float(wp[0]), float(wp[1])) for wp in waypoints]

            # Modify graph to avoid disruption
            self._modify_graph_for_disruption(G_mod, disruption, 1.5)

            # Find nodes for start, end, and waypoints
            try:
                start_node = ox.nearest_nodes(G_mod, X=start_point[1], Y=start_point[0])
                end_node = ox.nearest_nodes(G_mod, X=end_point[1], Y=end_point[0])
                waypoint_nodes = [ox.nearest_nodes(G_mod, X=wp[1], Y=wp[0]) for wp in waypoints]
            except Exception as e:
                print(f"Error finding nodes: {e}")
                return None

            # Find paths between consecutive points
            path_segments = []

            # Start to first waypoint
            try:
                path1 = nx.shortest_path(G_mod, start_node, waypoint_nodes[0], weight='travel_time')
                path_segments.append(path1)
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"No path to first waypoint: {e}")
                return None

            # Between waypoints
            for i in range(len(waypoint_nodes) - 1):
                try:
                    path_segment = nx.shortest_path(
                        G_mod,
                        waypoint_nodes[i],
                        waypoint_nodes[i + 1],
                        weight='travel_time'
                    )
                    path_segments.append(path_segment[1:])  # Skip first node (duplicate)
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    print(f"No path between waypoints {i} and {i + 1}: {e}")
                    return None

            # Last waypoint to end
            try:
                path_last = nx.shortest_path(G_mod, waypoint_nodes[-1], end_node, weight='travel_time')
                path_segments.append(path_last[1:])  # Skip first node (duplicate)
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"No path from last waypoint to end: {e}")
                return None

            # Combine path segments
            combined_path = []
            for segment in path_segments:
                combined_path.extend(segment)

            # Convert to coordinates
            route_points = []
            for node in combined_path:
                if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                    lat = float(G_mod.nodes[node]['y'])
                    lon = float(G_mod.nodes[node]['x'])
                    route_points.append((lat, lon))

            # Ensure start and end points are exact
            if route_points:
                route_points[0] = start_point
                route_points[-1] = end_point

            return route_points

        except Exception as e:
            print(f"Error in find_path_through_waypoints: {e}")
            import traceback
            traceback.print_exc()

        # Fallback - direct path through waypoints
        fallback_path = [start_point]
        fallback_path.extend(waypoints)
        fallback_path.append(end_point)
        return fallback_path

    def _modify_graph_for_disruption(self, graph, disruption, radius_multiplier):
        """Modify graph by increasing travel times around the disruption"""
        if not graph:
            return

        effective_radius = disruption.affected_area_radius * radius_multiplier
        disruption_location = disruption.location

        # Find nodes near the disruption
        disruption_nodes = []

        for node, data in graph.nodes(data=True):
            if 'y' not in data or 'x' not in data:
                continue

            node_point = (data['y'], data['x'])
            distance = self._calculate_distance(node_point, disruption_location)
            if distance <= effective_radius:
                disruption_nodes.append(node)

        # Determine weight multiplier based on disruption type
        if disruption.type == DisruptionType.TRAFFIC_JAM:
            weight_multiplier = 1.0 + disruption.severity * 5.0
        elif disruption.type == DisruptionType.ROAD_CLOSURE:
            weight_multiplier = 100.0
        else:
            weight_multiplier = 3.0

        # Apply weights to edges
        for node in disruption_nodes:
            for neighbor in list(graph.neighbors(node)):
                if graph.has_edge(node, neighbor):
                    for edge_key in list(graph[node][neighbor].keys()):
                        if 'travel_time' in graph[node][neighbor][edge_key]:
                            original_time = float(graph[node][neighbor][edge_key]['travel_time'])
                            graph[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier

    def _find_path_avoiding_disruption(self, graph, start_point, end_point, disruption, radius_multiplier):
        """Enhanced path finding that thoroughly explores routing options"""
        if not graph:
            return [start_point, end_point]

        try:
            # Convert coordinates to float
            start_point = (float(start_point[0]), float(start_point[1]))
            end_point = (float(end_point[0]), float(end_point[1]))

            G_mod = graph.copy()

            # Get nearest nodes
            try:
                start_node = ox.nearest_nodes(G_mod, X=start_point[1], Y=start_point[0])
                end_node = ox.nearest_nodes(G_mod, X=end_point[1], Y=end_point[0])
            except Exception as e:
                print(f"Error finding nearest nodes: {e}")
                return [start_point, end_point]

            # Apply the custom radius multiplier
            effective_radius = disruption.affected_area_radius * radius_multiplier

            # Find nodes near the disruption
            disruption_nodes = []
            affected_edges = 0

            for node, data in G_mod.nodes(data=True):
                if 'y' not in data or 'x' not in data:
                    continue

                node_point = (data['y'], data['x'])
                distance = self._calculate_distance(node_point, disruption.location)
                if distance <= effective_radius:
                    disruption_nodes.append(node)

            # Determine weight multiplier based on disruption type
            if disruption.type == DisruptionType.TRAFFIC_JAM:
                weight_multiplier = 1.0 + (9.0 * disruption.severity)
            elif disruption.type == DisruptionType.ROAD_CLOSURE:
                weight_multiplier = 100.0
            else:
                weight_multiplier = 5.0

            # Apply weights to edges
            for node in disruption_nodes:
                for neighbor in list(G_mod.neighbors(node)):
                    if G_mod.has_edge(node, neighbor):
                        for edge_key in list(G_mod[node][neighbor].keys()):
                            if 'travel_time' in G_mod[node][neighbor][edge_key]:
                                original_time = float(G_mod[node][neighbor][edge_key]['travel_time'])
                                G_mod[node][neighbor][edge_key]['travel_time'] = original_time * weight_multiplier
                                affected_edges += 1

            # Find path using modified graph
            try:
                path = nx.shortest_path(G_mod, start_node, end_node, weight='travel_time')

                # Convert to coordinates
                route_points = []
                for node in path:
                    if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                        lat = float(G_mod.nodes[node]['y'])
                        lon = float(G_mod.nodes[node]['x'])
                        route_points.append((lat, lon))

                # Ensure start and end points are exact
                if route_points:
                    route_points[0] = start_point
                    route_points[-1] = end_point

                return route_points

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Try the waypoint approach
                try:
                    # Create a waypoint perpendicular to the disruption center
                    disruption_lat, disruption_lon = disruption.location

                    # Midpoint between start and end
                    mid_lat = (start_point[0] + end_point[0]) / 2
                    mid_lon = (start_point[1] + end_point[1]) / 2

                    # Vector from disruption to midpoint
                    dx = mid_lon - disruption_lon
                    dy = mid_lat - disruption_lat

                    # Normalize and create perpendicular vector
                    magnitude = (dx ** 2 + dy ** 2) ** 0.5
                    if magnitude > 0:
                        dx, dy = dx / magnitude, dy / magnitude
                        perp_dx, perp_dy = -dy, dx
                    else:
                        dx, dy = 1, 0
                        perp_dx, perp_dy = 0, 1

                    # Create waypoint at a safe distance
                    safe_distance = effective_radius * 1.5

                    # Test which side is better (farther from disruption)
                    wp1_lat = mid_lat + perp_dy * safe_distance * 0.00001
                    wp1_lon = mid_lon + perp_dx * safe_distance * 0.00001

                    wp2_lat = mid_lat - perp_dy * safe_distance * 0.00001
                    wp2_lon = mid_lon - perp_dx * safe_distance * 0.00001

                    dist1 = self._calculate_distance((wp1_lat, wp1_lon), disruption.location)
                    dist2 = self._calculate_distance((wp2_lat, wp2_lon), disruption.location)

                    if dist1 > dist2:
                        waypoint_lat, waypoint_lon = wp1_lat, wp1_lon
                    else:
                        waypoint_lat, waypoint_lon = wp2_lat, wp2_lon

                    # Find nearest node to waypoint
                    waypoint_node = ox.nearest_nodes(G_mod, X=waypoint_lon, Y=waypoint_lat)

                    # Find paths to and from waypoint
                    path1 = nx.shortest_path(G_mod, start_node, waypoint_node, weight='travel_time')
                    path2 = nx.shortest_path(G_mod, waypoint_node, end_node, weight='travel_time')

                    # Combine paths
                    combined_path = path1[:-1] + path2

                    # Convert to coordinates
                    route_points = []
                    for node in combined_path:
                        if 'y' in G_mod.nodes[node] and 'x' in G_mod.nodes[node]:
                            lat = float(G_mod.nodes[node]['y'])
                            lon = float(G_mod.nodes[node]['x'])
                            route_points.append((lat, lon))

                    # Ensure start and end points are exact
                    if route_points:
                        route_points[0] = start_point
                        route_points[-1] = end_point

                    return route_points

                except Exception as e:
                    print(f"Waypoint routing failed: {e}")

        except Exception as e:
            print(f"Error in find_path_avoiding_disruption: {e}")

        # Advanced fallback: Create a smooth curve around the disruption
        try:
            # Direct vector from start to end
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]

            # Get disruption location
            disrupt_lat, disrupt_lon = disruption.location

            # Create intermediate points with perpendicular offset
            steps = 5
            detour_points = [start_point]

            for i in range(1, steps):
                t = i / steps

                # Linear interpolation along direct path
                basic_lat = start_point[0] + dx * t
                basic_lon = start_point[1] + dy * t

                # Vector from basic point to disruption
                to_disrupt_x = disrupt_lat - basic_lat
                to_disrupt_y = disrupt_lon - basic_lon

                # Normalize
                dist = (to_disrupt_x ** 2 + to_disrupt_y ** 2) ** 0.5
                if dist > 0:
                    to_disrupt_x /= dist
                    to_disrupt_y /= dist

                # Create perpendicular offset
                # Bell-shaped curve offset (max in middle, 0 at endpoints)
                bell_factor = 4 * t * (1 - t)  # Max at t=0.5

                # Calculate safe distance with radius multiplier
                safe_distance = disruption.affected_area_radius * radius_multiplier * 1.2

                # Direction away from disruption
                detour_lat = basic_lat - to_disrupt_x * safe_distance * bell_factor * 0.00001
                detour_lon = basic_lon - to_disrupt_y * safe_distance * bell_factor * 0.00001

                detour_points.append((detour_lat, detour_lon))

            detour_points.append(end_point)
            return detour_points

        except Exception as e:
            print(f"Advanced fallback failed: {e}")
            # Simple fallback - direct line with midpoint offset
            mid_lat = (start_point[0] + end_point[0]) / 2
            mid_lon = (start_point[1] + end_point[1]) / 2

            # Simple offset
            offset_lat = mid_lat + 0.001  # ~100m
            offset_lon = mid_lon + 0.001

            return [start_point, (offset_lat, offset_lon), end_point]

    def _segment_near_disruption(self, start, end, disruption, custom_radius=None):
        """Check if a route segment passes near a disruption using specified radius"""
        try:
            radius = custom_radius if custom_radius is not None else disruption.affected_area_radius

            if (self._calculate_distance(start, disruption.location) <= radius or
                    self._calculate_distance(end, disruption.location) <= radius):
                return True

            closest_distance = self._point_to_segment_distance(disruption.location, start, end)
            return closest_distance <= radius
        except Exception as e:
            print(f"Error in segment_near_disruption: {e}")
            return True  # Conservative approach - assume segment is affected if error

    def _point_to_segment_distance(self, point, line_start, line_end):
        """Calculate the shortest distance from a point to a line segment"""
        try:
            p_lat, p_lon = float(point[0]), float(point[1])
            s_lat, s_lon = float(line_start[0]), float(line_start[1])
            e_lat, e_lon = float(line_end[0]), float(line_end[1])

            if s_lat == e_lat and s_lon == e_lon:
                return self._calculate_distance(point, line_start)

            line_length_sq = (e_lat - s_lat) ** 2 + (e_lon - s_lon) ** 2

            if line_length_sq < 1e-10:
                return self._calculate_distance(point, line_start)

            t = max(0, min(1, ((p_lat - s_lat) * (e_lat - s_lat) +
                               (p_lon - s_lon) * (e_lon - s_lon)) / line_length_sq))

            closest_lat = s_lat + t * (e_lat - s_lat)
            closest_lon = s_lon + t * (e_lon - s_lon)

            return self._calculate_distance(point, (closest_lat, closest_lon))
        except Exception as e:
            print(f"Error in distance calculation: {e}")
            try:
                return self._calculate_distance(point, line_start)
            except:
                return float('inf')

    def _points_equal(self, p1, p2, tolerance=1e-6):
        """Check if two points are equal within a tolerance"""
        try:
            return (abs(float(p1[0]) - float(p2[0])) < tolerance and
                    abs(float(p1[1]) - float(p2[1])) < tolerance)
        except Exception:
            return False

    def _evaluate_solution(self, actions, disruption, state):
        """Evaluate solution quality with multiple metrics"""
        if not actions:
            return 0.5  # Neutral score for no action

        score = 0.0

        # For reroute actions
        for action in actions:
            if not hasattr(action, 'action_type'):
                continue

            if action.action_type == ActionType.REROUTE:
                # Route length (shorter is better)
                route_length = self._calculate_route_length(action.new_route)
                length_score = max(0.1, min(1.0, 10000.0 / max(route_length, 1.0)))

                # Safety (distance from disruption)
                min_distance = float('inf')
                for point in action.new_route:
                    distance = self._calculate_distance(point, disruption.location)
                    min_distance = min(min_distance, distance)

                safety_factor = min_distance / max(disruption.affected_area_radius, 1.0)
                safety_score = min(1.0, safety_factor)

                # Combined score with weights
                score = 0.4 * length_score + 0.6 * safety_score

            elif action.action_type == ActionType.RECIPIENT_UNAVAILABLE:
                # Base score for this action type
                score = 0.7

                # Adjust based on wait duration (don't wait too long or too short)
                optimal_duration = disruption.duration
                duration_ratio = action.duration / optimal_duration

                if duration_ratio < 0.5:
                    score *= 0.7  # Penalty for waiting too little
                elif duration_ratio > 1.5:
                    score *= 0.8  # Smaller penalty for waiting too long

        return score

    def _calculate_route_length(self, route):
        """Calculate total route length in meters"""
        total_length = 0.0
        for i in range(len(route) - 1):
            total_length += self._calculate_distance(route[i], route[i + 1])
        return total_length

    def _calculate_distance(self, point1, point2):
        """Calculate Haversine distance between two points in meters"""
        import math

        try:
            lat1, lon1 = float(point1[0]), float(point1[1])
            lat2, lon2 = float(point2[0]), float(point2[1])

            lat1, lon1 = math.radians(lat1), math.radians(lon1)
            lat2, lon2 = math.radians(lat2), math.radians(lon2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371000  # Earth radius in meters

            return c * r
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 10000  # Default large distance

    def _get_affected_drivers(self, disruption, state):
        """Find drivers affected by a disruption"""
        # Similar to the rule-based resolver implementation
        affected_drivers = set()

        for driver_id, route_data in state.driver_routes.items():
            if 'points' not in route_data:
                continue

            points = route_data['points']
            for point in points:
                distance = self._calculate_distance(point, disruption.location)
                if distance <= disruption.affected_area_radius * 2:
                    affected_drivers.add(driver_id)
                    break

        return affected_drivers

    def resolve_disruptions(self, state, active_disruptions):
        """Handle multiple disruptions"""
        all_actions = []
        for disruption in active_disruptions:
            actions = self.on_disruption_detected(disruption, state)
            all_actions.extend(actions)
        return all_actions


class TrainingDataGenerator:
    """Generates training data using a supercharged rule-based resolver."""

    def __init__(self, graph, warehouse_location):
        # Regular resolver for comparison
        self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)

        # Supercharged resolver for generating high-quality training examples
        self.supercharged_resolver = SuperchargedResolver(
            graph,
            warehouse_location,
            computation_budget=5.0  # Allow 5 seconds per disruption
        )

        self.feature_extractor = FeatureExtractor()

    def generate_training_data(self, states_disruptions):
        """Generate features and labels from state-disruption pairs."""
        features = []
        labels = []

        improvement_threshold = 1.2  # Only use examples with 20% or better quality
        improvements = []

        print(f"Generating training data from {len(states_disruptions)} samples...")
        for i, (state, disruption) in enumerate(states_disruptions):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(states_disruptions)}")

            # Extract features
            feature_vector = self.feature_extractor.extract_features(state, disruption)

            # Get baseline solution from standard resolver
            base_actions = self.rule_based_resolver.on_disruption_detected(disruption, state)
            base_quality = self._evaluate_solution_quality(base_actions, disruption, state)

            # Get enhanced solution from supercharged resolver
            super_actions = self.supercharged_resolver.on_disruption_detected(disruption, state)
            super_quality = self._evaluate_solution_quality(super_actions, disruption, state)

            # Calculate quality improvement
            if base_quality > 0:
                improvement = super_quality / base_quality
            else:
                improvement = 1.0 if super_quality == 0 else float('inf')

            # Only use examples with significant improvement
            if improvement >= improvement_threshold:
                improvements.append(improvement)

                # Determine label based on action
                if not super_actions:
                    label = -1  # No action needed
                else:
                    # Take the first (highest priority) action
                    action = super_actions[0]
                    label = action.action_type.value

                features.append(feature_vector)
                labels.append(label)

                # Log major improvements
                if improvement >= 1.5:
                    base_type = "None" if not base_actions else base_actions[0].action_type.name
                    super_type = "None" if not super_actions else super_actions[0].action_type.name

                    print(f"Found high quality example (improvement: {improvement:.2f}x)")
                    print(f"  Disruption: {disruption.type.value} (severity: {disruption.severity:.2f})")
                    print(f"  Base action: {base_type} → Enhanced action: {super_type}")

        # Report statistics
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Selected {len(features)}/{len(states_disruptions)} examples with quality improvement")
            print(f"Average quality improvement: {avg_improvement:.2f}x")
        else:
            print("Warning: No examples with significant quality improvement found")

        if len(features) == 0:
            print("Warning: No training examples generated, falling back to all examples")
            return self._generate_fallback_data(states_disruptions)

        return np.array(features), np.array(labels)

    def _evaluate_solution_quality(self, actions, disruption, state):
        """Evaluate the quality of a solution"""
        if not actions:
            return 0.5  # Neutral score for no action

        score = 0.0

        # For reroute actions
        for action in actions:
            if not hasattr(action, 'action_type'):
                continue

            if action.action_type == ActionType.REROUTE:
                # Route length (shorter is better)
                route_length = self._calculate_route_length(action.new_route)
                length_score = max(0.1, min(1.0, 10000.0 / max(route_length, 1.0)))

                # Safety (distance from disruption)
                min_distance = float('inf')
                for point in action.new_route:
                    distance = self._calculate_distance(point, disruption.location)
                    min_distance = min(min_distance, distance)

                safety_factor = min_distance / max(disruption.affected_area_radius, 1.0)
                safety_score = min(1.0, safety_factor)

                # Combined score with weights
                score = 0.4 * length_score + 0.6 * safety_score

            elif action.action_type == ActionType.RECIPIENT_UNAVAILABLE:
                # Base score for this action type
                score = 0.7

                # Adjust based on wait duration (don't wait too long or too short)
                optimal_duration = disruption.duration
                duration_ratio = action.duration / optimal_duration

                if duration_ratio < 0.5:
                    score *= 0.7  # Penalty for waiting too little
                elif duration_ratio > 1.5:
                    score *= 0.8  # Smaller penalty for waiting too long

        return score

    def _calculate_route_length(self, route):
        """Calculate total route length in meters"""
        total_length = 0.0
        for i in range(len(route) - 1):
            total_length += self._calculate_distance(route[i], route[i + 1])
        return total_length

    def _calculate_distance(self, point1, point2):
        """Calculate Haversine distance between two points in meters"""
        import math

        try:
            lat1, lon1 = float(point1[0]), float(point1[1])
            lat2, lon2 = float(point2[0]), float(point2[1])

            lat1, lon1 = math.radians(lat1), math.radians(lon1)
            lat2, lon2 = math.radians(lat2), math.radians(lon2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371000  # Earth radius in meters

            return c * r
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 10000  # Default large distance

    def _generate_fallback_data(self, states_disruptions):
        """Generate fallback training data when no quality improvements found"""
        print("Using standard rule-based resolver for training data")
        features = []
        labels = []

        for state, disruption in states_disruptions:
            # Extract features
            feature_vector = self.feature_extractor.extract_features(state, disruption)

            # Get action from standard resolver
            actions = self.rule_based_resolver.on_disruption_detected(disruption, state)

            # Determine label based on action
            if not actions:
                label = -1  # No action needed
            else:
                # Take the first (highest priority) action
                action = actions[0]
                label = action.action_type.value

            features.append(feature_vector)
            labels.append(label)

        return np.array(features), np.array(labels)


class ClassifierBasedResolver(DisruptionResolver):
    """
    Disruption resolver that uses a trained classifier to predict optimal actions.
    Combines the speed of ML inference with the reasoning power of rule-based systems.
    """

    def __init__(self, graph, warehouse_location, model_path=None):
        self.graph = graph
        self.warehouse_location = warehouse_location

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.classifier = ClassifierModel()

        # Fallback rule-based resolver for cases where the classifier is uncertain
        self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)

        # Action generator to create concrete actions from predictions
        self.action_generator = ActionGenerator(self.rule_based_resolver)

        # Performance metrics
        self.decision_times = []
        self.fallback_count = 0
        self.decision_count = 0

        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.classifier.load(model_path)

    def should_recalculate(self, state: DeliverySystemState, disruption: Disruption) -> bool:
        """
        Determine if recalculation is worth the computational cost.

        The classifier-based resolver always returns True since the inference is fast.
        """
        return True

    def on_disruption_detected(self, disruption, state):
        """Process a newly detected disruption and determine the best action.

        Uses the trained classifier to predict actions, with improved confidence handling.
        """
        self.decision_count += 1
        start_time = time.time()

        try:
            # If the classifier isn't trained, use rule-based resolver
            if not self.classifier.is_trained:
                actions = self.rule_based_resolver.on_disruption_detected(disruption, state)
                decision_time = time.time() - start_time
                self.decision_times.append(decision_time)
                self.fallback_count += 1
                return actions

            # Extract features from the state and disruption
            features = self.feature_extractor.extract_features(state, disruption)
            features = features.reshape(1, -1)  # Reshape for single prediction

            # Predict action type and confidence
            action_type = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = max(probabilities) if len(probabilities) > 0 else 0

            # Lower confidence threshold - allow the model more freedom
            # This will reduce fallbacks and let the model make more decisions
            if confidence < 0.4:  # Was 0.65, now more permissive
                self.fallback_count += 1
                actions = self.rule_based_resolver.on_disruption_detected(disruption, state)
                decision_time = time.time() - start_time
                self.decision_times.append(decision_time)
                return actions

            # Generate concrete action based on prediction
            action = self.action_generator.generate_action(action_type, state, disruption)

            # No valid action generated, fall back to rule-based
            if action is None:
                self.fallback_count += 1
                actions = self.rule_based_resolver.on_disruption_detected(disruption, state)
                decision_time = time.time() - start_time
                self.decision_times.append(decision_time)
                return actions

            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            return [action]

        except Exception as e:
            print(f"Error in classifier-based resolver: {str(e)}")
            # Fall back to rule-based resolver on error
            self.fallback_count += 1
            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            return self.rule_based_resolver.on_disruption_detected(disruption, state)

    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """Process multiple active disruptions."""
        actions = []

        for disruption in active_disruptions:
            disruption_actions = self.on_disruption_detected(disruption, state)
            if disruption_actions:
                actions.extend(disruption_actions)

        return actions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the resolver."""
        stats = {
            "decision_count": self.decision_count,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(self.decision_count, 1),
            "avg_decision_time_ms": np.mean(self.decision_times) * 1000 if self.decision_times else 0,
            "median_decision_time_ms": np.median(self.decision_times) * 1000 if self.decision_times else 0,
            "max_decision_time_ms": max(self.decision_times) * 1000 if self.decision_times else 0
        }
        return stats

    def train(self, training_data_generator: TrainingDataGenerator,
              states_disruptions: List[Tuple[DeliverySystemState, Disruption]]):
        """Train the classifier with generated data."""
        features, labels = training_data_generator.generate_training_data(states_disruptions)
        self.classifier.train(features, labels)

    def save_model(self, model_path: str):
        """Save the trained classifier model."""
        self.classifier.save(model_path)

    def load_model(self, model_path: str):
        """Load a trained classifier model."""
        self.classifier.load(model_path)
