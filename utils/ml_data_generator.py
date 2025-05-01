import os
import pickle
import math
import numpy as np
import pandas as pd
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction, ActionType
)
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import calculate_route_length, calculate_travel_time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLDataGenerator:
    """
    Generates training data for the ML classifier by simulating various disruption scenarios
    and evaluating different reroute actions.
    """
    
    OUTPUT_DIR = os.path.join('models', 'resolvers', 'training_data')
    
    def __init__(self, graph, warehouse_location, delivery_points, num_samples=1000, random_seed=42):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.num_samples = num_samples
        self.random_seed = random_seed
        
        # Create resolver for calculating routes with silent mode
        self.resolver = RuleBasedResolver(graph, warehouse_location)
        # Disable resolver's print statements
        self.resolver._print_enabled = False
        
        # Ensure output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Initialize random generator
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Set up logging to only show errors
        logging.getLogger().setLevel(logging.ERROR)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.ERROR)
    
    def generate_training_data(self):
        """
        Generate a dataset of disruption scenarios, actions, and outcomes for training
        
        The dataset includes:
        1. Features about the disruption and driver state
        2. Different action types tried
        3. Outcome metrics for each action
        4. Best action based on outcome metrics
        """
        # Data storage
        feature_rows = []
        stats = {
            'failed_scenarios': 0,
            'failed_features': 0,
            'failed_actions': 0,
            'successful_samples': 0
        }
        
        # Use tqdm for progress tracking
        pbar = tqdm(total=self.num_samples, desc="Generating samples")
        
        while len(feature_rows) < self.num_samples:
            # Generate a random disruption scenario
            disruption, state, driver_id = self._generate_random_scenario()
            
            if disruption is None or state is None or driver_id is None:
                stats['failed_scenarios'] += 1
                pbar.set_description(f"Failed scenarios: {stats['failed_scenarios']}")
                continue
            
            # Extract features for this scenario
            features_dict = self._extract_features(driver_id, disruption, state)
            
            if features_dict is None:
                stats['failed_features'] += 1
                pbar.set_description(f"Failed features: {stats['failed_features']}")
                continue
            
            # Try each action type and evaluate the outcome
            action_outcomes = self._evaluate_all_actions(driver_id, disruption, state)
            
            if not action_outcomes:
                stats['failed_actions'] += 1
                pbar.set_description(f"Failed actions: {stats['failed_actions']}")
                continue
            
            # Determine best action based on outcomes
            best_action = self._determine_best_action(action_outcomes)
            
            # Save the training sample with features and best action
            sample_data = {
                "disruption_type_road_closure": features_dict.get('disruption_type_road_closure', 0.0),
                "disruption_type_traffic_jam": features_dict.get('disruption_type_traffic_jam', 0.0),
                "disruption_severity": features_dict.get('severity', 0.0),
                "distance_to_disruption": features_dict.get('distance_to_disruption', 0.0),
                "remaining_deliveries": features_dict.get('remaining_deliveries', 0.0),
                "time_impact": features_dict.get('distance_to_next_delivery', 0.0),
                "network_density_nodes": features_dict.get('alternative_route_density', 0.0),
                "network_density_edges": features_dict.get('urban_density', 0.0),
                "route_progress": features_dict.get('route_progress', 0.5),  # Default value
                "best_action": best_action,
                "disruption_id": disruption.id,
                "driver_id": driver_id,
                "all_actions_travel_time": {k: v["travel_time"] for k, v in action_outcomes.items()},
                "all_actions_route_length": {k: v["route_length"] for k, v in action_outcomes.items()},
                "all_actions_detour_factor": {k: v["detour_factor"] for k, v in action_outcomes.items()}
            }
            
            feature_rows.append(sample_data)
            stats['successful_samples'] += 1
            pbar.set_description(f"Success: {stats['successful_samples']} | Failed: {stats['failed_scenarios'] + stats['failed_features'] + stats['failed_actions']}")
            pbar.update(1)
        
        pbar.close()
        
        # Check if we have any data before trying to create DataFrames
        if not feature_rows:
            logger.error("No valid samples were generated. Please try increasing the number of samples or check for errors.")
            return None, None
        
        # Save the training data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.OUTPUT_DIR, f"training_data_{timestamp}.csv")
        
        df = pd.DataFrame(feature_rows)
        df.to_csv(output_file, index=False)
        
        # Create another file with just features and labels for easy training
        X_cols = [col for col in df.columns if col not in ['best_action', 'disruption_id', 'driver_id', 
                                                          'all_actions_travel_time', 'all_actions_route_length',
                                                          'all_actions_detour_factor']]
        y_col = 'best_action'
        
        simple_df = df[X_cols + [y_col]]
        simple_output = os.path.join(self.OUTPUT_DIR, f"simple_training_data_{timestamp}.csv")
        simple_df.to_csv(simple_output, index=False)
        
        # Save generation metadata
        metadata = {
            "timestamp": timestamp,
            "num_samples": len(feature_rows),
            "failed_scenarios": stats['failed_scenarios'],
            "failed_features": stats['failed_features'],
            "failed_actions": stats['failed_actions'],
            "action_distribution": df['best_action'].value_counts().to_dict(),
            "feature_columns": X_cols,
            "label_column": y_col
        }
        
        metadata_file = os.path.join(self.OUTPUT_DIR, f"generation_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return output_file, simple_output
    
    def _generate_random_scenario(self) -> Tuple[Optional[Disruption], Optional[DeliverySystemState], Optional[int]]:
        """
        Generate a random disruption scenario with a delivery system state and affected driver
        """
        try:
            # Create empty lists/dicts for initialization
            drivers = []  # Empty drivers list for simulation
            all_delivery_points = list(self.delivery_points)
            random.shuffle(all_delivery_points)
            disruptions = []  # Empty disruptions list
            
            # Create a simulated state with all required parameters
            state = DeliverySystemState(
                drivers=drivers,
                deliveries=all_delivery_points,
                disruptions=disruptions,
                simulation_time=0,  # Time doesn't matter anymore
                graph=self.G,
                warehouse_location=self.warehouse_location
            )
            
            # Set up a more realistic number of drivers (3-8)
            num_drivers = random.randint(3, 8)
            
            # Assign random subset of delivery points to each driver
            state.driver_assignments = {}
            state.driver_positions = {}
            state.driver_routes = {}
            
            # Distribute delivery points among drivers more evenly
            points_per_driver = max(1, len(all_delivery_points) // num_drivers)
            remaining_points = len(all_delivery_points) % num_drivers
            
            start_idx = 0
            for driver_id in range(num_drivers):
                # Add extra point to some drivers to distribute remaining points
                extra_point = 1 if driver_id < remaining_points else 0
                end_idx = min(start_idx + points_per_driver + extra_point, len(all_delivery_points))
                
                # Get indices of assigned delivery points
                driver_delivery_indices = list(range(start_idx, end_idx))
                state.driver_assignments[driver_id] = driver_delivery_indices
                
                # Set more realistic driver position with weighted scenarios
                driver_scenario = random.choices(
                    ["at_warehouse", "near_delivery", "between_points", "stuck_in_traffic"],
                    weights=[0.15, 0.35, 0.35, 0.15],
                    k=1
                )[0]
                
                if driver_scenario == "at_warehouse":
                    position = self.warehouse_location
                elif driver_scenario == "near_delivery" and driver_delivery_indices:
                    delivery_idx = random.choice(driver_delivery_indices)
                    delivery_point = all_delivery_points[delivery_idx]
                    # Add noise to position
                    noise_lat = (random.random() - 0.5) * 0.001  # ~100m noise
                    noise_lon = (random.random() - 0.5) * 0.001
                    position = (delivery_point[0] + noise_lat, delivery_point[1] + noise_lon)
                elif driver_scenario == "between_points" and driver_delivery_indices:
                    # Choose two consecutive points and place between them
                    if len(driver_delivery_indices) >= 2:
                        idx1 = random.randint(0, len(driver_delivery_indices) - 2)
                        idx2 = idx1 + 1
                        point1 = all_delivery_points[driver_delivery_indices[idx1]]
                        point2 = all_delivery_points[driver_delivery_indices[idx2]]
                        # Random point between them
                        ratio = random.random()
                        position = (
                            point1[0] * (1-ratio) + point2[0] * ratio,
                            point1[1] * (1-ratio) + point2[1] * ratio
                        )
                    else:
                        # Fallback if only one delivery
                        position = self.warehouse_location
                elif driver_scenario == "stuck_in_traffic":
                    # We'll place the driver near where we'll generate disruption later
                    # This will be updated after disruption generation
                    position = self.warehouse_location  # Temporary
                else:
                    position = self.warehouse_location
                
                state.driver_positions[driver_id] = position
                
                # Generate a more realistic route for the driver
                route_points = [self.warehouse_location]
                for idx in driver_delivery_indices:
                    if idx < len(all_delivery_points):  # Safety check
                        delivery_point = all_delivery_points[idx]
                        route_points.append(delivery_point)
                
                # Add intermediate route points with more detail
                detailed_route = []
                for i in range(len(route_points) - 1):
                    # Find path between consecutive delivery points
                    start, end = route_points[i], route_points[i+1]
                    try:
                        start_node = ox.nearest_nodes(self.G, X=start[1], Y=start[0])
                        end_node = ox.nearest_nodes(self.G, X=end[1], Y=end[0])
                        path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
                        
                        # Convert path to points with more detail
                        path_points = []
                        for node in path:
                            if 'y' in self.G.nodes[node] and 'x' in self.G.nodes[node]:
                                lat = self.G.nodes[node]['y']
                                lon = self.G.nodes[node]['x']
                                path_points.append((lat, lon))
                        
                        detailed_route.extend(path_points)
                    except:
                        # If path finding fails, use direct line with some intermediate points
                        t_values = np.linspace(0, 1, 10)  # 10 intermediate points
                        for t in t_values:
                            lat = start[0] * (1-t) + end[0] * t
                            lon = start[1] * (1-t) + end[1] * t
                            detailed_route.append((lat, lon))
                
                # Add route progress
                progress = random.random()  # Random progress between 0-1
                
                state.driver_routes[driver_id] = {
                    'points': detailed_route,
                    'progress': progress,
                    'delivery_indices': driver_delivery_indices
                }
                
                start_idx = end_idx
            
            # Generate a more realistic disruption with balanced distribution
            # Select disruption type with weighted probabilities
            disruption_type = random.choices(
                [DisruptionType.ROAD_CLOSURE, DisruptionType.TRAFFIC_JAM],
                weights=[0.4, 0.6],  # Favor traffic jams slightly
                k=1
            )[0]
            
            # Set more realistic location with different scenarios
            disruption_scenario = random.choices(
                ["along_route", "near_delivery", "urban_area", "random"],
                weights=[0.5, 0.25, 0.15, 0.1],
                k=1
            )[0]
            
            if disruption_scenario == "along_route":
                # Choose a random driver and use a point along their route
                if state.driver_routes:
                    driver_id = random.choice(list(state.driver_routes.keys()))
                    route = state.driver_routes[driver_id]['points']
                    
                    if len(route) > 2:
                        # Choose a random segment of the route
                        segment_start = random.randint(0, len(route) - 2)
                        segment_end = segment_start + 1
                        
                        # Random point along segment
                        t = random.random()
                        lat = route[segment_start][0] * (1-t) + route[segment_end][0] * t
                        lon = route[segment_start][1] * (1-t) + route[segment_end][1] * t
                        
                        disruption_location = (lat, lon)
                    else:
                        disruption_location = random.choice(all_delivery_points)
                else:
                    disruption_location = random.choice(all_delivery_points)
            elif disruption_scenario == "near_delivery":
                # Random delivery point with small offset
                delivery_point = random.choice(all_delivery_points)
                noise_lat = (random.random() - 0.5) * 0.002  # ~200m noise
                noise_lon = (random.random() - 0.5) * 0.002
                disruption_location = (delivery_point[0] + noise_lat, delivery_point[1] + noise_lon)
            elif disruption_scenario == "urban_area":
                # Try to find a densely connected area in the graph
                try:
                    # Get a random node to start
                    nodes = list(self.G.nodes)
                    start_node = random.choice(nodes)
                    
                    # Get a subgraph around this node
                    subgraph = nx.ego_graph(self.G, start_node, radius=5)
                    
                    # Find node with highest degree (most connected)
                    if len(subgraph.nodes) > 0:
                        degrees = dict(subgraph.degree())
                        urban_node = max(degrees.items(), key=lambda x: x[1])[0]
                        
                        if 'y' in self.G.nodes[urban_node] and 'x' in self.G.nodes[urban_node]:
                            lat = self.G.nodes[urban_node]['y']
                            lon = self.G.nodes[urban_node]['x']
                            disruption_location = (lat, lon)
                        else:
                            disruption_location = random.choice(all_delivery_points)
                    else:
                        disruption_location = random.choice(all_delivery_points)
                except:
                    disruption_location = random.choice(all_delivery_points)
            else:  # random location
                # Generate a random point within the bounds of delivery points
                lats = [p[0] for p in all_delivery_points]
                lons = [p[1] for p in all_delivery_points]
                
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                rand_lat = min_lat + random.random() * (max_lat - min_lat)
                rand_lon = min_lon + random.random() * (max_lon - min_lon)
                
                disruption_location = (rand_lat, rand_lon)
            
            # Update stuck_in_traffic driver position if that scenario was chosen
            for driver_id, scenario in enumerate(["at_warehouse", "near_delivery", "between_points", "stuck_in_traffic"]):
                if driver_id < num_drivers and scenario == "stuck_in_traffic":
                    # Place driver near disruption
                    noise_lat = (random.random() - 0.5) * 0.0005  # ~50m noise
                    noise_lon = (random.random() - 0.5) * 0.0005
                    state.driver_positions[driver_id] = (
                        disruption_location[0] + noise_lat,
                        disruption_location[1] + noise_lon
                    )
            
            # Set realistic severity with a distribution favoring medium values
            beta_a, beta_b = 2, 2  # Parameters for beta distribution centered at 0.5
            severity = random.betavariate(beta_a, beta_b)
            
            # Adjust severity based on disruption type
            if disruption_type == DisruptionType.ROAD_CLOSURE:
                # Road closures tend to be more severe
                severity = 0.7 + severity * 0.3  # Range 0.7-1.0
            
            # Set more realistic radius based on disruption type and scenario
            if disruption_type == DisruptionType.ROAD_CLOSURE:
                if disruption_scenario == "urban_area":
                    radius = random.uniform(50, 150)  # Smaller in urban areas
                else:
                    radius = random.uniform(100, 300)  # Normal range
            else:  # Traffic jam
                if disruption_scenario == "urban_area":
                    radius = random.uniform(150, 500)  # Traffic affects wider area in cities
                else:
                    radius = random.uniform(200, 800)  # Normal range
            
            # Set more realistic duration based on type and severity
            if disruption_type == DisruptionType.ROAD_CLOSURE:
                duration = int(30 + severity * 150)  # 30-180 mins based on severity
            else:
                duration = int(15 + severity * 105)  # 15-120 mins based on severity
            
            disruption = Disruption(
                id=random.randint(1000, 9999),
                location=disruption_location,
                type=disruption_type,
                severity=severity,
                affected_area_radius=radius,
                duration=duration,
                metadata={
                    'scenario': disruption_scenario
                }
            )
            
            # Pick an affected driver
            affected_drivers = self.resolver._get_affected_drivers(disruption, state)
            
            if not affected_drivers:
                # If no driver is affected, try to find one that's close enough
                min_distance = float('inf')
                closest_driver = None
                
                for driver_id, position in state.driver_positions.items():
                    distance = calculate_haversine_distance(position, disruption_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_driver = driver_id
                
                if closest_driver is not None and min_distance < 2000:  # Within 2km
                    affected_driver_id = closest_driver
                else:
                    return None, None, None
            else:
                affected_driver_id = random.choice(list(affected_drivers))
            
            return disruption, state, affected_driver_id
            
        except Exception as e:
            logger.error(f"Error generating random scenario: {e}")
            return None, None, None
    
    def _extract_features(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[Dict[str, float]]:
        """
        Extract enhanced features for ML training
        Including all the new features added to the ML resolver
        
        Returns:
        - Dictionary of feature names and values
        """
        try:
            features = {}
            
            # 1. Disruption type (one-hot encoded)
            if disruption.type == DisruptionType.ROAD_CLOSURE:
                features['disruption_type_road_closure'] = 1.0
                features['disruption_type_traffic_jam'] = 0.0
            elif disruption.type == DisruptionType.TRAFFIC_JAM:
                features['disruption_type_road_closure'] = 0.0
                features['disruption_type_traffic_jam'] = 1.0
            else:
                features['disruption_type_road_closure'] = 0.0
                features['disruption_type_traffic_jam'] = 0.0
                
            # 2. Disruption severity (normalized 0-1)
            features['severity'] = disruption.severity
            
            # 3. Distance from driver to disruption (normalized)
            position = state.driver_positions.get(driver_id)
            if position is None:
                logger.warning(f"Driver {driver_id} has no position")
                return None
                
            distance_to_disruption = calculate_haversine_distance(position, disruption.location)
            # Normalize distance (assuming max reasonable distance is 10km)
            max_distance = 10000  # meters
            normalized_distance = min(1.0, distance_to_disruption / max_distance)
            features['distance_to_disruption'] = normalized_distance
            
            # 4. Number of delivery points remaining (normalized)
            remaining_deliveries = len(state.driver_assignments.get(driver_id, []))
            # Normalize by assuming max 20 deliveries per driver
            features['remaining_deliveries'] = min(1.0, remaining_deliveries / 20.0)
            
            # 5. Distance to next delivery (new feature)
            next_delivery_distance = float('inf')
            delivery_indices = state.driver_assignments.get(driver_id, [])
            for idx in delivery_indices:
                if idx < len(state.deliveries):
                    delivery_point = state.deliveries[idx]
                    dist = calculate_haversine_distance(position, delivery_point)
                    next_delivery_distance = min(next_delivery_distance, dist)
            
            if next_delivery_distance == float('inf'):
                next_delivery_distance = max_distance  # If no deliveries, use max
            
            features['distance_to_next_delivery'] = min(1.0, next_delivery_distance / max_distance)
            
            # 6. Alternative route density (new feature)
            try:
                # Find nodes near the disruption
                disruption_node = ox.nearest_nodes(self.G, X=disruption.location[1], Y=disruption.location[0])
                
                # Calculate a local density measure around the disruption
                radius = disruption.affected_area_radius * 3  # Look wider than the disruption itself
                disruption_area = nx.ego_graph(self.G, disruption_node, radius=10, distance='travel_time')
                
                # Measure alternative path density as a ratio of edges to nodes
                if len(disruption_area.nodes) > 0:
                    edge_to_node_ratio = len(disruption_area.edges) / len(disruption_area.nodes)
                    # Normalize to 0-1 (typical values range from 1-3 for road networks)
                    features['alternative_route_density'] = min(1.0, edge_to_node_ratio / 3.0)
                else:
                    features['alternative_route_density'] = 0.0
            except Exception as e:
                # Fallback if density calculation fails
                features['alternative_route_density'] = 0.5  # Default to medium
            
            # 7. Urban density estimate (new feature)
            try:
                # Count nearby nodes as a proxy for urban density
                disruption_radius_m = disruption.affected_area_radius
                nearby_nodes = [
                    n for n in self.G.nodes 
                    if 'y' in self.G.nodes[n] and 'x' in self.G.nodes[n] and
                    calculate_haversine_distance(
                        (self.G.nodes[n]['y'], self.G.nodes[n]['x']),
                        disruption.location
                    ) <= disruption_radius_m * 5  # Look at wider area for density
                ]
                
                # Normalize node count (0-100 nodes is typical range)
                node_count = len(nearby_nodes)
                features['urban_density'] = min(1.0, node_count / 100.0)
            except Exception as e:
                features['urban_density'] = 0.5  # Default to medium
                
            # 8. Add route progress (new feature)
            route_data = state.driver_routes.get(driver_id, {})
            progress = route_data.get('progress', 0.5)  # Default to middle of route
            features['route_progress'] = progress
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_action_metrics(self, action: DisruptionAction, original_route: List, original_length: float, original_time: float, disruption: Disruption) -> Optional[Dict[str, float]]:
        """
        Evaluate the metrics for a specific action
        
        Parameters:
        - action: The disruption action to evaluate
        - original_route: The original route points before applying the action
        - original_length: The original route length in meters
        - original_time: The original travel time in seconds
        - disruption: The disruption object
        
        Returns:
        - Dictionary with metrics: travel_time, route_length, detour_factor
        """
        try:
            # Get the new route based on action type
            if hasattr(action, 'new_route'):
                # For reroute actions which have a new_route attribute
                new_route = action.new_route
            elif action.action_type == ActionType.NO_ACTION:
                # For NoAction, maintain the original route
                new_route = original_route
            else:
                # Fallback if we can't determine the route
                return None
                
            if not new_route or len(new_route) < 2:
                return None
                
            # Calculate new metrics
            new_length = calculate_route_length(new_route)
            new_time = calculate_travel_time(new_route, self.G)
            
            # Calculate detour factor (how much longer the new route is)
            detour_factor = new_length / max(1.0, original_length)
            
            # Check if the new route avoids disruption
            avoids_disruption = True
            disruption_radius = disruption.affected_area_radius
            
            # Sample points along the route to check if they avoid the disruption
            for point in new_route:
                distance = calculate_haversine_distance(point, disruption.location)
                # Consider the route point to be affected if it's within the disruption radius
                if distance <= disruption_radius:
                    avoids_disruption = False
                    break
            
            return {
                "travel_time": new_time,
                "route_length": new_length,
                "detour_factor": detour_factor,
                "avoids_disruption": avoids_disruption
            }
            
        except Exception as e:
            logger.error(f"Error evaluating action metrics: {e}")
            return None
    
    def _evaluate_all_actions(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all possible actions for a given scenario
        Include parameterized avoidance with different buffer ratios
        """
        try:
            # Get original route data
            route_points = state.driver_routes.get(driver_id, {}).get('points', [])
            original_length = calculate_route_length(route_points)
            original_time = calculate_travel_time(route_points, self.G)
            
            # Initialize outcome storage
            outcomes = {}
            
            # 1. No action
            try:
                no_action = self.resolver._create_no_action(driver_id, disruption)
                if no_action:
                    metrics = self._evaluate_action_metrics(
                        no_action, route_points, original_length, original_time, disruption
                    )
                    if metrics:
                        outcomes['no_action'] = metrics
            except Exception as e:
                logger.error(f"Error evaluating no_action: {e}")
            
            # 2. Basic Reroute
            try:
                basic_action = self.resolver._create_reroute_action(driver_id, disruption, state)
                if basic_action:
                    metrics = self._evaluate_action_metrics(
                        basic_action, route_points, original_length, original_time, disruption
                    )
                    if metrics:
                        outcomes['basic_reroute'] = metrics
            except Exception as e:
                logger.error(f"Error evaluating basic_reroute: {e}")
            
            # 3. Multiple parameterized avoidance actions with different buffer ratios
            buffer_ratios = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
            for ratio in buffer_ratios:
                try:
                    # Create parameterized action with this buffer ratio
                    if hasattr(self.resolver, '_create_parameterized_avoidance_action'):
                        param_action = self.resolver._create_parameterized_avoidance_action(
                            driver_id, disruption, state, buffer_ratio=ratio
                        )
                        
                        if param_action:
                            metrics = self._evaluate_action_metrics(
                                param_action, route_points, original_length, original_time, disruption
                            )
                            if metrics:
                                outcomes[f'parameterized_avoidance_{ratio:.1f}'] = metrics
                    else:
                        # Fallback to legacy tight/wide if parameterized not available
                        if ratio <= 0.8:
                            action = self.resolver._create_tight_avoidance_action(driver_id, disruption, state)
                            if action:
                                metrics = self._evaluate_action_metrics(
                                    action, route_points, original_length, original_time, disruption
                                )
                                if metrics:
                                    outcomes['tight_avoidance'] = metrics
                        elif ratio >= 2.0:
                            action = self.resolver._create_wide_avoidance_action(driver_id, disruption, state)
                            if action:
                                metrics = self._evaluate_action_metrics(
                                    action, route_points, original_length, original_time, disruption
                                )
                                if metrics:
                                    outcomes['wide_avoidance'] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating parameterized avoidance with ratio {ratio}: {e}")
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error evaluating all actions: {e}")
            return {}
    
    def _determine_best_action(self, action_outcomes: Dict[str, Dict[str, float]]) -> str:
        """
        Determine the best action based on outcome metrics
        
        Uses a weighted combination of travel time and route length
        """
        if not action_outcomes:
            return "basic_reroute"  # Default
        
        # Define weights for different metrics
        time_weight = 0.7  # Higher weight for travel time
        length_weight = 0.3  # Lower weight for route length
        
        # Find the minimum values for normalization
        min_time = min(outcome["travel_time"] for outcome in action_outcomes.values())
        min_length = min(outcome["route_length"] for outcome in action_outcomes.values())
        
        # Find maximum values for normalization
        max_time = max(outcome["travel_time"] for outcome in action_outcomes.values())
        max_length = max(outcome["route_length"] for outcome in action_outcomes.values())
        
        # Calculate normalized scores for each action
        best_action = None
        best_score = float('inf')
        
        for action_type, metrics in action_outcomes.items():
            # Normalize values to 0-1 range
            norm_time = (metrics["travel_time"] - min_time) / max(1, max_time - min_time)
            norm_length = (metrics["route_length"] - min_length) / max(1, max_length - min_length)
            
            # Calculate weighted score (lower is better)
            score = time_weight * norm_time + length_weight * norm_length
            
            # Update best action if this one is better
            if score < best_score:
                best_score = score
                best_action = action_type
        
        return best_action or "basic_reroute"


# Command-line execution
if __name__ == "__main__":
    import argparse
    from config.config import Config
    from models.services.graph import load_graph, get_largest_connected_component
    
    parser = argparse.ArgumentParser(description='Generate training data for ML classifier')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Load graph
    print("Loading graph...")
    graph_path = config.get_osm_file_path()
    try:
        graph = load_graph(graph_path)
        graph = get_largest_connected_component(graph)
    except FileNotFoundError:
        print(f"Graph file not found at {graph_path}")
        print("Please ensure the graph file exists before running this script.")
        exit(1)
    
    # Create data generator
    warehouse_location = config.get_warehouse_location()
    delivery_points = config.get_delivery_points()
    
    print(f"Creating data generator with {args.samples} samples, seed {args.seed}")
    generator = MLDataGenerator(
        graph=graph,
        warehouse_location=warehouse_location,
        delivery_points=delivery_points,
        num_samples=args.samples,
        random_seed=args.seed
    )
    
    # Generate training data
    generator.generate_training_data() 