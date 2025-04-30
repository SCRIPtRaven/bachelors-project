import os
import pickle
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Set

import networkx as nx
import osmnx as ox

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoRerouteAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction
)
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance


class MLClassifierResolver(RuleBasedResolver):
    """
    Machine learning-based resolver that uses a trained classifier to select 
    the optimal reroute action for traffic jams and road closures.
    
    For recipient unavailable disruptions, it behaves the same as the RuleBasedResolver.
    """

    MODEL_PATH = os.path.join('models', 'resolvers', 'saved_models', 'reroute_classifier.pkl')
    
    def __init__(self, graph, warehouse_location, max_computation_time=1.0):
        super().__init__(graph, warehouse_location, max_computation_time)
        self.classifier = self._load_classifier()
        
    def _load_classifier(self):
        """Load the trained classifier model if available"""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Warning: ML Classifier model not found at {self.MODEL_PATH}. "
                      f"Will use fallback rules for decision making.")
                return None
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            return None
            
    def has_classifier(self):
        """Check if a valid classifier model is loaded"""
        return self.classifier is not None
    
    def resolve_disruptions(self, state: DeliverySystemState, active_disruptions: List[Disruption]) -> List[
        DisruptionAction]:
        """Use classifier to determine optimal actions for disruptions"""
        actions = []

        for disruption in active_disruptions:
            try:
                affected_drivers = self._get_affected_drivers(disruption, state)

                print(f"ML Resolver processing {disruption.type.value} affecting {len(affected_drivers)} drivers")

                if disruption.type in [DisruptionType.ROAD_CLOSURE, DisruptionType.TRAFFIC_JAM]:
                    for driver_id in affected_drivers:
                        if self.classifier is not None:
                            action = self._select_optimal_action(driver_id, disruption, state)
                            if action:
                                actions.append(action)
                        else:
                            # Fallback to basic reroute if no model is available
                            action = self._create_reroute_action(driver_id, disruption, state)
                            if action:
                                actions.append(action)
                elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                    # Handle recipient unavailable the same way as rule-based resolver
                    for driver_id in affected_drivers:
                        delivery_idx = self._find_affected_delivery(driver_id, disruption, state)
                        if delivery_idx is not None:
                            action = RecipientUnavailableAction(
                                driver_id=driver_id,
                                delivery_index=delivery_idx,
                                disruption_id=disruption.id,
                                duration=disruption.duration
                            )
                            actions.append(action)

            except Exception as e:
                print(f"Error handling disruption {disruption.id} in ML resolver: {e}")
                import traceback
                traceback.print_exc()
                continue

        return actions
    
    def _select_optimal_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[DisruptionAction]:
        """
        Use the trained classifier to select the best action for a given disruption
        """
        try:
            # Extract features for the classifier
            features = self._extract_features(driver_id, disruption, state)
            
            if features is None:
                print(f"Could not extract features for driver {driver_id} and disruption {disruption.id}")
                return self._create_reroute_action(driver_id, disruption, state)
            
            # Make prediction with the classifier
            action_type = self._predict_action_type(features)
            
            # Create and return the appropriate action
            if action_type == 'no_reroute':
                return self._create_no_reroute_action(driver_id, disruption)
            elif action_type == 'tight_avoidance':
                return self._create_tight_avoidance_action(driver_id, disruption, state)
            elif action_type == 'wide_avoidance':
                return self._create_wide_avoidance_action(driver_id, disruption, state)
            else:  # 'basic_reroute' or fallback
                return self._create_reroute_action(driver_id, disruption, state)
                
        except Exception as e:
            print(f"Error selecting optimal action: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic reroute
            return self._create_reroute_action(driver_id, disruption, state)
    
    def _extract_features(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[pd.DataFrame]:
        """
        Extract features from the current state, driver, and disruption for the classifier
        Only using the features the model was trained on:
        1. Disruption type (one-hot encoded)
        2. Distance from driver to disruption (normalized)
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
                
            # 2. Distance from driver to disruption (normalized)
            position = state.driver_positions.get(driver_id)
            if position is None:
                return None
                
            distance_to_disruption = calculate_haversine_distance(position, disruption.location)
            # Normalize distance (assuming max reasonable distance is 10km)
            max_distance = 10000
            features['distance_to_disruption'] = min(1.0, distance_to_disruption / max_distance)
            
            # Convert to DataFrame with correct column order
            feature_names = [
                'disruption_type_road_closure',
                'disruption_type_traffic_jam',
                'distance_to_disruption'
            ]
            
            return pd.DataFrame([features])[feature_names]
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_action_type(self, features: pd.DataFrame) -> str:
        """
        Use the classifier to predict the best action type
        
        Returns one of: 'basic_reroute', 'no_reroute', 'tight_avoidance', 'wide_avoidance'
        """
        if self.classifier is None:
            # Default to basic reroute if no classifier is available
            return 'basic_reroute'
            
        try:
            # Get prediction probabilities
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(features)[0]
                max_prob = max(probs)
                
                # If confidence is too low, fall back to basic reroute
                if max_prob < 0.4:  # 40% confidence threshold
                    return 'basic_reroute'
                    
                # Get the predicted class
                prediction = self.classifier.classes_[np.argmax(probs)]
            else:
                # If no predict_proba, use regular predict
                prediction = self.classifier.predict(features)[0]
            
            if isinstance(prediction, (int, np.integer)):
                # If classifier returns a class index
                action_map = {
                    0: 'basic_reroute',
                    1: 'no_reroute',
                    2: 'tight_avoidance',
                    3: 'wide_avoidance'
                }
                return action_map.get(prediction, 'basic_reroute')
            else:
                # If classifier returns the class name directly
                return prediction
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return 'basic_reroute'  # Default to basic on error 