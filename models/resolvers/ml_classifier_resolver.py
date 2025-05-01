import os
import pickle
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Set

import networkx as nx
import osmnx as ox

from models.entities.delivery import Delivery
from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction, ActionType
)
from models.resolvers.resolver import DisruptionResolver
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance

# For neural networks
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


class MLClassifierResolver(RuleBasedResolver):
    """
    Machine learning-based resolver that uses a trained classifier to select 
    the optimal reroute action for traffic jams and road closures.
    
    For recipient unavailable disruptions, it behaves the same as the RuleBasedResolver.
    """

    MODEL_DIR = os.path.join('models', 'resolvers', 'saved_models')
    RF_MODEL_PATH = os.path.join(MODEL_DIR, 'reroute_classifier.pkl')
    NN_MODEL_PATH = os.path.join(MODEL_DIR, 'reroute_classifier_nn.pkl')
    
    def __init__(self, graph, warehouse_location, max_computation_time=1.0, model_type='random_forest'):
        super().__init__(graph, warehouse_location, max_computation_time)
        self.model_type = model_type
        
        if model_type == 'neural_network' and not HAS_TF:
            print(f"Warning: Neural network requested but TensorFlow not available.")
            print("Falling back to random forest. Please install tensorflow to use neural networks.")
            self.model_type = 'random_forest'
            
        # Load the appropriate classifier
        self.classifier, self.scaler, self.label_encoder = self._load_classifier()
        
    def _load_classifier(self):
        """Load the trained classifier model if available"""
        try:
            model_path = self.NN_MODEL_PATH if self.model_type == 'neural_network' else self.RF_MODEL_PATH
            
            if os.path.exists(model_path):
                # Check for modern format with .keras file
                keras_path = model_path.replace('.pkl', '.keras')
                if self.model_type == 'neural_network' and os.path.exists(keras_path):
                    print(f"Loading neural network model from {keras_path} (modern Keras format)")
                    
                    # Load the keras model
                    keras_model = tf.keras.models.load_model(keras_path)
                    
                    # Load metadata from the pickle file
                    with open(model_path, 'rb') as f:
                        wrapper_data = pickle.load(f)
                    
                    # Extract label encoder and other info
                    label_encoder = wrapper_data.get('label_encoder')
                    
                    # Load scaler if available
                    scaler = None
                    meta_path = model_path.replace('.pkl', '_meta.pkl')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'rb') as f:
                            metadata = pickle.load(f)
                            scaler = metadata.get('scaler')
                    
                    return keras_model, scaler, label_encoder
                
                # Legacy loading for older model formats
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # For neural network models that are stored as dictionaries
                if isinstance(model_data, dict) and self.model_type == 'neural_network':
                    if 'model_type' in model_data and model_data['model_type'] == 'neural_network':
                        print(f"Loading neural network model from {model_path} (legacy format)")
                        keras_model = model_data.get('keras_model')
                        scaler = model_data.get('scaler')
                        label_encoder = model_data.get('label_encoder')
                        return keras_model, scaler, label_encoder
                    elif 'model_path' in model_data:
                        # For models using the newer wrapper format
                        keras_path = model_data.get('model_path')
                        if os.path.exists(keras_path):
                            print(f"Loading neural network model from {keras_path}")
                            keras_model = tf.keras.models.load_model(keras_path)
                            label_encoder = model_data.get('label_encoder')
                            
                            # Try to load scaler
                            scaler = None
                            meta_path = model_path.replace('.pkl', '_meta.pkl')
                            if os.path.exists(meta_path):
                                with open(meta_path, 'rb') as f:
                                    metadata = pickle.load(f)
                                    scaler = metadata.get('scaler')
                            
                            return keras_model, scaler, label_encoder
                    else:
                        print(f"Warning: Neural network model requested but not found at {model_path}.")
                        print("Will use fallback rules for decision making.")
                        return None, None, None
                else:
                    # For scikit-learn models like RandomForest
                    print(f"Loading random forest model from {model_path}")
                    return model_data, None, None
            else:
                print(f"Warning: ML Classifier model not found at {model_path}. "
                      f"Will use fallback rules for decision making.")
                return None, None, None
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            
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
            if action_type == 'no_action':
                return self._create_no_action(driver_id, disruption)
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
            
            # Check for additional features from training data
            additional_features = [
                'severity',
                'remaining_deliveries',
                'distance_to_next_delivery',
                'alternative_route_density',
                'urban_density',
                'route_progress'
            ]
            
            # Try to extract additional features if available
            try:
                # Extract remaining deliveries
                if 'remaining_deliveries' in additional_features:
                    remaining_deliveries = len(state.driver_assignments.get(driver_id, []))
                    features['remaining_deliveries'] = min(1.0, remaining_deliveries / 20.0)  # Normalize
                
                # Extract route progress
                if 'route_progress' in additional_features:
                    route_data = state.driver_routes.get(driver_id, {})
                    progress = route_data.get('progress', 0.5)  # Default to middle of route
                    features['route_progress'] = progress
                
                # Add default values for features we can't easily compute now
                if 'severity' in additional_features:
                    features['severity'] = disruption.severity if hasattr(disruption, 'severity') else 0.5
                
                if 'distance_to_next_delivery' in additional_features:
                    features['distance_to_next_delivery'] = 0.5  # Default middle value
                
                if 'alternative_route_density' in additional_features:
                    features['alternative_route_density'] = 0.5  # Default middle value
                
                if 'urban_density' in additional_features:
                    features['urban_density'] = 0.5  # Default middle value
            except:
                # If additional features extraction fails, ignore them
                pass
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # For neural network models, apply the scaler
            if self.model_type == 'neural_network' and self.scaler is not None:
                # Scale the features
                df_array = self.scaler.transform(df)
                return df_array
            
            return df
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_action_type(self, features: pd.DataFrame) -> str:
        """
        Use the classifier to predict the best action type
        
        Returns one of: 'basic_reroute', 'no_action', 'tight_avoidance', 'wide_avoidance'
        """
        if self.classifier is None:
            # Default to basic reroute if no classifier is available
            return 'basic_reroute'
            
        try:
            if self.model_type == 'neural_network':
                # Neural network prediction
                if isinstance(features, np.ndarray):
                    probs = self.classifier.predict(features)[0]
                    max_prob = np.max(probs)
                    
                    # If confidence is too low, fall back to basic reroute
                    if max_prob < 0.4:  # 40% confidence threshold
                        return 'basic_reroute'
                    
                    # Get predicted class index
                    prediction_idx = np.argmax(probs)
                    
                    # Convert to class name using label encoder
                    if self.label_encoder is not None:
                        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
                    else:
                        # Fallback to index mapping if no encoder
                        action_map = {
                            0: 'basic_reroute',
                            1: 'no_action',
                            2: 'tight_avoidance',
                            3: 'wide_avoidance'
                        }
                        prediction = action_map.get(prediction_idx, 'basic_reroute')
                else:
                    return 'basic_reroute'
            else:
                # Standard sklearn model prediction
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
                    1: 'no_action',
                    2: 'tight_avoidance',
                    3: 'wide_avoidance'
                }
                return action_map.get(prediction, 'basic_reroute')
            else:
                # If classifier returns the class name directly
                # Convert old 'no_reroute' to 'no_action' if found
                if prediction == 'no_reroute':
                    return 'no_action'
                return prediction
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return 'basic_reroute'  # Default to basic on error 