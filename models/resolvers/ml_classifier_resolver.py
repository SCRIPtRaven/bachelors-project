import os
import pickle
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, RecipientUnavailableAction
)
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance

try:
    import tensorflow as tf
    from tensorflow import keras

    HAS_TF = True
except ImportError:
    HAS_TF = False


class MLClassifierResolver(RuleBasedResolver):
    MODEL_DIR = os.path.join('models', 'resolvers', 'saved_models')
    RF_MODEL_PATH = os.path.join(MODEL_DIR, 'reroute_classifier.pkl')
    NN_MODEL_PATH = os.path.join(MODEL_DIR, 'reroute_classifier_nn.pkl')

    NN_EXPECTED_FEATURES = [
        'disruption_type_road_closure', 'disruption_type_traffic_jam',
        'disruption_severity', 'distance_to_disruption_center',
        'remaining_deliveries', 'distance_along_route_to_disruption',
        'distance_to_next_delivery_along_route', 'next_delivery_before_disruption',
        'alternative_route_density', 'urban_density'
    ]
    RF_FALLBACK_EXPECTED_FEATURES = [
        'disruption_type_road_closure', 'disruption_type_traffic_jam',
        'disruption_severity', 'distance_to_disruption_center',
        'distance_along_route_to_disruption',
        'alternative_route_density'
    ]

    def __init__(self, graph, warehouse_location, max_computation_time=1.0, model_type='random_forest'):
        super().__init__(graph, warehouse_location, max_computation_time)
        self.model_type = model_type
        self.model_expected_features: Optional[List[str]] = None

        if model_type == 'neural_network' and not HAS_TF:
            print(f"Warning: Neural network requested but TensorFlow not available.")
            print("Falling back to random forest. Please install tensorflow to use neural networks.")
            self.model_type = 'random_forest'

        self.classifier, self.scaler, self.label_encoder = self._load_classifier()

        if self.classifier and not self.model_expected_features:
            print(
                f"CRITICAL WARNING: Classifier '{self.model_type}' loaded, but its expected feature list was not determined. Predictions will likely fail.")

    def _load_classifier(self):
        try:
            model_path = self.NN_MODEL_PATH if self.model_type == 'neural_network' else self.RF_MODEL_PATH

            if os.path.exists(model_path):
                if self.model_type == 'neural_network':
                    keras_path_modern = model_path.replace('.pkl', '.keras')
                    if os.path.exists(keras_path_modern):
                        print(f"Loading neural network model from {keras_path_modern} (modern Keras format)")
                        keras_model = tf.keras.models.load_model(keras_path_modern, compile=False)

                        with open(model_path, 'rb') as f:
                            wrapper_data = pickle.load(f)
                        label_encoder = wrapper_data.get('label_encoder')

                        scaler = None
                        meta_path = model_path.replace('.pkl', '_meta.pkl')
                        if os.path.exists(meta_path):
                            with open(meta_path, 'rb') as f:
                                metadata = pickle.load(f)
                                scaler = metadata.get('scaler')

                        if keras_model:
                            self.model_expected_features = self.NN_EXPECTED_FEATURES
                        return keras_model, scaler, label_encoder

                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    if isinstance(model_data, dict) and model_data.get('model_type') == 'neural_network':
                        print(f"Loading neural network model from {model_path} (legacy dict format)")
                        keras_model = model_data.get('keras_model')
                        scaler = model_data.get('scaler')
                        label_encoder = model_data.get('label_encoder')
                        if keras_model:
                            self.model_expected_features = self.NN_EXPECTED_FEATURES
                        return keras_model, scaler, label_encoder
                    elif isinstance(model_data, dict) and 'model_path' in model_data:
                        keras_path_wrapper = model_data.get('model_path')
                        if os.path.exists(keras_path_wrapper):
                            print(f"Loading neural network model from {keras_path_wrapper} (wrapper format)")
                            keras_model = tf.keras.models.load_model(keras_path_wrapper, compile=False)
                            label_encoder = model_data.get('label_encoder')
                            scaler = None
                            meta_path = model_path.replace('.pkl', '_meta.pkl')
                            if os.path.exists(meta_path):
                                with open(meta_path, 'rb') as f:
                                    metadata = pickle.load(f)
                                    scaler = metadata.get('scaler')
                            if keras_model:
                                self.model_expected_features = self.NN_EXPECTED_FEATURES
                            return keras_model, scaler, label_encoder

                    print(f"Warning: Neural network model format at {model_path} not recognized or model missing.")
                    print("Will use fallback rules for decision making.")
                    return None, None, None

                else:
                    print(f"Loading scikit-learn ({self.model_type}) model from {model_path}")
                    with open(model_path, 'rb') as f:
                        sklearn_model = pickle.load(f)

                    if sklearn_model:
                        if hasattr(sklearn_model, 'feature_names_in_') and sklearn_model.feature_names_in_ is not None:
                            self.model_expected_features = list(sklearn_model.feature_names_in_)
                            print(
                                f"DEBUG: Features from {self.model_type} model's feature_names_in_: {self.model_expected_features}")
                        else:
                            print(
                                f"Warning: {self.model_type} model at {model_path} does not have 'feature_names_in_'.")
                            if self.model_type == 'random_forest':
                                print("Using hardcoded fallback feature list for Random Forest.")
                                self.model_expected_features = self.RF_FALLBACK_EXPECTED_FEATURES
                            else:
                                print(
                                    f"Warning: No hardcoded fallback feature list for model type '{self.model_type}'. Feature extraction might be incorrect.")
                    return sklearn_model, None, None
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
        return self.classifier is not None

    def resolve_disruptions(self,
                            state: DeliverySystemState,
                            active_disruptions: List[Disruption],
                            force_process_driver_id: Optional[int] = None
                            ) -> List[DisruptionAction]:
        actions = []

        for disruption in active_disruptions:
            try:
                if force_process_driver_id is not None:
                    drivers_to_process = [force_process_driver_id]
                    print(f"ML Resolver processing {disruption.type.value} for FORCED driver {force_process_driver_id}")
                else:
                    drivers_to_process = self._get_affected_drivers(disruption, state)
                    print(f"ML Resolver processing {disruption.type.value} affecting {len(drivers_to_process)} drivers")

                if disruption.type in [DisruptionType.ROAD_CLOSURE, DisruptionType.TRAFFIC_JAM]:
                    for driver_id in drivers_to_process:
                        if driver_id not in state.driver_positions or driver_id not in state.driver_routes:
                            print(
                                f"Warning (ML): Skipping driver {driver_id} for disruption {disruption.id} as they are missing from state dicts.")
                            continue

                        if self.classifier is not None:
                            action = self._select_optimal_action(driver_id, disruption, state)
                            if action:
                                actions.append(action)
                        else:
                            action = self._create_reroute_action(driver_id, disruption, state)
                            if action:
                                actions.append(action)
                elif disruption.type == DisruptionType.RECIPIENT_UNAVAILABLE:
                    for driver_id in drivers_to_process:
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

    def _select_optimal_action(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        DisruptionAction]:
        try:
            features = self._extract_features(driver_id, disruption, state)

            if features is None:
                print(
                    f"DEBUG (ML Fallback): Feature extraction failed for driver {driver_id}, disruption {disruption.id}. Falling back to basic reroute.")
                return self._create_reroute_action(driver_id, disruption, state)

            action_type, all_probs = self._predict_action_type(features)

            debug_confidences_str = "N/A"
            if all_probs is not None:
                class_names_for_probs = None
                if self.model_type == 'neural_network':
                    if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                        class_names_for_probs = self.label_encoder.classes_
                elif hasattr(self.classifier, 'classes_'):
                    class_names_for_probs = self.classifier.classes_

                if class_names_for_probs is not None and len(class_names_for_probs) == len(all_probs):
                    confidence_details = [f"{name}: {prob:.4f}" for name, prob in zip(class_names_for_probs, all_probs)]
                    debug_confidences_str = ", ".join(confidence_details)
                elif len(all_probs) > 0:
                    default_action_map_for_probs = {
                        0: 'basic_reroute', 1: 'no_action',
                        2: 'tight_avoidance', 3: 'wide_avoidance'
                    }
                    if len(all_probs) <= len(default_action_map_for_probs):
                        confidence_details = [f"{default_action_map_for_probs.get(i, f'Class_{i}')}: {prob:.4f}" for
                                              i, prob in enumerate(all_probs)]
                    else:
                        confidence_details = [f"Class_{i}: {prob:.4f}" for i, prob in enumerate(all_probs)]
                    debug_confidences_str = ", ".join(confidence_details)
                else:
                    debug_confidences_str = "Probs array empty or invalid"

            print(
                f"DEBUG (ML Predict): Predicted action type: {action_type} (Confidences: {debug_confidences_str}) for driver {driver_id}, disruption {disruption.id}")

            if action_type == 'no_action':
                return self._create_no_action(driver_id, disruption)
            elif action_type == 'tight_avoidance':
                return self._create_tight_avoidance_action(driver_id, disruption, state)
            elif action_type == 'wide_avoidance':
                return self._create_wide_avoidance_action(driver_id, disruption, state)
            else:
                return self._create_reroute_action(driver_id, disruption, state)

        except Exception as e:
            print(f"Error selecting optimal action: {e}")
            import traceback
            traceback.print_exc()
            return self._create_reroute_action(driver_id, disruption, state)

    def _extract_features(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        pd.DataFrame]:
        try:
            features = {}

            features['disruption_type_road_closure'] = 1.0 if disruption.type == DisruptionType.ROAD_CLOSURE else 0.0
            features['disruption_type_traffic_jam'] = 1.0 if disruption.type == DisruptionType.TRAFFIC_JAM else 0.0
            features['disruption_severity'] = disruption.severity

            driver_pos = state.driver_positions.get(driver_id)
            driver_route_info = state.driver_routes.get(driver_id)
            assigned_delivery_indices = state.driver_assignments.get(driver_id, [])
            driver_deliveries = [state.deliveries[i] for i in assigned_delivery_indices if i < len(state.deliveries)]

            if driver_pos is None or driver_route_info is None:
                print(f"Warning: Missing position or route info for driver {driver_id}")
                return None

            current_route_points = driver_route_info.get('points', [])
            current_route_nodes = driver_route_info.get('nodes', [])
            current_delivery_index = driver_route_info.get('current_delivery_index', 0)

            graph_to_use = state.graph
            if graph_to_use is None:
                print("Error: Graph object is None in DeliverySystemState!")
                return None

            if not current_route_nodes and current_route_points:
                try:
                    current_route_nodes = [ox.nearest_nodes(graph_to_use, X=p[1], Y=p[0]) for p in current_route_points]
                except Exception as e:
                    print(f"Warning: Could not determine nodes for driver {driver_id}'s route from points. Error: {e}")

            distance_to_center = calculate_haversine_distance(driver_pos, disruption.location)
            max_distance_norm = 10000
            features['distance_to_disruption_center'] = min(1.0, distance_to_center / max_distance_norm)

            dist_along_route = distance_to_center
            next_delivery_dist_along_route = float('inf')
            next_delivery_before_disruption = 0.0

            if current_route_nodes:
                try:
                    disruption_node = ox.nearest_nodes(graph_to_use, X=disruption.location[1], Y=disruption.location[0])
                    driver_node = ox.nearest_nodes(graph_to_use, X=driver_pos[1], Y=driver_pos[0])

                    current_node_index_in_route = -1
                    if driver_node in current_route_nodes:
                        current_node_index_in_route = current_route_nodes.index(driver_node)

                    disruption_on_route = False
                    nodes_to_disruption = []
                    temp_dist_along_route = 0.0
                    next_delivery_node_found_on_path = None

                    for i in range(max(0, current_node_index_in_route), len(current_route_nodes) - 1):
                        u, v = current_route_nodes[i], current_route_nodes[i + 1]
                        edge_data = graph_to_use.get_edge_data(u, v)
                        if edge_data:
                            edge_data = edge_data.get(0, edge_data)
                            segment_length = edge_data.get('length', 0)
                            nodes_to_disruption.append(u)
                            temp_dist_along_route += segment_length

                            if next_delivery_node_found_on_path is None and i >= current_delivery_index:
                                if current_delivery_index < len(driver_deliveries):
                                    next_del_obj = driver_deliveries[current_delivery_index]
                                    if hasattr(next_del_obj, 'location') and isinstance(next_del_obj.location, tuple):
                                        next_delivery_point_coords = next_del_obj.location
                                        temp_next_delivery_node = ox.nearest_nodes(graph_to_use,
                                                                                   X=next_delivery_point_coords[1],
                                                                                   Y=next_delivery_point_coords[0])
                                        if v == temp_next_delivery_node:
                                            next_delivery_dist_along_route = temp_dist_along_route
                                            next_delivery_node_found_on_path = temp_next_delivery_node
                                    else:
                                        pass

                            if v == disruption_node:
                                disruption_on_route = True
                                nodes_to_disruption.append(v)
                                break

                    if disruption_on_route:
                        dist_along_route = temp_dist_along_route
                        if next_delivery_node_found_on_path is not None and next_delivery_node_found_on_path in nodes_to_disruption:
                            next_delivery_before_disruption = 1.0
                    elif driver_node and disruption_node:
                        try:
                            dist_along_route = nx.shortest_path_length(graph_to_use, source=driver_node,
                                                                       target=disruption_node, weight='length')
                        except nx.NetworkXNoPath:
                            dist_along_route = distance_to_center
                    else:
                        dist_along_route = distance_to_center


                except Exception as e:
                    print(f"Warning: Error calculating distance along route for driver {driver_id}: {e}")

            features['distance_along_route_to_disruption'] = min(1.0, dist_along_route / max_distance_norm)
            features['distance_to_next_delivery_along_route'] = min(1.0,
                                                                    next_delivery_dist_along_route / max_distance_norm) if next_delivery_dist_along_route != float(
                'inf') else 1.0
            features['next_delivery_before_disruption'] = next_delivery_before_disruption

            max_deliveries_norm = 50
            remaining_deliveries_count = len(driver_deliveries) - current_delivery_index
            features['remaining_deliveries'] = min(1.0, max(0, remaining_deliveries_count) / max_deliveries_norm)

            route_progress = driver_route_info.get('progress')
            if route_progress is None and driver_deliveries:
                route_progress = current_delivery_index / max(1, len(driver_deliveries))
            features['route_progress'] = route_progress if route_progress is not None else 0.0

            features['alternative_route_density'] = 0.5
            features['urban_density'] = 0.5

            df = pd.DataFrame([features])

            if not self.model_expected_features:
                print(
                    f"DEBUG (ML Fallback): Model's expected feature list not available for {self.model_type}. Cannot create correctly shaped feature DataFrame.")
                return None

            try:
                df = df.reindex(columns=self.model_expected_features, fill_value=0.0)
            except Exception as reindex_err:
                print(
                    f"DEBUG (ML Fallback): Error during feature DataFrame reindexing for {self.model_type}: {reindex_err}")
                print(f"  DataFrame columns before reindex: {list(df.columns)}")
                print(f"  Target model_expected_features: {self.model_expected_features}")
                return None

            if df.shape[1] != len(self.model_expected_features):
                print(f"DEBUG (ML Warning): Post-reindex DataFrame column count ({df.shape[1]}) "
                      f"does not match length of model_expected_features ({len(self.model_expected_features)}) for {self.model_type}.")
                print(f"  DataFrame columns: {list(df.columns)}")
                print(f"  Expected columns: {self.model_expected_features}")

            return df

        except Exception as e:
            print(f"Error extracting features for {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_action_type(self, features: pd.DataFrame) -> Tuple[str, Optional[np.ndarray]]:
        if self.classifier is None:
            return 'basic_reroute', None

        try:
            probs: Optional[np.ndarray] = None
            prediction: str = 'basic_reroute'

            if self.model_type == 'neural_network':
                if not HAS_TF or not isinstance(self.classifier, tf.keras.Model):
                    print(
                        "DEBUG (ML Fallback): NN model type selected but classifier is not a valid Keras model. Falling back.")
                    return 'basic_reroute', None

                try:
                    features_input = features
                    raw_probs = self.classifier.predict(features_input)
                    probs = raw_probs[0] if raw_probs.ndim == 2 and raw_probs.shape[0] == 1 else raw_probs

                except Exception as pred_err:
                    print(f"DEBUG (ML Fallback): NN prediction call failed: {pred_err}. Falling back.")
                    return 'basic_reroute', None

                max_prob_val = np.max(probs)

                if max_prob_val < 0.4:
                    print(f"DEBUG (ML Fallback): NN confidence ({max_prob_val:.2f}) below threshold. Falling back.")
                    return 'basic_reroute', probs

                prediction_idx = np.argmax(probs)

                if self.label_encoder is not None:
                    prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
                else:
                    action_map = {0: 'basic_reroute', 1: 'no_action', 2: 'tight_avoidance', 3: 'wide_avoidance'}
                    prediction = action_map.get(prediction_idx, 'basic_reroute')
            else:
                if not hasattr(self.classifier, 'predict'):
                    print("DEBUG (ML Fallback): Sklearn classifier missing 'predict' method. Falling back.")
                    return 'basic_reroute', None

                if hasattr(self.classifier, 'predict_proba'):
                    try:
                        raw_probs = self.classifier.predict_proba(features)
                        probs = raw_probs[0] if raw_probs.ndim == 2 and raw_probs.shape[0] == 1 else raw_probs
                    except Exception as pred_err:
                        print(f"DEBUG (ML Fallback): Sklearn predict_proba call failed: {pred_err}. Falling back.")
                        return 'basic_reroute', None

                    max_prob_val = np.max(probs)

                    if max_prob_val < 0.4:
                        print(
                            f"DEBUG (ML Fallback): Sklearn confidence ({max_prob_val:.2f}) below threshold. Falling back.")
                        return 'basic_reroute', probs

                    prediction = self.classifier.classes_[np.argmax(probs)]
                else:
                    try:
                        prediction = self.classifier.predict(features)[0]
                    except Exception as pred_err:
                        print(f"DEBUG (ML Fallback): Sklearn predict call failed: {pred_err}. Falling back.")
                        return 'basic_reroute', None

            if isinstance(prediction, (int, np.integer)):
                action_map = {0: 'basic_reroute', 1: 'no_action', 2: 'tight_avoidance', 3: 'wide_avoidance'}
                final_prediction_str = action_map.get(prediction, 'basic_reroute')
            else:
                final_prediction_str = str(prediction)
                if final_prediction_str == 'no_reroute':
                    final_prediction_str = 'no_action'

            return final_prediction_str, probs

        except Exception as e:
            print(f"DEBUG (ML Fallback): Error during prediction processing: {e}")
            import traceback
            traceback.print_exc()
            return 'basic_reroute', None
