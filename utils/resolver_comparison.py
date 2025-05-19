import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models.services import graph_service

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.ml_classifier_resolver import MLClassifierResolver
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction,
    NoAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction,
    ActionType
)
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import calculate_route_length, calculate_travel_time, \
    find_route_enter_disruption_index
from utils.ml_data_generator import MLDataGenerator
from config.config import Config

COMPARISON_TARGET_ACTION_TYPES = [
    ActionType.NO_ACTION.display_name,
    ActionType.REROUTE_BASIC.display_name,
    ActionType.REROUTE_TIGHT_AVOIDANCE.display_name,
    ActionType.REROUTE_WIDE_AVOIDANCE.display_name
]


def get_current_model_type() -> str:
    config_file = os.path.join('config', 'ml_model_config.pkl')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
                if 'model_type' in config:
                    return config['model_type']
        except Exception as e:
            print(f"Error reading model configuration: {e}")

    return 'random_forest'


def _safe_json_loads(json_string, default=None):
    if pd.isna(json_string):
        return default
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


class ResolverComparison:
    RESULTS_DIR = os.path.join('models', 'resolvers', 'comparison_results')

    def __init__(self, graph_obj, warehouse_location, delivery_points, num_scenarios=100,
                 random_seed=42,
                 model_type=None, input_csv_path: Optional[str] = None):
        self.G = graph_obj
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.num_scenarios = num_scenarios
        self.random_seed = random_seed
        self.input_csv_path = input_csv_path

        self.model_type = model_type or get_current_model_type()
        print(f"Using model type: {self.model_type}")

        os.makedirs(self.RESULTS_DIR, exist_ok=True)

        self.rule_based_resolver = RuleBasedResolver(self.G, warehouse_location)
        self.ml_resolver = MLClassifierResolver(self.G, warehouse_location,
                                                model_type=self.model_type)

        self.ml_model_loaded = self.ml_resolver.has_classifier()
        if not self.ml_model_loaded:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Failed to load the specified ML model ('{self.model_type}').")
            print("Comparison cannot proceed accurately without the ML model.")
            print("Please check model files and configuration.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if not self.input_csv_path:
            self.data_generator = MLDataGenerator(
                graph=self.G,
                warehouse_location=warehouse_location,
                delivery_points=delivery_points,
                num_samples=num_scenarios,
                random_seed=random_seed,
                save_full_scenario_data=False
            )
            print("Initialized MLDataGenerator for random scenario generation.")
        else:
            self.data_generator = None
            print(f"Will load scenarios from: {self.input_csv_path}")

        self.scenario_results = []

    def run_comparison(self) -> str:
        print(f"Starting comparison for {self.num_scenarios} scenarios...")

        if not self.ml_model_loaded:
            print("ERROR: ML model failed to load during initialization. Aborting comparison.")
            return "Failed - ML Model Load Error"

        if self.input_csv_path:
            success = self._load_and_process_scenarios_from_csv()
            if not success:
                print("Comparison failed due to issues processing the input CSV.")
                return "Failed"
        else:
            if not self.data_generator:
                print("Error: Data generator not initialized and no input CSV provided.")
                return "Failed"

            print(f"Generating {self.num_scenarios} random comparison scenarios...")
            progress_step = max(1, self.num_scenarios // 10)
            scenarios_processed = 0
            scenarios_failed_generation = 0

            while scenarios_processed < self.num_scenarios:
                if (scenarios_processed + scenarios_failed_generation) % progress_step == 0:
                    print(
                        f"Attempting scenario {scenarios_processed + scenarios_failed_generation + 1} / {self.num_scenarios} target... ({scenarios_processed} successful)")

                disruption, state, driver_id = self.data_generator._generate_random_scenario_on_route()

                if not disruption or not state or driver_id is None:
                    scenarios_failed_generation += 1
                    if scenarios_failed_generation > self.num_scenarios * 5:
                        print("Error: Excessive scenario generation failures. Aborting.")
                        break
                    continue

                result = self._compare_resolvers_on_scenario(disruption, state, driver_id)
                if result:
                    self.scenario_results.append(result)
                    scenarios_processed += 1

            print(
                f"Finished processing random scenarios. Processed: {scenarios_processed}, Failed generation: {scenarios_failed_generation}")

        return self._analyze_and_save_results()

    def _load_and_process_scenarios_from_csv(self) -> bool:
        try:
            print(f"Loading scenarios from {self.input_csv_path}...")
            all_scenarios_df = pd.read_csv(self.input_csv_path)
            print(f"Loaded {len(all_scenarios_df)} total scenarios from CSV.")
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {self.input_csv_path}")
            return False
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

        if 'best_action' not in all_scenarios_df.columns:
            print("Error: Input CSV must contain a 'best_action' column for selection.")
            return False

        print(
            f"Selecting up to {self.num_scenarios} scenarios, aiming for balance across {len(COMPARISON_TARGET_ACTION_TYPES)} types...")
        num_target_types = len(COMPARISON_TARGET_ACTION_TYPES)
        samples_per_type = self.num_scenarios // num_target_types
        remainder = self.num_scenarios % num_target_types

        selected_rows = []
        actual_counts = Counter()

        for i, action_type in enumerate(COMPARISON_TARGET_ACTION_TYPES):
            target_count = samples_per_type + (1 if i < remainder else 0)

            type_df = all_scenarios_df[all_scenarios_df['best_action'] == action_type].copy()

            if len(type_df) == 0:
                print(
                    f"Warning: No scenarios found for action type '{action_type}' in the input CSV.")
                continue

            if len(type_df) >= target_count:
                selected_sample = type_df.sample(n=target_count, random_state=self.random_seed)
            else:
                print(
                    f"Warning: Only found {len(type_df)} scenarios for '{action_type}', requested {target_count}. Using all available.")
                selected_sample = type_df

            selected_rows.append(selected_sample)
            actual_counts[action_type] += len(selected_sample)

        if not selected_rows:
            print("Error: No suitable scenarios selected from the CSV. Cannot proceed.")
            return False

        final_selection_df = pd.concat(selected_rows, ignore_index=True)
        total_selected = len(final_selection_df)
        print(f"Selected a total of {total_selected} scenarios.")
        print(f"Selected counts per action type: {dict(actual_counts)}")

        print(f"Processing {total_selected} selected scenarios...")
        scenarios_processed = 0
        scenarios_failed_reconstruction = 0
        progress_step = max(1, total_selected // 10)

        for index, row in final_selection_df.iterrows():
            if (scenarios_processed + scenarios_failed_reconstruction) % progress_step == 0:
                print(
                    f"Processing selected scenario {scenarios_processed + scenarios_failed_reconstruction + 1}/{total_selected}...")

            disruption, state, driver_id = self._reconstruct_scenario_from_row(row)

            if disruption and state and driver_id is not None:
                result = self._compare_resolvers_on_scenario(disruption, state, driver_id)
                if result:
                    self.scenario_results.append(result)
                    scenarios_processed += 1
            else:
                scenarios_failed_reconstruction += 1

        print(
            f"Finished processing scenarios from CSV. Reconstructed & Compared: {scenarios_processed}, Failed Reconstruction: {scenarios_failed_reconstruction}")
        return True

    def _reconstruct_scenario_from_row(self, row: pd.Series) -> Tuple[
        Optional[Disruption], Optional[DeliverySystemState], Optional[int]]:
        required_cols = [
            'disruption_id', 'target_driver_id', 'disruption_raw_type',
            'disruption_location_lat', 'disruption_location_lon',
            'disruption_severity', 'disruption_affected_area_radius', 'disruption_duration',
            'initial_driver_position_lat', 'initial_driver_position_lon',
            'initial_driver_route_points_json', 'all_delivery_points_original_json',
            'warehouse_location_original_json'
        ]

        if not all(col in row.index for col in required_cols):
            print(
                f"Error: Missing required columns in CSV row for reconstruction. Required: {required_cols}")
            return None, None, None

        try:
            disruption_id = int(row['disruption_id'])
            target_driver_id = int(row['target_driver_id'])
            disruption_type_str = row['disruption_raw_type']
            try:
                disruption_type = DisruptionType[disruption_type_str]
            except KeyError:
                print(f"Error: Invalid disruption_raw_type '{disruption_type_str}' in CSV.")
                return None, None, None

            disruption = Disruption(
                id=disruption_id,
                location=(row['disruption_location_lat'], row['disruption_location_lon']),
                type=disruption_type,
                severity=row['disruption_severity'],
                affected_area_radius=row['disruption_affected_area_radius'],
                duration=int(row['disruption_duration'])
            )

            warehouse_loc = _safe_json_loads(row['warehouse_location_original_json'])
            all_deliveries = _safe_json_loads(row['all_delivery_points_original_json'])
            initial_route_points = _safe_json_loads(row['initial_driver_route_points_json'])
            original_initial_pos = (row['initial_driver_position_lat'],
                                    row['initial_driver_position_lon'])

            if warehouse_loc is None or all_deliveries is None or initial_route_points is None or \
                    pd.isna(original_initial_pos[0]) or pd.isna(original_initial_pos[1]):
                print(
                    f"Error: Failed to parse required fields from CSV for state reconstruction (warehouse, deliveries, route, position).")
                return None, None, None

            initial_pos = original_initial_pos
            if initial_route_points and len(initial_route_points) > 1:
                enter_index = find_route_enter_disruption_index(
                    route_points=initial_route_points,
                    disruption_location=disruption.location,
                    disruption_radius=disruption.affected_area_radius,
                    start_index=0
                )
                if enter_index > 0:
                    placement_offset = 2
                    driver_pos_index = max(0, enter_index - placement_offset)
                    if driver_pos_index < len(initial_route_points):
                        initial_pos = initial_route_points[driver_pos_index]
                    else:
                        print(
                            f"Warning: Calculated driver position index {driver_pos_index} is out of bounds for route length {len(initial_route_points)}. Using original position.")
                        initial_pos = original_initial_pos
                elif enter_index == 0:
                    print(
                        f"Warning: Route starts inside disruption (entry_index=0). Using original position.")
                    initial_pos = original_initial_pos
                else:
                    initial_pos = original_initial_pos
            else:
                print(
                    f"Warning: Initial route points list is too short or missing. Using original position.")
                initial_pos = original_initial_pos
            target_driver_assignments = {str(target_driver_id): []}

            state = DeliverySystemState(
                drivers=[],
                deliveries=all_deliveries,
                disruptions=[disruption],
                simulation_time=0,
                graph=self.G,
                warehouse_location=warehouse_loc
            )

            state.driver_assignments = target_driver_assignments
            state.driver_positions = {target_driver_id: initial_pos}
            state.driver_routes = {
                target_driver_id: {'points': initial_route_points, 'nodes': [], 'progress': 0.0}}

            return disruption, state, target_driver_id

        except Exception as e:
            print(
                f"Error during scenario reconstruction from row {row.get('disruption_id', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _compare_resolvers_on_scenario(self,
                                       disruption: Disruption,
                                       state: DeliverySystemState,
                                       driver_id: int) -> Dict[str, Any]:
        try:
            features = self.ml_resolver._extract_features(driver_id, disruption, state)

            if features is None:
                return None

            rule_based_start = time.time()
            rule_based_actions = self.rule_based_resolver.resolve_disruptions(state, [disruption],
                                                                              force_process_driver_id=driver_id)
            rule_based_time = time.time() - rule_based_start

            ml_start = time.time()
            ml_actions = self.ml_resolver.resolve_disruptions(state, [disruption],
                                                              force_process_driver_id=driver_id)
            ml_time = time.time() - ml_start

            rule_based_action = next(
                (a for a in rule_based_actions if
                 hasattr(a, 'driver_id') and a.driver_id == driver_id), None)
            ml_action = next(
                (a for a in ml_actions if hasattr(a, 'driver_id') and a.driver_id == driver_id),
                None)

            rule_based_type = self._get_action_type(rule_based_action)
            ml_type = self._get_action_type(ml_action)

            ml_confidence = None
            if self.ml_resolver.classifier is not None and hasattr(self.ml_resolver.classifier,
                                                                   'predict_proba'):
                try:
                    if isinstance(features, pd.DataFrame):
                        if len(features) == 1:
                            features = features.iloc[0:1]
                    elif isinstance(features, np.ndarray):
                        feature_names = [
                            "disruption_type_road_closure",
                            "disruption_type_traffic_jam",
                            "distance_to_disruption"
                        ]
                        if len(features.shape) == 1:
                            features = pd.DataFrame([features], columns=feature_names)
                        else:
                            features = pd.DataFrame(features, columns=feature_names)
                    else:
                        feature_names = [
                            "disruption_type_road_closure",
                            "disruption_type_traffic_jam",
                            "distance_to_disruption"
                        ]
                        features = pd.DataFrame([features], columns=feature_names)

                    probs = self.ml_resolver.classifier.predict_proba(features)[0]
                    ml_confidence = max(probs)
                except Exception as e:
                    print(f"Error getting confidence scores: {e}")
                    import traceback
                    traceback.print_exc()

            original_route = state.driver_routes.get(driver_id, {}).get('points', [])
            original_length = calculate_route_length(original_route)
            original_time = calculate_travel_time(original_route, self.G)

            rule_based_metrics = self._evaluate_action(rule_based_action, original_route,
                                                       original_length,
                                                       original_time, disruption)
            ml_metrics = self._evaluate_action(ml_action, original_route, original_length,
                                               original_time, disruption)

            time_improvement = 0
            length_improvement = 0
            length_diff_meters = 0
            time_diff_seconds = 0

            if rule_based_metrics and ml_metrics:
                length_diff_meters = rule_based_metrics["route_length"] - ml_metrics["route_length"]
                time_diff_seconds = rule_based_metrics["travel_time"] - ml_metrics["travel_time"]

                if rule_based_metrics["travel_time"] > 0:
                    time_improvement = (time_diff_seconds / rule_based_metrics["travel_time"]) * 100

                if rule_based_metrics["route_length"] > 0:
                    if rule_based_type == "no_action" and ml_type != "no_action":
                        length_improvement = -100.0
                    elif rule_based_type != "no_action" and ml_type == "no_action":
                        length_improvement = 100.0
                    else:
                        raw_improvement = (length_diff_meters / rule_based_metrics[
                            "route_length"]) * 100

                        length_improvement = max(-100.0, min(100.0, raw_improvement))

            result = {
                "disruption_type": disruption.type.value,
                "disruption_severity": disruption.severity,
                "disruption_id": disruption.id,
                "driver_id": driver_id,

                "rule_based_action": rule_based_type,
                "ml_action": ml_type,
                "actions_match": rule_based_type == ml_type,

                "rule_based_time_ms": rule_based_time * 1000,
                "ml_time_ms": ml_time * 1000,

                "original_route_length": original_length,
                "original_travel_time": original_time,

                "rule_based_route_length": rule_based_metrics.get(
                    "route_length") if rule_based_metrics else None,
                "rule_based_travel_time": rule_based_metrics.get(
                    "travel_time") if rule_based_metrics else None,

                "ml_route_length": ml_metrics.get("route_length") if ml_metrics else None,
                "ml_travel_time": ml_metrics.get("travel_time") if ml_metrics else None,

                "time_improvement_pct": time_improvement,
                "length_improvement_pct": length_improvement,
                "length_diff_meters": length_diff_meters,
                "time_diff_seconds": time_diff_seconds,

                "ml_confidence": ml_confidence
            }

            if features is not None:
                feature_names = [
                    "disruption_type_road_closure",
                    "disruption_type_traffic_jam",
                    "distance_to_disruption"
                ]

                if isinstance(features, pd.DataFrame):
                    features = features.values.flatten()
                elif isinstance(features, np.ndarray):
                    features = features.flatten()

                for i, name in enumerate(feature_names):
                    if i < len(features):
                        result[f"feature_{name}"] = features[i]

            return result

        except Exception as e:
            print(f"Error comparing resolvers on scenario: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_action_type(self, action: DisruptionAction) -> str:
        if action is None:
            return "no_action"
        elif isinstance(action, NoAction):
            return "no_action"
        elif isinstance(action, RerouteTightAvoidanceAction):
            return "tight_avoidance"
        elif isinstance(action, RerouteWideAvoidanceAction):
            return "wide_avoidance"
        elif isinstance(action, RerouteBasicAction):
            return "basic_reroute"
        else:
            return f"unknown_{type(action).__name__}"

    def _evaluate_action(self, action: Optional[DisruptionAction],
                         original_route: List[Tuple[float, float]],
                         original_length: float,
                         original_time: float,
                         disruption: Disruption) -> Optional[Dict[str, float]]:
        original_graph_state = None
        try:
            if action is None:
                action_to_eval = NoAction(driver_id=-1, affected_disruption_id=disruption.id)
            else:
                action_to_eval = action

            original_graph_state = self._apply_disruption_to_graph(disruption)

            if isinstance(action_to_eval, NoAction):
                new_route = original_route
                if not new_route or len(new_route) < 2:
                    new_length = 0.0
                    new_time = 0.0 if new_length == 0.0 else float('inf')
                else:
                    new_length = calculate_route_length(new_route)
                    new_time = calculate_travel_time(new_route, self.G, disruption=disruption)

            elif hasattr(action_to_eval, 'new_route') and action_to_eval.new_route:
                new_route = action_to_eval.new_route
                if not new_route or len(new_route) < 2:
                    new_length = 0.0
                    new_time = 0.0 if new_length == 0.0 else float('inf')
                else:
                    new_length = calculate_route_length(new_route)
                    new_time = calculate_travel_time(new_route, self.G, disruption=disruption)
            else:
                print(f"Warning: Cannot evaluate action of type {type(action_to_eval)}")
                return None

            detour_length_factor = new_length / max(0.1,
                                                    original_length) if original_length > 0 else 0.0

            return {
                "route_length": new_length,
                "travel_time": new_time,
                "detour_factor": detour_length_factor
            }

        except Exception as e:
            print(f"Error evaluating action: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if original_graph_state is not None:
                self._revert_graph_changes(original_graph_state)

    def _analyze_and_save_results(self) -> str:
        if not self.scenario_results:
            print("No results to analyze.")
            return None

        df = pd.DataFrame(self.scenario_results)

        total_scenarios = len(df)
        matching_actions = df['actions_match'].sum()
        matching_pct = matching_actions / total_scenarios * 100

        avg_time_improvement = df['time_improvement_pct'].mean()
        median_time_improvement = df['time_improvement_pct'].median()

        avg_length_improvement = df['length_improvement_pct'].mean()
        median_length_improvement = df['length_improvement_pct'].median()

        avg_length_diff = df['length_diff_meters'].mean()
        avg_time_diff = df['time_diff_seconds'].mean()

        rule_based_avg_time = df['rule_based_time_ms'].mean()
        ml_avg_time = df['ml_time_ms'].mean()

        confidence_stats = {}
        if 'ml_confidence' in df.columns:
            confidence_stats = {
                'avg_confidence': df['ml_confidence'].mean(),
                'min_confidence': df['ml_confidence'].min(),
                'max_confidence': df['ml_confidence'].max(),
                'low_confidence_count': (df['ml_confidence'] < 0.4).sum(),
                'low_confidence_pct': (df['ml_confidence'] < 0.4).sum() / total_scenarios * 100
            }

        feature_importance = {}
        if self.ml_resolver.classifier is not None and hasattr(self.ml_resolver.classifier,
                                                               'predict_proba'):
            try:
                feature_columns = [col for col in df.columns if col.startswith('feature_')]
                feature_names = [col.replace('feature_', '') for col in feature_columns]

                X = df[feature_columns].values
                y = df['ml_action'].values

                result = permutation_importance(
                    self.ml_resolver.classifier,
                    X,
                    y,
                    n_repeats=10,
                    random_state=42
                )

                for i, name in enumerate(feature_names):
                    feature_importance[name] = {
                        'importance': result.importances_mean[i],
                        'std': result.importances_std[i]
                    }
            except Exception as e:
                print(f"Error calculating feature importance: {e}")

        summary = {
            "total_scenarios": total_scenarios,
            "matching_actions": matching_actions,
            "matching_actions_pct": matching_pct,
            "avg_time_improvement_pct": avg_time_improvement,
            "median_time_improvement_pct": median_time_improvement,
            "avg_length_improvement_pct": avg_length_improvement,
            "median_length_improvement_pct": median_length_improvement,
            "avg_length_diff_meters": avg_length_diff,
            "avg_time_diff_seconds": avg_time_diff,
            "rule_based_avg_time_ms": rule_based_avg_time,
            "ml_avg_time_ms": ml_avg_time,
            "confidence_stats": confidence_stats,
            "feature_importance": feature_importance
        }

        rule_based_counts = df['rule_based_action'].value_counts()
        ml_counts = df['ml_action'].value_counts()

        action_metrics = {}
        for action in set(df['ml_action'].unique()):
            if action != "no_action":
                action_subset = df[df['ml_action'] == action]
                if len(action_subset) > 0:
                    action_metrics[action] = {
                        "count": len(action_subset),
                        "avg_length_improvement": action_subset['length_improvement_pct'].mean(),
                        "median_length_improvement": action_subset[
                            'length_improvement_pct'].median(),
                        "avg_time_improvement": action_subset['time_improvement_pct'].mean(),
                        "median_time_improvement": action_subset['time_improvement_pct'].median()
                    }

        print("\nComparison Results Summary:")
        print(f"Total scenarios: {total_scenarios}")
        print(f"Matching actions: {matching_actions} ({matching_pct:.2f}%)")

        print("\nTime Performance:")
        print(f"Average time improvement: {avg_time_improvement:.2f}%")
        print(f"Median time improvement: {median_time_improvement:.2f}%")
        print(f"Average time difference: {avg_time_diff:.2f} seconds")

        print("\nRoute Length Performance:")
        print(f"Average length improvement: {avg_length_improvement:.2f}% (capped)")
        print(f"Median length improvement: {median_length_improvement:.2f}%")
        print(f"Average length difference: {avg_length_diff:.2f} meters")

        print("\nExecution Performance:")
        print(f"Rule-based resolver average execution time: {rule_based_avg_time:.2f} ms")
        print(f"ML resolver average execution time: {ml_avg_time:.2f} ms")

        if confidence_stats:
            print("\nML Confidence Statistics:")
            print(f"Average confidence: {confidence_stats['avg_confidence']:.2f}")
            print(f"Minimum confidence: {confidence_stats['min_confidence']:.2f}")
            print(f"Maximum confidence: {confidence_stats['max_confidence']:.2f}")
            print(
                f"Low confidence decisions (<40%): {confidence_stats['low_confidence_count']} ({confidence_stats['low_confidence_pct']:.2f}%)")

        if feature_importance:
            print("\nFeature Importance:")
            for feature, importance in feature_importance.items():
                print(f"  {feature}: {importance['importance']:.4f} ± {importance['std']:.4f}")

        print("\nRule-based resolver action distribution:")
        for action, count in rule_based_counts.items():
            print(f"  {action}: {count} ({count / total_scenarios * 100:.2f}%)")

        print("\nML resolver action distribution:")
        for action, count in ml_counts.items():
            print(f"  {action}: {count} ({count / total_scenarios * 100:.2f}%)")

        if action_metrics:
            print("\nML Action Type Performance:")
            for action, metrics in action_metrics.items():
                print(f"  {action} ({metrics['count']} instances):")
                print(
                    f"    Length improvement: {metrics['avg_length_improvement']:.2f}% (avg), {metrics['median_length_improvement']:.2f}% (median)")
                print(
                    f"    Time improvement: {metrics['avg_time_improvement']:.2f}% (avg), {metrics['median_time_improvement']:.2f}% (median)")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.RESULTS_DIR, f"comparison_results_{timestamp}.csv")
        summary_file = os.path.join(self.RESULTS_DIR, f"comparison_summary_{timestamp}.txt")

        df.to_csv(results_file, index=False)

        with open(summary_file, 'w') as f:
            f.write("Resolver Comparison Summary\n")
            f.write("===========================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total scenarios: {total_scenarios}\n")

            f.write("Performance Metrics:\n")
            f.write(f"Matching actions: {matching_actions} ({matching_pct:.2f}%)\n\n")

            f.write("Time Performance:\n")
            f.write(f"Average time improvement: {avg_time_improvement:.2f}%\n")
            f.write(f"Median time improvement: {median_time_improvement:.2f}%\n")
            f.write(f"Average time difference: {avg_time_diff:.2f} seconds\n\n")

            f.write("Route Length Performance:\n")
            f.write(f"Average length improvement: {avg_length_improvement:.2f}% (capped)\n")
            f.write(f"Median length improvement: {median_length_improvement:.2f}%\n")
            f.write(f"Average length difference: {avg_length_diff:.2f} meters\n\n")

            f.write("Execution Performance:\n")
            f.write(f"Rule-based resolver average execution time: {rule_based_avg_time:.2f} ms\n")
            f.write(f"ML resolver average execution time: {ml_avg_time:.2f} ms\n\n")

            if confidence_stats:
                f.write("ML Confidence Statistics:\n")
                f.write(f"Average confidence: {confidence_stats['avg_confidence']:.2f}\n")
                f.write(f"Minimum confidence: {confidence_stats['min_confidence']:.2f}\n")
                f.write(f"Maximum confidence: {confidence_stats['max_confidence']:.2f}\n")
                f.write(
                    f"Low confidence decisions (<40%): {confidence_stats['low_confidence_count']} ({confidence_stats['low_confidence_pct']:.2f}%)\n\n")

            if feature_importance:
                f.write("Feature Importance:\n")
                for feature, importance in feature_importance.items():
                    f.write(
                        f"  {feature}: {importance['importance']:.4f} ± {importance['std']:.4f}\n")
                f.write("\n")

            f.write("Rule-based resolver action distribution:\n")
            for action, count in rule_based_counts.items():
                f.write(f"  {action}: {count} ({count / total_scenarios * 100:.2f}%)\n")

            f.write("\nML resolver action distribution:\n")
            for action, count in ml_counts.items():
                f.write(f"  {action}: {count} ({count / total_scenarios * 100:.2f}%)\n")

            if action_metrics:
                f.write("\nML Action Type Performance:\n")
                for action, metrics in action_metrics.items():
                    f.write(f"  {action} ({metrics['count']} instances):\n")
                    f.write(
                        f"    Length improvement: {metrics['avg_length_improvement']:.2f}% (avg), {metrics['median_length_improvement']:.2f}% (median)\n")
                    f.write(
                        f"    Time improvement: {metrics['avg_time_improvement']:.2f}% (avg), {metrics['median_time_improvement']:.2f}% (median)\n")

        print(f"\nDetailed results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")

        return summary_file

    def _apply_disruption_to_graph(self, disruption: Disruption) -> Dict[str, Any]:
        original_elements = {
            'removed_nodes_with_edges': [],
            'modified_edge_travel_times': []
        }

        disruption_radius = float(disruption.affected_area_radius)
        potential_nodes_in_disruption_influence = []
        search_radius_multiplier = 1.5

        for node, data in self.G.nodes(data=True):
            if 'y' in data and 'x' in data:
                if calculate_haversine_distance((data['y'], data['x']),
                                                disruption.location) <= disruption_radius * search_radius_multiplier:
                    potential_nodes_in_disruption_influence.append(node)

        if disruption.type == DisruptionType.ROAD_CLOSURE:
            nodes_to_remove_this_step = []
            strict_radius = disruption_radius
            for node in potential_nodes_in_disruption_influence:
                if node not in self.G: continue
                node_data = self.G.nodes[node]
                if 'y' in node_data and 'x' in node_data:
                    if calculate_haversine_distance((node_data['y'], node_data['x']),
                                                    disruption.location) <= strict_radius:
                        nodes_to_remove_this_step.append(node)

            if nodes_to_remove_this_step:
                for node_to_remove in nodes_to_remove_this_step:
                    if node_to_remove not in self.G: continue
                    incident_edges_data = []
                    for u, v, k, data in list(self.G.edges(node_to_remove, data=True, keys=True)):
                        incident_edges_data.append((u, v, k, data))
                    for u, v, k, data in list(
                            self.G.in_edges(node_to_remove, data=True, keys=True)):
                        is_duplicate = any(
                            item[0] == u and item[1] == v and item[2] == k for item in
                            incident_edges_data)
                        if not is_duplicate:
                            incident_edges_data.append((u, v, k, data))

                    original_elements['removed_nodes_with_edges'].append(
                        (node_to_remove, self.G.nodes[node_to_remove].copy(), incident_edges_data))

                self.G.remove_nodes_from(nodes_to_remove_this_step)

        elif disruption.type == DisruptionType.TRAFFIC_JAM:
            weight_multiplier = 1.0 + (9.0 * disruption.severity)
            affected_edges_count = 0
            edges_to_check = set()
            for node_u in potential_nodes_in_disruption_influence:
                if node_u not in self.G: continue
                for u, v, k, data in self.G.edges(node_u, data=True, keys=True):
                    if 'travel_time' in data:
                        node_v_data = self.G.nodes.get(v, {})
                        if not ('x' in self.G.nodes[u] and 'y' in self.G.nodes[u] and \
                                'x' in node_v_data and 'y' in node_v_data): continue
                        mid_lon = (self.G.nodes[u]['x'] + node_v_data['x']) / 2
                        mid_lat = (self.G.nodes[u]['y'] + node_v_data['y']) / 2
                        if calculate_haversine_distance((mid_lat, mid_lon),
                                                        disruption.location) <= disruption_radius:
                            edges_to_check.add((u, v, k))

            for u, v, k in edges_to_check:
                if not self.G.has_edge(u, v, k): continue
                try:
                    original_time = self.G[u][v][k]['travel_time']
                    if not isinstance(original_time, (int, float)): original_time = float(
                        original_time)

                    original_elements['modified_edge_travel_times'].append((u, v, k, original_time))
                    self.G[u][v][k]['travel_time'] = original_time * weight_multiplier
                    affected_edges_count += 1
                except (TypeError, ValueError, KeyError) as e:
                    print(
                        f"Warning (Eval): Could not process travel_time for edge ({u}, {v}, {k}): {e}")

        return original_elements

    def _revert_graph_changes(self, original_elements: Dict[str, Any]):
        for node, node_data, incident_edges_data in reversed(
                original_elements.get('removed_nodes_with_edges', [])):
            self.G.add_node(node, **node_data)
            edges_to_add_formatted = []
            for u, v, k, data in incident_edges_data:
                edges_to_add_formatted.append((u, v, k, data))
            self.G.add_edges_from(edges_to_add_formatted)

        for u, v, k, original_travel_time in original_elements.get('modified_edge_travel_times',
                                                                   []):
            if self.G.has_edge(u, v, k):
                self.G[u][v][k]['travel_time'] = original_travel_time
            else:
                print(f"Warning (Eval): Edge ({u},{v},{k}) not found during reversion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare ML and Rule-based disruption resolvers.')
    parser.add_argument('--scenarios', type=int, default=100,
                        help='Number of scenarios to generate or select from CSV for comparison.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for scenario generation.')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['random_forest', 'neural_network', 'neural_network_simple'],
                        help='Specify model type to use (overrides config). Default is from ml_model_config.pkl.')
    parser.add_argument('--input_csv', type=str, default=None,
                        help='Optional path to a CSV file containing pre-generated scenarios (full_scenario format expected).')

    args = parser.parse_args()

    config_instance = Config()

    print("Loading graph for comparison...")
    try:
        graph_path = config_instance.get_osm_file_path()
        loaded_graph = graph_service.load_graph(graph_path)
        main_graph_component = graph_service.get_largest_connected_component(loaded_graph)
        print("Graph loaded successfully.")
    except Exception as e:
        print(f"Error loading graph: {e}. Exiting.")
        sys.exit(1)

    warehouse_location = config_instance.get_warehouse_location()
    delivery_points = config_instance.get_delivery_points()

    print(f"Creating comparison with {args.scenarios} scenarios, seed {args.seed}")
    comparison = ResolverComparison(
        graph_obj=main_graph_component,
        warehouse_location=warehouse_location,
        delivery_points=delivery_points,
        num_scenarios=args.scenarios,
        random_seed=args.seed,
        model_type=args.model_type,
        input_csv_path=args.input_csv
    )

    results_file = comparison.run_comparison()
