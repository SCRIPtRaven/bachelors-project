import itertools
import json
import logging
import multiprocessing
import os
import random
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.actions import (
    DisruptionAction, ActionType
)
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import calculate_route_length, calculate_travel_time
from utils.route_utils import find_closest_point_index_on_route, \
    find_route_enter_disruption_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TARGET_ACTION_TYPES = [
    ActionType.NO_ACTION.display_name,
    ActionType.REROUTE_BASIC.display_name,
    ActionType.REROUTE_TIGHT_AVOIDANCE.display_name,
    ActionType.REROUTE_WIDE_AVOIDANCE.display_name
]

INTERMEDIATE_SAVE_INTERVAL = 500


def _generate_one_sample_worker(args: Tuple[int, Any, Tuple, List, RuleBasedResolver, Optional[str], bool]) -> Tuple[
    str, Optional[str], Optional[Dict]]:
    seed, graph, warehouse_location, delivery_points, resolver, generation_hint, save_full_scenario_data = args

    random.seed(seed)
    np.random.seed(seed)

    dummy_generator = MLDataGenerator(graph, warehouse_location, delivery_points, num_samples=1, random_seed=seed)
    dummy_generator.resolver = resolver

    try:
        disruption, state, driver_id = dummy_generator._generate_random_scenario_on_route(hint=generation_hint)
        if disruption is None or state is None or driver_id is None:
            return 'fail_scenario', None, None

        features_dict = dummy_generator._extract_features(driver_id, disruption, state)
        if features_dict is None:
            return 'fail_feature', None, None

        action_outcomes = dummy_generator._evaluate_all_actions(driver_id, disruption, state)
        if not action_outcomes:
            return 'fail_action', None, None

        best_action = dummy_generator._determine_best_action(action_outcomes)

        sample_data = {
            "disruption_type_road_closure": features_dict.get('disruption_type_road_closure', 0.0),
            "disruption_type_traffic_jam": features_dict.get('disruption_type_traffic_jam', 0.0),
            "disruption_severity": features_dict.get('severity', 0.0),
            "distance_to_disruption_center": features_dict.get('distance_to_disruption_center', 0.0),
            "remaining_deliveries": features_dict.get('remaining_deliveries', 0.0),
            "distance_along_route_to_disruption": features_dict.get('distance_along_route_to_disruption', 0.0),
            "distance_to_next_delivery_along_route": features_dict.get('distance_to_next_delivery_along_route', 0.0),
            "next_delivery_before_disruption": features_dict.get('next_delivery_before_disruption', 0.0),
            "alternative_route_density": features_dict.get('alternative_route_density', 0.0),
            "urban_density": features_dict.get('urban_density', 0.0),
            "best_action": best_action,
            "disruption_id": disruption.id,
            "target_driver_id": driver_id,
            "all_actions_travel_time": {k: v["travel_time"] for k, v in action_outcomes.items()},
            "all_actions_route_length": {k: v["route_length"] for k, v in action_outcomes.items()},
        }

        if save_full_scenario_data:
            sample_data.update({
                "disruption_raw_type": disruption.type.name,
                "disruption_location_lat": disruption.location[0],
                "disruption_location_lon": disruption.location[1],
                "disruption_affected_area_radius": disruption.affected_area_radius,
                "disruption_duration": disruption.duration,

                "initial_driver_position_lat": state.driver_positions[driver_id][
                    0] if driver_id in state.driver_positions else None,
                "initial_driver_position_lon": state.driver_positions[driver_id][
                    1] if driver_id in state.driver_positions else None,
                "initial_driver_route_points_json": json.dumps(
                    state.driver_routes[driver_id]['points']) if driver_id in state.driver_routes and 'points' in
                                                                 state.driver_routes[driver_id] else None,

                "all_delivery_points_original_json": json.dumps(delivery_points),
                "driver_assignments_json": json.dumps(state.driver_assignments) if hasattr(state,
                                                                                           'driver_assignments') else None,
                "warehouse_location_original_json": json.dumps(warehouse_location)
            })

        return 'success', best_action, sample_data

    except Exception as e:
        return 'fail_exception', None, None


class MLDataGenerator:
    OUTPUT_DIR = os.path.join('models', 'resolvers', 'training_data')

    def __init__(self, graph, warehouse_location, delivery_points, num_samples=1000, random_seed=42,
                 save_full_scenario_data: bool = False):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.save_full_scenario_data = save_full_scenario_data

        logger.info("Pre-calculating nearest nodes for warehouse and delivery points...")
        try:
            self.warehouse_node = ox.nearest_nodes(self.G, X=self.warehouse_location[1], Y=self.warehouse_location[0])
            self.delivery_point_nodes = [
                ox.nearest_nodes(self.G, X=pt[1], Y=pt[0]) for pt in self.delivery_points
            ]
            self.delivery_idx_to_node_id = {idx: node_id for idx, node_id in enumerate(self.delivery_point_nodes)}
            logger.info("Nearest nodes pre-calculation complete.")
        except Exception as e:
            logger.error(f"Error during nearest_nodes pre-calculation: {e}. This may impact route generation.")
            self.warehouse_node = None
            self.delivery_point_nodes = []
            self.delivery_idx_to_node_id = {}

        self.resolver = RuleBasedResolver(graph, warehouse_location)
        self.resolver._print_enabled = False

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        logging.getLogger().setLevel(logging.ERROR)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.ERROR)

    def _save_data_to_csv(self, feature_rows_to_save: List[Dict],
                          is_intermediate: bool = False,
                          intermediate_file_tag: Optional[str] = None):
        if not feature_rows_to_save:
            logger.info("No data to save.")
            return

        logger.info(f"Saving {len(feature_rows_to_save)} collected samples...")

        try:
            df = pd.DataFrame(feature_rows_to_save)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            logger.error("Data saving failed. Raw data rows (first 3):")
            for i, row in enumerate(feature_rows_to_save[:3]):
                logger.error(f"Row {i}: {row}")
            return

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        base_filename_prefix = "training_data"

        if self.save_full_scenario_data:
            base_filename_prefix += "_full_scenario"

        if is_intermediate:
            if intermediate_file_tag:
                tag = str(intermediate_file_tag).replace(" ", "_")
                base_filename_prefix = f"intermediate_{base_filename_prefix}_{tag}"
            else:
                base_filename_prefix = f"intermediate_{base_filename_prefix}"

        output_file = os.path.join(self.OUTPUT_DIR, f"{base_filename_prefix}_{timestamp}.csv")
        simple_output_file = os.path.join(self.OUTPUT_DIR, f"simple_{base_filename_prefix}_{timestamp}.csv")
        metadata_file = os.path.join(self.OUTPUT_DIR, f"metadata_{base_filename_prefix}_{timestamp}.json")

        try:
            logger.info(f"Saving main data to {output_file}")
            df.to_csv(output_file, index=False)
            logger.info(f"Main data saved successfully to {output_file}")

            X_cols = [col for col in df.columns if col not in [
                'best_action', 'disruption_id', 'target_driver_id',
                'all_actions_travel_time', 'all_actions_route_length',
                'disruption_raw_type', 'disruption_location_lat', 'disruption_location_lon',
                'disruption_affected_area_radius', 'disruption_duration',
                'initial_driver_position_lat', 'initial_driver_position_lon',
                'initial_driver_route_points_json', 'all_delivery_points_original_json',
                'driver_assignments_json', 'warehouse_location_original_json'
            ]]
            y_col = 'best_action'

            if X_cols and y_col in df.columns and df.shape[0] > 0:
                simple_df = df[X_cols + [y_col]]
                logger.info(f"Saving simple data to {simple_output_file}")
                simple_df.to_csv(simple_output_file, index=False)
                logger.info(f"Simple data saved successfully to {simple_output_file}")
            else:
                logger.warning("Could not create simple_df due to missing columns. Skipping simple save.")
                X_cols = []
                y_col = ''

            current_stats = self._current_stats if hasattr(self, '_current_stats') else {}
            current_action_counts = self._current_action_counts if hasattr(self, '_current_action_counts') else {}

            metadata = {
                "filename_timestamp": timestamp,
                "save_type": "intermediate" if is_intermediate else "final",
                "intermediate_tag": intermediate_file_tag if is_intermediate else None,
                "includes_full_scenario_data": self.save_full_scenario_data,
                "num_samples_in_file": len(feature_rows_to_save),
                "target_samples_per_action": self.num_samples,
                "total_samples_generated_so_far": current_stats.get('successful_samples', 0),
                "current_action_counts": current_action_counts,
                "total_iterations_so_far": current_stats.get('iterations', 0),
                "worker_exceptions": current_stats.get('worker_exceptions', 0),
                "failed_scenarios": current_stats.get('failed_scenarios', 0),
                "failed_features": current_stats.get('failed_features', 0),
                "failed_actions": current_stats.get('failed_actions', 0),
                "action_distribution_in_file": df[
                    'best_action'].value_counts().to_dict() if 'best_action' in df.columns else {},
                "feature_columns": X_cols,
                "label_column": y_col
            }
            logger.info(f"Saving metadata to {metadata_file}")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved successfully to {metadata_file}")

        except Exception as e:
            logger.error(f"Error during file saving operations: {e}")
            import traceback
            traceback.print_exc()

    def generate_training_data(self):
        feature_rows = []
        self._current_stats = {
            'failed_scenarios': 0,
            'failed_features': 0,
            'failed_actions': 0,
            'successful_samples': 0,
            'iterations': 0,
            'worker_exceptions': 0
        }
        self._current_action_counts = {action_name: 0 for action_name in TARGET_ACTION_TYPES}

        target_samples_per_action = self.num_samples
        max_iterations = target_samples_per_action * len(TARGET_ACTION_TYPES) * 500

        total_target_samples = target_samples_per_action * len(TARGET_ACTION_TYPES)
        pbar = tqdm(total=total_target_samples, desc="Generating samples", unit="sample")

        num_workers = max(1, os.cpu_count() - 4)
        logger.info(f"Using {num_workers} worker processes.")

        samples_since_last_save = 0
        intermediate_batch_count = 0

        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                seed_generator = itertools.count(self.random_seed)
                worker_args_base = (self.G, self.warehouse_location, self.delivery_points, self.resolver)

                def generate_task_args_with_hints():
                    HINT_APPLICATION_RATE = 4
                    for i in itertools.count():
                        seed = next(seed_generator)
                        hint_for_this_task = None

                        wide_needed = self._current_action_counts[
                                          ActionType.REROUTE_WIDE_AVOIDANCE.display_name] < target_samples_per_action
                        wide_lagging = self._current_action_counts[ActionType.REROUTE_WIDE_AVOIDANCE.display_name] < \
                                       self._current_action_counts[ActionType.REROUTE_BASIC.display_name] * 0.6

                        tight_needed = self._current_action_counts[
                                           ActionType.REROUTE_TIGHT_AVOIDANCE.display_name] < target_samples_per_action
                        tight_lagging = self._current_action_counts[ActionType.REROUTE_TIGHT_AVOIDANCE.display_name] < \
                                        self._current_action_counts[ActionType.REROUTE_BASIC.display_name] * 0.6

                        current_active_hints = []
                        if wide_needed and wide_lagging:
                            current_active_hints.append(ActionType.REROUTE_WIDE_AVOIDANCE.display_name)
                        if tight_needed and tight_lagging:
                            current_active_hints.append(ActionType.REROUTE_TIGHT_AVOIDANCE.display_name)

                        if current_active_hints and (i % HINT_APPLICATION_RATE == 0):
                            hint_for_this_task = random.choice(current_active_hints)

                        yield (seed, *worker_args_base, hint_for_this_task, self.save_full_scenario_data)

                tasks_iterator = generate_task_args_with_hints()

                results_iterator = pool.imap_unordered(_generate_one_sample_worker, tasks_iterator)

                for status, best_action, sample_data in results_iterator:
                    self._current_stats['iterations'] += 1

                    if status == 'success':
                        if best_action in self._current_action_counts and self._current_action_counts[
                            best_action] < target_samples_per_action:
                            feature_rows.append(sample_data)
                            self._current_action_counts[best_action] += 1
                            self._current_stats['successful_samples'] += 1
                            samples_since_last_save += 1
                            pbar.update(1)
                    elif status == 'fail_scenario':
                        self._current_stats['failed_scenarios'] += 1
                    elif status == 'fail_feature':
                        self._current_stats['failed_features'] += 1
                    elif status == 'fail_action':
                        self._current_stats['failed_actions'] += 1
                    elif status == 'fail_exception':
                        self._current_stats['worker_exceptions'] += 1

                    counts_str = ", ".join([f"{k}={v}" for k, v in self._current_action_counts.items()])
                    pbar.set_description(
                        f"Counts: [{counts_str}] | Target: {target_samples_per_action} | Iter: {self._current_stats['iterations']}")

                    if samples_since_last_save >= INTERMEDIATE_SAVE_INTERVAL:
                        intermediate_batch_count += 1
                        logger.info(
                            f"Reached {samples_since_last_save} new samples. Performing intermediate save (batch {intermediate_batch_count})...")
                        self._save_data_to_csv(list(feature_rows), is_intermediate=True,
                                               intermediate_file_tag=f"batch_{intermediate_batch_count}")
                        samples_since_last_save = 0

                    if all(count >= target_samples_per_action for count in self._current_action_counts.values()):
                        logger.info("Target counts reached for all action types.")
                        break

                    if self._current_stats['iterations'] >= max_iterations:
                        logger.warning(f"Maximum iterations ({max_iterations}) reached. Stopping generation.")
                        break

                logger.info("Exited main generation loop. Terminating worker pool...")
                pool.terminate()
                pool.join()
                logger.info("Worker pool terminated.")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received! Attempting to save partial data...")
            self._save_data_to_csv(list(feature_rows), is_intermediate=True, intermediate_file_tag="interrupt")
            logger.warning("Partial data saved due to KeyboardInterrupt. Exiting.")
            pbar.close()
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during data generation: {e}")
            import traceback
            traceback.print_exc()
            logger.error("Attempting to save any collected data before exiting...")
            self._save_data_to_csv(list(feature_rows), is_intermediate=True, intermediate_file_tag="error_exit")
            pbar.close()
            sys.exit(1)

        pbar.close()

        logger.info("Generation process complete or interrupted. Performing final data save.")
        self._save_data_to_csv(feature_rows, is_intermediate=False)

        total_generated = sum(self._current_action_counts.values())
        logger.info(f"--- Generation Summary ---")
        logger.info(f"Target samples per action: {target_samples_per_action}")
        logger.info(f"Actual counts: {self._current_action_counts}")
        logger.info(f"Total valid samples generated: {total_generated} / {self._current_stats['successful_samples']}")
        logger.info(f"Total iterations: {self._current_stats['iterations']}")
        logger.info(f"Failed scenarios: {self._current_stats['failed_scenarios']}")
        logger.info(f"Failed features: {self._current_stats['failed_features']}")
        logger.info(f"Failed actions: {self._current_stats['failed_actions']}")
        if self._current_stats['worker_exceptions'] > 0:
            logger.warning(f"Worker exceptions encountered: {self._current_stats['worker_exceptions']}")

        return "Data generation process finished (see logs for saved files).", "Check output directory."

    def _generate_random_scenario_on_route(self, hint: Optional[str] = None) -> Tuple[
        Optional[Disruption], Optional[DeliverySystemState], Optional[int]]:
        try:
            disruption_radius_multiplier = 1.0
            severity_bias_min, severity_bias_max = 0.0, 0.0
            disruption_type_weights = [0.4, 0.6]

            if hint == ActionType.REROUTE_WIDE_AVOIDANCE.display_name:
                logger.debug(f"Scenario generation hint: WIDE_AVOIDANCE")
                disruption_radius_multiplier = random.uniform(1.5, 3.0)
                severity_bias_min, severity_bias_max = 0.15, 0.35
                disruption_type_weights = [0.6, 0.4]
            elif hint == ActionType.REROUTE_TIGHT_AVOIDANCE.display_name:
                logger.debug(f"Scenario generation hint: TIGHT_AVOIDANCE")
                disruption_radius_multiplier = random.uniform(0.7, 1.3)
                severity_bias_min, severity_bias_max = 0.05, 0.20
                disruption_type_weights = [0.5, 0.5]

            drivers = []
            all_delivery_points = list(self.delivery_points)
            random.shuffle(all_delivery_points)
            state = DeliverySystemState(
                drivers=drivers, deliveries=all_delivery_points, disruptions=[],
                simulation_time=0, graph=self.G, warehouse_location=self.warehouse_location
            )
            num_drivers = random.randint(3, 8)
            state.driver_assignments = {}
            state.driver_positions = {}
            state.driver_routes = {}
            points_per_driver = max(1, len(all_delivery_points) // num_drivers)
            remaining_points = len(all_delivery_points) % num_drivers
            start_idx = 0
            driver_ids = list(range(num_drivers))

            for driver_id in driver_ids:
                valid_route = True
                extra_point = 1 if driver_id < remaining_points else 0
                end_idx = min(start_idx + points_per_driver + extra_point, len(all_delivery_points))
                driver_delivery_indices = list(range(start_idx, end_idx))
                state.driver_assignments[driver_id] = driver_delivery_indices
                start_idx = end_idx

                current_route_nodes = [self.warehouse_node]
                for delivery_idx in driver_delivery_indices:
                    if delivery_idx < len(self.delivery_point_nodes):
                        current_route_nodes.append(self.delivery_point_nodes[delivery_idx])
                    else:
                        logger.warning(
                            f"Delivery index {delivery_idx} out of bounds for pre-calculated delivery nodes. Skipping for driver {driver_id}.")
                        valid_route = False;
                        break
                if not valid_route: continue

                route_stops_points = [self.warehouse_location] + [self.delivery_points[idx] for idx in
                                                                  driver_delivery_indices]

                detailed_route_points = []
                for i in range(len(current_route_nodes) - 1):
                    start_node, end_node = current_route_nodes[i], current_route_nodes[i + 1]
                    if start_node is None or end_node is None:
                        logger.warning(
                            f"Missing pre-calculated start/end node for driver {driver_id}. Skipping route segment.")
                        valid_route = False;
                        break
                    try:
                        path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
                        segment_points = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path if
                                          'y' in self.G.nodes[node] and 'x' in self.G.nodes[node]]
                        if i > 0: segment_points = segment_points[1:]
                        detailed_route_points.extend(segment_points)
                    except Exception as e:
                        logger.error(f"Error calculating route segment for driver {driver_id}: {e}")
                        valid_route = False
                        break

                if valid_route and detailed_route_points:
                    route_nodes_for_detailed_path = []
                    if detailed_route_points:
                        try:
                            route_nodes_for_detailed_path = [ox.nearest_nodes(self.G, X=p[1], Y=p[0]) for p in
                                                             detailed_route_points]
                        except Exception as e:
                            logger.warning(
                                f"Could not get nodes for detailed_route_points for driver {driver_id}: {e}. Route nodes will be incomplete.")

                    state.driver_routes[driver_id] = {
                        'points': detailed_route_points,
                        'nodes': route_nodes_for_detailed_path,
                        'progress': 0.0,
                        'delivery_indices': driver_delivery_indices
                    }
                else:
                    state.driver_assignments.pop(driver_id, None)

            valid_driver_ids = list(state.driver_routes.keys())
            if not valid_driver_ids:
                logger.warning("Could not generate valid routes for any driver.")
                return None, None, None

            target_driver_id = random.choice(valid_driver_ids)
            target_route = state.driver_routes[target_driver_id]['points']

            if len(target_route) < 2:
                logger.warning(f"Selected driver {target_driver_id} has route with < 2 points.")
                return None, None, None

            segment_start_idx = random.randint(1, len(target_route) - 2)
            segment_end_idx = segment_start_idx + 1
            segment_start_pt = target_route[segment_start_idx]
            segment_end_pt = target_route[segment_end_idx]

            t = random.random()
            if not isinstance(segment_start_pt, (list, tuple)) or len(segment_start_pt) != 2 or \
                    not isinstance(segment_end_pt, (list, tuple)) or len(segment_end_pt) != 2:
                logger.warning(
                    f"Invalid segment points for disruption generation: {segment_start_pt}, {segment_end_pt}. Skipping scenario.")
                return None, None, None

            disruption_lat = segment_start_pt[0] * (1 - t) + segment_end_pt[0] * t
            disruption_lon = segment_start_pt[1] * (1 - t) + segment_end_pt[1] * t
            disruption_location = (disruption_lat, disruption_lon)

            disruption_type = random.choices(
                [DisruptionType.ROAD_CLOSURE, DisruptionType.TRAFFIC_JAM],
                weights=disruption_type_weights, k=1
            )[0]

            base_severity = random.betavariate(2, 2)
            applied_severity_bias = random.uniform(severity_bias_min, severity_bias_max)
            severity = max(0.05, min(0.95, base_severity + applied_severity_bias))
            if disruption_type == DisruptionType.ROAD_CLOSURE:
                severity = max(0.7, severity)
                radius = random.uniform(50, 250) * disruption_radius_multiplier
            else:  # Traffic jam
                radius = random.uniform(100, 600) * disruption_radius_multiplier

            radius = max(30, min(1000, radius))

            if disruption_type == DisruptionType.ROAD_CLOSURE:
                duration = int(30 + severity * 150)
            else:
                duration = int(15 + severity * 105)

            disruption = Disruption(
                id=random.randint(1000, 9999),
                location=disruption_location,
                type=disruption_type,
                severity=severity,
                affected_area_radius=radius,
                duration=duration,
                metadata={'target_driver': target_driver_id, 'on_segment_index': segment_start_idx}
            )
            state.disruptions = [disruption]

            max_pos_index = segment_start_idx
            driver_pos_index = random.randint(0, max_pos_index)
            state.driver_positions[target_driver_id] = target_route[driver_pos_index]
            total_route_len = calculate_route_length(target_route)
            dist_to_pos = calculate_route_length(target_route[:driver_pos_index + 1])
            state.driver_routes[target_driver_id][
                'progress'] = dist_to_pos / total_route_len if total_route_len > 0 else 0.0

            for driver_id in valid_driver_ids:
                if driver_id != target_driver_id:
                    other_route = state.driver_routes[driver_id]['points']
                    if other_route:
                        other_pos_index = random.randint(0, len(other_route) - 1)
                        state.driver_positions[driver_id] = other_route[other_pos_index]
                        other_total_len = calculate_route_length(other_route)
                        other_dist_to_pos = calculate_route_length(other_route[:other_pos_index + 1])
                        state.driver_routes[driver_id][
                            'progress'] = other_dist_to_pos / other_total_len if other_total_len > 0 else 0.0
                    else:
                        state.driver_positions[driver_id] = self.warehouse_location
                        state.driver_routes[driver_id]['progress'] = 0.0

            return disruption, state, target_driver_id

        except Exception as e:
            logger.error(f"Error generating targeted scenario: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _extract_features(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Optional[
        Dict[str, float]]:
        try:
            features = {}

            if disruption.type == DisruptionType.ROAD_CLOSURE:
                features['disruption_type_road_closure'] = 1.0
                features['disruption_type_traffic_jam'] = 0.0
            elif disruption.type == DisruptionType.TRAFFIC_JAM:
                features['disruption_type_road_closure'] = 0.0
                features['disruption_type_traffic_jam'] = 1.0
            else:
                features['disruption_type_road_closure'] = 0.0
                features['disruption_type_traffic_jam'] = 0.0

            features['severity'] = disruption.severity

            position = state.driver_positions.get(driver_id)
            if position is None:
                logger.warning(f"Driver {driver_id} has no position")
                return None

            route_data = state.driver_routes.get(driver_id, {})
            route_points = route_data.get('points', [])
            delivery_indices = route_data.get('delivery_indices', [])

            if not route_points:
                logger.warning(f"Driver {driver_id} has no route points")
                return None

            current_route_index = find_closest_point_index_on_route(route_points, position)
            if current_route_index == -1:
                current_route_index = 0

            distance_to_disruption_center = calculate_haversine_distance(position, disruption.location)
            max_distance_norm = 10000
            features['distance_to_disruption_center'] = min(1.0, distance_to_disruption_center / max_distance_norm)

            remaining_deliveries = len(delivery_indices)
            features['remaining_deliveries'] = min(1.0, remaining_deliveries / 20.0)

            disruption_radius = float(disruption.affected_area_radius)
            enter_disruption_index = find_route_enter_disruption_index(
                route_points, disruption.location, disruption_radius, current_route_index
            )

            distance_along_route = float('inf')
            if enter_disruption_index != -1:
                dist_start_to_current = calculate_route_length(route_points[:current_route_index + 1])
                dist_start_to_disruption = calculate_route_length(route_points[:enter_disruption_index + 1])
                distance_along_route = max(0, dist_start_to_disruption - dist_start_to_current)

            features['distance_along_route_to_disruption'] = min(1.0, distance_along_route / max_distance_norm)

            next_delivery_route_index = -1
            next_delivery_point = None
            if delivery_indices:
                first_delivery_idx = delivery_indices[0]
                if first_delivery_idx < len(state.deliveries):
                    next_delivery_dest = state.deliveries[first_delivery_idx]
                    min_dist_to_dest = float('inf')
                    for i in range(current_route_index, len(route_points)):
                        dist = calculate_haversine_distance(route_points[i], next_delivery_dest)
                        if dist < min_dist_to_dest:
                            min_dist_to_dest = dist
                            next_delivery_route_index = i
                            next_delivery_point = route_points[i]

            distance_to_next_delivery_along_route = float('inf')
            if next_delivery_route_index != -1:
                dist_start_to_current = calculate_route_length(route_points[:current_route_index + 1])
                dist_start_to_next_delivery = calculate_route_length(route_points[:next_delivery_route_index + 1])
                distance_to_next_delivery_along_route = max(0, dist_start_to_next_delivery - dist_start_to_current)

            features['distance_to_next_delivery_along_route'] = min(1.0,
                                                                    distance_to_next_delivery_along_route / max_distance_norm)

            next_delivery_before_disruption = 0.0
            if next_delivery_route_index != -1 and enter_disruption_index != -1:
                if next_delivery_route_index < enter_disruption_index:
                    next_delivery_before_disruption = 1.0
            elif next_delivery_route_index != -1 and enter_disruption_index == -1:
                next_delivery_before_disruption = 1.0

            features['next_delivery_before_disruption'] = next_delivery_before_disruption

            try:
                disruption_node = ox.nearest_nodes(self.G, X=disruption.location[1], Y=disruption.location[0])
                spatial_radius_density = disruption_radius * 3
                disruption_area = nx.ego_graph(self.G, disruption_node, radius=spatial_radius_density,
                                               distance='length')

                if len(disruption_area.nodes) > 1:
                    edge_to_node_ratio = len(disruption_area.edges) / len(disruption_area.nodes)
                    features['alternative_route_density'] = min(1.0, max(0.0, edge_to_node_ratio / 3.0))
                else:
                    features['alternative_route_density'] = 0.0
            except Exception as e:
                features['alternative_route_density'] = 0.5

            try:
                spatial_radius_density = disruption_radius * 5
                nearby_nodes_count = 0
                for node, data in self.G.nodes(data=True):
                    if 'y' in data and 'x' in data:
                        if calculate_haversine_distance((data['y'], data['x']),
                                                        disruption.location) <= spatial_radius_density:
                            nearby_nodes_count += 1

                features['urban_density'] = min(1.0, max(0.0, nearby_nodes_count / 100.0))
            except Exception as e:
                features['urban_density'] = 0.5

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                    for u, v, k, data in list(self.G.in_edges(node_to_remove, data=True, keys=True)):
                        is_duplicate = any(
                            item[0] == u and item[1] == v and item[2] == k for item in incident_edges_data)
                        if not is_duplicate:
                            incident_edges_data.append((u, v, k, data))

                    original_elements['removed_nodes_with_edges'].append(
                        (node_to_remove, self.G.nodes[node_to_remove].copy(), incident_edges_data))

                self.G.remove_nodes_from(nodes_to_remove_this_step)
                logger.debug(
                    f"Applied ROAD_CLOSURE: Removed {len(nodes_to_remove_this_step)} nodes for disruption {disruption.id}")

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
                        if calculate_haversine_distance((mid_lat, mid_lon), disruption.location) <= disruption_radius:
                            edges_to_check.add((u, v, k))

            for u, v, k in edges_to_check:
                if not self.G.has_edge(u, v, k): continue
                try:
                    original_time = self.G[u][v][k]['travel_time']
                    if not isinstance(original_time, (int, float)): original_time = float(original_time)

                    original_elements['modified_edge_travel_times'].append((u, v, k, original_time))
                    self.G[u][v][k]['travel_time'] = original_time * weight_multiplier
                    affected_edges_count += 1
                except (TypeError, ValueError, KeyError) as e:
                    logger.warning(
                        f"Could not process travel_time for edge ({u}, {v}, {k}) during disruption apply: {e}")

            if affected_edges_count > 0:
                logger.debug(
                    f"Applied TRAFFIC_JAM: Modified {affected_edges_count} edges for disruption {disruption.id}")
        return original_elements

    def _revert_graph_changes(self, original_elements: Dict[str, Any]):
        for node, node_data, incident_edges_data in reversed(original_elements.get('removed_nodes_with_edges', [])):
            self.G.add_node(node, **node_data)
            edges_to_add_formatted = []
            for u, v, k, data in incident_edges_data:
                edges_to_add_formatted.append((u, v, k, data))
            self.G.add_edges_from(edges_to_add_formatted)
        if original_elements.get('removed_nodes_with_edges'):
            logger.debug(f"Reverted node removals.")

        for u, v, k, original_travel_time in original_elements.get('modified_edge_travel_times', []):
            if self.G.has_edge(u, v, k):
                self.G[u][v][k]['travel_time'] = original_travel_time
            else:
                logger.warning(
                    f"Edge ({u},{v},{k}) not found during travel_time reversion. Node might not have been restored correctly.")
        if original_elements.get('modified_edge_travel_times'):
            logger.debug(f"Reverted edge travel_time modifications.")

    def _evaluate_action_metrics(self, action: DisruptionAction, original_route: List, original_length: float,
                                 original_time: float, disruption: Disruption) -> Optional[Dict[str, float]]:
        try:
            if hasattr(action, 'new_route'):
                new_route = action.new_route
            elif action.action_type == ActionType.NO_ACTION:
                new_route = original_route
            else:
                return None

            if not new_route or len(new_route) < 2:
                return None
            new_length = calculate_route_length(new_route)
            new_time = calculate_travel_time(new_route, self.G, disruption=disruption)

            return {
                "travel_time": new_time,
                "route_length": new_length
            }

        except Exception as e:
            logger.error(f"Error evaluating action metrics: {e}")
            return None

    def _evaluate_all_actions(self, driver_id: int, disruption: Disruption, state: DeliverySystemState) -> Dict[
        str, Dict[str, float]]:
        original_graph_elements = None
        try:
            original_graph_elements = self._apply_disruption_to_graph(disruption)

            route_points = state.driver_routes.get(driver_id, {}).get('points', [])

            original_length = calculate_route_length(route_points)
            if route_points and len(route_points) >= 2:
                original_time_on_modified_graph = calculate_travel_time(route_points, self.G, disruption=disruption)
            else:
                original_time_on_modified_graph = float('inf')

            outcomes = {}

            try:
                no_action = self.resolver._create_no_action(driver_id, disruption)
                if no_action:
                    metrics = {
                        "travel_time": original_time_on_modified_graph,
                        "route_length": original_length
                    }
                    if metrics["travel_time"] != float('inf'):
                        outcomes[ActionType.NO_ACTION.display_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating no_action: {e}")

            try:
                basic_action = self.resolver._create_reroute_action(driver_id, disruption, state)
                if basic_action:
                    metrics = self._evaluate_action_metrics(
                        basic_action, route_points, original_length, original_time_on_modified_graph, disruption
                    )
                    if metrics:
                        outcomes[ActionType.REROUTE_BASIC.display_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating basic_reroute: {e}")

            buffer_ratios = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
            for ratio in buffer_ratios:
                try:
                    if hasattr(self.resolver, '_create_parameterized_avoidance_action'):
                        param_action = self.resolver._create_parameterized_avoidance_action(
                            driver_id, disruption, state, buffer_ratio=ratio
                        )

                        if param_action:
                            metrics = self._evaluate_action_metrics(
                                param_action, route_points, original_length, original_time_on_modified_graph, disruption
                            )
                            if metrics:
                                outcomes[f'parameterized_avoidance_{ratio:.1f}'] = metrics
                    else:
                        if ratio <= 0.8:
                            action = self.resolver._create_tight_avoidance_action(driver_id, disruption, state)
                            if action:
                                metrics = self._evaluate_action_metrics(
                                    action, route_points, original_length, original_time_on_modified_graph, disruption
                                )
                                if metrics:
                                    outcomes[ActionType.REROUTE_TIGHT_AVOIDANCE.display_name] = metrics
                        elif ratio >= 2.0:
                            action = self.resolver._create_wide_avoidance_action(driver_id, disruption, state)
                            if action:
                                metrics = self._evaluate_action_metrics(
                                    action, route_points, original_length, original_time_on_modified_graph, disruption
                                )
                                if metrics:
                                    outcomes[ActionType.REROUTE_WIDE_AVOIDANCE.display_name] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating parameterized avoidance with ratio {ratio}: {e}")

            return outcomes

        except Exception as e:
            logger.error(f"Error evaluating all actions: {e}")
            return {}
        finally:
            if original_graph_elements is not None:
                self._revert_graph_changes(original_graph_elements)
                logger.debug(f"Reverted graph changes for disruption {disruption.id}.")

    def _determine_best_action(self, action_outcomes: Dict[str, Dict[str, float]]) -> str:
        default_best = ActionType.REROUTE_BASIC.display_name
        if not action_outcomes:
            return default_best

        time_weight = 0.92
        length_weight = 0.08

        min_time = min(outcome["travel_time"] for outcome in action_outcomes.values())
        min_length = min(outcome["route_length"] for outcome in action_outcomes.values())

        max_time = max(outcome["travel_time"] for outcome in action_outcomes.values())
        max_length = max(outcome["route_length"] for outcome in action_outcomes.values())

        best_action_key = None
        best_score = float('inf')

        for action_type_key, metrics in action_outcomes.items():
            norm_time = (metrics["travel_time"] - min_time) / max(1, max_time - min_time)
            norm_length = (metrics["route_length"] - min_length) / max(1, max_length - min_length)

            score = time_weight * norm_time + length_weight * norm_length

            if score < best_score:
                best_score = score
                best_action_key = action_type_key

        if best_action_key is None:
            return default_best

        if best_action_key == ActionType.NO_ACTION.display_name:
            return ActionType.NO_ACTION.display_name
        elif best_action_key == ActionType.REROUTE_BASIC.display_name:
            return ActionType.REROUTE_BASIC.display_name
        elif best_action_key == ActionType.REROUTE_TIGHT_AVOIDANCE.display_name:
            return ActionType.REROUTE_TIGHT_AVOIDANCE.display_name
        elif best_action_key == ActionType.REROUTE_WIDE_AVOIDANCE.display_name:
            return ActionType.REROUTE_WIDE_AVOIDANCE.display_name
        elif best_action_key.startswith('parameterized_avoidance_'):
            try:
                ratio = float(best_action_key.split('_')[-1])
                if ratio <= 1.0:
                    return ActionType.REROUTE_TIGHT_AVOIDANCE.display_name
                else:
                    return ActionType.REROUTE_WIDE_AVOIDANCE.display_name
            except ValueError:
                return default_best
        else:
            return default_best


if __name__ == "__main__":
    import argparse
    from config.config import Config
    from models.services.graph import load_graph, get_largest_connected_component

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Generate training data for ML classifier')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file')
    args = parser.parse_args()

    config = Config()

    print("Loading graph...")
    graph_path = config.get_osm_file_path()
    try:
        graph = load_graph(graph_path)
        graph = get_largest_connected_component(graph)
    except FileNotFoundError:
        print(f"Graph file not found at {graph_path}")
        print("Please ensure the graph file exists before running this script.")
        exit(1)

    warehouse_location = config.get_warehouse_location()
    delivery_points = config.get_delivery_points()

    print(f"Creating data generator with {args.samples} samples, seed {args.seed}")
    generator = MLDataGenerator(
        graph=graph,
        warehouse_location=warehouse_location,
        delivery_points=delivery_points,
        num_samples=args.samples,
        random_seed=args.seed,
        save_full_scenario_data=True
    )

    generator.generate_training_data()
