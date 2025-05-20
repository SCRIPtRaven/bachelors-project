import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import osmnx as ox

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.services.optimization.simulated_annealing import SimulatedAnnealingOptimizer
from models.entities.driver import Driver
from config.config import OptimizationConfig
from models.services.graph_service import load_graph, get_largest_connected_component

CITY_NAME_FOR_WAREHOUSE = "Paris, France"
DEFAULT_WAREHOUSE_FROM_GEOCODE = ox.geocode(CITY_NAME_FOR_WAREHOUSE)
GEOCODED_WAREHOUSE_COORDS = (DEFAULT_WAREHOUSE_FROM_GEOCODE[0], DEFAULT_WAREHOUSE_FROM_GEOCODE[1])

GRAPH_DIR_RELATIVE_PATH = os.path.join("resources", "data", "graphs")

def load_drivers_from_file(file_path):
    drivers = []
    if not os.path.exists(file_path):
        print(f"Error: Driver file not found at {file_path}")
        return drivers
    try:
        with open(file_path, 'r') as f:
            driver_data_list = json.load(f)
        for data in driver_data_list:
            drivers.append(Driver(
                id=data['id'],
                weight_capacity=data['weight_capacity'],
                volume_capacity=data['volume_capacity']
            ))
        print(f"Successfully loaded {len(drivers)} drivers from {file_path}")
    except Exception as e:
        print(f"Error loading drivers from {file_path}: {e}")
    return drivers


def load_deliveries_from_file(file_path):
    delivery_points = []
    warehouse_location = None
    city_name = None
    if not os.path.exists(file_path):
        print(f"Error: Delivery file not found at {file_path}")
        return delivery_points, warehouse_location, city_name
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if 'points' not in data:
            print(f"Error: 'points' key missing in delivery file {file_path}")
            return [], None, None

        raw_points = data.get('points', [])
        if not isinstance(raw_points, list):
            print(f"Error: 'points' in {file_path} is not a list. Found: {type(raw_points)}")
            return [], None, None

        delivery_points = []
        for i, p_data in enumerate(raw_points):
            if isinstance(p_data, dict) and \
                    'latitude' in p_data and 'longitude' in p_data and \
                    'weight' in p_data and 'volume' in p_data:
                try:
                    lat = float(p_data['latitude'])
                    lon = float(p_data['longitude'])
                    weight = float(p_data['weight'])
                    volume = float(p_data['volume'])
                    delivery_points.append((lat, lon, weight, volume))
                except (ValueError, TypeError) as e_cast:
                    print(
                        f"Error: Point {i} in {file_path} has non-numeric data: {p_data}. Error: {e_cast}")
            elif isinstance(p_data, list) and len(
                    p_data) == 4:
                try:
                    delivery_points.append(
                        (float(p_data[0]), float(p_data[1]), float(p_data[2]), float(p_data[3])))
                except (ValueError, TypeError) as e_cast:
                    print(
                        f"Error: Point {i} in {file_path} (list format) has non-numeric data: {p_data}. Error: {e_cast}")
            else:
                print(
                    f"Error: Point {i} in {file_path} is not a valid dictionary or list of 4 elements. Found: {p_data}")

        if not delivery_points and raw_points:
            print(
                f"Warning: No valid delivery points processed from {file_path} despite raw_points being present.")
        elif not raw_points:
            print(f"Warning: 'points' array is empty in {file_path}.")

        warehouse_location = data.get('warehouse')
        if warehouse_location and isinstance(warehouse_location, list) and len(
                warehouse_location) == 2:
            warehouse_location = tuple(warehouse_location)
        city_name = data.get('city')
        print(
            f"Successfully loaded {len(delivery_points)} delivery points from {file_path} for city: {city_name or 'N/A'}.")
        if warehouse_location:
            print(f"Warehouse location from file: {warehouse_location}")
    except Exception as e:
        print(f"Error loading deliveries from {file_path}: {e}")
    return delivery_points, warehouse_location, city_name


def run_single_optimization_iteration(scenario_name, drivers, delivery_points, warehouse_location,
                                      graph, sa_params_override, default_sa_settings):
    current_sa_settings = deepcopy(default_sa_settings)
    current_sa_settings.update(sa_params_override)
    current_sa_settings['VISUALIZE_PROCESS'] = False

    original_global_sa_settings = deepcopy(OptimizationConfig.SETTINGS)
    OptimizationConfig.SETTINGS = current_sa_settings

    optimizer = SimulatedAnnealingOptimizer(
        delivery_drivers=drivers,
        snapped_delivery_points=delivery_points,
        G=graph,
        warehouse_coords=warehouse_location
    )

    exec_time = 0.0
    time_improvement = 0.0
    solution_details = None
    unassigned_details = None

    print(f"Starting optimization run for {scenario_name}...")
    try:
        solution_details, unassigned_details, exec_time, time_improvement = optimizer.optimize()
    except Exception as e:
        print(f"Error during optimization for {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        OptimizationConfig.SETTINGS = original_global_sa_settings
    return solution_details, unassigned_details, exec_time, time_improvement


def main():
    parser = argparse.ArgumentParser(description="Run batch Simulated Annealing optimizations.")
    parser.add_argument(
        '--graph_city',
        type=str,
        default="Paris, France",
        help="City name to load the graph for (e.g., 'Paris, France'). GraphML file should exist."
    )
    parser.add_argument(
        '--runs_per_scenario',
        type=int,
        default=5,
        help="Number of times to run each scenario configuration."
    )
    args = parser.parse_args()

    BASE_SCENARIOS = [
        {
            "name_prefix": "S1_Small", "num_drivers": 2, "num_deliveries": 50,
            "driver_file": "saved_configurations/drivers_2_20250514_224608.json",
            "delivery_file": "saved_configurations/deliveries_50_Paris_20250514_224554.json",
        },
        {
            "name_prefix": "S2_Medium", "num_drivers": 5, "num_deliveries": 125,
            "driver_file": "saved_configurations/drivers_5_20250514_224604.json",
            "delivery_file": "saved_configurations/deliveries_125_Paris_20250514_224546.json",
        },
        {
            "name_prefix": "S3_Large", "num_drivers": 10, "num_deliveries": 250,
            "driver_file": "saved_configurations/drivers_10_20250514_224558.json",
            "delivery_file": "saved_configurations/deliveries_250_Paris_20250514_224540.json",
        }
    ]

    HYPERPARAMETER_SETS = {
        "H1_Fast": {
            "INITIAL_TEMPERATURE": 100.0,
            "COOLING_RATE": 0.90,
            "MIN_TEMPERATURE": 1.0,
            "ITERATIONS_PER_TEMPERATURE": 50,
        },
        "H2_Balanced": {
            "INITIAL_TEMPERATURE": 500.0,
            "COOLING_RATE": 0.95,
            "MIN_TEMPERATURE": 0.5,
            "ITERATIONS_PER_TEMPERATURE": 100,
        },
        "H3_Accurate": {
            "INITIAL_TEMPERATURE": 1000.0,
            "COOLING_RATE": 0.99,
            "MIN_TEMPERATURE": 0.1,
            "ITERATIONS_PER_TEMPERATURE": 200,
        }
    }

    graph_file_name = "paris.graphml"
    graph_path = os.path.join(project_root, GRAPH_DIR_RELATIVE_PATH, graph_file_name)

    if not os.path.exists(graph_path):
        print(f"ERROR: Graph file not found at {graph_path}. Please ensure it exists.")
        return
    print(f"Loading graph for {CITY_NAME_FOR_WAREHOUSE} from {graph_path}...")
    G = load_graph(filename=graph_path)
    if G is None:
        print(f"Failed to load graph for {CITY_NAME_FOR_WAREHOUSE}.")
        return
    G = get_largest_connected_component(G)
    print(f"Graph for {CITY_NAME_FOR_WAREHOUSE} loaded successfully.")

    original_global_sa_settings = deepcopy(OptimizationConfig.SETTINGS)
    all_scenario_results = []

    for base_scenario_info in BASE_SCENARIOS:
        print(
            f"\nProcessing Base Scenario: {base_scenario_info['name_prefix']} (Drivers: {base_scenario_info['num_drivers']}, Deliveries: {base_scenario_info['num_deliveries']})")

        driver_file = base_scenario_info['driver_file']
        delivery_file = base_scenario_info['delivery_file']

        drivers = load_drivers_from_file(driver_file)
        if not drivers:
            print(
                f"Skipping base scenario {base_scenario_info['name_prefix']} due to driver loading error from {driver_file}.")
            continue

        delivery_points, file_warehouse_coords, _ = load_deliveries_from_file(delivery_file)
        if not delivery_points:
            print(
                f"Skipping base scenario {base_scenario_info['name_prefix']} due to delivery loading error from {delivery_file}.")
            continue

        warehouse_loc_to_use = file_warehouse_coords
        if warehouse_loc_to_use is None:
            print(
                f"INFO: Warehouse location not found in delivery file {delivery_file} for scenario {base_scenario_info['name_prefix']}. Using geocoded location for {CITY_NAME_FOR_WAREHOUSE}.")
            warehouse_loc_to_use = GEOCODED_WAREHOUSE_COORDS

        print(f"Using Warehouse: {warehouse_loc_to_use} for {base_scenario_info['name_prefix']}")

        for hp_name, hp_params in HYPERPARAMETER_SETS.items():
            full_scenario_name = f"{base_scenario_info['name_prefix']}_{hp_name}"
            print(f"\n  Testing Hyperparameters: {hp_name} for {base_scenario_info['name_prefix']}")

            scenario_runs_exec_time = []
            scenario_runs_improvement = []

            for i in range(args.runs_per_scenario):
                print(f"    Run {i + 1}/{args.runs_per_scenario} for {full_scenario_name}")
                solution_details, unassigned_details, exec_time, improvement = run_single_optimization_iteration(
                    full_scenario_name + f"_Run{i + 1}",
                    drivers,
                    delivery_points,
                    warehouse_loc_to_use,
                    G,
                    hp_params,
                    original_global_sa_settings
                )
                if exec_time is not None and improvement is not None:
                    scenario_runs_exec_time.append(exec_time)
                    scenario_runs_improvement.append(improvement)

            if scenario_runs_exec_time:
                avg_exec_time = np.mean(scenario_runs_exec_time)
                avg_time_improvement = np.mean(scenario_runs_improvement)
                avg_optimality = (avg_time_improvement / avg_exec_time) if avg_exec_time > 0 else 0

                result_summary = (
                    f"\n  Results for {full_scenario_name} (after {len(scenario_runs_exec_time)} successful runs):"
                    f"\n    Average Execution Time: {avg_exec_time:.2f} seconds"
                    f"\n    Average Time Improvement: {avg_time_improvement:.2f} %"
                    f"\n    Average Optimality (Improvement/ExecTime): {avg_optimality:.4f}"
                )
                print(result_summary)
                all_scenario_results.append(result_summary)
            else:
                print(
                    f"\n  No successful runs for {full_scenario_name} to calculate average results.")
            print("\n" + "-" * 60 + "\n")

    print("\n" + "=" * 80 + "\nBatch optimization process completed.")
    print("Summary of All Scenario Averages:")
    for res in all_scenario_results:
        print(res)


if __name__ == "__main__":
    os.makedirs("saved_configurations",
                exist_ok=True)
    main()
