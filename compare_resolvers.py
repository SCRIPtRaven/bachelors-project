import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from models.resolvers.classifier_based_resolver import ClassifierBasedResolver
from models.resolvers.rule_based_resolver import RuleBasedResolver
from train_classifier import generate_synthetic_states_disruptions, preprocess_graph


def haversine_distance(point1, point2):
    """Calculate the Haversine distance between two points in meters"""
    lat1, lon1 = float(point1[0]), float(point1[1])
    lat2, lon2 = float(point2[0]), float(point2[1])

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth radius in meters

    return c * r


def calculate_route_length(route):
    """Calculate the total length of a route in meters"""
    if not route or len(route) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(route) - 1):
        total_length += haversine_distance(route[i], route[i + 1])

    return total_length


def min_distance_to_disruption(route, disruption):
    """Calculate the minimum distance from any point in the route to the disruption"""
    if not route:
        return float('inf')

    min_dist = float('inf')
    for point in route:
        dist = haversine_distance(point, disruption.location)
        min_dist = min(min_dist, dist)

    return min_dist


def compare_resolvers(graph, warehouse_location, model_path, num_samples=100):
    """Compare performance of rule-based and classifier-based resolvers."""
    # Load resolvers
    rule_based_resolver = RuleBasedResolver(graph, warehouse_location)
    classifier_resolver = ClassifierBasedResolver(graph, warehouse_location, model_path)

    # Generate test cases
    print(f"Generating {num_samples} test cases...")
    test_cases = generate_synthetic_states_disruptions(graph, warehouse_location, num_samples)

    # Performance metrics
    rule_times = []
    classifier_times = []
    agreement_count = 0
    action_type_counts = {
        "rule_based": {-1: 0, 1: 0, 2: 0},  # -1: no action, 1: RECIPIENT_UNAVAILABLE, 2: REROUTE
        "classifier": {-1: 0, 1: 0, 2: 0}
    }

    # Route quality metrics for matched reroute actions
    quality_metrics = []

    # Disruption type statistics
    disruption_type_stats = {}

    print("Comparing resolvers on test cases...")
    for state, disruption in tqdm(test_cases):
        # Record disruption type for analysis
        disruption_type = disruption.type.value if hasattr(disruption, 'type') else "unknown"

        # Track statistics by disruption type
        if disruption_type not in disruption_type_stats:
            disruption_type_stats[disruption_type] = {
                "count": 0,
                "rule_actions": {-1: 0, 1: 0, 2: 0},
                "classifier_actions": {-1: 0, 1: 0, 2: 0},
                "agreement_count": 0
            }
        disruption_type_stats[disruption_type]["count"] += 1

        # Test rule-based resolver
        start_time = time.time()
        rule_actions = rule_based_resolver.on_disruption_detected(disruption, state)
        rule_time = time.time() - start_time
        rule_times.append(rule_time)

        # Determine action type from rule-based resolver
        rule_action_type = -1  # Default: no action
        if rule_actions:
            rule_action_type = rule_actions[0].action_type.value
        action_type_counts["rule_based"][rule_action_type] += 1
        disruption_type_stats[disruption_type]["rule_actions"][rule_action_type] += 1

        # Test classifier-based resolver
        start_time = time.time()
        classifier_actions = classifier_resolver.on_disruption_detected(disruption, state)
        classifier_time = time.time() - start_time
        classifier_times.append(classifier_time)

        # Determine action type from classifier-based resolver
        classifier_action_type = -1  # Default: no action
        if classifier_actions:
            classifier_action_type = classifier_actions[0].action_type.value
        action_type_counts["classifier"][classifier_action_type] += 1
        disruption_type_stats[disruption_type]["classifier_actions"][classifier_action_type] += 1

        # Check agreement
        if rule_action_type == classifier_action_type:
            agreement_count += 1
            disruption_type_stats[disruption_type]["agreement_count"] += 1

            # If both resolvers recommend a reroute, compare route quality
            if rule_action_type == 2 and rule_actions and classifier_actions:  # REROUTE action
                try:
                    # Extract routes
                    rule_route = rule_actions[0].new_route
                    classifier_route = classifier_actions[0].new_route

                    # Calculate route metrics
                    rule_length = calculate_route_length(rule_route)
                    classifier_length = calculate_route_length(classifier_route)

                    rule_min_dist = min_distance_to_disruption(rule_route, disruption)
                    classifier_min_dist = min_distance_to_disruption(classifier_route, disruption)

                    quality_metrics.append({
                        "disruption_type": disruption_type,
                        "disruption_severity": disruption.severity if hasattr(disruption, 'severity') else 0.5,
                        "rule_route_length": rule_length,
                        "classifier_route_length": classifier_length,
                        "rule_route_points": len(rule_route),
                        "classifier_route_points": len(classifier_route),
                        "rule_min_distance": rule_min_dist,
                        "classifier_min_distance": classifier_min_dist,
                        "length_ratio": classifier_length / rule_length if rule_length > 0 else 1.0,
                        "distance_ratio": classifier_min_dist / rule_min_dist if rule_min_dist > 0 else 1.0
                    })
                except Exception as e:
                    print(f"Error calculating route metrics: {e}")

    # Calculate statistics
    rule_times_ms = np.array(rule_times) * 1000
    classifier_times_ms = np.array(classifier_times) * 1000

    avg_rule_time = np.mean(rule_times_ms)
    avg_classifier_time = np.mean(classifier_times_ms)

    speedup = avg_rule_time / avg_classifier_time if avg_classifier_time > 0 else float('inf')
    agreement_rate = (agreement_count / len(test_cases)) * 100

    # Print results
    print("\nPerformance Results:")
    print(f"Rule-based resolver average time: {avg_rule_time:.2f} ms")
    print(f"Classifier-based resolver average time: {avg_classifier_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Agreement rate: {agreement_rate:.2f}%")

    print("\nAction Type Distribution:")
    print(f"Rule-based: No Action: {action_type_counts['rule_based'][-1]}, "
          f"Recipient Unavailable: {action_type_counts['rule_based'][1]}, "
          f"Reroute: {action_type_counts['rule_based'][2]}")
    print(f"Classifier: No Action: {action_type_counts['classifier'][-1]}, "
          f"Recipient Unavailable: {action_type_counts['classifier'][1]}, "
          f"Reroute: {action_type_counts['classifier'][2]}")

    # Additional classifier statistics
    print("\nClassifier Statistics:")
    stats = classifier_resolver.get_performance_stats()
    print(f"Fallback rate: {stats['fallback_rate'] * 100:.2f}%")
    print(f"Average decision time: {stats['avg_decision_time_ms']:.2f} ms")

    # Print disruption type analysis
    print("\nAnalysis by Disruption Type:")
    for disruption_type, data in disruption_type_stats.items():
        type_agreement_rate = data["agreement_count"] / data["count"] * 100 if data["count"] > 0 else 0
        print(f"\n  {disruption_type} ({data['count']} instances, {type_agreement_rate:.2f}% agreement):")
        print(f"    Rule-based: No Action: {data['rule_actions'][-1]}, "
              f"Recipient Unavailable: {data['rule_actions'][1]}, "
              f"Reroute: {data['rule_actions'][2]}")
        print(f"    Classifier: No Action: {data['classifier_actions'][-1]}, "
              f"Recipient Unavailable: {data['classifier_actions'][1]}, "
              f"Reroute: {data['classifier_actions'][2]}")

    # Print route quality analysis
    if quality_metrics:
        print("\nRoute Quality Analysis:")
        avg_rule_length = sum(m["rule_route_length"] for m in quality_metrics) / len(quality_metrics)
        avg_classifier_length = sum(m["classifier_route_length"] for m in quality_metrics) / len(quality_metrics)

        avg_rule_points = sum(m["rule_route_points"] for m in quality_metrics) / len(quality_metrics)
        avg_classifier_points = sum(m["classifier_route_points"] for m in quality_metrics) / len(quality_metrics)

        avg_rule_distance = sum(m["rule_min_distance"] for m in quality_metrics) / len(quality_metrics)
        avg_classifier_distance = sum(m["classifier_min_distance"] for m in quality_metrics) / len(quality_metrics)

        print(
            f"  Route Length: Rule={avg_rule_length:.2f}m, Classifier={avg_classifier_length:.2f}m, Ratio={avg_classifier_length / avg_rule_length:.2f}")
        print(
            f"  Route Points: Rule={avg_rule_points:.2f}, Classifier={avg_classifier_points:.2f}, Ratio={avg_classifier_points / avg_rule_points:.2f}")
        print(
            f"  Min Disruption Distance: Rule={avg_rule_distance:.2f}m, Classifier={avg_classifier_distance:.2f}m, Ratio={avg_classifier_distance / avg_rule_distance:.2f}")

        # Analyze route quality by disruption type
        print("\n  Route Quality by Disruption Type:")
        disruption_types = set(m["disruption_type"] for m in quality_metrics)
        for disruption_type in disruption_types:
            type_metrics = [m for m in quality_metrics if m["disruption_type"] == disruption_type]
            if not type_metrics:
                continue

            avg_length_ratio = sum(m["length_ratio"] for m in type_metrics) / len(type_metrics)
            avg_distance_ratio = sum(m["distance_ratio"] for m in type_metrics) / len(type_metrics)

            print(f"    {disruption_type} ({len(type_metrics)} routes):")
            print(f"      Route Length Ratio (Classifier/Rule): {avg_length_ratio:.2f}")
            print(f"      Disruption Distance Ratio (Classifier/Rule): {avg_distance_ratio:.2f}")

    # Create plots directory
    os.makedirs("plots", exist_ok=True)

    # Plot timing comparison
    plt.figure(figsize=(10, 6))
    plt.hist([rule_times_ms, classifier_times_ms], bins=20,
             label=['Rule-based', 'Classifier-based'], alpha=0.7)
    plt.axvline(avg_rule_time, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(avg_classifier_time, color='orange', linestyle='dashed', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Resolver Execution Time Comparison')
    plt.legend()
    plt.savefig('plots/timing_comparison.png')
    plt.close()

    # Plot action type distribution
    plt.figure(figsize=(12, 6))
    labels = ['No Action', 'Recipient Unavailable', 'Reroute']
    rule_values = [action_type_counts['rule_based'][i] for i in [-1, 1, 2]]
    classifier_values = [action_type_counts['classifier'][i] for i in [-1, 1, 2]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, rule_values, width, label='Rule-based')
    rects2 = ax.bar(x + width / 2, classifier_values, width, label='Classifier-based')

    ax.set_xlabel('Action Type')
    ax.set_ylabel('Count')
    ax.set_title('Action Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('plots/action_distribution.png')
    plt.close()

    # Plot route quality comparison if applicable
    if quality_metrics:
        # Plot route length comparison
        plt.figure(figsize=(10, 6))
        rule_lengths = [m["rule_route_length"] for m in quality_metrics]
        classifier_lengths = [m["classifier_route_length"] for m in quality_metrics]

        plt.scatter(rule_lengths, classifier_lengths, alpha=0.7)

        # Add diagonal line for reference
        max_length = max(max(rule_lengths), max(classifier_lengths))
        plt.plot([0, max_length], [0, max_length], 'k--', alpha=0.5)

        plt.xlabel('Rule-based Route Length (m)')
        plt.ylabel('Classifier Route Length (m)')
        plt.title('Route Length Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/route_length_comparison.png')
        plt.close()

        # Plot minimum distance to disruption
        plt.figure(figsize=(10, 6))
        rule_distances = [m["rule_min_distance"] for m in quality_metrics]
        classifier_distances = [m["classifier_min_distance"] for m in quality_metrics]

        plt.scatter(rule_distances, classifier_distances, alpha=0.7)

        # Add diagonal line for reference
        max_distance = max(max(rule_distances), max(classifier_distances))
        plt.plot([0, max_distance], [0, max_distance], 'k--', alpha=0.5)

        plt.xlabel('Rule-based Min Distance to Disruption (m)')
        plt.ylabel('Classifier Min Distance to Disruption (m)')
        plt.title('Disruption Avoidance Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/disruption_distance_comparison.png')
        plt.close()

    return {
        'rule_times_ms': rule_times_ms.tolist(),
        'classifier_times_ms': classifier_times_ms.tolist(),
        'avg_rule_time': avg_rule_time,
        'avg_classifier_time': avg_classifier_time,
        'speedup': speedup,
        'agreement_rate': agreement_rate,
        'action_counts': action_type_counts,
        'disruption_type_stats': disruption_type_stats,
        'quality_metrics': quality_metrics if quality_metrics else None
    }


if __name__ == "__main__":
    import sys

    default_graph_file = "resources/data/graphs/kaunas.graphml"
    default_model_path = "models/trained_resolver.joblib"
    default_samples = 100

    print(f"Usage: python {sys.argv[0]} <graph_file> <model_path> [num_samples]")
    print(f"  graph_file: Path to the graph file (default: {default_graph_file})")
    print(f"  model_path: Path to the trained model (default: {default_model_path})")
    print(f"  num_samples: Number of test samples to generate (default: {default_samples})")
    print()

    if len(sys.argv) >= 2:
        graph_file = sys.argv[1]
    else:
        graph_file = default_graph_file
        print(f"No graph file specified, using default: {graph_file}")

    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    else:
        model_path = default_model_path
        print(f"No model path specified, using default: {model_path}")

    if len(sys.argv) >= 4:
        try:
            num_samples = int(sys.argv[3])
        except ValueError:
            print(f"Invalid number of samples, using default: {default_samples}")
            num_samples = default_samples
    else:
        num_samples = default_samples
        print(f"No sample count specified, using default: {num_samples}")

    # Check if files exist
    if not os.path.exists(graph_file):
        print(f"Error: Graph file '{graph_file}' not found.")
        print("Please ensure the graph file exists or specify a different path.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please ensure the model file exists or train a model first using train_classifier.py.")
        sys.exit(1)

    # Create directories for results
    os.makedirs("plots", exist_ok=True)

    # Load graph
    print(f"Loading graph from {graph_file}...")
    try:
        file_extension = os.path.splitext(graph_file)[1].lower()
        if file_extension == '.graphml':
            # Load GraphML file
            graph = nx.read_graphml(graph_file)
            print("Loaded GraphML file successfully")
        elif file_extension == '.pickle':
            # Load pickle file
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            print("Loaded pickle file successfully")
        else:
            print(f"Unsupported file extension: {file_extension}")
            print("Supported formats: .graphml, .pickle")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading graph file: {e}")
        sys.exit(1)

    graph = preprocess_graph(graph)

    try:
        node_coords = []
        for _, data in graph.nodes(data=True):
            if 'y' in data and 'x' in data:
                try:
                    lat = float(data['y'])
                    lon = float(data['x'])
                    node_coords.append((lat, lon))
                except (ValueError, TypeError):
                    continue

        if not node_coords:
            print("Warning: No valid coordinates found in graph nodes.")
            # Fallback to a default location (example coordinates for Kaunas)
            warehouse_location = (54.8985, 23.9036)
            print(f"Using fallback warehouse location: {warehouse_location}")
        else:
            center_lat = sum(lat for lat, _ in node_coords) / len(node_coords)
            center_lon = sum(lon for _, lon in node_coords) / len(node_coords)
            warehouse_location = (center_lat, center_lon)
            print(f"Using warehouse location at center of graph: ({center_lat:.6f}, {center_lon:.6f})")

        print(f"Starting comparison with {num_samples} test samples...")
        # Run the comparison
        results = compare_resolvers(graph, warehouse_location, model_path, num_samples)

        # Save results
        with open('comparison_results.json', 'w') as f:
            import json


            # Convert any numpy arrays or non-serializable objects to lists or primitive types
            def clean_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(clean_for_json(v) for v in obj)
                else:
                    return obj


            # Clean the results for JSON serialization
            clean_results = clean_for_json(results)
            json.dump(clean_results, f, indent=2)

        print("\nResults saved to comparison_results.json")
        print("Visualizations saved to plots/timing_comparison.png and plots/action_distribution.png")
        if results.get('quality_metrics'):
            print(
                "Route quality visualizations saved to plots/route_length_comparison.png and plots/disruption_distance_comparison.png")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
