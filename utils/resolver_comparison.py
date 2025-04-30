import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.inspection import permutation_importance

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.rule_based_resolver import RuleBasedResolver
from models.resolvers.ml_classifier_resolver import MLClassifierResolver
from models.resolvers.actions import (
    DisruptionAction, RerouteBasicAction, RecipientUnavailableAction,
    NoRerouteAction, RerouteTightAvoidanceAction, RerouteWideAvoidanceAction
)
from models.resolvers.state import DeliverySystemState
from utils.geo_utils import calculate_haversine_distance
from utils.route_utils import calculate_route_length, calculate_travel_time
from utils.ml_data_generator import MLDataGenerator
from config.config import Config
from workers.graph_loader import GraphLoadWorker
from models.services import graph


class ResolverComparison:
    """
    Compares the performance of the ML-based resolver against the rule-based resolver
    on the same set of disruption scenarios.
    """
    
    RESULTS_DIR = os.path.join('models', 'resolvers', 'comparison_results')
    
    def __init__(self, graph, warehouse_location, delivery_points, num_scenarios=100, random_seed=42):
        self.G = graph
        self.warehouse_location = warehouse_location
        self.delivery_points = delivery_points
        self.num_scenarios = num_scenarios
        self.random_seed = random_seed
        
        # Create output directory
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # Create resolvers
        self.rule_based_resolver = RuleBasedResolver(graph, warehouse_location)
        self.ml_resolver = MLClassifierResolver(graph, warehouse_location)
        
        # Check if ML resolver has a classifier
        self.ml_available = self.ml_resolver.has_classifier()
        
        # Create data generator for scenarios
        self.data_generator = MLDataGenerator(
            graph=graph,
            warehouse_location=warehouse_location,
            delivery_points=delivery_points,
            num_samples=num_scenarios,
            random_seed=random_seed
        )
        
        # Results storage
        self.scenario_results = []
        
    def run_comparison(self) -> str:
        """
        Run the comparison between resolvers and save results
        
        Returns:
            Path to the saved results file
        """
        if not self.ml_available:
            print("WARNING: ML Classifier model not available. Will use rule-based resolver for both cases.")
        
        print(f"Generating {self.num_scenarios} comparison scenarios...")
        
        # Progress tracking
        progress_step = max(1, self.num_scenarios // 10)
        
        # Generate and process scenarios
        for i in range(self.num_scenarios):
            if i % progress_step == 0:
                print(f"Processing scenario {i+1}/{self.num_scenarios} ({(i+1)/self.num_scenarios*100:.1f}%)")
            
            # Generate a random scenario
            disruption, state, driver_id = self.data_generator._generate_random_scenario()
            
            if disruption is None or state is None or driver_id is None:
                print("Failed to generate a valid scenario, skipping...")
                continue
            
            # Process with both resolvers
            result = self._compare_resolvers_on_scenario(disruption, state, driver_id)
            
            if result:
                self.scenario_results.append(result)
        
        # Analyze and save results
        return self._analyze_and_save_results()
    
    def _compare_resolvers_on_scenario(self, 
                                       disruption: Disruption, 
                                       state: DeliverySystemState, 
                                       driver_id: int) -> Dict[str, Any]:
        """
        Compare both resolvers on a single scenario
        
        Returns:
            Dictionary with comparison results
        """
        try:
            # Extract features for reference using the same method as ML resolver
            features = self.ml_resolver._extract_features(driver_id, disruption, state)
            
            if features is None:
                return None
            
            # Get action from rule-based resolver
            rule_based_start = time.time()
            rule_based_actions = self.rule_based_resolver.resolve_disruptions(state, [disruption])
            rule_based_time = time.time() - rule_based_start
            
            # Get action from ML resolver
            ml_start = time.time()
            ml_actions = self.ml_resolver.resolve_disruptions(state, [disruption])
            ml_time = time.time() - ml_start
            
            # Find relevant actions for the driver
            rule_based_action = next((a for a in rule_based_actions if hasattr(a, 'driver_id') and a.driver_id == driver_id), None)
            ml_action = next((a for a in ml_actions if hasattr(a, 'driver_id') and a.driver_id == driver_id), None)
            
            # Determine action types
            rule_based_type = self._get_action_type(rule_based_action)
            ml_type = self._get_action_type(ml_action)
            
            # Get ML confidence scores if available
            ml_confidence = None
            if self.ml_resolver.classifier is not None and hasattr(self.ml_resolver.classifier, 'predict_proba'):
                try:
                    # Ensure features are in the correct shape and maintain feature names
                    if isinstance(features, pd.DataFrame):
                        # Keep DataFrame format to maintain feature names
                        if len(features) == 1:
                            features = features.iloc[0:1]  # Ensure single row
                    elif isinstance(features, np.ndarray):
                        # Convert to DataFrame with correct feature names
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
                        # Convert to DataFrame with correct feature names
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
            
            # Evaluate outcomes
            original_route = state.driver_routes.get(driver_id, {}).get('points', [])
            original_length = calculate_route_length(original_route)
            original_time = calculate_travel_time(original_route, self.G)
            
            rule_based_metrics = self._evaluate_action(rule_based_action, original_route, original_length, original_time, disruption)
            ml_metrics = self._evaluate_action(ml_action, original_route, original_length, original_time, disruption)
            
            # Determine relative performance
            time_improvement = 0
            length_improvement = 0
            length_diff_meters = 0
            time_diff_seconds = 0
            
            if rule_based_metrics and ml_metrics:
                # Calculate absolute differences
                length_diff_meters = rule_based_metrics["route_length"] - ml_metrics["route_length"]
                time_diff_seconds = rule_based_metrics["travel_time"] - ml_metrics["travel_time"]
                
                # Calculate improvement percentages with safeguards
                if rule_based_metrics["travel_time"] > 0:
                    time_improvement = (time_diff_seconds / rule_based_metrics["travel_time"]) * 100
                
                # For length improvement, handle edge cases and outliers
                if rule_based_metrics["route_length"] > 0:
                    # If rule-based is no_reroute but ML creates a route, or vice versa, handle specially
                    if rule_based_type == "none" and ml_type != "none":
                        # ML added a route when rule-based did nothing - this is a negative improvement
                        # Cap at -100% (can't be worse than doubling the original)
                        length_improvement = -100.0
                    elif rule_based_type != "none" and ml_type == "none":
                        # ML did nothing when rule-based added a route - this is a positive improvement
                        length_improvement = 100.0
                    else:
                        # Regular case - both made some decision
                        raw_improvement = (length_diff_meters / rule_based_metrics["route_length"]) * 100
                        
                        # Cap extreme values to prevent misleading percentages
                        # Limit to ±100% which means "twice as good/bad"
                        length_improvement = max(-100.0, min(100.0, raw_improvement))
            
            # Prepare result data
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
                
                "rule_based_route_length": rule_based_metrics.get("route_length") if rule_based_metrics else None,
                "rule_based_travel_time": rule_based_metrics.get("travel_time") if rule_based_metrics else None,
                
                "ml_route_length": ml_metrics.get("route_length") if ml_metrics else None,
                "ml_travel_time": ml_metrics.get("travel_time") if ml_metrics else None,
                
                "time_improvement_pct": time_improvement,
                "length_improvement_pct": length_improvement,
                "length_diff_meters": length_diff_meters,
                "time_diff_seconds": time_diff_seconds,
                
                "ml_confidence": ml_confidence
            }
            
            # Add features for reference
            if features is not None:
                feature_names = [
                    "disruption_type_road_closure",
                    "disruption_type_traffic_jam",
                    "distance_to_disruption"
                ]
                
                # Convert features to array for storage
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
        """Get the string representation of an action type"""
        if action is None:
            return "none"
        elif isinstance(action, NoRerouteAction):
            return "no_reroute"
        elif isinstance(action, RerouteTightAvoidanceAction):
            return "tight_avoidance"
        elif isinstance(action, RerouteWideAvoidanceAction):
            return "wide_avoidance"
        elif isinstance(action, RerouteBasicAction):
            return "basic_reroute"
        elif isinstance(action, RecipientUnavailableAction):
            return "recipient_unavailable"
        else:
            return f"unknown_{type(action).__name__}"
    
    def _evaluate_action(self, action: DisruptionAction, 
                         original_route: List[Tuple[float, float]], 
                         original_length: float, 
                         original_time: float,
                         disruption: Disruption) -> Dict[str, float]:
        """
        Evaluate the outcome metrics for an action
        """
        try:
            if action is None:
                return None
            
            if isinstance(action, NoRerouteAction):
                # For no_reroute, use original route but apply disruption penalty
                new_route = original_route
                
                # Apply penalty based on disruption type and severity
                if disruption.type == DisruptionType.ROAD_CLOSURE:
                    time_factor = 2.0  # Significant delay but not infinite
                elif disruption.type == DisruptionType.TRAFFIC_JAM:
                    time_factor = 1.0 + disruption.severity * 1.5
                else:
                    time_factor = 1.0
                
                new_length = original_length
                new_time = original_time * time_factor
                
            elif hasattr(action, 'new_route') and action.new_route:
                # For reroute actions, use the new route
                new_route = action.new_route
                new_length = calculate_route_length(new_route)
                new_time = calculate_travel_time(new_route, self.G)
                
                # Apply any necessary adjustments for specific action types
                if isinstance(action, RerouteTightAvoidanceAction) and disruption.type == DisruptionType.TRAFFIC_JAM:
                    # Tight avoidance might still be affected by traffic
                    new_time *= (1.0 + disruption.severity * 0.5)
            else:
                return None
            
            # Calculate metrics
            detour_length_factor = new_length / max(0.1, original_length)
            
            return {
                "route_length": new_length,
                "travel_time": new_time,
                "detour_factor": detour_length_factor
            }
            
        except Exception as e:
            print(f"Error evaluating action: {e}")
            return None
    
    def _analyze_and_save_results(self) -> str:
        """
        Analyze comparison results and save to file
        
        Returns:
            Path to the saved results file
        """
        if not self.scenario_results:
            print("No results to analyze.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.scenario_results)
        
        # Calculate summary statistics
        total_scenarios = len(df)
        matching_actions = df['actions_match'].sum()
        matching_pct = matching_actions / total_scenarios * 100
        
        # Calculate improvement metrics with outlier handling
        avg_time_improvement = df['time_improvement_pct'].mean()
        median_time_improvement = df['time_improvement_pct'].median()
        
        # Calculate average length improvement (using the capped values)
        avg_length_improvement = df['length_improvement_pct'].mean()
        median_length_improvement = df['length_improvement_pct'].median()
        
        # Calculate absolute metrics
        avg_length_diff = df['length_diff_meters'].mean()
        avg_time_diff = df['time_diff_seconds'].mean()
        
        rule_based_avg_time = df['rule_based_time_ms'].mean()
        ml_avg_time = df['ml_time_ms'].mean()
        
        # Calculate confidence statistics
        confidence_stats = {}
        if 'ml_confidence' in df.columns:
            confidence_stats = {
                'avg_confidence': df['ml_confidence'].mean(),
                'min_confidence': df['ml_confidence'].min(),
                'max_confidence': df['ml_confidence'].max(),
                'low_confidence_count': (df['ml_confidence'] < 0.4).sum(),
                'low_confidence_pct': (df['ml_confidence'] < 0.4).sum() / total_scenarios * 100
            }
        
        # Calculate feature importance if ML model is available
        feature_importance = {}
        if self.ml_resolver.classifier is not None and hasattr(self.ml_resolver.classifier, 'predict_proba'):
            try:
                # Get feature names
                feature_columns = [col for col in df.columns if col.startswith('feature_')]
                feature_names = [col.replace('feature_', '') for col in feature_columns]
                
                # Prepare data for importance calculation
                X = df[feature_columns].values
                y = df['ml_action'].values
                
                # Calculate permutation importance
                result = permutation_importance(
                    self.ml_resolver.classifier,
                    X,
                    y,
                    n_repeats=10,
                    random_state=42
                )
                
                # Store importance scores
                for i, name in enumerate(feature_names):
                    feature_importance[name] = {
                        'importance': result.importances_mean[i],
                        'std': result.importances_std[i]
                    }
            except Exception as e:
                print(f"Error calculating feature importance: {e}")
        
        # Generate summary report
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
            "ml_available": self.ml_available,
            "confidence_stats": confidence_stats,
            "feature_importance": feature_importance
        }
        
        # Action type distributions
        rule_based_counts = df['rule_based_action'].value_counts()
        ml_counts = df['ml_action'].value_counts()
        
        # Calculate paired metrics by action type
        action_metrics = {}
        for action in set(df['ml_action'].unique()):
            if action != "none":
                action_subset = df[df['ml_action'] == action]
                if len(action_subset) > 0:
                    action_metrics[action] = {
                        "count": len(action_subset),
                        "avg_length_improvement": action_subset['length_improvement_pct'].mean(),
                        "median_length_improvement": action_subset['length_improvement_pct'].median(),
                        "avg_time_improvement": action_subset['time_improvement_pct'].mean(),
                        "median_time_improvement": action_subset['time_improvement_pct'].median()
                    }
        
        # Print summary
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
            print(f"Low confidence decisions (<40%): {confidence_stats['low_confidence_count']} ({confidence_stats['low_confidence_pct']:.2f}%)")
        
        if feature_importance:
            print("\nFeature Importance:")
            for feature, importance in feature_importance.items():
                print(f"  {feature}: {importance['importance']:.4f} ± {importance['std']:.4f}")
        
        print("\nRule-based resolver action distribution:")
        for action, count in rule_based_counts.items():
            print(f"  {action}: {count} ({count/total_scenarios*100:.2f}%)")
        
        print("\nML resolver action distribution:")
        for action, count in ml_counts.items():
            print(f"  {action}: {count} ({count/total_scenarios*100:.2f}%)")
        
        if action_metrics:
            print("\nML Action Type Performance:")
            for action, metrics in action_metrics.items():
                print(f"  {action} ({metrics['count']} instances):")
                print(f"    Length improvement: {metrics['avg_length_improvement']:.2f}% (avg), {metrics['median_length_improvement']:.2f}% (median)")
                print(f"    Time improvement: {metrics['avg_time_improvement']:.2f}% (avg), {metrics['median_time_improvement']:.2f}% (median)")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.RESULTS_DIR, f"comparison_results_{timestamp}.csv")
        summary_file = os.path.join(self.RESULTS_DIR, f"comparison_summary_{timestamp}.txt")
        
        # Save detailed results
        df.to_csv(results_file, index=False)
        
        # Save summary
        with open(summary_file, 'w') as f:
            f.write("Resolver Comparison Summary\n")
            f.write("===========================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total scenarios: {total_scenarios}\n")
            f.write(f"ML resolver available: {self.ml_available}\n\n")
            
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
                f.write(f"Low confidence decisions (<40%): {confidence_stats['low_confidence_count']} ({confidence_stats['low_confidence_pct']:.2f}%)\n\n")
            
            if feature_importance:
                f.write("Feature Importance:\n")
                for feature, importance in feature_importance.items():
                    f.write(f"  {feature}: {importance['importance']:.4f} ± {importance['std']:.4f}\n")
                f.write("\n")
            
            f.write("Rule-based resolver action distribution:\n")
            for action, count in rule_based_counts.items():
                f.write(f"  {action}: {count} ({count/total_scenarios*100:.2f}%)\n")
            
            f.write("\nML resolver action distribution:\n")
            for action, count in ml_counts.items():
                f.write(f"  {action}: {count} ({count/total_scenarios*100:.2f}%)\n")
            
            if action_metrics:
                f.write("\nML Action Type Performance:\n")
                for action, metrics in action_metrics.items():
                    f.write(f"  {action} ({metrics['count']} instances):\n")
                    f.write(f"    Length improvement: {metrics['avg_length_improvement']:.2f}% (avg), {metrics['median_length_improvement']:.2f}% (median)\n")
                    f.write(f"    Time improvement: {metrics['avg_time_improvement']:.2f}% (avg), {metrics['median_time_improvement']:.2f}% (median)\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        
        return summary_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare ML and rule-based resolvers')
    parser.add_argument('--scenarios', type=int, default=100, help='Number of scenarios to compare')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Load graph
    print("Loading graph...")
    graph_path = config.get_osm_file_path()
    try:
        G = graph.load_graph(graph_path)
        G = graph.get_largest_connected_component(G)
    except FileNotFoundError:
        print(f"Graph file not found at {graph_path}")
        print("Please ensure the graph file exists before running this script.")
        exit(1)
    
    # Create comparison runner
    warehouse_location = config.get_warehouse_location()
    delivery_points = config.get_delivery_points()
    
    print(f"Creating comparison with {args.scenarios} scenarios, seed {args.seed}")
    comparison = ResolverComparison(
        graph=G,
        warehouse_location=warehouse_location,
        delivery_points=delivery_points,
        num_scenarios=args.scenarios,
        random_seed=args.seed
    )
    
    # Run comparison
    comparison.run_comparison() 