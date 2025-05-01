#!/usr/bin/env python
"""
Wrapper script to run the resolver comparison from the project root directory.
This ensures Python can find all the necessary modules.
"""

import os
import sys

# Ensure we're running from the project root
project_root = os.path.abspath(os.path.dirname(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Now import and run the comparison
from utils.resolver_comparison import ResolverComparison
from config.config import Config
from models.services.graph import load_graph, get_largest_connected_component


def main():
    """Run the resolver comparison"""
    # Load configuration
    config = Config()
    
    # Load graph
    print("Loading graph...")
    graph_path = config.get_osm_file_path()
    try:
        graph = load_graph(graph_path)
        graph = get_largest_connected_component(graph)
    except FileNotFoundError:
        print(f"ERROR: Graph file not found at {graph_path}")
        print("Please ensure the graph file exists before running this script.")
        return 1
    
    # Get warehouse and delivery points
    warehouse_location = config.get_warehouse_location()
    delivery_points = config.get_delivery_points()
    
    # Create and run the comparison
    try:
        comparison = ResolverComparison(
            graph=graph,
            warehouse_location=warehouse_location,
            delivery_points=delivery_points,
            num_scenarios=50  # Use a smaller number for quicker comparison
        )
        
        # Run comparison and get results path
        results_path = comparison.run_comparison()
        print(f"Comparison results saved to: {results_path}")
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to complete comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 