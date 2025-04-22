import pickle
import random
import time

from models.entities.disruption import Disruption, DisruptionType
from models.resolvers.classifier_based_resolver import (
    ClassifierBasedResolver, TrainingDataGenerator
)
from models.resolvers.state import DeliverySystemState
from models.services.geolocation import GeolocationService


def generate_synthetic_states_disruptions(graph, warehouse_location, num_samples=1000):
    """Generate synthetic state-disruption pairs for training data."""

    states_disruptions = []

    print(f"Generating {num_samples} training samples...")

    # Get bounds from graph for generating random points - convert coordinates to float
    node_coords = []
    for _, data in graph.nodes(data=True):
        try:
            lat = float(data['y'])
            lon = float(data['x'])
            node_coords.append((lat, lon))
        except (KeyError, ValueError, TypeError):
            continue  # Skip nodes without valid coordinates

    if not node_coords:
        print("Error: No valid coordinates found in graph")
        return []

    bounds = (
        min(lat for lat, _ in node_coords),
        max(lat for lat, _ in node_coords),
        min(lon for _, lon in node_coords),
        max(lon for _, lon in node_coords)
    )

    print(f"Graph bounds: {bounds}")

    # Generate diverse training scenarios
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples")

        try:
            # Random number of delivery points (5-30)
            num_points = random.randint(5, 30)
            delivery_points = GeolocationService.generate_delivery_points(bounds, num_points)

            # Random number of drivers (1-8)
            num_drivers = random.randint(1, 8)
            drivers = GeolocationService.generate_delivery_drivers(num_drivers)

            # Random driver assignments
            driver_assignments = {}
            for idx in range(len(delivery_points)):
                driver_id = random.randint(1, num_drivers)
                if driver_id not in driver_assignments:
                    driver_assignments[driver_id] = []
                driver_assignments[driver_id].append(idx)

            # Random driver positions
            driver_positions = {}
            for driver in drivers:
                if driver.id in driver_assignments and driver_assignments[driver.id]:
                    # Position near an assigned delivery
                    delivery_idx = random.choice(driver_assignments[driver.id])
                    delivery = delivery_points[delivery_idx]

                    # Ensure coordinates are float values
                    base_lat = float(delivery.coordinates[0])
                    base_lon = float(delivery.coordinates[1])

                    # Add small random offset
                    lat_offset = random.uniform(-0.001, 0.001)
                    lon_offset = random.uniform(-0.001, 0.001)

                    driver_positions[driver_id] = (
                        float(base_lat) + float(lat_offset),
                        float(base_lon) + float(lon_offset)
                    )
                else:
                    # Position near warehouse - ensure warehouse coordinates are float
                    warehouse_lat = float(warehouse_location[0])
                    warehouse_lon = float(warehouse_location[1])

                    lat_offset = random.uniform(-0.003, 0.003)
                    lon_offset = random.uniform(-0.003, 0.003)

                    driver_positions[driver.id] = (
                        warehouse_lat + lat_offset,
                        warehouse_lon + lon_offset
                    )

            # Create random disruption
            disruption_type = random.choice(list(DisruptionType))

            # Create disruption in a relevant location (near routes or deliveries)
            if disruption_type == DisruptionType.RECIPIENT_UNAVAILABLE and delivery_points:
                # Place at a random delivery point
                delivery_idx = random.randint(0, len(delivery_points) - 1)
                delivery = delivery_points[delivery_idx]

                # Ensure coordinates are float
                location = (float(delivery.coordinates[0]), float(delivery.coordinates[1]))

                disruption = Disruption(
                    id=i + 1,
                    type=disruption_type,
                    location=location,
                    affected_area_radius=random.uniform(5, 20),
                    duration=random.randint(300, 3600),
                    severity=random.uniform(0.2, 1.0),
                    metadata={"delivery_point_index": delivery_idx}
                )
            else:
                # Random location, prioritizing areas with deliveries
                if delivery_points and random.random() < 0.7:
                    # Near a random delivery
                    delivery = random.choice(delivery_points)
                    base_lat = float(delivery.coordinates[0])
                    base_lon = float(delivery.coordinates[1])
                    base_location = (base_lat, base_lon)
                else:
                    # Random location within bounds
                    lat = random.uniform(float(bounds[0]), float(bounds[1]))
                    lon = random.uniform(float(bounds[2]), float(bounds[3]))
                    base_location = (lat, lon)

                # Add small random offset
                lat_offset = random.uniform(-0.002, 0.002)
                lon_offset = random.uniform(-0.002, 0.002)

                location = (
                    float(base_location[0]) + float(lat_offset),
                    float(base_location[1]) + float(lon_offset)
                )

                # Parameters based on disruption type
                if disruption_type == DisruptionType.TRAFFIC_JAM:
                    radius = random.uniform(50, 300)
                    duration = random.randint(900, 3600)  # 15min to 1hr
                elif disruption_type == DisruptionType.ROAD_CLOSURE:
                    radius = random.uniform(20, 150)
                    duration = random.randint(1800, 7200)  # 30min to 2hrs
                else:
                    radius = random.uniform(10, 50)
                    duration = random.randint(300, 1800)  # 5min to 30min

                disruption = Disruption(
                    id=i + 1,
                    type=disruption_type,
                    location=(float(location[0]), float(location[1])),
                    affected_area_radius=float(radius),
                    duration=duration,
                    severity=random.uniform(0.2, 1.0)
                )

            # Random time of day (6 AM to 8 PM)
            simulation_time = random.randint(6, 20) * 3600 + random.randint(0, 3599)

            driver_routes = {}
            for driver_id, assignments in driver_assignments.items():
                if assignments:
                    route_points = [warehouse_location]
                    for delivery_idx in assignments:
                        if delivery_idx < len(delivery_points):
                            delivery = delivery_points[delivery_idx]
                            route_points.append(delivery.coordinates)
                    route_points.append(warehouse_location)

                    driver_routes[driver_id] = {
                        'points': route_points,
                        'times': [300] * (len(route_points) - 1),  # Estimated travel times
                        'delivery_indices': list(range(1, len(assignments) + 1)),
                        'segments': []
                    }

            # Create the state
            state = DeliverySystemState(
                drivers=drivers,
                deliveries=delivery_points,
                disruptions=[disruption],
                simulation_time=simulation_time,
                graph=graph,
                warehouse_location=warehouse_location,
                driver_positions=driver_positions,
                driver_assignments=driver_assignments,
                driver_routes=driver_routes
            )

            states_disruptions.append((state, disruption))

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Successfully generated {len(states_disruptions)} training samples")
    return states_disruptions


def train_and_save_model(graph, warehouse_location, model_path, num_samples=1000):
    """Train and save a classifier-based resolver model."""
    print("Initializing training components...")

    # Create the training data generator with supercharged resolver
    training_generator = TrainingDataGenerator(graph, warehouse_location)

    # Create the resolver
    resolver = ClassifierBasedResolver(graph, warehouse_location)

    # Generate synthetic training data
    print(f"Generating {num_samples} synthetic training states...")
    states_disruptions = generate_synthetic_states_disruptions(graph, warehouse_location, num_samples)

    # Train the model
    print("Generating high-quality training data using supercharged resolver...")
    print("This may take some time as each sample is evaluated with multiple solution variations")
    start_time = time.time()
    resolver.train(training_generator, states_disruptions)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    # Save the model
    print(f"Saving model to {model_path}")
    resolver.save_model(model_path)

    return resolver


def preprocess_graph(graph):
    """Ensure all node coordinates and edge travel times in the graph are stored as floats."""
    print("Preprocessing graph: Converting coordinates and travel times to float...")

    # Convert node coordinates
    for node_id, data in graph.nodes(data=True):
        if 'y' in data and not isinstance(data['y'], (int, float)):
            try:
                data['y'] = float(data['y'])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert y-coordinate to float for node {node_id}")

        if 'x' in data and not isinstance(data['x'], (int, float)):
            try:
                data['x'] = float(data['x'])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert x-coordinate to float for node {node_id}")

    # Convert edge travel times
    edge_count = 0
    converted_count = 0

    for u, v, k, data in graph.edges(data=True, keys=True):
        edge_count += 1
        if 'travel_time' in data:
            if not isinstance(data['travel_time'], (int, float)):
                try:
                    data['travel_time'] = float(data['travel_time'])
                    converted_count += 1
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert travel_time to float for edge ({u}, {v}, {k})")

    if converted_count > 0:
        print(f"Converted {converted_count} of {edge_count} edge travel times to float")

    return graph


if __name__ == "__main__":
    import sys
    import os
    import networkx as nx

    # Define sensible defaults
    default_graph_file = "resources/data/graphs/kaunas.graphml"
    default_model_path = "models/trained_resolver.joblib"
    default_samples = 5

    # Show usage information regardless
    print(f"Usage: python {sys.argv[0]} <graph_file> <output_model_path> [num_samples]")
    print(f"  graph_file: Path to the graph file (default: {default_graph_file})")
    print(f"  output_model_path: Path where the model will be saved (default: {default_model_path})")
    print(f"  num_samples: Number of training samples to generate (default: {default_samples})")
    print()

    # Use command line arguments if provided, otherwise use defaults
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

    # Create parent directories for model path if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    # Check if graph file exists
    if not os.path.exists(graph_file):
        print(f"Error: Graph file '{graph_file}' not found.")
        print("Please ensure the graph file exists or specify a different path.")
        sys.exit(1)

    # Load graph based on file extension
    print(f"Loading graph from {graph_file}...")
    try:
        file_extension = os.path.splitext(graph_file)[1].lower()
        if file_extension == '.graphml':
            # Load GraphML file
            graph = nx.read_graphml(graph_file)
            graph = preprocess_graph(graph)
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

    # Determine warehouse location (center of graph)
    try:
        # Determine warehouse location (center of graph)
        node_coords = []
        for _, data in graph.nodes(data=True):
            if 'y' in data and 'x' in data:
                try:
                    # Convert coordinates to float explicitly
                    lat = float(data['y'])
                    lon = float(data['x'])
                    node_coords.append((lat, lon))
                except (ValueError, TypeError):
                    # Skip invalid coordinates
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

        print(f"Starting training on {num_samples} samples...")
        # Train the model
        train_and_save_model(graph, warehouse_location, model_path, num_samples)

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
