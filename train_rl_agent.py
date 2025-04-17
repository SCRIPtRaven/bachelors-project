import argparse
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from models.rl.actions import ActionType
from models.rl.agents.dqn_agent import DQNAgent
from models.rl.environment import DeliveryEnvironment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL agent for disruption resolution')
    parser.add_argument('--episodes', type=int, default=250, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--load', type=str, help='Load model from file')
    parser.add_argument('--save', type=str, default='dqn_model.h5', help='Save model to file')
    parser.add_argument('--city', type=str, default='Kaunas, Lithuania', help='City for graph data')
    parser.add_argument('--viz', action='store_true', help='Visualize training')
    return parser.parse_args()


def load_data(city_name):
    """
    Load graph, delivery points, and drivers for the specified city

    Args:
        city_name: Name of the city

    Returns:
        Dictionary with graph, delivery points, and drivers
    """
    from config.paths import get_graph_file_path
    import osmnx as ox
    from models.services.geolocation import GeolocationService

    # Load graph
    graph_path = get_graph_file_path(city_name)
    if not os.path.exists(graph_path):
        print(f"Downloading graph for {city_name}...")
        from models.services.graph import download_and_save_graph
        download_and_save_graph(city_name)

    G = ox.load_graphml(graph_path)

    # Calculate bounding box for delivery points
    nodes = list(G.nodes(data=True))
    if not nodes:
        raise ValueError("Graph has no nodes")

    lats = [data['y'] for _, data in nodes]
    lons = [data['x'] for _, data in nodes]

    bounds = (
        min(lats),
        max(lats),
        min(lons),
        max(lons)
    )

    # Generate delivery points
    num_deliveries = 30
    delivery_points = GeolocationService.generate_delivery_points(bounds, num_deliveries)

    # Convert to list of tuples
    delivery_points_list = []
    for point in delivery_points:
        lat, lon = point.coordinates
        delivery_points_list.append((lat, lon, point.weight, point.volume))

    # Generate drivers
    num_drivers = 5
    drivers = GeolocationService.generate_delivery_drivers(num_drivers)

    # Calculate warehouse location
    center_lat = (bounds[0] + bounds[1]) / 2
    center_lon = (bounds[2] + bounds[3]) / 2

    warehouse_node = ox.nearest_nodes(G, X=center_lon, Y=center_lat)
    warehouse_coords = (G.nodes[warehouse_node]['y'], G.nodes[warehouse_node]['x'])

    return {
        "graph": G,
        "delivery_points": delivery_points_list,
        "drivers": drivers,
        "warehouse_location": warehouse_coords
    }


def make_disruption_generator(delivery_points, driver_capacity=3, fix_affected_drivers=True):
    """
    Create a disruption generator function that ensures disruptions affect drivers

    Args:
        delivery_points: List of delivery points
        driver_capacity: Maximum number of drivers affected
        fix_affected_drivers: Whether to ensure disruptions affect drivers

    Returns:
        Function that generates disruptions
    """
    from models.entities.disruption import Disruption, DisruptionType
    import random

    next_id = 1

    def generate_disruption(current_time):
        nonlocal next_id

        disruption_type = random.choice(list(DisruptionType))

        # Ensure we use a delivery point that's likely on a route
        point_idx = random.randint(0, min(10, len(delivery_points) - 1))
        location = delivery_points[point_idx][0:2]

        radius = random.uniform(100, 500)  # 100m to 500m
        duration = random.randint(900, 3600)  # 15min to 60min
        severity = random.uniform(0.3, 1.0)

        # Create a set of driver IDs that are affected
        affected_driver_ids = set()
        if fix_affected_drivers:
            # Ensure at least one driver is affected
            for i in range(1, driver_capacity + 1):
                if random.random() < 0.6:  # 60% chance to affect each driver
                    affected_driver_ids.add(i)

            # Always ensure at least one driver is affected
            if not affected_driver_ids:
                affected_driver_ids.add(random.randint(1, driver_capacity))

        disruption = Disruption(
            id=next_id,
            type=disruption_type,
            location=location,
            affected_area_radius=radius,
            duration=duration,
            severity=severity,
            affected_driver_ids=affected_driver_ids
        )

        next_id += 1
        return disruption

    return generate_disruption


def haversine_distance(point1, point2):
    """Calculate distance between two points in meters using Haversine formula"""
    import math

    lat1, lon1 = point1
    lat2, lon2 = point2

    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth radius in meters

    return c * r


def get_rule_based_action(env, state):
    """
    Get the action that a rule-based resolver would take in the current state.
    This is used for imitation learning during early training.

    Args:
        env: The environment instance
        state: The current state vector

    Returns:
        int: Action ID or None if no action would be taken
    """
    try:
        # Create a temporary rule-based resolver
        from models.rl.rule_based_resolver import RuleBasedResolver

        resolver = RuleBasedResolver(env.graph, env.warehouse_location)

        # If there are no disruptions, no action is needed
        if not env.active_disruptions:
            return 0

        disruption = env.active_disruptions[0] if env.active_disruptions else None

        if not disruption:
            return 0

        # Check if the resolver would take action
        if not resolver.should_recalculate(env.current_state, disruption):
            return 0

        # Get rule-based actions
        actions = resolver.resolve_disruptions(env.current_state, [disruption])

        if not actions:
            return 0

        # Find the corresponding action ID
        for action in actions:
            if action.action_type == ActionType.REROUTE and hasattr(action, 'driver_id'):
                driver_id = action.driver_id

                # Convert to action ID based on environment's encoding
                strategies_per_driver = 5
                # Base action for the first strategy of this driver
                base_action = (driver_id - 1) * strategies_per_driver + 1

                # Return the middle strategy as default (index 2)
                return base_action + 2

        return 0
    except Exception as e:
        print(f"Error getting rule-based action: {e}")
        return None


def main():
    """Enhanced training process with proper progress reporting and message suppression"""
    from tqdm.auto import tqdm
    import os
    import tensorflow as tf
    import logging

    # Silence TensorFlow and logging output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    # Redirect disruptive log messages
    logging.basicConfig(filename='training_details.log', level=logging.INFO)
    logger = logging.getLogger('training')

    args = parse_args()

    print(f"Loading data for {args.city}...")
    data = load_data(args.city)

    # Create environment with improved configuration
    env_config = {
        "graph": data["graph"],
        "warehouse_location": data["warehouse_location"],
        "delivery_points": data["delivery_points"],
        "drivers": data["drivers"],
        "disruption_generator": make_disruption_generator(data["delivery_points"], fix_affected_drivers=True),
        "simulation_steps": args.steps
    }

    env = DeliveryEnvironment(env_config)

    # Setup agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    if args.load:
        print(f"Loading model from {args.load}...")
        agent.load(args.load)
        agent.epsilon = 0.1

    # Metrics tracking
    episodes = args.episodes
    batch_size = args.batch_size
    rewards = []
    avg_rewards = []
    success_rates = []
    losses = []

    # Curriculum stages
    curriculum_stages = [
        {"name": "Basic", "episodes": episodes // 4, "disruption_types": ["traffic_jam"], "complexity": 0.3},
        {"name": "Intermediate", "episodes": episodes // 4, "disruption_types": ["traffic_jam", "road_closure"],
         "complexity": 0.6},
        {"name": "Advanced", "episodes": episodes // 2, "disruption_types": ["traffic_jam", "road_closure"],
         "complexity": 1.0}
    ]

    episode_counter = 0

    # Create test scenarios for evaluation
    print("Creating test scenarios...")
    test_scenarios = create_test_scenarios(data, env)

    # Redirect disruptive output
    original_print = print

    def filtered_print(*args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        if any(pattern in msg for pattern in [
            "Processing traffic_jam",
            "Processing road_closure",
            "Processing recipient_unavailable"
        ]):
            logger.info(msg)  # Write to log file instead
        else:
            original_print(*args, **kwargs)

    # Temporarily replace print function
    import builtins
    builtins.print = filtered_print

    try:
        # Main progress bar
        with tqdm(total=episodes, desc="Total Progress", ncols=100) as main_pbar:

            # Train through curriculum stages
            for stage_idx, stage in enumerate(curriculum_stages):
                print(f"\n{'=' * 50}")
                print(f"Starting Stage {stage_idx + 1}/{len(curriculum_stages)}: {stage['name']}")
                print(f"Disruption types: {stage['disruption_types']}, Complexity: {stage['complexity']}")
                print(f"{'=' * 50}")

                # Configure environment for this stage
                env.disruption_types = stage["disruption_types"]
                env.complexity_factor = stage["complexity"]
                stage_episodes = stage["episodes"]

                # Training progress bar for this stage
                with tqdm(total=stage_episodes, desc=f"Stage: {stage['name']}",
                          position=0, leave=False, ncols=100) as stage_pbar:

                    # Training loop
                    for e in range(stage_episodes):
                        with tf.keras.utils.custom_object_scope({'tf': tf}):
                            state = env.reset()
                            total_reward = 0
                            actions_taken = 0
                            successful_actions = 0

                            # Episode steps
                            for step in range(args.steps):
                                # Determine action
                                if np.random.random() < 0.3 and episode_counter < episodes // 3:
                                    rb_action = get_rule_based_action(env, state)
                                    action = rb_action if rb_action is not None else agent.act(state)
                                else:
                                    action = agent.act(state)

                                # Take step
                                next_state, reward, done, info = env.step(action)

                                if action > 0:
                                    actions_taken += 1
                                    if info.get('action_success', False):
                                        successful_actions += 1

                                # Store experience
                                agent.remember(state, action, reward, next_state, done)
                                state = next_state
                                total_reward += reward

                                if done:
                                    break

                            # Train on experiences
                            if len(agent.memory) >= batch_size:
                                agent.replay(batch_size)

                            # Update target network periodically
                            if e % 10 == 0:
                                agent.update_target_model()

                            # Calculate metrics
                            success_rate = successful_actions / max(1, actions_taken) * 100
                            rewards.append(total_reward)
                            avg_rewards.append(np.mean(rewards[-100:]))
                            success_rates.append(success_rate)

                            # Update progress displays
                            main_pbar.update(1)
                            stage_pbar.update(1)

                            # Update metrics in progress bar
                            stage_pbar.set_postfix({
                                'reward': f"{total_reward:.1f}",
                                'avg': f"{avg_rewards[-1]:.1f}",
                                'ε': f"{agent.epsilon:.3f}",
                                'success': f"{success_rate:.0f}%"
                            })

                            # Periodically log detailed metrics
                            if e % 10 == 0:
                                logger.info(f"Episode {episode_counter}: reward={total_reward:.2f}, "
                                            f"avg_reward={avg_rewards[-1]:.2f}, "
                                            f"success_rate={success_rate:.2f}%, epsilon={agent.epsilon:.4f}")

                                if hasattr(env, 'active_disruptions'):
                                    for i, d in enumerate(env.active_disruptions):
                                        affect_str = "affects drivers" if d.affected_driver_ids else "NO DRIVERS AFFECTED"
                                        logger.info(f"  Disruption {i}: {d.type.value} at {d.location} - {affect_str}")

                                if 'final_stats' in info:
                                    stats = info['final_stats']
                                    logger.info(f"  Deliveries: {stats['successful_deliveries']} completed, "
                                                f"{stats['failed_deliveries']} failed, "
                                                f"Time saved: {stats['time_saved'] / 60:.2f} min")

                                logger.info(f"  Solution feasibility check: {env.check_solution_feasibility()}")

                            # Periodically evaluate
                            if episode_counter % 50 == 0 and episode_counter > 0:
                                print(f"\n{'-' * 50}")
                                print(f"EVALUATION AT EPISODE {episode_counter}/{episodes}")

                                # Add a dedicated progress bar for evaluation
                                with tqdm(total=len(test_scenarios), desc="Evaluating",
                                          leave=False, ncols=80) as eval_pbar:
                                    eval_results = evaluate_agent(agent, test_scenarios, env,
                                                                  progress_bar=eval_pbar)

                                print(f"Avg. Reward: {eval_results['avg_reward']:.2f}")
                                print(f"Avg. Success Rate: {eval_results['avg_success_rate']:.2f}%")
                                print(f"Avg. Time Saved: {eval_results['avg_time_saved']:.2f} min")

                                affected_rate = eval_results.get('disruptions_affecting_drivers_pct', 0)
                                print(f"Disruptions affecting drivers: {affected_rate:.1f}%")
                                print(f"{'-' * 50}")

                            # Save model checkpoints
                            if e % 100 == 0 and e > 0:
                                save_path = f"{args.save.split('.')[0]}_{episode_counter}.h5"
                                agent.save(save_path)
                                print(f"\nModel checkpoint saved to {save_path}")

                            episode_counter += 1

                print(f"Completed stage: {stage['name']}")

        # Save final model
        agent.save(args.save)
        print(f"\nTraining complete! Final model saved to {args.save}")

        # Plot results
        plot_training_results(rewards, avg_rewards, success_rates, losses)

    finally:
        # Restore original print function
        builtins.print = original_print
        print(f"Detailed logs saved to training_details.log")


def create_test_scenarios(data, env):
    """Create standardized test scenarios for consistent evaluation"""
    scenarios = []

    for severity in [0.4, 0.7, 0.9]:
        scenarios.append({
            "name": f"Traffic Jam (Severity {severity})",
            "type": "traffic_jam",
            "severity": severity,
            "location": select_strategic_location(data, "high_traffic"),
            "radius": 200,
            "duration": 1800
        })

    for radius in [100, 200, 300]:
        scenarios.append({
            "name": f"Road Closure (Radius {radius}m)",
            "type": "road_closure",
            "severity": 1.0,
            "location": select_strategic_location(data, "critical_junction"),
            "radius": radius,
            "duration": 3600
        })

    scenarios.append({
        "name": "Multiple Disruptions",
        "disruptions": [
            {
                "type": "traffic_jam",
                "severity": 0.6,
                "location": select_strategic_location(data, "high_traffic"),
                "radius": 150,
                "duration": 2400
            },
            {
                "type": "road_closure",
                "severity": 1.0,
                "location": select_strategic_location(data, "critical_junction", exclude=scenarios[0]["location"]),
                "radius": 200,
                "duration": 3600
            }
        ]
    })

    return scenarios


def select_strategic_location(data, location_type, exclude=None):
    """Select a strategic location from the map based on the type needed"""
    graph = data["graph"]

    if location_type == "high_traffic":
        node_degrees = dict(graph.degree())
        high_degree_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]

        selected_node = random.choice(high_degree_nodes)[0]

    elif location_type == "critical_junction":
        betweenness = nx.betweenness_centrality(graph, k=100, endpoints=False)
        critical_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]

        selected_node = random.choice(critical_nodes)[0]

    else:
        selected_node = random.choice(list(graph.nodes()))

    lat = graph.nodes[selected_node]['y']
    lon = graph.nodes[selected_node]['x']
    location = (lat, lon)

    if exclude is not None:
        attempts = 0
        while attempts < 10 and haversine_distance(location, exclude) < 500:
            if location_type == "high_traffic":
                selected_node = random.choice(high_degree_nodes)[0]
            elif location_type == "critical_junction":
                selected_node = random.choice(critical_nodes)[0]
            else:
                selected_node = random.choice(list(graph.nodes()))

            lat = graph.nodes[selected_node]['y']
            lon = graph.nodes[selected_node]['x']
            location = (lat, lon)
            attempts += 1

    return location


def evaluate_agent(agent, test_scenarios, env, progress_bar=None):
    """Evaluate agent performance on test scenarios with better disruption metrics"""
    results = {
        'rewards': [],
        'success_rates': [],
        'time_saved': [],
        'disruptions_affecting_drivers': 0,
        'total_disruptions': 0
    }

    for i, scenario in enumerate(test_scenarios):
        # Start with base config and customize for this scenario
        env_config = {
            "graph": env.graph,
            "warehouse_location": env.warehouse_location,
            "delivery_points": env.delivery_points,
            "drivers": env.drivers,
            "disruption_generator": None,
            "simulation_steps": 100,
            "step_size": 300
        }

        if "disruptions" in scenario:
            env_config["predefined_disruptions"] = scenario["disruptions"]
            results['total_disruptions'] += len(scenario["disruptions"])
        else:
            env_config["predefined_disruptions"] = [{
                "type": scenario["type"],
                "severity": scenario["severity"],
                "location": scenario["location"],
                "radius": scenario["radius"],
                "duration": scenario["duration"]
            }]
            results['total_disruptions'] += 1

        try:
            test_env = DeliveryEnvironment(env_config)

            # Run the scenario
            state = test_env.reset()
            total_reward = 0
            actions_taken = 0
            successful_actions = 0

            # Count disruptions affecting drivers
            for disruption in test_env.active_disruptions:
                if hasattr(disruption, 'affected_driver_ids') and disruption.affected_driver_ids:
                    results['disruptions_affecting_drivers'] += 1

            done = False
            while not done:
                action = agent.act(state, training=False)
                next_state, reward, done, info = test_env.step(action)

                if action > 0:
                    actions_taken += 1
                    if info.get('action_success', False):
                        successful_actions += 1

                state = next_state
                total_reward += reward

            # Collect results
            success_rate = successful_actions / max(1, actions_taken) * 100
            time_saved = 0

            if 'final_stats' in info:
                time_saved = info['final_stats']['time_saved'] / 60  # Convert to minutes

            results['rewards'].append(total_reward)
            results['success_rates'].append(success_rate)
            results['time_saved'].append(time_saved)

            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f"{total_reward:.1f}",
                    'success': f"{success_rate:.0f}%"
                })

        except Exception as e:
            print(f"Error evaluating scenario '{scenario.get('name', 'unnamed')}': {e}")
            # Continue with other scenarios

    # If no scenarios were successful, return zeros
    if not results['rewards']:
        return {
            'avg_reward': 0.0,
            'avg_success_rate': 0.0,
            'avg_time_saved': 0.0,
            'disruptions_affecting_drivers_pct': 0.0
        }

    # Calculate percentage of disruptions affecting drivers
    disruption_affect_pct = 0
    if results['total_disruptions'] > 0:
        disruption_affect_pct = (results['disruptions_affecting_drivers'] /
                                 results['total_disruptions']) * 100

    # Calculate averages
    return {
        'avg_reward': np.mean(results['rewards']),
        'avg_success_rate': np.mean(results['success_rates']),
        'avg_time_saved': np.mean(results['time_saved']),
        'disruptions_affecting_drivers_pct': disruption_affect_pct
    }


def plot_training_results(rewards, avg_rewards, success_rates, losses):
    """Create more detailed visualization of training progress"""
    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.ylabel('Reward')

    plt.subplot(4, 1, 2)
    plt.plot(avg_rewards)
    plt.title('Average Reward (last 100 episodes)')
    plt.ylabel('Avg Reward')

    plt.subplot(4, 1, 3)
    plt.plot(success_rates)
    plt.title('Action Success Rate (%)')
    plt.ylabel('Success Rate')

    plt.subplot(4, 1, 4)
    if losses:
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.savefig('training_results.pdf')
    plt.show()


if __name__ == "__main__":
    main()
