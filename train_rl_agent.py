import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from models.rl.agents.dqn_agent import DQNAgent
from models.rl.environment import DeliveryEnvironment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL agent for disruption resolution')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
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


def make_disruption_generator(delivery_points, driver_capacity=3):
    """
    Create a disruption generator function

    Args:
        delivery_points: List of delivery points
        driver_capacity: Maximum number of drivers affected

    Returns:
        Function that generates disruptions
    """
    from models.entities.disruption import Disruption, DisruptionType
    import random

    next_id = 1

    def generate_disruption(current_time):
        nonlocal next_id

        disruption_type = random.choice(list(DisruptionType))

        point_idx = random.randint(0, len(delivery_points) - 1)
        location = delivery_points[point_idx][0:2]

        radius = random.uniform(100, 500)  # 100m to 500m
        duration = random.randint(900, 3600)  # 15min to 60min
        severity = random.uniform(0.3, 1.0)

        affected_driver_ids = set()

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


def main():
    """Main training function"""
    args = parse_args()

    print(f"Loading data for {args.city}...")
    data = load_data(args.city)

    disruption_generator = make_disruption_generator(data["delivery_points"])

    env_config = {
        "graph": data["graph"],
        "warehouse_location": data["warehouse_location"],
        "delivery_points": data["delivery_points"],
        "drivers": data["drivers"],
        "disruption_generator": disruption_generator,
        "simulation_steps": args.steps
    }

    env = DeliveryEnvironment(env_config)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    if args.load:
        print(f"Loading model from {args.load}...")
        agent.load(args.load)
        agent.epsilon = 0.1

    episodes = args.episodes
    batch_size = args.batch_size

    rewards = []
    avg_rewards = []
    success_rates = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        actions_taken = 0
        successful_actions = 0

        for step in range(args.steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            if action > 0:
                actions_taken += 1
                if info.get('action_success', False):
                    successful_actions += 1

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if args.viz:
                env.render()

            if done:
                break

        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

        if e % 10 == 0:
            agent.update_target_model()

        success_rate = successful_actions / max(1, actions_taken) * 100

        rewards.append(total_reward)
        avg_rewards.append(np.mean(rewards[-100:]))
        success_rates.append(success_rate)

        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Avg Reward: {avg_rewards[-1]:.2f}")
            print(f"  Success Rate: {success_rate:.2f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}")

            if 'final_stats' in info:
                stats = info['final_stats']
                print(f"  Deliveries: {stats['successful_deliveries']} completed, {stats['failed_deliveries']} failed")
                time_saved = stats['time_saved'] / 60
                print(f"  Time Saved: {time_saved:.2f} minutes")

        if e % 100 == 0 and e > 0:
            save_path = f"{args.save.split('.')[0]}_{e}.h5"
            agent.save(save_path)
            print(f"Model saved to {save_path}")

    agent.save(args.save)
    print(f"Final model saved to {args.save}")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.ylabel('Reward')

    plt.subplot(3, 1, 2)
    plt.plot(avg_rewards)
    plt.title('Average Reward (last 100 episodes)')
    plt.ylabel('Avg Reward')

    plt.subplot(3, 1, 3)
    plt.plot(success_rates)
    plt.title('Action Success Rate (%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


if __name__ == "__main__":
    main()
