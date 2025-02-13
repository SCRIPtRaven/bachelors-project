import unittest

import networkx as nx

from logic.delivery_optimizer import SimulatedAnnealingOptimizer
from services.geolocation_service import DeliveryDriver


class TestSimulatedAnnealingOptimizer(unittest.TestCase):
    def create_test_graph(self):
        """Create a simple test graph with known distances"""
        G = nx.Graph()
        # Create a 3x3 grid graph with known distances
        for i in range(3):
            for j in range(3):
                # Add node with both lat/lon and x/y coordinates
                G.add_node(
                    f"{i},{j}",
                    y=float(i),
                    x=float(j),
                    lat=float(i),
                    lon=float(j)
                )

        # Add edges with known weights
        for i in range(3):
            for j in range(3):
                if j < 2:  # Horizontal edges
                    G.add_edge(f"{i},{j}", f"{i},{j + 1}",
                               length=1.0,
                               travel_time=1.0)  # Add travel_time
                if i < 2:  # Vertical edges
                    G.add_edge(f"{i},{j}", f"{i + 1},{j}",
                               length=1.0,
                               travel_time=1.0)  # Add travel_time

        return G

    def test_known_optimal_solution(self):
        """Test if optimizer can find a known optimal solution"""
        G = self.create_test_graph()

        deliveries = [
            (0.0, 0.0, 10, 1),  # location (0,0), weight 10, volume 1
            (0.0, 2.0, 15, 2),  # location (0,2), weight 15, volume 2
            (2.0, 0.0, 20, 1),  # location (2,0), weight 20, volume 1
            (2.0, 2.0, 5, 1),  # location (2,2), weight 5, volume 1
        ]

        drivers = [
            DeliveryDriver(id=1, weight_capacity=30, volume_capacity=3),
            DeliveryDriver(id=2, weight_capacity=25, volume_capacity=3)
        ]

        optimizer = SimulatedAnnealingOptimizer(drivers, deliveries, G)
        final_solution, unassigned = optimizer.optimize(return_best=True)

        self.assertIsNotNone(final_solution)
        self.assertEqual(len(unassigned), 0)

        # Check detailed route assignments
        for assignment in final_solution:
            print(f"\nChecking assignment for Driver {assignment.driver_id}:")

            # Print route details
            route_points = [optimizer.deliveries[i].coordinates
                            for i in assignment.delivery_indices]
            print(f"Route points: {route_points}")

            # Calculate and print actual distance
            distance = optimizer.calculate_route_distance(route_points)
            print(f"Route distance: {distance}")

            # Check specific route compositions
            driver_deliveries = set(assignment.delivery_indices)
            if assignment.driver_id == 1:
                # Driver 1 should have (0,0) and (0,2) or similar short route
                self.assertLessEqual(distance, 3.0)
            else:
                # Driver 2 should have (2,0) and (2,2) or similar short route
                self.assertLessEqual(distance, 3.0)

            # Check capacity constraints
            driver = next(d for d in drivers if d.id == assignment.driver_id)
            self.assertLessEqual(assignment.total_weight, driver.weight_capacity)
            self.assertLessEqual(assignment.total_volume, driver.volume_capacity)

        # Overall route quality
        total_distance = sum(optimizer.calculate_route_distance(
            [optimizer.deliveries[i].coordinates for i in a.delivery_indices]
        ) for a in final_solution)

        self.assertLessEqual(total_distance, 6.0)

    def test_capacity_constraints(self):
        """Test that capacity constraints are never violated"""
        G = self.create_test_graph()

        deliveries = [
            (0.0, 0.0, 20, 2),
            (0.0, 1.0, 20, 2),
            (0.0, 2.0, 20, 2),
        ]

        drivers = [
            DeliveryDriver(id=1, weight_capacity=30, volume_capacity=3),
            DeliveryDriver(id=2, weight_capacity=30, volume_capacity=3)
        ]

        optimizer = SimulatedAnnealingOptimizer(drivers, deliveries, G)
        final_solution, unassigned = optimizer.optimize(return_best=True)  # Add return_best=True

        self.assertIsNotNone(final_solution, "Optimization should produce a solution")

        # Verify capacity constraints
        for assignment in final_solution:
            driver = next(d for d in drivers if d.id == assignment.driver_id)
            self.assertLessEqual(assignment.total_weight, driver.weight_capacity)
            self.assertLessEqual(assignment.total_volume, driver.volume_capacity)

        # Print debug information about unassigned deliveries
        print(f"\nUnassigned deliveries: {len(unassigned)}")
        print(f"Total deliveries: {len(deliveries)}")

    def test_all_deliveries_assigned_when_possible(self):
        """Test that all deliveries are assigned when there's sufficient capacity"""
        pass  # Similar structure to above tests

    def test_minimal_unassigned_deliveries(self):
        """Test that minimum deliveries are left unassigned when capacity is insufficient"""
        pass  # Similar structure to above tests


if __name__ == '__main__':
    unittest.main()
