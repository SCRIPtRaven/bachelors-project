import copy
import time

from PyQt5 import QtCore

from core.viewmodel import ViewModel
from logic.delivery_optimizer import SimulatedAnnealingOptimizer
from logic.greedy_delivery_optimizer import GreedyOptimizer
from utils.route_color_manager import RouteColorManager


class VisualizationViewModel(ViewModel):
    """
    ViewModel for handling optimization and visualization of delivery routes.
    This is the bridge between optimization algorithms and map display.
    """
    # Signals
    visualization_updated = QtCore.pyqtSignal(object)  # Emitted with visualization data
    optimization_started = QtCore.pyqtSignal()  # Emitted when optimization starts
    optimization_progress = QtCore.pyqtSignal(float, object)  # Emitted with progress updates
    optimization_completed = QtCore.pyqtSignal(object, object)  # Emitted when optimization completes
    simulation_started = QtCore.pyqtSignal(object)  # Emitted when simulation starts

    def __init__(self):
        super().__init__()
        self._optimization_in_progress = False
        self._current_progress = 0.0
        self._current_solution = None
        self._unassigned_deliveries = None
        self._simulated_annealing_solution = None
        self._simulated_annealing_unassigned = None
        self._greedy_solution = None
        self._greedy_unassigned = None
        self._selected_algorithm = "Simulated Annealing"
        self._selected_driver_id = None
        self._warehouse_location = None
        self._graph = None
        self._route_color_manager = RouteColorManager()
        self._start_time = None

    @property
    def optimization_in_progress(self):
        return self._optimization_in_progress

    @property
    def current_progress(self):
        return self._current_progress

    @property
    def current_solution(self):
        return self._current_solution

    @property
    def unassigned_deliveries(self):
        return self._unassigned_deliveries

    @property
    def selected_algorithm(self):
        return self._selected_algorithm

    @selected_algorithm.setter
    def selected_algorithm(self, algorithm):
        if self._selected_algorithm != algorithm:
            self._selected_algorithm = algorithm
            self.property_changed.emit('_selected_algorithm')

            # Switch between different solution types
            if algorithm == "Simulated Annealing" and self._simulated_annealing_solution:
                self._current_solution = self._simulated_annealing_solution
                self._unassigned_deliveries = self._simulated_annealing_unassigned
                self._update_visualization()
            elif algorithm == "Greedy" and self._greedy_solution:
                self._current_solution = self._greedy_solution
                self._unassigned_deliveries = self._greedy_unassigned
                self._update_visualization()

    @property
    def selected_driver_id(self):
        return self._selected_driver_id

    @selected_driver_id.setter
    def selected_driver_id(self, driver_id):
        if self._selected_driver_id != driver_id:
            self._selected_driver_id = driver_id
            self.property_changed.emit('_selected_driver_id')
            self._update_visualization()

    def set_warehouse_location(self, location):
        """
        Set the warehouse location for route calculations.

        Args:
            location: Tuple of (latitude, longitude)
        """
        self._warehouse_location = location

    def set_graph(self, graph):
        """
        Set the graph for route calculations.

        Args:
            graph: NetworkX graph representing the road network
        """
        self._graph = graph

    def start_optimization(self, delivery_points, drivers, graph):
        """
        Start the optimization process.

        Args:
            delivery_points: The delivery points to optimize
            drivers: The drivers to assign deliveries to
            graph: The graph representing the road network
        """
        if self._optimization_in_progress or not graph:
            return

        # Store references for optimization
        self._graph = graph

        # Set state to optimizing
        self.set_property('_optimization_in_progress', True)
        self.set_property('_current_progress', 0.0)
        self._start_time = time.time()

        # Signal that optimization has started
        self.optimization_started.emit()

        # Run the optimization in a background thread
        def run_optimization():
            try:
                # Create and configure the optimizers
                sa_optimizer = SimulatedAnnealingOptimizer(
                    drivers, delivery_points, graph, None
                )

                # Connect progress updates if the optimizer supports them
                if hasattr(sa_optimizer, 'update_visualization'):
                    sa_optimizer.update_visualization.connect(self._on_optimization_progress)

                # Connect the finished signal
                sa_optimizer.finished.connect(self._on_sa_optimization_completed)

                # Run the simulated annealing optimization
                self._simulated_annealing_solution, self._simulated_annealing_unassigned = sa_optimizer.optimize()

                # Also run a greedy optimization for comparison
                greedy_optimizer = GreedyOptimizer(
                    drivers, delivery_points, graph, None
                )
                greedy_optimizer.finished.connect(self._on_greedy_optimization_completed)
                self._greedy_solution, self._greedy_unassigned = greedy_optimizer.optimize()

                return {
                    'success': True,
                    'sa_solution': self._simulated_annealing_solution,
                    'sa_unassigned': self._simulated_annealing_unassigned,
                    'greedy_solution': self._greedy_solution,
                    'greedy_unassigned': self._greedy_unassigned
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(e)
                }

        # Run the optimization asynchronously
        self.run_async(
            run_optimization,
            on_result=self._on_optimization_complete,
            on_error=self._on_optimization_error
        )

    def _on_sa_optimization_completed(self, solution, unassigned):
        """Handle completion of simulated annealing optimization."""
        self._simulated_annealing_solution = solution
        self._simulated_annealing_unassigned = unassigned

        # If greedy optimization is also complete or not needed, finalize
        if self._greedy_solution is not None:
            self._finalize_optimization()

    def _on_greedy_optimization_completed(self, solution, unassigned):
        """Handle completion of greedy optimization."""
        self._greedy_solution = solution
        self._greedy_unassigned = unassigned

        # If SA optimization is also complete, finalize
        if self._simulated_annealing_solution is not None:
            self._finalize_optimization()

    def _finalize_optimization(self):
        """Finalize the optimization process."""
        # Set the current solution based on the selected algorithm
        if self._selected_algorithm == "Simulated Annealing":
            self._current_solution = self._simulated_annealing_solution
            self._unassigned_deliveries = self._simulated_annealing_unassigned
        else:
            self._current_solution = self._greedy_solution
            self._unassigned_deliveries = self._greedy_unassigned

        # Update the visualization
        self._update_visualization()

        # Signal completion
        self.optimization_completed.emit(self._current_solution, self._unassigned_deliveries)

        # Reset state
        self.set_property('_optimization_in_progress', False)
        self.set_property('_current_progress', 1.0)

    def _on_optimization_progress(self, solution, unassigned):
        """
        Handle progress updates from the optimizer.

        Args:
            solution: The current best solution
            unassigned: Set of unassigned delivery indices
        """
        # Store the current progress solution
        temp_solution = copy.deepcopy(solution)
        temp_unassigned = copy.deepcopy(unassigned)

        # Calculate a progress value (this is an approximation)
        if hasattr(self, '_last_progress_time'):
            time_diff = time.time() - self._last_progress_time
            if time_diff < 0.5:  # Limit update frequency
                return

        self._last_progress_time = time.time()

        # Update progress based on time elapsed
        elapsed = time.time() - self._start_time
        progress = min(0.9, elapsed / 60.0)  # Assume max 60 seconds
        self.set_property('_current_progress', progress)

        # Update visualization with progress
        self._current_solution = temp_solution
        self._unassigned_deliveries = temp_unassigned
        self._update_visualization()

        # Signal progress
        self.optimization_progress.emit(progress, (temp_solution, temp_unassigned))

    def _on_optimization_complete(self, result):
        """
        Handle completion of the optimization process.

        Args:
            result: Dictionary containing optimization results
        """
        if result['success']:
            self._simulated_annealing_solution = result['sa_solution']
            self._simulated_annealing_unassigned = result['sa_unassigned']
            self._greedy_solution = result['greedy_solution']
            self._greedy_unassigned = result['greedy_unassigned']

            self._finalize_optimization()
        else:
            # Handle error
            self.set_property('_optimization_in_progress', False)
            print(f"Optimization error: {result.get('error', 'Unknown error')}")

    def _on_optimization_error(self, error):
        """Handle optimization errors."""
        self.set_property('_optimization_in_progress', False)
        print(f"Optimization error: {str(error)}")

        import traceback
        traceback.print_exc()

    def _update_visualization(self):
        """Update the visualization based on the current solution and settings."""
        if not self._current_solution:
            return

        try:
            # Prepare visualization data
            visualization_data = self._prepare_visualization_data()

            # Emit the visualization data for the view to consume
            self.visualization_updated.emit(visualization_data)

        except Exception as e:
            print(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _prepare_visualization_data(self):
        """
        Prepare visualization data for the map display.

        Returns:
            Dictionary with route and delivery point data
        """
        # This will contain all data needed for visualization
        data = {
            'delivery_points': [],
            'routes': [],
            'warehouse': self._warehouse_location
        }

        if not self._current_solution or not self._graph:
            return data

        try:
            # Process delivery points
            delivery_driver_map = {}
            driver_style_map = {}
            total_drivers = len([d for d in self._current_solution if d.driver_id])

            for assignment in self._current_solution:
                driver_id = assignment.driver_id
                driver_style = self._route_color_manager.get_route_style(driver_id - 1, total_drivers)

                # Apply selection highlighting if needed
                if self._selected_driver_id is not None:
                    if driver_id == self._selected_driver_id:
                        driver_style['opacity'] = 1.0
                    else:
                        driver_style = driver_style.copy()
                        driver_style['opacity'] = 0.15

                driver_style_map[driver_id] = driver_style

                for delivery_idx in assignment.delivery_indices:
                    if delivery_idx not in self._unassigned_deliveries:
                        delivery_driver_map[delivery_idx] = driver_id

            # Process all delivery points
            for idx, point in enumerate(self._delivery_points):
                lat, lon, weight, volume = point

                point_opacity = 0.7

                if idx in self._unassigned_deliveries:
                    color = "black"
                    popup = f"Delivery Point {idx + 1}<br>UNASSIGNED<br>Weight: {weight}kg<br>Volume: {volume}m³"
                elif idx in delivery_driver_map:
                    driver_id = delivery_driver_map[idx]
                    color = driver_style_map[driver_id]['color']

                    if self._selected_driver_id is not None:
                        if driver_id != self._selected_driver_id:
                            point_opacity = 0.15

                    popup = f"Delivery Point {idx + 1}<br>Assigned to Driver {driver_id}<br>Weight: {weight}kg<br>Volume: {volume}m³"
                else:
                    color = "gray"
                    popup = f"Delivery Point {idx + 1}<br>Status: Unknown<br>Weight: {weight}kg<br>Volume: {volume}m³"

                data['delivery_points'].append({
                    "lat": lat,
                    "lng": lon,
                    "weight": weight,
                    "volume": volume,
                    "index": idx,
                    "color": color,
                    "opacity": point_opacity,
                    "popup": popup
                })

            # Process routes
            for i, assignment in enumerate(self._current_solution):
                if not assignment.delivery_indices:
                    continue

                driver_id = assignment.driver_id
                route_style = driver_style_map[driver_id]

                # Calculate route points
                route_points = self._calculate_route_points(assignment)

                if route_points:
                    popup = f"Driver {driver_id} Route<br>"
                    popup += f"Deliveries: {len(assignment.delivery_indices)}<br>"
                    popup += f"Weight: {assignment.total_weight:.1f}kg<br>"
                    popup += f"Volume: {assignment.total_volume:.3f}m³"

                    data['routes'].append({
                        "id": i,
                        "driver_id": driver_id,
                        "path": route_points,
                        "style": route_style,
                        "popup": popup
                    })

            return data

        except Exception as e:
            print(f"Error preparing visualization data: {str(e)}")
            import traceback
            traceback.print_exc()
            return data

    def _calculate_route_points(self, assignment):
        """
        Calculate the route points for a driver assignment.

        Args:
            assignment: The DeliveryAssignment to calculate route for

        Returns:
            List of points (lat, lon) for the route
        """
        if not self._warehouse_location or not self._graph:
            return []

        try:
            import networkx as nx
            import osmnx as ox

            # Filter out unassigned deliveries
            delivery_indices = [idx for idx in assignment.delivery_indices
                                if idx not in self._unassigned_deliveries]

            if not delivery_indices:
                return []

            # Get delivery points
            delivery_points = []
            for idx in delivery_indices:
                lat, lon, _, _ = self._delivery_points[idx]
                delivery_points.append((lat, lon))

            # Create a route starting and ending at the warehouse
            waypoints = [self._warehouse_location] + delivery_points + [self._warehouse_location]

            # Calculate route using shortest paths
            route_points = []

            for i in range(len(waypoints) - 1):
                start_waypoint = waypoints[i]
                end_waypoint = waypoints[i + 1]

                try:
                    start_node = ox.nearest_nodes(self._graph, X=start_waypoint[1], Y=start_waypoint[0])
                    end_node = ox.nearest_nodes(self._graph, X=end_waypoint[1], Y=end_waypoint[0])

                    # Find shortest path
                    path = nx.shortest_path(self._graph, start_node, end_node, weight='length')

                    # Extract coordinates for each node in the path
                    for node in path:
                        lat = self._graph.nodes[node]['y']
                        lon = self._graph.nodes[node]['x']
                        route_points.append((lat, lon))

                except Exception as e:
                    print(f"Error calculating path segment: {str(e)}")
                    # Fall back to direct line if path finding fails
                    route_points.extend([start_waypoint, end_waypoint])

            return route_points

        except Exception as e:
            print(f"Error calculating route: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def start_simulation(self):
        """Start a simulation of the current routes."""
        if not self._current_solution:
            return

        # Prepare simulation data
        simulation_data = self._prepare_simulation_data()

        # Signal that simulation has started with the data
        self.simulation_started.emit(simulation_data)

    def _prepare_simulation_data(self):
        """
        Prepare data for route simulation.

        Returns:
            List of route data for simulation
        """
        if not self._current_solution or not self._warehouse_location or not self._graph:
            return []

        try:
            import networkx as nx
            import osmnx as ox

            simulation_data = []
            total_drivers = len([d for d in self._current_solution if d.driver_id])

            for assignment in self._current_solution:
                if not assignment.delivery_indices:
                    continue

                # Calculate routes with travel times
                route_points, travel_times = self._calculate_route_with_times(assignment)

                if not route_points or len(route_points) < 2:
                    continue

                # Find indices where deliveries occur
                delivery_indices = []
                delivery_waypoints = []

                for idx in assignment.delivery_indices:
                    if idx not in self._unassigned_deliveries:
                        lat, lon, _, _ = self._delivery_points[idx]
                        delivery_waypoints.append((lat, lon))

                # Mark delivery points in the route
                for i, point in enumerate(route_points):
                    for waypoint in delivery_waypoints:
                        if self._points_close(point, waypoint):
                            delivery_indices.append(i)
                            break

                # Get route style
                route_style = self._route_color_manager.get_route_style(
                    assignment.driver_id - 1, total_drivers
                )

                simulation_data.append({
                    "driverId": assignment.driver_id,
                    "path": route_points,
                    "travelTimes": travel_times,
                    "deliveryIndices": delivery_indices,
                    "style": route_style
                })

            return simulation_data

        except Exception as e:
            print(f"Error preparing simulation data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_route_with_times(self, assignment):
        """
        Calculate route points and travel times for a driver assignment.

        Args:
            assignment: The DeliveryAssignment to calculate route for

        Returns:
            Tuple of (route_points, travel_times)
        """
        if not self._warehouse_location or not self._graph:
            return [], []

        try:
            import networkx as nx
            import osmnx as ox

            # Filter out unassigned deliveries
            delivery_indices = [idx for idx in assignment.delivery_indices
                                if idx not in self._unassigned_deliveries]

            if not delivery_indices:
                return [], []

            # Get delivery points
            delivery_points = []
            for idx in delivery_indices:
                lat, lon, _, _ = self._delivery_points[idx]
                delivery_points.append((lat, lon))

            # Create a route starting and ending at the warehouse
            waypoints = [self._warehouse_location] + delivery_points + [self._warehouse_location]

            # Calculate route using shortest paths
            route_points = []
            travel_times = []

            for i in range(len(waypoints) - 1):
                start_waypoint = waypoints[i]
                end_waypoint = waypoints[i + 1]

                try:
                    start_node = ox.nearest_nodes(self._graph, X=start_waypoint[1], Y=start_waypoint[0])
                    end_node = ox.nearest_nodes(self._graph, X=end_waypoint[1], Y=end_waypoint[0])

                    # Find shortest path
                    path = nx.shortest_path(self._graph, start_node, end_node, weight='travel_time')

                    # Extract coordinates and travel times for each segment in the path
                    segment_points = []
                    segment_times = []

                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]

                        node_u = self._graph.nodes[u]
                        lat_u, lon_u = node_u['y'], node_u['x']
                        segment_points.append((lat_u, lon_u))

                        # Get travel time for this edge
                        edge_data = self._graph.get_edge_data(u, v, 0)
                        if 'travel_time' in edge_data:
                            travel_time = edge_data['travel_time']
                        else:
                            # Calculate approximate travel time if not available
                            if 'length' in edge_data and 'speed_kph' in edge_data:
                                length = edge_data['length']  # meters
                                speed = edge_data['speed_kph']  # km/h
                                travel_time = (length / 1000) / (speed / 3600)  # seconds
                            else:
                                # Default to 30 seconds if no data available
                                travel_time = 30

                        segment_times.append(travel_time)

                    # Add the last point
                    last_node = path[-1]
                    lat_last, lon_last = self._graph.nodes[last_node]['y'], self._graph.nodes[last_node]['x']
                    segment_points.append((lat_last, lon_last))

                    # Append the segment to our route
                    if i > 0 and route_points and segment_points:
                        # Avoid duplicating points at segment boundaries
                        segment_points = segment_points[1:]

                    route_points.extend(segment_points)
                    travel_times.extend(segment_times)

                except Exception as e:
                    print(f"Error calculating path segment: {str(e)}")
                    # Fall back to direct line if path finding fails
                    route_points.extend([start_waypoint, end_waypoint])

                    # Estimate travel time for direct line
                    dx = end_waypoint[1] - start_waypoint[1]
                    dy = end_waypoint[0] - start_waypoint[0]
                    distance_km = ((dx ** 2 + dy ** 2) ** 0.5) * 111  # Rough conversion to km
                    travel_time = (distance_km / 50) * 3600  # Assume 50 km/h speed
                    travel_times.append(travel_time)

            return route_points, travel_times

        except Exception as e:
            print(f"Error calculating route with times: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

    def _points_close(self, point1, point2, threshold=0.0001):
        """
        Check if two geographic points are close to each other.

        Args:
            point1: First point (lat, lon)
            point2: Second point (lat, lon)
            threshold: Distance threshold

        Returns:
            Boolean indicating if points are close
        """
        lat1, lon1 = point1
        lat2, lon2 = point2

        return abs(lat1 - lat2) < threshold and abs(lon1 - lon2) < threshold

    def set_delivery_points(self, delivery_points):
        """
        Set the delivery points for route calculations.

        Args:
            delivery_points: List of (lat, lon, weight, volume) tuples
        """
        self._delivery_points = delivery_points

    def calculate_total_time(self, solution):
        """
        Calculate the total travel time for a solution.

        Args:
            solution: The optimization solution

        Returns:
            Total travel time in seconds
        """
        if not solution or not self._graph or not self._warehouse_location:
            return 0

        import networkx as nx
        import osmnx as ox

        total_time = 0
        node_cache = {}

        def get_node(coords):
            if coords not in node_cache:
                node_cache[coords] = ox.nearest_nodes(self._graph, X=coords[1], Y=coords[0])
            return node_cache[coords]

        for assignment in solution:
            if not assignment.delivery_indices:
                continue

            # Get points for the route
            route_points = [self._warehouse_location]

            for idx in assignment.delivery_indices:
                if idx not in self._unassigned_deliveries:
                    lat, lon, _, _ = self._delivery_points[idx]
                    route_points.append((lat, lon))

            route_points.append(self._warehouse_location)

            # Calculate travel time for each segment
            for i in range(len(route_points) - 1):
                start = route_points[i]
                end = route_points[i + 1]

                try:
                    start_node = get_node(start)
                    end_node = get_node(end)

                    try:
                        # Try to find path with travel_time
                        path = nx.shortest_path(self._graph, start_node, end_node, weight='travel_time')

                        # Sum up travel time along the path
                        segment_time = 0
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            edge_data = self._graph.get_edge_data(u, v, 0)
                            if 'travel_time' in edge_data:
                                segment_time += edge_data['travel_time']
                            else:
                                # Estimate based on length and speed
                                if 'length' in edge_data:
                                    length = edge_data['length']  # meters
                                    speed = edge_data.get('speed_kph', 50)  # km/h
                                    edge_time = (length / 1000) / (speed / 3600)  # seconds
                                    segment_time += edge_time

                        total_time += segment_time

                    except nx.NetworkXNoPath:
                        # Fallback to direct distance calculation
                        dx = end[1] - start[1]
                        dy = end[0] - start[0]
                        distance_km = ((dx ** 2 + dy ** 2) ** 0.5) * 111  # Rough conversion to km
                        segment_time = (distance_km / 50) * 3600  # Assume 50 km/h speed
                        total_time += segment_time

                except Exception as e:
                    print(f"Error calculating travel time: {e}")
                    # Add a default time as fallback
                    total_time += 60  # Add 1 minute as fallback

        return total_time

    def calculate_total_distance(self, solution):
        """
        Calculate the total distance for a solution.

        Args:
            solution: The optimization solution

        Returns:
            Total distance in kilometers
        """
        if not solution or not self._graph or not self._warehouse_location:
            return 0

        import networkx as nx
        import osmnx as ox

        total_distance = 0
        node_cache = {}

        def get_node(coords):
            if coords not in node_cache:
                node_cache[coords] = ox.nearest_nodes(self._graph, X=coords[1], Y=coords[0])
            return node_cache[coords]

        for assignment in solution:
            if not assignment.delivery_indices:
                continue

            # Get points for the route
            route_points = [self._warehouse_location]

            for idx in assignment.delivery_indices:
                if idx not in self._unassigned_deliveries:
                    lat, lon, _, _ = self._delivery_points[idx]
                    route_points.append((lat, lon))

            route_points.append(self._warehouse_location)

            # Calculate distance for each segment
            for i in range(len(route_points) - 1):
                start = route_points[i]
                end = route_points[i + 1]

                try:
                    start_node = get_node(start)
                    end_node = get_node(end)

                    try:
                        # Try to find path with length
                        path_length = nx.shortest_path_length(self._graph, start_node, end_node, weight='length')
                        total_distance += path_length / 1000  # Convert to kilometers

                    except nx.NetworkXNoPath:
                        # Fallback to direct distance calculation
                        dx = end[1] - start[1]
                        dy = end[0] - start[0]
                        distance_km = ((dx ** 2 + dy ** 2) ** 0.5) * 111  # Rough conversion to km
                        total_distance += distance_km

                except Exception as e:
                    print(f"Error calculating distance: {e}")
                    # Don't add anything if we can't calculate

        return total_distance

    def calculate_driver_statistics(self, solution):
        """
        Calculate detailed statistics for each driver.

        Args:
            solution: The optimization solution

        Returns:
            Dictionary mapping driver IDs to their statistics
        """
        if not solution or not self._graph or not self._warehouse_location:
            return {}

        import networkx as nx
        import osmnx as ox

        driver_stats = {}
        node_cache = {}

        def get_node(coords):
            if coords not in node_cache:
                node_cache[coords] = ox.nearest_nodes(self._graph, X=coords[1], Y=coords[0])
            return node_cache[coords]

        for assignment in solution:
            driver_id = assignment.driver_id

            if not assignment.delivery_indices:
                driver_stats[driver_id] = {
                    'travel_time': 0,
                    'distance': 0,
                    'deliveries': 0,
                    'weight': 0,
                    'volume': 0
                }
                continue

            travel_time = 0
            distance = 0

            # Get points for the route
            route_points = [self._warehouse_location]

            for idx in assignment.delivery_indices:
                if idx not in self._unassigned_deliveries:
                    lat, lon, _, _ = self._delivery_points[idx]
                    route_points.append((lat, lon))

            route_points.append(self._warehouse_location)

            # Calculate metrics for each segment
            for i in range(len(route_points) - 1):
                start = route_points[i]
                end = route_points[i + 1]

                try:
                    start_node = get_node(start)
                    end_node = get_node(end)

                    # Calculate distance
                    try:
                        path_length = nx.shortest_path_length(self._graph, start_node, end_node, weight='length')
                        distance += path_length / 1000  # Convert to kilometers
                    except Exception:
                        # Fallback
                        dx = end[1] - start[1]
                        dy = end[0] - start[0]
                        segment_distance = ((dx ** 2 + dy ** 2) ** 0.5) * 111
                        distance += segment_distance

                    # Calculate travel time
                    try:
                        path = nx.shortest_path(self._graph, start_node, end_node, weight='travel_time')

                        segment_time = 0
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            edge_data = self._graph.get_edge_data(u, v, 0)
                            if 'travel_time' in edge_data:
                                segment_time += edge_data['travel_time']
                            else:
                                # Estimate based on length and speed
                                if 'length' in edge_data:
                                    length = edge_data['length']  # meters
                                    speed = edge_data.get('speed_kph', 50)  # km/h
                                    edge_time = (length / 1000) / (speed / 3600)  # seconds
                                    segment_time += edge_time

                        travel_time += segment_time

                    except Exception:
                        # Fallback
                        segment_distance = distance  # Use the distance we calculated above
                        segment_time = (segment_distance / 50) * 3600  # Assume 50 km/h
                        travel_time += segment_time

                except Exception as e:
                    print(f"Error calculating driver statistics: {e}")

            driver_stats[driver_id] = {
                'travel_time': travel_time,
                'distance': distance,
                'deliveries': len(assignment.delivery_indices),
                'weight': assignment.total_weight,
                'volume': assignment.total_volume
            }

        return driver_stats

    def get_statistics(self):
        """
        Get comprehensive statistics for the current solution.

        Returns:
            Dictionary with solution statistics
        """
        if not self._current_solution:
            return {
                'total_time': 0,
                'total_distance': 0,
                'total_deliveries': 0,
                'unassigned_deliveries': 0,
                'driver_stats': {}
            }

        total_time = self.calculate_total_time(self._current_solution)
        total_distance = self.calculate_total_distance(self._current_solution)
        driver_stats = self.calculate_driver_statistics(self._current_solution)

        total_deliveries = sum(len(assignment.delivery_indices) for assignment in self._current_solution)
        unassigned_deliveries = len(self._unassigned_deliveries) if self._unassigned_deliveries else 0

        return {
            'total_time': total_time,
            'total_distance': total_distance,
            'total_deliveries': total_deliveries,
            'unassigned_deliveries': unassigned_deliveries,
            'driver_stats': driver_stats
        }
