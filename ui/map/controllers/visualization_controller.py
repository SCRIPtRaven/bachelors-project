# ui/map/controllers/visualization_controller.py
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import networkx as nx
import osmnx as ox
from PyQt5 import QtCore, QtWidgets

from config import validate_settings, OPTIMIZATION_SETTINGS
from logic.delivery_optimizer import SimulatedAnnealingOptimizer


class VisualizationController(QtCore.QObject):
    optimization_finished = QtCore.pyqtSignal(object, object)
    _route_calculation_executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, base_map):
        super().__init__()
        self.base_map = base_map
        self.route_color_manager = base_map.route_color_manager

        self.current_solution = None
        self.unassigned_deliveries = None
        self.optimizer = None
        self.start_time = None

        self.delivery_drivers = None
        self.snapped_delivery_points = None
        self.G = None

        self.sa_solution = None
        self.sa_unassigned = None
        self.greedy_solution = None
        self.greedy_unassigned = None
        self.current_view = "Simulated Annealing"

    def prepare_optimization(self, delivery_drivers, snapped_delivery_points, map_widget):
        """Prepare data for optimization"""
        self.delivery_drivers = delivery_drivers
        self.snapped_delivery_points = snapped_delivery_points
        self.G = self.base_map.G
        self.start_time = time.time()
        self.map_widget = map_widget

        self.optimization_finished.connect(self.on_optimization_finished)

        self.base_map.clear_all_layers()

        warehouse_location = self.base_map.get_warehouse_location()
        if warehouse_location:
            print(f"Adding warehouse at: {warehouse_location}")
            self.base_map.add_warehouse(warehouse_location[0], warehouse_location[1])

        self.base_map.add_delivery_points(self.snapped_delivery_points)

    def run_optimization(self):
        """Run the optimization process in a worker thread"""
        try:
            validate_settings()
            self.base_map.execute_js("if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

            self.sa_solution = None
            self.sa_unassigned = None
            self.greedy_solution = None
            self.greedy_unassigned = None

            self.optimizer = SimulatedAnnealingOptimizer(
                self.delivery_drivers,
                self.snapped_delivery_points,
                self.G,
                self.map_widget
            )

            self.last_update_time = 0
            self.update_interval = 1.0

            self.optimizer.update_visualization.connect(self.rate_limited_visualization_update)

            self.optimizer.finished.connect(
                lambda sol, unassigned: self.on_sa_finished(sol, unassigned)
            )

            if OPTIMIZATION_SETTINGS['VALIDATE']:
                from logic.greedy_delivery_optimizer import GreedyOptimizer

                self.greedy_optimizer = GreedyOptimizer(
                    self.delivery_drivers,
                    self.snapped_delivery_points,
                    self.G,
                    self.map_widget
                )

                self.greedy_optimizer.finished.connect(
                    lambda sol, unassigned: self.on_greedy_finished(sol, unassigned)
                )

                self.greedy_optimizer.optimize()

            self.optimizer.optimize()

        except Exception as e:
            print(f"Optimization error: {e}")
            self.base_map.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")
            print(f"Optimization error: {e}")
            import traceback
            traceback.print_exc()
            self.optimization_finished.emit(None, None)

    def rate_limited_visualization_update(self, solution, unassigned):
        """Handle visualization updates with rate limiting to prevent UI freezing"""
        current_time = time.time()

        if not hasattr(self, 'update_interval'):
            self.update_interval = 2.0

        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time

            QtCore.QMetaObject.invokeMethod(
                self.base_map,
                'addToVisualizationQueue',
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, (solution, unassigned))
            )

    def update_visualization(self, solution, unassigned_deliveries=None):
        """Update the visualization with the given solution using background processing"""
        try:
            if isinstance(solution, tuple):
                solution, unassigned = solution
            else:
                unassigned = unassigned_deliveries

            if hasattr(self, 'last_visualized_solution'):
                if self._solutions_equal(solution, self.last_visualized_solution) and \
                        self.last_visualized_unassigned == unassigned:
                    return

            self.last_visualized_solution = deepcopy(solution)
            self.last_visualized_unassigned = unassigned if isinstance(unassigned, set) else set(unassigned or [])

            self._route_calculation_executor.submit(
                self._process_visualization_in_background,
                deepcopy(solution),
                unassigned
            )

        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()

    def _solutions_equal(self, sol1, sol2):
        """Compare two solutions to see if they're functionally equivalent"""
        if len(sol1) != len(sol2):
            return False

        for a1, a2 in zip(sol1, sol2):
            if a1.driver_id != a2.driver_id:
                return False

            if len(a1.delivery_indices) != len(a2.delivery_indices):
                return False

            if not all(i1 == i2 for i1, i2 in zip(a1.delivery_indices, a2.delivery_indices)):
                return False

        return True

    def _process_visualization_in_background(self, solution, unassigned_deliveries):
        """Process route calculations in a background thread"""
        try:
            if unassigned_deliveries is None:
                unassigned_set = set()
            elif isinstance(unassigned_deliveries, (int, float)):
                unassigned_set = {unassigned_deliveries}
            else:
                unassigned_set = set(unassigned_deliveries)

            selected_driver_id = self.base_map.selected_driver_id

            delivery_driver_map = {}
            driver_style_map = {}
            total_drivers = len(self.delivery_drivers)

            for assignment in solution:
                driver_id = assignment.driver_id
                driver_style = self.route_color_manager.get_route_style(driver_id - 1, total_drivers)

                if selected_driver_id is not None:
                    if driver_id == selected_driver_id:
                        driver_style['opacity'] = 1.0
                    else:
                        driver_style = driver_style.copy()
                        driver_style['opacity'] = 0.15

                driver_style_map[driver_id] = driver_style

                for delivery_idx in assignment.delivery_indices:
                    if delivery_idx not in unassigned_set:
                        delivery_driver_map[delivery_idx] = driver_id

            delivery_points_data = []
            for idx, point in enumerate(self.snapped_delivery_points):
                lat, lon, weight, volume = point

                point_opacity = 0.7

                if idx in unassigned_set:
                    color = "black"
                    popup = f"Delivery Point {idx + 1}<br>UNASSIGNED<br>Weight: {weight}kg<br>Volume: {volume}m³"
                elif idx in delivery_driver_map:
                    driver_id = delivery_driver_map[idx]
                    color = driver_style_map[driver_id]['color']

                    if selected_driver_id is not None:
                        if driver_id != selected_driver_id:
                            point_opacity = 0.15

                    popup = f"Delivery Point {idx + 1}<br>Assigned to Driver {driver_id}<br>Weight: {weight}kg<br>Volume: {volume}m³"
                else:
                    color = "gray"
                    popup = f"Delivery Point {idx + 1}<br>Status: Unknown<br>Weight: {weight}kg<br>Volume: {volume}m³"

                delivery_points_data.append({
                    "lat": lat,
                    "lng": lon,
                    "weight": weight,
                    "volume": volume,
                    "index": idx,
                    "color": color,
                    "opacity": point_opacity,
                    "popup": popup
                })

            route_data = []
            warehouse_location = self.base_map.get_warehouse_location()

            for i, assignment in enumerate(solution):
                if not assignment.delivery_indices:
                    continue

                driver_id = assignment.driver_id
                route_style = driver_style_map[driver_id]

                route_data_tuple = self._calculate_route_points(assignment, warehouse_location, unassigned_set)

                if isinstance(route_data_tuple, tuple) and len(route_data_tuple) >= 1:
                    detailed_route = route_data_tuple[0]
                else:
                    detailed_route = route_data_tuple

                if detailed_route:
                    popup = f"Driver {driver_id} Route<br>"
                    popup += f"Deliveries: {len(assignment.delivery_indices)}<br>"
                    popup += f"Weight: {assignment.total_weight:.1f}kg<br>"
                    popup += f"Volume: {assignment.total_volume:.3f}m³"

                    route_data.append({
                        "id": i,
                        "driver_id": driver_id,
                        "path": detailed_route,
                        "style": route_style,
                        "popup": popup
                    })

            QtCore.QMetaObject.invokeMethod(
                self.base_map,
                'updateVisualizationFromBackground',
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, {
                    "delivery_points": delivery_points_data,
                    "routes": route_data
                })
            )

        except Exception as e:
            print(f"Error in background visualization processing: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_route_points(self, assignment, warehouse_location, unassigned_set):
        """
        Calculate route points at the edge level of the road network graph.
        Each segment in the result represents a single road edge with proper travel time.
        """
        if not warehouse_location:
            return [], [], []

        delivery_points = []
        for idx in assignment.delivery_indices:
            if idx not in unassigned_set:
                lat, lon, _, _ = self.snapped_delivery_points[idx]
                delivery_points.append((lat, lon))

        if not delivery_points:
            return [], [], []

        waypoints = [warehouse_location] + delivery_points + [warehouse_location]

        route_points = []
        travel_times = []
        delivery_indices = []

        delivery_waypoint_indices = set(range(1, len(waypoints) - 1))

        if not hasattr(self, '_route_cache'):
            self._route_cache = {}

        current_point_index = 0

        for i in range(len(waypoints) - 1):
            start_waypoint = waypoints[i]
            end_waypoint = waypoints[i + 1]

            cache_key = (start_waypoint, end_waypoint)

            if cache_key in self._route_cache:
                segment_points, segment_times = self._route_cache[cache_key]
            else:
                try:
                    start_node = ox.nearest_nodes(self.G, X=start_waypoint[1], Y=start_waypoint[0])
                    end_node = ox.nearest_nodes(self.G, X=end_waypoint[1], Y=end_waypoint[0])

                    path = nx.shortest_path(self.G, start_node, end_node, weight='length')

                    segment_points = []
                    segment_times = []

                    for j in range(len(path)):
                        node = path[j]
                        lat = self.G.nodes[node]['y']
                        lon = self.G.nodes[node]['x']
                        segment_points.append((lat, lon))

                        if j < len(path) - 1:
                            next_node = path[j + 1]
                            edge_data = self.G.get_edge_data(node, next_node, 0)

                            if 'travel_time' in edge_data:
                                travel_time = edge_data['travel_time']
                            else:
                                if 'length' in edge_data:
                                    length = edge_data['length']
                                    speed = edge_data.get('speed', 50)
                                    travel_time = length / (speed * 1000 / 3600)
                                else:
                                    next_lat = self.G.nodes[next_node]['y']
                                    next_lon = self.G.nodes[next_node]['x']
                                    dx = next_lon - lon
                                    dy = next_lat - lat
                                    length = ((dx ** 2 + dy ** 2) ** 0.5) * 111000
                                    travel_time = length / (50 * 1000 / 3600)

                                travel_time = max(travel_time, 0.1)

                            segment_times.append(travel_time)

                    if len(self._route_cache) < 10000:
                        self._route_cache[cache_key] = (segment_points, segment_times)

                except nx.NetworkXNoPath:
                    print(f"No path found between {start_waypoint} and {end_waypoint}, using direct line")
                    segment_points = [start_waypoint, end_waypoint]

                    dx = end_waypoint[1] - start_waypoint[1]
                    dy = end_waypoint[0] - start_waypoint[0]
                    distance = ((dx ** 2 + dy ** 2) ** 0.5) * 111000
                    travel_time = distance / (50 * 1000 / 3600)

                    segment_times = [travel_time]

                except Exception as e:
                    print(f"Error calculating path: {e}")
                    segment_points = [start_waypoint, end_waypoint]

                    dx = end_waypoint[1] - start_waypoint[1]
                    dy = end_waypoint[0] - start_waypoint[0]
                    distance = ((dx ** 2 + dy ** 2) ** 0.5) * 111000
                    travel_time = distance / (50 * 1000 / 3600)

                    segment_times = [travel_time]

            if route_points and segment_points:
                first_point = segment_points[0]
                last_added = route_points[-1]

                if (abs(first_point[0] - last_added[0]) < 1e-6 and
                        abs(first_point[1] - last_added[1]) < 1e-6):
                    segment_points = segment_points[1:]

            if i in delivery_waypoint_indices:
                if segment_points:
                    segment_points[0] = start_waypoint

                    delivery_indices.append(current_point_index)

            if i + 1 in delivery_waypoint_indices:
                if segment_points:
                    segment_points[-1] = end_waypoint

                    delivery_indices.append(current_point_index + len(segment_points) - 1)

            route_points.extend(segment_points)
            travel_times.extend(segment_times)

            current_point_index = len(route_points)

        return route_points, travel_times, delivery_indices

    def run_simulation(self):
        """Run a simulation of drivers delivering packages using actual travel times"""
        if not self.current_solution:
            return

        try:
            self.base_map.execute_js("if (typeof showLoadingIndicator === 'function') { showLoadingIndicator(); }")

            simulation_data = []
            warehouse_location = self.base_map.get_warehouse_location()
            total_drivers = len(self.delivery_drivers)

            for assignment in self.current_solution:
                if not assignment.delivery_indices:
                    continue

                route_points, travel_times, delivery_indices = self._calculate_route_points(
                    assignment,
                    warehouse_location,
                    self.unassigned_deliveries
                )

                if not route_points or len(route_points) < 2:
                    print(f"Warning: No valid route found for driver {assignment.driver_id}")
                    continue

                while len(travel_times) < len(route_points) - 1:
                    travel_times.append(1.0)

                route_style = self.route_color_manager.get_route_style(
                    assignment.driver_id - 1,
                    total_drivers
                )

                simulation_data.append({
                    "driverId": assignment.driver_id,
                    "path": route_points,
                    "travelTimes": travel_times,
                    "deliveryIndices": delivery_indices,
                    "style": route_style
                })

            self.base_map.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

            if simulation_data:
                self.base_map.start_simulation(simulation_data)

                try:
                    from config.app_settings import SIMULATION_SPEED
                    speed = SIMULATION_SPEED
                except (ImportError, AttributeError):
                    speed = 25.0

                self.base_map.set_simulation_speed(speed)
            else:
                QtWidgets.QMessageBox.warning(
                    self.base_map,
                    "Simulation Error",
                    "No valid routes to simulate."
                )

        except Exception as e:
            self.base_map.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()

            QtWidgets.QMessageBox.critical(
                self.base_map,
                "Simulation Error",
                f"An error occurred while starting the simulation:\n{str(e)}"
            )

    def on_sa_finished(self, solution, unassigned):
        """Handle SA optimization completion"""
        self.sa_solution = solution
        self.sa_unassigned = unassigned

        if not OPTIMIZATION_SETTINGS['VALIDATE'] or self.greedy_solution is not None:
            self.optimization_finished.emit(solution, unassigned)

    def on_greedy_finished(self, solution, unassigned):
        """Handle Greedy optimization completion"""
        self.greedy_solution = solution
        self.greedy_unassigned = unassigned

        if self.sa_solution is not None:
            self.optimization_finished.emit(self.sa_solution, self.sa_unassigned)

    def switch_solution_view(self, solution_type):
        """Switch between SA and Greedy solutions"""
        if solution_type == "Simulated Annealing":
            if hasattr(self, 'sa_solution') and self.sa_solution is not None:
                self.current_view = solution_type
                unassigned = self.sa_unassigned if isinstance(self.sa_unassigned, (set, list)) else set()

                self.current_solution = self.sa_solution
                self.unassigned_deliveries = unassigned

                self.update_visualization(self.sa_solution, unassigned)
                self._update_statistics(self.sa_solution)
        else:
            if hasattr(self, 'greedy_solution') and self.greedy_solution is not None:
                self.current_view = solution_type
                unassigned = self.greedy_unassigned if isinstance(self.greedy_unassigned, (set, list)) else set()

                self.current_solution = self.greedy_solution
                self.unassigned_deliveries = unassigned

                self.update_visualization(self.greedy_solution, unassigned)
                self._update_statistics(self.greedy_solution)

    def on_optimization_finished(self, final_solution, unassigned):
        """Handle optimization completion"""
        try:
            self.base_map.execute_js("if (typeof hideLoadingIndicator === 'function') { hideLoadingIndicator(); }")

            self.base_map.setEnabled(True)

            if final_solution is None:
                QtWidgets.QMessageBox.critical(
                    self.base_map,
                    "Error",
                    "Optimization failed"
                )
                return

            if hasattr(self, 'last_update_time'):
                delattr(self, 'last_update_time')

            self.current_solution = final_solution
            self.unassigned_deliveries = unassigned

            main_window = self.base_map.get_main_window()
            if (OPTIMIZATION_SETTINGS['VALIDATE'] and main_window and
                    hasattr(self, 'sa_solution') and hasattr(self, 'greedy_solution')):
                main_window.solution_switch.show()
                main_window.solution_switch.setEnabled(True)

            if main_window and hasattr(main_window, 'btn_simulate'):
                main_window.btn_simulate.setEnabled(True)

            self._update_statistics(final_solution)
            self._show_optimization_summary(final_solution, unassigned)

            self.base_map.visualization_queue.append((final_solution, unassigned))

        except Exception as e:
            print(f"Error in optimization finished handler: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self.base_map,
                "Error",
                f"An error occurred while finalizing the optimization:\n{str(e)}"
            )

    def _update_statistics(self, final_solution):
        """Update the statistics labels with solution metrics"""
        if not hasattr(self, 'optimizer') or not self.optimizer:
            return 0, 0

        total_time = self.optimizer.calculate_total_time(final_solution)

        try:
            total_distance = sum(
                nx.shortest_path_length(
                    self.G,
                    ox.nearest_nodes(self.G, X=start[1], Y=start[0]),
                    ox.nearest_nodes(self.G, X=end[1], Y=end[0]),
                    weight='length'
                )
                for assignment in final_solution
                for start, end in zip(
                    [self.base_map.get_warehouse_location()] +
                    [self.snapped_delivery_points[i][0:2] for i in assignment.delivery_indices],
                    [self.snapped_delivery_points[i][0:2] for i in assignment.delivery_indices] +
                    [self.base_map.get_warehouse_location()]
                )
                if assignment.delivery_indices
            )
        except Exception as e:
            print(f"Error calculating total distance: {e}")
            total_distance = 0

        if hasattr(self.base_map, 'time_label'):
            computation_time = time.time() - self.start_time
            self.base_map.time_label.setText(
                f"Routes computed in {computation_time:.2f} seconds"
            )

        if hasattr(self.base_map, 'travel_time_label'):
            total_minutes = total_time / 60
            self.base_map.travel_time_label.setText(
                f"Total travel time: {total_minutes:.2f} minutes"
            )

        if hasattr(self.base_map, 'distance_label'):
            total_kilometers = total_distance / 1000
            self.base_map.distance_label.setText(
                f"Total distance: {total_kilometers:.2f} km"
            )

        return total_distance, total_time

    def _show_optimization_summary(self, final_solution, unassigned):
        """Show a summary of the optimization results"""
        total_distance, total_time = self._update_statistics(final_solution)

        assigned_count = sum(len(assignment.delivery_indices)
                             for assignment in final_solution)
        unassigned_count = len(unassigned)
        total_deliveries = assigned_count + unassigned_count

        summary = (
            f"Optimization Complete\n\n"
            f"Total Deliveries: {total_deliveries}\n"
            f"Successfully Assigned: {assigned_count}\n"
            f"Unassigned: {unassigned_count}\n"
            f"Total Distance: {total_distance / 1000:.2f} km\n"
            f"Estimated Time: {total_time / 60:.2f} minutes"
        )

        QtWidgets.QMessageBox.information(
            self.base_map,
            "Optimization Results",
            summary
        )
