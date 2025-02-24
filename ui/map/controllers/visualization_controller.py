import time

import folium
import networkx as nx
import osmnx as ox
from PyQt5 import QtCore, QtWidgets

from config import validate_settings, OPTIMIZATION_SETTINGS
from logic.delivery_optimizer import SimulatedAnnealingOptimizer
from logic.greedy_delivery_optimizer import GreedyOptimizer
from utils.geolocation import get_city_coordinates


class VisualizationController(QtCore.QObject):
    optimization_finished = QtCore.pyqtSignal(object, object)

    def __init__(self, base_map):
        super().__init__()
        self.base_map = base_map
        self.route_color_manager = base_map.route_color_manager
        self.delivery_layer = None
        self.routes_layer = None
        self.optimization_map_initialized = False
        self.map_zoom = None

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

    def run_optimization(self):
        try:
            validate_settings()

            self.optimizer = SimulatedAnnealingOptimizer(
                self.delivery_drivers,
                self.snapped_delivery_points,
                self.G,
                self.map_widget
            )

            if OPTIMIZATION_SETTINGS['VALIDATE']:
                self.greedy_optimizer = GreedyOptimizer(
                    self.delivery_drivers,
                    self.snapped_delivery_points,
                    self.G,
                    self.map_widget
                )

                self.optimizer.finished.connect(self.on_sa_finished)
                self.greedy_optimizer.finished.connect(self.on_greedy_finished)

                self.optimizer.optimize()
                self.greedy_optimizer.optimize()
            else:
                self.optimizer.update_visualization.connect(
                    lambda sol, unassigned: QtCore.QMetaObject.invokeMethod(
                        self.base_map,
                        'addToVisualizationQueue',
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(object, (sol, unassigned))
                    )
                )

                self.optimizer.finished.connect(
                    lambda sol, unassigned: self.optimization_finished.emit(sol, unassigned)
                )
                self.optimizer.optimize()

        except Exception as e:
            print(f"Optimization error: {e}")
            self.optimization_finished.emit(None, None)

    def update_visualization(self, update_data, unassigned_deliveries=None):
        """
        Handle visualization updates by processing complete solution states.
        All updates should provide a complete solution object, ensuring consistent visualization.
        """
        try:
            if isinstance(update_data, tuple):
                solution, unassigned = update_data
            else:
                solution = update_data

            self._handle_solution_visualization(solution, unassigned_deliveries)

        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()

    def _handle_solution_visualization(self, solution, unassigned_deliveries):
        """Handle the actual visualization of a solution"""
        self._initialize_map()
        self._setup_layers()

        if unassigned_deliveries is None:
            unassigned_set = set()
        elif isinstance(unassigned_deliveries, (int, float)):
            unassigned_set = {unassigned_deliveries}
        else:
            unassigned_set = set(unassigned_deliveries)

        self.current_solution = solution
        self.unassigned_deliveries = unassigned_set

        self._visualize_routes(solution, unassigned_set)
        self._visualize_problematic_deliveries(unassigned_set)

        self.base_map.load_map()
        QtWidgets.QApplication.processEvents()

    def _initialize_map(self):
        """Initialize the map for visualization"""
        city_center, zoom = get_city_coordinates(
            self.base_map.current_city or "Kaunas, Lithuania"
        )

        self.base_map.map = folium.Map(
            location=city_center,
            zoom_start=zoom
        )

        warehouse_location = self.base_map.get_warehouse_location()
        if warehouse_location:
            folium.Marker(
                location=warehouse_location,
                icon=folium.Icon(color='red', icon='home', prefix='fa'),
                popup='Warehouse (Start/End Point)'
            ).add_to(self.base_map.map)

    def _setup_layers(self):
        self.delivery_layer = folium.FeatureGroup(name='delivery_points')
        self.routes_layer = folium.FeatureGroup(name='routes')
        self.base_map.map.add_child(self.delivery_layer)
        self.base_map.map.add_child(self.routes_layer)

    def _visualize_routes(self, current_solution, unassigned_set):
        total_drivers = len(self.delivery_drivers)

        for assignment in current_solution:
            if not assignment.delivery_indices:
                continue

            route_style = self.route_color_manager.get_route_style(assignment.driver_id - 1, total_drivers)

            if hasattr(self.base_map, 'selected_driver_id') and self.base_map.selected_driver_id is not None:
                if assignment.driver_id != self.base_map.selected_driver_id:
                    reduced_opacity_style = route_style.copy()
                    reduced_opacity_style['opacity'] = 0.25
                    route_style = reduced_opacity_style

            driver_points = self._get_driver_points(
                assignment, unassigned_set
            )

            if driver_points:
                self._add_point_markers(driver_points, assignment, route_style)
                self._add_route_line(driver_points, route_style, assignment)

    def _get_driver_points(self, assignment, unassigned_set):
        points = []
        for delivery_idx in assignment.delivery_indices:
            if delivery_idx not in unassigned_set:
                try:
                    lat, lon, weight, volume = self.base_map.snapped_delivery_points[delivery_idx]
                    points.append({
                        'idx': delivery_idx,
                        'coords': (lat, lon),
                        'weight': weight,
                        'volume': volume
                    })
                except Exception as e:
                    print(f"Error processing delivery point {delivery_idx}: {e}")
        return points

    def _add_point_markers(self, points, assignment, style):
        for point in points:
            folium.CircleMarker(
                location=point['coords'],
                radius=6,
                color=style['color'],
                fill=True,
                fill_color=style['color'],
                opacity=style['opacity'],
                fill_opacity=style['opacity'],
                popup=self._create_point_popup(point, assignment)
            ).add_to(self.delivery_layer)

    def _create_point_popup(self, point, assignment):
        return (f'Delivery Point {point["idx"] + 1}<br>'
                f'Assigned to Driver {assignment.driver_id}<br>'
                f'Weight: {point["weight"]}kg<br>'
                f'Volume: {point["volume"]}m³')

    def _add_route_line(self, driver_points, route_style, assignment):
        if not driver_points:
            return

        warehouse_location = self.base_map.get_warehouse_location()
        if not warehouse_location:
            return

        # Create full route: warehouse -> deliveries -> warehouse
        full_route = [{'coords': warehouse_location}]
        full_route.extend(driver_points)
        full_route.append({'coords': warehouse_location})
        detailed_route = self._calculate_detailed_route(full_route)

        folium.PolyLine(
            locations=detailed_route,
            color=route_style['color'],
            weight=route_style['weight'],
            dash_array=route_style['dash_array'],
            opacity=route_style['opacity'],
            popup=f'Driver {assignment.driver_id} Route'
        ).add_to(self.routes_layer)

    def _calculate_detailed_route(self, driver_points):
        detailed_route = []
        for i in range(len(driver_points) - 1):
            start = driver_points[i]['coords']
            end = driver_points[i + 1]['coords']

            try:
                start_node = ox.nearest_nodes(self.base_map.G, X=start[1], Y=start[0])
                end_node = ox.nearest_nodes(self.base_map.G, X=end[1], Y=end[0])
                path = nx.shortest_path(self.base_map.G, start_node, end_node, weight='length')
                path_coords = [(self.base_map.G.nodes[node]['y'], self.base_map.G.nodes[node]['x'])
                               for node in path]
                detailed_route.extend(path_coords)
            except nx.NetworkXNoPath:
                detailed_route.extend([start, end])

        return detailed_route

    def _visualize_problematic_deliveries(self, unassigned_deliveries):
        for delivery_idx in unassigned_deliveries:
            self._add_unassigned_delivery_marker(delivery_idx)

    def _add_unassigned_delivery_marker(self, delivery_idx):
        lat, lon, weight, volume = self.base_map.snapped_delivery_points[delivery_idx]
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color='black',
            fill=True,
            fill_opacity=0.7,
            popup=(f'Delivery Point {delivery_idx + 1}<br>'
                   f'UNASSIGNED<br>'
                   f'Weight: {weight}kg<br>'
                   f'Volume: {volume}m³')
        ).add_to(self.delivery_layer)

    def on_sa_finished(self, solution, unassigned):
        self.sa_solution = solution
        self.sa_unassigned = unassigned

        if self.greedy_solution is not None:
            self.optimization_finished.emit(solution, unassigned)

    def on_greedy_finished(self, solution, unassigned):
        self.greedy_solution = solution
        self.greedy_unassigned = unassigned

        if self.sa_solution is not None:
            self.optimization_finished.emit(self.sa_solution, self.sa_unassigned)

    def switch_solution_view(self, solution_type):
        if solution_type == "Simulated Annealing":
            if hasattr(self, 'sa_solution') and self.sa_solution is not None:
                self.current_view = solution_type
                unassigned = self.sa_unassigned if isinstance(self.sa_unassigned, (set, list)) else set()
                self.update_visualization(self.sa_solution, unassigned)
                self._update_statistics(self.sa_solution)
        else:
            if hasattr(self, 'greedy_solution') and self.greedy_solution is not None:
                self.current_view = solution_type
                unassigned = self.greedy_unassigned if isinstance(self.greedy_unassigned, (set, list)) else set()
                self.update_visualization(self.greedy_solution, unassigned)
                self._update_statistics(self.greedy_solution)

    def on_optimization_finished(self, final_solution, unassigned):
        try:
            self.base_map.setEnabled(True)

            if final_solution is None:
                QtWidgets.QMessageBox.critical(
                    self.base_map,
                    "Error",
                    "Optimization failed"
                )
                return

            self.current_solution = final_solution
            self.unassigned_deliveries = unassigned

            main_window = self.base_map.get_main_window()
            if (OPTIMIZATION_SETTINGS['VALIDATE'] and main_window and
                    hasattr(self, 'sa_solution') and hasattr(self, 'greedy_solution')):
                main_window.solution_switch.show()
                main_window.solution_switch.setEnabled(True)

            self._update_statistics(final_solution)
            self._show_optimization_summary(final_solution, unassigned)

            self.base_map.visualization_queue.append((final_solution, unassigned))

        except Exception as e:
            print(f"Error in optimization finished handler: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self.base_map,
                "Error",
                f"An error occurred while finalizing the optimization:\n{str(e)}"
            )
        finally:
            pass

    def _update_statistics(self, final_solution):
        """Directly use optimizer's time calculation method"""
        total_time = self.optimizer.calculate_total_time(final_solution)
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
                [self.base_map.snapped_delivery_points[i][0:2] for i in assignment.delivery_indices],
                [self.base_map.snapped_delivery_points[i][0:2] for i in assignment.delivery_indices] +
                [self.base_map.get_warehouse_location()]
            )
        )

        if self.base_map.time_label:
            computation_time = time.time() - self.start_time
            self.base_map.time_label.setText(
                f"Routes computed in {computation_time:.2f} seconds"
            )

        if self.base_map.travel_time_label:
            total_minutes = total_time / 60
            self.base_map.travel_time_label.setText(
                f"Total travel time: {total_minutes:.2f} minutes"
            )

        if self.base_map.distance_label:
            total_kilometers = total_distance / 1000
            self.base_map.distance_label.setText(
                f"Total distance: {total_kilometers:.2f} km"
            )

        return total_distance, total_time

    def _show_optimization_summary(self, final_solution, unassigned):
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
