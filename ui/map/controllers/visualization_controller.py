import time

import folium
import networkx as nx
import osmnx as ox
from PyQt5 import QtCore, QtWidgets

from logic.delivery_optimizer import SimulatedAnnealingOptimizer
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

    def prepare_optimization(self, delivery_drivers, snapped_delivery_points):
        """Prepare data for optimization"""
        self.delivery_drivers = delivery_drivers
        self.snapped_delivery_points = snapped_delivery_points
        self.G = self.base_map.G
        self.start_time = time.time()

        self.optimization_finished.connect(self.on_optimization_finished)

    def run_optimization(self):
        """Run the optimization process in the worker thread"""
        try:
            self.optimizer = SimulatedAnnealingOptimizer(
                self.delivery_drivers,
                self.snapped_delivery_points,
                self.G
            )

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

    def update_visualization(self, current_solution, unassigned_deliveries):
        """Entry point for visualization updates during optimization."""
        try:
            self._initialize_map()
            self._setup_layers()

            unassigned_set = set(unassigned_deliveries)
            conflicting_deliveries, delivery_to_driver = self._identify_conflicts(current_solution)

            self._visualize_routes(
                current_solution,
                unassigned_set,
                delivery_to_driver,
                conflicting_deliveries
            )

            self._visualize_problematic_deliveries(
                conflicting_deliveries,
                unassigned_deliveries
            )

            self.base_map.load_map()
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()

    def _initialize_map(self):
        city_center, zoom = get_city_coordinates(
            self.base_map.current_city or "Kaunas, Lithuania"
        )

        if not self.optimization_map_initialized:
            self.base_map.init_map(city_center, zoom)
            self.optimization_map_initialized = True
            self.map_zoom = zoom

        self.base_map.map = folium.Map(
            location=self.base_map.map.location,
            zoom_start=self.map_zoom
        )

    def _setup_layers(self):
        self.delivery_layer = folium.FeatureGroup(name='delivery_points')
        self.routes_layer = folium.FeatureGroup(name='routes')
        self.base_map.map.add_child(self.delivery_layer)
        self.base_map.map.add_child(self.routes_layer)

    def _identify_conflicts(self, solution):
        delivery_to_driver = {}
        conflicting_deliveries = set()

        for driver_idx, assignment in enumerate(solution):
            for delivery_idx in assignment.delivery_indices:
                if delivery_idx in delivery_to_driver:
                    conflicting_deliveries.add(delivery_idx)
                else:
                    delivery_to_driver[delivery_idx] = (driver_idx, assignment.driver_id)

        return conflicting_deliveries, delivery_to_driver

    def _visualize_routes(self, current_solution, unassigned_set, delivery_to_driver, conflicting_deliveries):
        active_drivers = [a for a in current_solution if a.delivery_indices]
        total_drivers = len(active_drivers)

        for driver_idx, assignment in enumerate(current_solution):
            if not assignment.delivery_indices:
                continue

            route_style = self.route_color_manager.get_route_style(driver_idx, total_drivers)

            if hasattr(self.base_map, 'selected_driver_id') and self.base_map.selected_driver_id is not None:
                if assignment.driver_id != self.base_map.selected_driver_id:
                    reduced_opacity_style = route_style.copy()
                    reduced_opacity_style['opacity'] = 0.25
                    route_style = reduced_opacity_style

            driver_points = self._get_driver_points(
                assignment, driver_idx, delivery_to_driver,
                unassigned_set, conflicting_deliveries
            )

            if driver_points:
                self._add_point_markers(driver_points, assignment, route_style)
                self._add_route_line(driver_points, route_style, assignment)

    def _get_driver_points(self, assignment, driver_idx, delivery_to_driver, unassigned_set, conflicting_deliveries):
        points = []
        for delivery_idx in assignment.delivery_indices:
            if delivery_idx not in unassigned_set and delivery_idx not in conflicting_deliveries:
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
        if len(driver_points) > 1:
            detailed_route = self._calculate_detailed_route(driver_points)

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

    def _visualize_problematic_deliveries(self, conflicting_deliveries, unassigned_deliveries):
        for delivery_idx in conflicting_deliveries:
            self._add_conflicting_delivery_marker(delivery_idx)

        for delivery_idx in unassigned_deliveries:
            if delivery_idx not in conflicting_deliveries:
                self._add_unassigned_delivery_marker(delivery_idx)

    def _add_conflicting_delivery_marker(self, delivery_idx):
        lat, lon, weight, volume = self.base_map.snapped_delivery_points[delivery_idx]
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=(f'Delivery Point {delivery_idx + 1}<br>'
                   f'ERROR: Multiple Assignments<br>'
                   f'Weight: {weight}kg<br>'
                   f'Volume: {volume}m³')
        ).add_to(self.delivery_layer)

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

    def on_optimization_finished(self, final_solution, unassigned):
        """Handle optimization completion in the main thread"""
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

            self._update_statistics(final_solution, unassigned)
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
            if self.optimizer:
                self.optimizer.deleteLater()
                self.optimizer = None

    def _update_statistics(self, final_solution, unassigned):
        city_center, _ = get_city_coordinates(self.base_map.current_city or "Kaunas, Lithuania")
        total_distance = 0
        total_time = 0

        for assignment in final_solution:
            if assignment.delivery_indices:
                delivery_coords = [
                    self.base_map.snapped_delivery_points[i][0:2]
                    for i in assignment.delivery_indices
                ]
                complete_route = [city_center] + delivery_coords + [city_center]
                route_length = self.optimizer.calculate_route_distance(complete_route)
                total_distance += route_length
                total_time += route_length / 10

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
        total_distance, total_time = self._update_statistics(final_solution, unassigned)

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
