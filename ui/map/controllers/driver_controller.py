from PyQt5 import QtCore, QtWidgets

from services.geolocation_service import GeolocationService
from ui.map.utils.clickable_label import ClickableLabel


class DriverController(QtCore.QObject):
    def __init__(self, base_map):
        super().__init__()
        self.base_map = base_map
        self.delivery_drivers = []
        self.driver_labels = {}
        self.selected_driver_id = None
        self.visualization_queue = None

    def set_visualization_queue(self, queue):
        self.visualization_queue = queue

    def generate_drivers(self, num_drivers):
        try:
            self.delivery_drivers = GeolocationService.generate_delivery_drivers(num_drivers)
            self._update_driver_ui()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.base_map, "Error", f"Error generating drivers: {e}")

    def _update_driver_ui(self):
        stats_layout = self._get_stats_layout()
        self._clear_existing_driver_widgets(stats_layout)
        self._create_driver_list(stats_layout)

    def _get_stats_layout(self):
        return self.base_map.time_label.parent().layout()

    def _clear_existing_driver_widgets(self, layout):
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QtWidgets.QScrollArea) or (
                    isinstance(widget, QtWidgets.QLabel) and widget.text() == "Delivery Drivers:"):
                widget.setParent(None)

    def _create_driver_list(self, layout):
        header_label = QtWidgets.QLabel("Delivery Drivers:")
        header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header_label)

        scroll_area = self._create_scroll_area()
        driver_container = QtWidgets.QWidget()
        driver_layout = QtWidgets.QVBoxLayout(driver_container)
        driver_layout.setSpacing(5)
        driver_layout.setContentsMargins(5, 5, 5, 5)

        self.driver_labels = {}
        for driver in self.delivery_drivers:
            driver_label = self._create_driver_label(driver)
            self.driver_labels[driver.id] = driver_label
            driver_layout.addWidget(driver_label)

        driver_layout.addStretch()
        scroll_area.setWidget(driver_container)
        layout.addWidget(scroll_area)

    def _create_scroll_area(self):
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        scroll_area.setMaximumHeight(300)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background: white;
            }
        """)
        return scroll_area

    def _create_driver_label(self, driver):
        label = ClickableLabel(
            f"Driver {driver.id}: Capacity {driver.weight_capacity}kg, {driver.volume_capacity}m³",
            driver_id=driver.id
        )
        label.doubleClicked.connect(self.on_driver_double_clicked)
        return label

    def update_driver_stats(self, solution_data):
        """
        Update driver labels with detailed statistics after optimization is complete.

        Args:
            solution_data: A dictionary containing driver statistics from the optimization
        """
        if not solution_data or not self.driver_labels:
            return

        for driver_id, stats in solution_data.items():
            if driver_id in self.driver_labels:
                driver = next((d for d in self.delivery_drivers if d.id == driver_id), None)
                if not driver:
                    continue

                hours = int(stats['travel_time'] // 3600)
                minutes = int((stats['travel_time'] % 3600) // 60)
                seconds = int(stats['travel_time'] % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                weight_pct = (stats['weight'] / driver.weight_capacity) * 100
                volume_pct = (stats['volume'] / driver.volume_capacity) * 100

                label_text = (
                    f"Driver {driver_id}: {stats['deliveries']} deliveries\n"
                    f"Time: {time_str}\n"
                    f"Distance: {stats['distance']:.2f} km\n"
                    f"Weight: {stats['weight']:.1f}/{driver.weight_capacity:.1f} kg ({weight_pct:.1f}%)\n"
                    f"Volume: {stats['volume']:.3f}/{driver.volume_capacity:.3f} m³ ({volume_pct:.1f}%)"
                )

                self.driver_labels[driver_id].setText(label_text)
                self.driver_labels[driver_id].setWordWrap(True)

                self.driver_labels[driver_id].setMinimumHeight(80)

    def on_driver_double_clicked(self, driver_id):
        """
        Handle driver selection by re-visualizing the current solution with updated selection state.
        """
        if not hasattr(self.base_map, 'current_solution') or self.base_map.current_solution is None:
            return

        self.selected_driver_id = None if self.selected_driver_id == driver_id else driver_id

        self.base_map._selected_driver_id = self.selected_driver_id

        for d_id, label in self.driver_labels.items():
            label.setSelected(d_id == self.selected_driver_id)

        if hasattr(self.base_map, 'visualization_controller'):
            current_solution = self.base_map.visualization_controller.current_solution
            unassigned = self.base_map.visualization_controller.unassigned_deliveries

            if hasattr(self.base_map.visualization_controller, 'last_visualized_solution'):
                delattr(self.base_map.visualization_controller, 'last_visualized_solution')

            from copy import deepcopy
            current_solution = deepcopy(current_solution)

            if current_solution:
                self.base_map.visualization_controller.update_visualization(
                    current_solution, unassigned
                )
