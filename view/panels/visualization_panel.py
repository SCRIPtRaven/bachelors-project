from PyQt5 import QtCore, QtWidgets


class VisualizationPanel(QtWidgets.QWidget):
    """
    Widget that displays visualization controls and statistics.
    Works directly with VisualizationViewModel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualization_viewmodel = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.layout = QtWidgets.QVBoxLayout(self)

        # Statistics section
        self.stats_group = QtWidgets.QGroupBox("Statistics")
        self.stats_layout = QtWidgets.QVBoxLayout(self.stats_group)

        self.time_label = QtWidgets.QLabel("Time taken to compute route: N/A")
        self.travel_time_label = QtWidgets.QLabel("Total travel time: N/A")
        self.distance_label = QtWidgets.QLabel("Total distance: N/A")
        self.deliveries_label = QtWidgets.QLabel("Deliveries: N/A")

        self.stats_layout.addWidget(self.time_label)
        self.stats_layout.addWidget(self.travel_time_label)
        self.stats_layout.addWidget(self.distance_label)
        self.stats_layout.addWidget(self.deliveries_label)

        # Controls section
        self.controls_group = QtWidgets.QGroupBox("Visualization Controls")
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_group)

        self.algorithm_toggle = QtWidgets.QPushButton("Simulated Annealing")
        self.algorithm_toggle.setCheckable(True)
        self.algorithm_toggle.setStyleSheet("""
            QPushButton {
                min-height: 35px;
                min-width: 200px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                text-align: center;
            }
            QPushButton:checked {
                background-color: #2196F3;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        self.simulate_button = QtWidgets.QPushButton("Simulate Deliveries")
        self.simulate_button.setStyleSheet("""
            QPushButton {
                min-height: 35px;
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        self.controls_layout.addWidget(self.algorithm_toggle, 0, QtCore.Qt.AlignCenter)
        self.controls_layout.addWidget(self.simulate_button, 0, QtCore.Qt.AlignCenter)

        # Add to main layout
        self.layout.addWidget(self.stats_group)
        self.layout.addWidget(self.controls_group)
        self.layout.addStretch()

        # Connect signals
        self.algorithm_toggle.clicked.connect(self._on_algorithm_toggled)
        self.simulate_button.clicked.connect(self._on_simulate_clicked)

    @property
    def visualization_viewmodel(self):
        return self._visualization_viewmodel

    @visualization_viewmodel.setter
    def visualization_viewmodel(self, viewmodel):
        """Set the visualization ViewModel and connect signals"""
        self._visualization_viewmodel = viewmodel

        if viewmodel:
            # Connect signals
            viewmodel.optimization_completed.connect(self.update_statistics)
            viewmodel.property_changed.connect(self._on_viewmodel_property_changed)

            # Disable simulation button initially
            self.simulate_button.setEnabled(viewmodel.current_solution is not None)

    def _on_algorithm_toggled(self, checked):
        """Handle algorithm toggle button click"""
        if self._visualization_viewmodel:
            if checked:
                self.algorithm_toggle.setText("Greedy")
                self._visualization_viewmodel.selected_algorithm = "Greedy"
            else:
                self.algorithm_toggle.setText("Simulated Annealing")
                self._visualization_viewmodel.selected_algorithm = "Simulated Annealing"

    def _on_simulate_clicked(self):
        """Handle simulate button click"""
        if self._visualization_viewmodel:
            self._visualization_viewmodel.start_simulation()

    def _on_viewmodel_property_changed(self, property_name):
        """Handle ViewModel property changes"""
        if property_name == '_current_solution':
            # Enable/disable simulate button
            self.simulate_button.setEnabled(self._visualization_viewmodel.current_solution is not None)

            # Update statistics
            self.update_statistics()

    def update_statistics(self):
        """Update statistics display"""
        if not self._visualization_viewmodel:
            return

        stats = self._visualization_viewmodel.get_statistics()

        # Format time values
        computation_time = 0  # This would come from somewhere else
        total_travel_time = stats['total_time']
        total_distance = stats['total_distance']

        # Update labels
        self.time_label.setText(f"Time taken to compute route: {computation_time:.2f} seconds")
        self.travel_time_label.setText(f"Total travel time: {total_travel_time / 60:.2f} minutes")
        self.distance_label.setText(f"Total distance: {total_distance:.2f} km")
        self.deliveries_label.setText(
            f"Deliveries: {stats['total_deliveries']} assigned, "
            f"{stats['unassigned_deliveries']} unassigned"
        )
