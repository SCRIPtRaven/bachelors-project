from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QWidget, QFrame

from view.map.map_manager import MapManager
from view.map.map_widget import MapWidget
from view.panels.driver_panel import DriverPanel
from view.panels.visualization_panel import VisualizationPanel
from view.windows.city_selector import CitySelector


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        from PyQt5 import QtWebEngineWidgets
        QtWebEngineWidgets.QWebEngineProfile.defaultProfile()

        self.setWindowTitle("Kaunas Route Planner")
        self.setFixedSize(1400, 800)

        self.delivery_viewmodel = None
        self.driver_viewmodel = None
        self.visualization_viewmodel = None
        self.map_viewmodel = None
        self.map_manager = None

        self.fully_initialized = False

        self.init_widgets()
        self.init_layout()

        QtCore.QTimer.singleShot(0, self.connect_signals)

    def setup_viewmodels(self, delivery_viewmodel, driver_viewmodel):
        """Set up ViewModels and connect signals."""
        # Store the view models
        self.delivery_viewmodel = delivery_viewmodel
        self.driver_viewmodel = driver_viewmodel

        # Get the application instance for other ViewModels
        if hasattr(QtCore.QCoreApplication.instance(), 'application'):
            app = QtCore.QCoreApplication.instance().application
            self.visualization_viewmodel = app.visualization_viewmodel
            self.map_viewmodel = app.map_viewmodel

        # Set ViewModels in components
        self.map_widget.delivery_viewmodel = self.delivery_viewmodel
        self.map_widget.driver_viewmodel = self.driver_viewmodel
        self.map_widget.visualization_viewmodel = self.visualization_viewmodel

        self.driver_panel.driver_viewmodel = self.driver_viewmodel
        self.visualization_panel.visualization_viewmodel = self.visualization_viewmodel

        # Create the map manager to connect map and ViewModels
        self.map_manager = MapManager(
            self.map_widget,
            self.delivery_viewmodel,
            self.driver_viewmodel,
            self.visualization_viewmodel,
            self.map_viewmodel
        )

        self.fully_initialized = True

    def toggle_solution_algorithm(self):
        """Toggle between different solution algorithms."""
        if self.solution_switch.isChecked():
            self.solution_switch.setText("Greedy")
            self.visualization_viewmodel.selected_algorithm = "Greedy"
        else:
            self.solution_switch.setText("Simulated Annealing")
            self.visualization_viewmodel.selected_algorithm = "Simulated Annealing"

    def init_widgets(self):
        """Initialize all widgets but keep them hidden initially"""
        self.map_widget = MapWidget(self)
        self.map_widget.hide()

        self.driver_panel = DriverPanel()
        self.visualization_panel = VisualizationPanel()

        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)
        self.stats_widget.hide()

        self.time_taken_label = QLabel("Time taken to compute route: N/A")
        self.total_travel_time_label = QLabel("Total travel time: N/A")
        self.total_distance_label = QLabel("Total distance: N/A")

        self.stats_layout.addWidget(self.time_taken_label)
        self.stats_layout.addWidget(self.total_travel_time_label)
        self.stats_layout.addWidget(self.total_distance_label)
        self.stats_layout.addStretch()

        self.btn_load = QPushButton("Load Map Data")
        self.btn_load.setFixedHeight(50)
        self.btn_load.setStyleSheet("QPushButton { font-size: 14px; }")

        self.delivery_input = QLineEdit()
        self.delivery_input.setPlaceholderText("Enter number of delivery points")
        self.delivery_input.hide()

        self.btn_generate_deliveries = QPushButton("Generate Deliveries")
        self.btn_generate_deliveries.hide()

        self.btn_tsp = QPushButton("Find Shortest Route")
        self.btn_tsp.hide()

        self.driver_input = QLineEdit()
        self.driver_input.setPlaceholderText("Enter number of drivers")
        self.driver_input.hide()

        self.btn_generate_drivers = QPushButton("Generate Drivers")
        self.btn_generate_drivers.hide()

        self.btn_simulate = QPushButton("Simulate Deliveries")
        self.btn_simulate.setEnabled(False)
        self.btn_simulate.hide()
        self.btn_simulate.setStyleSheet("""
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

        self.solution_switch = QtWidgets.QPushButton("Simulated Annealing")
        self.solution_switch.setCheckable(True)
        self.solution_switch.setEnabled(False)
        self.solution_switch.clicked.connect(self.on_solution_switch_clicked)
        self.solution_switch.hide()
        self.solution_switch.setStyleSheet("""
                QPushButton {
                    min-height: 35px;
                    min-width: 200px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 15px;
                    text-align: center;
                    transition: background-color 0.3s;
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

    def init_layout(self):
        """Set up the initial layout with just the load button"""
        self.main_layout = QVBoxLayout(self)

        bottom_frame = QFrame()
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.addWidget(self.btn_load)
        bottom_layout.setContentsMargins(20, 0, 20, 20)

        self.main_layout.addStretch()
        self.main_layout.addWidget(bottom_frame)

    def connect_signals(self):
        """Connect signals after initialization"""
        self.btn_load.clicked.connect(self.handle_load_data)
        self.map_widget.load_completed.connect(self.show_full_ui)
        self.btn_simulate.clicked.connect(self.start_simulation)
        self.solution_switch.clicked.connect(self.toggle_solution_algorithm)
        self.btn_generate_deliveries.clicked.connect(self.generate_deliveries)
        self.btn_generate_drivers.clicked.connect(self.generate_drivers)
        self.btn_tsp.clicked.connect(self.map_manager.find_shortest_route)

        self.delivery_viewmodel.generation_completed.connect(self.on_delivery_generation_completed)
        self.driver_viewmodel.generation_completed.connect(self.on_driver_generation_completed)

    def handle_load_data(self):
        """Safely handle map data loading"""
        if not self.fully_initialized:
            QtWidgets.QMessageBox.information(
                self,
                "Please Wait",
                "Application initialization is still in progress. Please try again in a moment."
            )
            return

        selector = CitySelector(self)
        selector.city_selected.connect(self.on_city_selected)
        selector.exec_()

    def on_city_selected(self, city_name):
        """Handle city selection in a thread-safe way"""
        if hasattr(self, 'map_manager') and self.map_manager is not None:
            self.map_manager.load_city(city_name)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Not Ready",
                "Map manager not initialized. Please wait a moment and try again."
            )

    def on_graph_loaded(self, graph):
        """Update the view model with the loaded graph"""
        self.delivery_viewmodel.set_graph(graph)

    def show_full_ui(self, success):
        """
        Creates the main application interface after successful data loading.
        The interface consists of a map panel on the left and a control/stats panel on the right.
        """
        if not success:
            return

        self.btn_load.setParent(None)

        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(0)

        # Map container (left side)
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.addWidget(self.map_widget)
        map_layout.setContentsMargins(0, 0, 0, 0)

        # Right panel container
        right_panel = QWidget()
        right_panel.setFixedWidth(400)
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #E5E5E5;
            }
        """)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Controls section
        controls_widget = QWidget()
        controls_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
                margin: 20px;
            }
            QPushButton {
                min-height: 35px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 15px;
                box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #f8f8f8;
                box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.15);
            }
            QLineEdit {
                min-height: 35px;
                padding: 0 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 13px;
            }
        """)

        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # Input controls
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Delivery controls
        delivery_controls = QHBoxLayout()
        self.delivery_input.setPlaceholderText("Enter number of delivery points")
        delivery_controls.addWidget(self.delivery_input, 3)
        delivery_controls.addWidget(self.btn_generate_deliveries, 1)

        # Driver controls
        driver_controls = QHBoxLayout()
        self.driver_input.setPlaceholderText("Enter number of drivers")
        driver_controls.addWidget(self.driver_input, 3)
        driver_controls.addWidget(self.btn_generate_drivers, 1)

        # Add all controls to the layout
        bottom_layout.addLayout(delivery_controls)
        bottom_layout.addLayout(driver_controls)
        bottom_layout.addWidget(self.btn_tsp)
        self.solution_switch.setFixedWidth(200)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.solution_switch, alignment=QtCore.Qt.AlignCenter)

        # Add everything to the controls layout
        controls_layout.addWidget(bottom_section, 0)

        # Build the right panel layout - IMPORTANT: Structure changes
        right_layout.addWidget(controls_widget, 0)  # Controls at the top with fixed size
        right_layout.addWidget(self.driver_panel, 1)  # Driver panel with stretch
        right_layout.addWidget(self.visualization_panel, 0)  # Visualization panel with fixed size

        # Assemble the main layout
        main_content_layout.addWidget(map_container, 1)
        main_content_layout.addWidget(right_panel, 0)

        # Replace the current layout
        QWidget().setLayout(self.main_layout)
        self.main_layout = main_content_layout
        self.setLayout(self.main_layout)

        # Show all widgets that were hidden
        self.map_widget.show()
        self.delivery_input.show()
        self.btn_generate_deliveries.show()
        self.btn_tsp.show()
        self.driver_input.show()
        self.btn_generate_drivers.show()
        self.solution_switch.show()

        # Enable the buttons that should be enabled after loading
        self.solution_switch.setEnabled(True)

    def generate_deliveries(self):
        """Generate the requested number of delivery points"""
        num_deliveries = self.delivery_input.text()
        if not num_deliveries.isdigit():
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of delivery points.")
            return

        num_deliveries = int(num_deliveries)

        self.delivery_viewmodel.generate_delivery_points(num_deliveries)

        self.btn_generate_deliveries.setEnabled(False)
        self.btn_generate_deliveries.setText("Generating...")

    def on_deliveries_changed(self, delivery_points):
        """Update the UI when delivery points change"""
        self.map_widget.add_delivery_points(delivery_points)

    def on_delivery_generation_completed(self, success, message):
        """Handle completion of delivery point generation"""
        self.btn_generate_deliveries.setEnabled(True)
        self.btn_generate_deliveries.setText("Generate Deliveries")

        if success:
            if "Skipped" in message:
                QMessageBox.information(self, "Deliveries Generated", message)
        else:
            QMessageBox.warning(self, "Generation Error", message)

    def generate_drivers(self):
        """Generate the requested number of delivery drivers using the ViewModel."""
        num_drivers = self.driver_input.text()
        if not num_drivers.isdigit():
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of drivers.")
            return

        num_drivers = int(num_drivers)

        # Use the view model instead of directly calling the map widget
        self.driver_viewmodel.generate_drivers(num_drivers)

        # Disable the button during generation
        self.btn_generate_drivers.setEnabled(False)
        self.btn_generate_drivers.setText("Generating...")

    def on_drivers_changed(self, drivers):
        """Handler for when the list of drivers changes."""
        # For now, we'll still use the existing map widget's methods
        # Later we'll refactor this to work directly with the ViewModel
        self.map_widget.update_drivers(drivers)

    def on_driver_generation_completed(self, success, message):
        """Handler for when driver generation completes."""
        self.btn_generate_drivers.setEnabled(True)
        self.btn_generate_drivers.setText("Generate Drivers")

        if success:
            QMessageBox.information(self, "Drivers Generated", message)
        else:
            QMessageBox.warning(self, "Generation Error", message)

    def on_driver_selected(self, driver_id):
        """Handler for when a driver is selected."""
        # This will be used to update the UI when a driver is selected
        self.map_widget.highlight_driver_route(driver_id)

    def on_solution_switch_clicked(self):
        """Handle solution algorithm switch"""
        if not hasattr(self, 'visualization_viewmodel'):
            return

        if self.solution_switch.isChecked():
            self.solution_switch.setText("Greedy")
            self.visualization_viewmodel.selected_algorithm = "Greedy"
        else:
            self.solution_switch.setText("Simulated Annealing")
            self.visualization_viewmodel.selected_algorithm = "Simulated Annealing"

    def find_shortest_route(self):
        """Start the optimization process using the MapManager"""
        if hasattr(self, 'map_manager'):
            self.map_manager.find_shortest_route()
        else:
            QMessageBox.warning(self, "Not Ready", "Map manager not initialized. Please wait for map to load.")

    def start_simulation(self):
        """Start simulation using the VisualizationViewModel"""
        if hasattr(self, 'visualization_viewmodel'):
            self.visualization_viewmodel.start_simulation()
        else:
            QMessageBox.warning(self, "Not Ready", "Visualization viewmodel not initialized.")
