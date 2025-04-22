from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QWidget, QFrame

from config.app_settings import UI_COLORS
from utils.ui_styles import get_button_style
from views.components.map.map_widget import MapWidget
from views.dialogs.city_selector import CitySelector


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("City Route Planner")

        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        width = int(screen_geometry.width() * 0.9)
        height = int(screen_geometry.height() * 0.9)
        self.resize(width, height)

        self.move(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2
        )

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        self.setMinimumSize(800, 600)

        self.init_widgets()
        self.init_layout()
        self.connect_signals()

    def init_widgets(self):
        """Initialize all widgets but keep them hidden initially"""
        self.map_widget = MapWidget(self)
        self.map_widget.hide()

        self.map_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

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
        self.btn_simulate.setStyleSheet(get_button_style('success'))

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
        """Connect all signal handlers"""
        self.btn_generate_deliveries.clicked.connect(self.generate_deliveries)
        self.btn_tsp.clicked.connect(self.map_widget.find_shortest_route)
        self.btn_load.clicked.connect(self.handle_load_data)
        self.map_widget.load_completed.connect(self.show_full_ui)
        self.btn_generate_drivers.clicked.connect(self.generate_drivers)
        self.btn_simulate.clicked.connect(self.map_widget.run_simulation)

    def handle_load_data(self):
        """Open city selector dialog and handle the city selection"""
        selector = CitySelector(self)
        selector.city_selected.connect(self.map_widget.load_graph_data)
        selector.exec_()

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

        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.addWidget(self.map_widget)
        map_layout.setContentsMargins(0, 0, 0, 0)

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
            }
            QPushButton:hover {
                background-color: #f8f8f8;
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

        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        delivery_controls = QHBoxLayout()
        self.delivery_input.setPlaceholderText("Enter number of delivery points")
        delivery_controls.addWidget(self.delivery_input, 3)
        delivery_controls.addWidget(self.btn_generate_deliveries, 1)

        driver_controls = QHBoxLayout()
        self.driver_input.setPlaceholderText("Enter number of drivers")
        driver_controls.addWidget(self.driver_input, 3)
        driver_controls.addWidget(self.btn_generate_drivers, 1)

        bottom_layout.addLayout(delivery_controls)
        bottom_layout.addLayout(driver_controls)
        bottom_layout.addWidget(self.btn_tsp)
        self.solution_switch.setFixedWidth(200)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.solution_switch, alignment=QtCore.Qt.AlignCenter)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.btn_simulate, alignment=QtCore.Qt.AlignCenter)
        self.btn_simulate.show()

        controls_layout.addWidget(bottom_section, 0)

        stats_widget = QWidget()
        stats_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
                margin: 20px;
            }
            QLabel {
                padding: 8px;
                margin: 5px;
                font-size: 14px;
                color: #333;
            }
        """)

        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(20, 20, 20, 20)

        self.time_taken_label = QLabel("Time taken to compute route: N/A")
        self.total_travel_time_label = QLabel("Total travel time: N/A")
        self.total_distance_label = QLabel("Total distance: N/A")

        stats_layout.addWidget(self.time_taken_label)
        stats_layout.addWidget(self.total_travel_time_label)
        stats_layout.addWidget(self.total_distance_label)
        stats_layout.addStretch()

        self.map_widget.set_stats_labels(
            self.time_taken_label,
            self.total_travel_time_label,
            self.total_distance_label
        )

        right_layout.addWidget(controls_widget, 1)
        right_layout.addWidget(stats_widget, 1)

        main_content_layout.addWidget(map_container, 1)
        main_content_layout.addWidget(right_panel, 0)

        QWidget().setLayout(self.main_layout)
        self.main_layout = main_content_layout
        self.setLayout(self.main_layout)

        self.map_widget.show()
        self.delivery_input.show()
        self.btn_generate_deliveries.show()
        self.btn_tsp.show()

        self.driver_input.show()
        self.btn_generate_drivers.show()

    def generate_deliveries(self):
        """Generate the requested number of delivery points"""
        success, message = self.map_widget.delivery_viewmodel.validate_and_generate_points(self.delivery_input.text())
        if not success:
            QMessageBox.warning(self, "Invalid Input", message)

    def generate_drivers(self):
        """Generate the requested number of delivery drivers"""
        success, message = self.map_widget.driver_viewmodel.validate_and_generate_drivers(self.driver_input.text())
        if not success:
            QMessageBox.warning(self, "Invalid Input", message)

    def on_solution_switch_clicked(self):
        is_greedy = self.solution_switch.isChecked()
        solution_name, button_text = self.map_widget.optimization_viewmodel.toggle_solution_view(is_greedy)
        self.solution_switch.setText(button_text)
