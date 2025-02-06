from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QWidget, QFrame

from ui.widgets.map_widget import MapWidget
from ui.windows.city_selector import CitySelector


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kaunas Route Planner")
        self.setFixedSize(1600, 1000)

        self.init_widgets()

        self.init_layout()

        self.connect_signals()

    def init_widgets(self):
        """Initialize all widgets but keep them hidden initially"""
        # Map widget
        self.map_widget = MapWidget(self)
        self.map_widget.hide()

        # Stats widget
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

        # Control widgets
        self.btn_load = QPushButton("Load Map Data")
        self.btn_load.setFixedHeight(50)
        self.btn_load.setStyleSheet("QPushButton { font-size: 14px; }")

        self.btn_route = QPushButton("Compute Route")
        self.btn_route.hide()

        self.delivery_input = QLineEdit()
        self.delivery_input.setPlaceholderText("Enter number of delivery points")
        self.delivery_input.hide()

        self.btn_generate_deliveries = QPushButton("Generate Deliveries")
        self.btn_generate_deliveries.hide()

        self.btn_tsp = QPushButton("Find Shortest Route")
        self.btn_tsp.hide()

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
        self.btn_route.clicked.connect(self.map_widget.compute_route)
        self.map_widget.load_completed.connect(self.show_full_ui)

    def handle_load_data(self):
        """Open city selector dialog and handle the city selection"""
        selector = CitySelector(self)
        selector.city_selected.connect(self.map_widget.load_graph_data)
        selector.exec_()

    def show_full_ui(self, success):
        """Transition to the full UI after successful data load"""
        if not success:
            return

        self.btn_load.setParent(None)

        main_content_layout = QVBoxLayout()

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.map_widget, stretch=3)
        content_layout.addWidget(self.stats_widget, stretch=1)

        button_panel = QWidget()
        button_panel_layout = QVBoxLayout(button_panel)

        route_layout = QHBoxLayout()
        route_layout.addWidget(self.btn_route)

        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(self.delivery_input)
        delivery_layout.addWidget(self.btn_generate_deliveries)
        delivery_layout.addWidget(self.btn_tsp)

        button_panel_layout.addLayout(route_layout)
        button_panel_layout.addLayout(delivery_layout)

        main_content_layout.addLayout(content_layout, stretch=1)
        main_content_layout.addWidget(button_panel)

        self.map_widget.show()
        self.stats_widget.show()
        self.btn_route.show()
        self.delivery_input.show()
        self.btn_generate_deliveries.show()
        self.btn_tsp.show()

        QWidget().setLayout(self.main_layout)
        self.main_layout = main_content_layout
        self.setLayout(self.main_layout)

        # Set up the stats labels
        self.map_widget.set_stats_labels(
            self.time_taken_label,
            self.total_travel_time_label,
            self.total_distance_label
        )

    def generate_deliveries(self):
        """Generate the requested number of delivery points"""
        num_deliveries = self.delivery_input.text()
        if not num_deliveries.isdigit():
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of delivery points.")
            return

        num_deliveries = int(num_deliveries)
        self.map_widget.generate_delivery_points(num_deliveries)
