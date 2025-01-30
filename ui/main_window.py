from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QWidget

from ui.map_widget import MapWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kaunas Route Planner")
        self.setFixedSize(1000, 700)

        self.map_widget = MapWidget(self)

        # Stats widget
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)

        self.time_taken_label = QLabel("Time taken to compute route: N/A")
        self.total_travel_time_label = QLabel("Total travel time: N/A")
        self.total_distance_label = QLabel("Total distance: N/A")

        self.stats_layout.addWidget(self.time_taken_label)
        self.stats_layout.addWidget(self.total_travel_time_label)
        self.stats_layout.addWidget(self.total_distance_label)
        self.stats_layout.addStretch()

        # Buttons and line edits
        self.btn_download = QPushButton("Download Data")
        self.btn_load = QPushButton("Load Data")
        self.btn_route = QPushButton("Compute Route")

        self.delivery_input = QLineEdit()
        self.delivery_input.setPlaceholderText("Enter number of delivery points")
        self.btn_generate_deliveries = QPushButton("Generate Deliveries")

        self.btn_tsp = QPushButton("Find Shortest Route")

        # Connect signals
        self.btn_generate_deliveries.clicked.connect(self.generate_deliveries)
        self.btn_tsp.clicked.connect(self.map_widget.find_shortest_route)
        self.btn_download.clicked.connect(self.map_widget.download_graph_data)
        self.btn_load.clicked.connect(self.map_widget.load_graph_data)
        self.btn_route.clicked.connect(self.map_widget.compute_route)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_download)
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_route)

        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(self.delivery_input)
        delivery_layout.addWidget(self.btn_generate_deliveries)
        delivery_layout.addWidget(self.btn_tsp)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(delivery_layout)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.map_widget, stretch=3)
        content_layout.addWidget(self.stats_widget, stretch=1)

        main_layout.addLayout(content_layout)

        # Let map widget update stats
        self.map_widget.set_stats_labels(
            self.time_taken_label,
            self.total_travel_time_label,
            self.total_distance_label
        )

    def generate_deliveries(self):
        """
        Generate the requested number of delivery points.
        """
        num_deliveries = self.delivery_input.text()
        if not num_deliveries.isdigit():
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of delivery points.")
            return

        num_deliveries = int(num_deliveries)
        self.map_widget.generate_delivery_points(num_deliveries)
