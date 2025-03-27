from PyQt5 import QtWidgets, QtCore, QtGui


class ActionLogDialog(QtWidgets.QDialog):
    """
    Dialog window for displaying action logs and disruption responses
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Disruption Resolution Log")
        self.setFixedSize(600, 700)
        self.setModal(False)

        if parent:
            parent_geometry = parent.geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)

        if hasattr(parent, 'map_widget') and hasattr(parent.map_widget, 'disruption_viewmodel'):
            viewmodel = parent.map_widget.disruption_viewmodel
            viewmodel.active_disruptions_changed.connect(self.update_active_disruptions)

        self.init_ui()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QtWidgets.QVBoxLayout(self)

        stats_frame = QtWidgets.QFrame()
        stats_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        stats_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        stats_layout = QtWidgets.QGridLayout(stats_frame)

        self.disruption_count_label = QtWidgets.QLabel("Disruptions: 0")
        self.action_count_label = QtWidgets.QLabel("Actions Taken: 0")
        self.recalculation_label = QtWidgets.QLabel("Recalculations: 0")
        self.time_impact_label = QtWidgets.QLabel("Time Impact: 0 min")

        stats_layout.addWidget(QtWidgets.QLabel("<b>Statistics:</b>"), 0, 0, 1, 2)
        stats_layout.addWidget(self.disruption_count_label, 1, 0)
        stats_layout.addWidget(self.action_count_label, 1, 1)
        stats_layout.addWidget(self.recalculation_label, 2, 0)
        stats_layout.addWidget(self.time_impact_label, 2, 1)

        layout.addWidget(stats_frame)

        log_label = QtWidgets.QLabel("Action Log:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(400)
        layout.addWidget(self.log_text)

        filter_frame = QtWidgets.QFrame()
        filter_layout = QtWidgets.QHBoxLayout(filter_frame)

        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["All", "Disruptions", "Reroutes", "Reassignments", "Waits", "Skips"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)

        clear_button = QtWidgets.QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        filter_layout.addWidget(clear_button)

        layout.addWidget(filter_frame)

        disruptions_label = QtWidgets.QLabel("Active Disruptions:")
        disruptions_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(disruptions_label)

        self.disruptions_list = QtWidgets.QListWidget()
        self.disruptions_list.setMaximumHeight(150)
        layout.addWidget(self.disruptions_list)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.hide)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                padding: 8px 16px;
                border-radius: 4px;
                border: 1px solid #cccccc;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)

    def add_log_entry(self, message, entry_type="action"):
        """Add an entry to the action log"""
        current_time = QtCore.QTime.currentTime().toString("HH:mm:ss")

        html_color = "#000000"
        if entry_type == "disruption":
            html_color = "#ff0000"  # Red for disruptions
        elif entry_type == "reroute":
            html_color = "#0000ff"  # Blue for reroutes
        elif entry_type == "reassign":
            html_color = "#008000"  # Green for reassignments
        elif entry_type == "wait":
            html_color = "#ffa500"  # Orange for waits
        elif entry_type == "skip":
            html_color = "#800080"  # Purple for skips

        html_message = f"<p><span style='color:#888888;'>[{current_time}]</span> " \
                       f"<span style='color:{html_color};'>{message}</span></p>"

        self.log_text.append(html_message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_stats(self, disruption_count, action_count, recalculation_count, time_impact):
        """Update the statistics display"""
        self.disruption_count_label.setText(f"Disruptions: {disruption_count}")
        self.action_count_label.setText(f"Actions Taken: {action_count}")
        self.recalculation_label.setText(f"Recalculations: {recalculation_count}")

        time_minutes = round(time_impact / 60, 1)
        self.time_impact_label.setText(f"Time Impact: {time_minutes} min")

    def update_active_disruptions(self, disruptions):
        """Update the list of active disruptions"""
        self.disruptions_list.clear()

        for disruption in disruptions:
            disruption_type = disruption.type.value.replace('_', ' ').title()
            severity = int(disruption.severity * 100)

            item_text = f"{disruption_type} (ID: {disruption.id}, Severity: {severity}%)"
            item = QtWidgets.QListWidgetItem(item_text)

            if disruption.type.value == 'traffic_jam':
                item.setForeground(QtGui.QColor("#FF5733"))
            elif disruption.type.value == 'road_closure':
                item.setForeground(QtGui.QColor("#900C3F"))
            elif disruption.type.value == 'recipient_unavailable':
                item.setForeground(QtGui.QColor("#C70039"))

            self.disruptions_list.addItem(item)

    def apply_filter(self, filter_text):
        """Apply a filter to the log display"""
        # Implementation would search through log entries
        # and show/hide based on the selected filter
        print(f"Filtering log by: {filter_text}")

    def clear_log(self):
        """Clear the action log"""
        self.log_text.clear()
