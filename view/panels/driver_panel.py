from PyQt5 import QtWidgets

from view.components.clickable_label import ClickableLabel


class DriverPanel(QtWidgets.QWidget):
    """
    Widget that displays driver information and allows driver selection.
    Works directly with DriverViewModel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._driver_viewmodel = None
        self.driver_labels = {}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.layout = QtWidgets.QVBoxLayout(self)

        self.header_label = QtWidgets.QLabel("Delivery Drivers:")
        self.header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.layout.addWidget(self.header_label)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)
        self.scroll_area.setMaximumHeight(300)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background: white;
            }
        """)

        self.driver_container = QtWidgets.QWidget()
        self.driver_layout = QtWidgets.QVBoxLayout(self.driver_container)
        self.driver_layout.setSpacing(5)
        self.driver_layout.setContentsMargins(5, 5, 5, 5)

        self.scroll_area.setWidget(self.driver_container)
        self.layout.addWidget(self.scroll_area)

    @property
    def driver_viewmodel(self):
        return self._driver_viewmodel

    @driver_viewmodel.setter
    def driver_viewmodel(self, viewmodel):
        """Set the driver ViewModel and connect signals"""
        self._driver_viewmodel = viewmodel

        if viewmodel:
            # Connect signals
            viewmodel.drivers_changed.connect(self.update_drivers)
            viewmodel.driver_selected.connect(self.on_driver_selected)

            # Initial update
            self.update_drivers(viewmodel.drivers)

    def update_drivers(self, drivers):
        """Update the driver list display"""
        # Clear existing labels
        self._clear_driver_labels()

        # Create new labels
        for driver in drivers:
            self._create_driver_label(driver)

        # Add stretch to push labels to the top
        self.driver_layout.addStretch()

    def _clear_driver_labels(self):
        """Clear all driver labels"""
        for label in self.driver_labels.values():
            label.setParent(None)

        self.driver_labels = {}

        # Clear layout
        while self.driver_layout.count():
            item = self.driver_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_driver_label(self, driver):
        """Create a label for a driver"""
        label_text = self._driver_viewmodel.format_driver_label_text(driver)

        label = ClickableLabel(label_text, driver_id=driver.id)
        label.doubleClicked.connect(self._on_driver_double_clicked)

        # Set selection state
        label.setSelected(driver.id == self._driver_viewmodel.selected_driver_id)

        self.driver_labels[driver.id] = label
        self.driver_layout.addWidget(label)

    def _on_driver_double_clicked(self, driver_id):
        """Handle driver label double click"""
        if self._driver_viewmodel:
            self._driver_viewmodel.select_driver(driver_id)

    def on_driver_selected(self, driver_id):
        """Handle driver selection from ViewModel"""
        # Update label selection states
        for d_id, label in self.driver_labels.items():
            label.setSelected(d_id == driver_id)
