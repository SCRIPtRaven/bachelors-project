from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QListWidget, QVBoxLayout


class CitySelector(QDialog):
    """
    A dialog window that allows users to select a city from a predefined list.
    The window appears centered relative to its parent window.
    """
    city_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select City")
        self.setModal(True)
        self.setFixedSize(800, 600)

        if parent:
            parent_geometry = parent.geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)

        self.init_ui()

    def init_ui(self):
        """Set up the user interface components"""
        layout = QVBoxLayout(self)

        self.city_list = QListWidget()
        self.city_list.setAlternatingRowColors(True)

        self.city_list.addItem("Kaunas, Lithuania")
        self.city_list.addItem("Vilnius, Lithuania")
        self.city_list.addItem("Paris, France")

        self.city_list.addItem("Jonava, Lithuania") # For ease of testing

        self.city_list.itemDoubleClicked.connect(self.handle_selection)

        layout.addWidget(self.city_list)

    def handle_selection(self, item):
        """
        Handle the double-click event on a city item.
        First closes the dialog, then emits the signal after a short delay
        to ensure proper visual feedback.
        """
        selected_city = item.text()
        self.accept()

        QTimer.singleShot(100, lambda: self.city_selected.emit(selected_city))
