from PyQt5 import QtCore, QtWidgets


class ClickableLabel(QtWidgets.QLabel):
    doubleClicked = QtCore.pyqtSignal(int)

    def __init__(self, text, driver_id, parent=None):
        super().__init__(text, parent)
        self.driver_id = driver_id
        self.selected = False
        self.setStyleSheet("padding: 5px;")
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.TextFormat.RichText)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.driver_id)
        super().mouseDoubleClickEvent(event)

    def enterEvent(self, event):
        if not self.selected:
            self.setStyleSheet("padding: 5px; background-color: #e0f7fa;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.selected:
            self.setStyleSheet("padding: 5px;")
        super().leaveEvent(event)

    def setSelected(self, selected):
        self.selected = selected
        if self.selected:
            self.setStyleSheet("padding: 5px; background-color: #80deea; font-weight: bold;")
        else:
            self.setStyleSheet("padding: 5px;")
