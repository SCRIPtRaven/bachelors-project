from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout


class Card(QFrame):
    """Enhanced card widget with shadows and hover effects"""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)

        # Setup basic appearance
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: white;
                border-radius: 10px;
                padding: 2px;
            }
        """)
        self.setFrameShape(QFrame.StyledPanel)

        # Shadow properties
        self.shadow_color = QColor(0, 0, 0, 30)
        self.shadow_radius = 15

        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Title if provided
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
            title_label.setStyleSheet("color: #424242; padding-bottom: 8px;")
            self.layout.addWidget(title_label)

            # Add separator line
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            separator.setStyleSheet("background-color: #e0e0e0;")
            separator.setMaximumHeight(1)
            self.layout.addWidget(separator)
            self.layout.addSpacing(5)

        # Content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.content)

        # Setup hover animation
        self._hover_animation = QPropertyAnimation(self, b"geometry")
        self._hover_animation.setDuration(100)
        self._hover_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._is_hovered = False

    def paintEvent(self, event):
        """Custom paint event to draw shadow"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(
            5, 5,
            self.width() - 10, self.height() - 10,
            10, 10
        )

        # Draw shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.shadow_color)
        painter.drawPath(path)

        # Draw card background
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(path)

        super().paintEvent(event)

    def enterEvent(self, event):
        """Handle mouse enter for hover effect"""
        self._is_hovered = True
        self.shadow_color = QColor(0, 0, 0, 60)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave for hover effect"""
        self._is_hovered = False
        self.shadow_color = QColor(0, 0, 0, 30)
        self.update()
        super().leaveEvent(event)


class StatsCard(Card):
    """Card optimized for displaying statistics"""

    def __init__(self, title="Statistics", parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            #card {
                background-color: white;
                border-radius: 10px;
                padding: 2px;
            }
            QLabel {
                padding: 5px;
                font-size: 14px;
            }
        """)

    def add_stat(self, label_text, value_text="", unit=""):
        """Add a statistic with label and value"""
        stat_widget = QWidget()
        stat_layout = QHBoxLayout(stat_widget)
        stat_layout.setContentsMargins(0, 5, 0, 5)

        label = QLabel(label_text)
        label.setStyleSheet("color: #757575; font-weight: normal;")

        value = QLabel(f"{value_text} {unit}".strip())
        value.setStyleSheet("color: #212121; font-weight: bold;")

        stat_layout.addWidget(label)
        stat_layout.addStretch()
        stat_layout.addWidget(value)

        self.content_layout.addWidget(stat_widget)
        return value  # Return so it can be updated later


class DriverCard(Card):
    """Card for displaying driver information"""

    def __init__(self, driver_id, parent=None):
        super().__init__(f"Driver {driver_id}", parent)
        self.driver_id = driver_id

        # Add a color indicator based on driver ID
        hue = (driver_id * 137.5) % 360  # Golden ratio to distribute colors
        self.indicator = QFrame()
        self.indicator.setFixedSize(15, 15)
        self.indicator.setStyleSheet(f"""
            background-color: hsla({hue}, 80%, 50%, 1.0);
            border-radius: 7px;
        """)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.indicator)
        header_layout.addWidget(QLabel(f"Driver {driver_id}"))
        header_layout.addStretch()

        self.content_layout.addWidget(header)
