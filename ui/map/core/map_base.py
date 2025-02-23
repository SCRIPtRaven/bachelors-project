import os

import folium
from PyQt5 import QtCore, QtWebEngineWidgets

from config.paths import MAP_HTML
from utils.geolocation import get_city_coordinates


class BaseMapWidget(QtWebEngineWidgets.QWebEngineView):
    load_completed = QtCore.pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.map = None
        self.G = None
        self.current_city = None

    def init_map(self, center=None, zoom=None):
        """Initialize or reinitialize the map with given parameters"""
        if center is None or zoom is None:
            center, zoom = get_city_coordinates("Kaunas, Lithuania")

        self.map = folium.Map(location=center, zoom_start=zoom)
        self.load_map()

    def load_map(self):
        self.map.save(MAP_HTML)
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(MAP_HTML))
        self.setUrl(url)
