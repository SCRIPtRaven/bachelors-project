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

        initial_coords, initial_zoom = get_city_coordinates("Kaunas, Lithuania")
        self.map = folium.Map(location=initial_coords, zoom_start=initial_zoom)

        self.map.save(MAP_HTML)
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(MAP_HTML))
        self.setUrl(url)

    def init_map(self, center=None, zoom=None):
        if center is None:
            center, zoom = get_city_coordinates("Kaunas, Lithuania")
        if zoom is None:
            zoom = 12

        self.map = folium.Map(location=center, zoom_start=zoom)
        self.load_map()

    def load_map(self):
        self.map.save(MAP_HTML)
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(MAP_HTML))
        self.setUrl(url)
