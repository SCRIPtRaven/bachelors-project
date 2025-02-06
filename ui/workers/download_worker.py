from PyQt5 import QtCore

from data.graph_manager import download_and_save_graph


class DownloadGraphWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, place_name="Kaunas, Lithuania"):
        super().__init__()
        self.place_name = place_name

    def run(self):
        try:
            download_and_save_graph(place_name=self.place_name)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, f"An error occurred while downloading data:\n{e}")
