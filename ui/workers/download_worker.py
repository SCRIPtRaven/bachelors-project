from PyQt5 import QtCore

from data.graph_manager import download_and_save_graph


class DownloadGraphWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, str)

    def run(self):
        try:
            download_and_save_graph()
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, f"An error occurred while downloading data:\n{e}")
