import sys

from PyQt5 import QtWidgets

from utils.ui_styles import get_app_stylesheet
from views.windows.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(get_app_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
