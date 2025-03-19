import sys

from PyQt5 import QtWidgets

from app.application import Application
from view.windows.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    application = Application()
    application.initialize()

    app.application = application

    window = MainWindow()

    window.setup_viewmodels(
        application.delivery_viewmodel,
        application.driver_viewmodel
    )

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
