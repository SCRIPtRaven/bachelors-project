from PyQt5 import QtCore


class ViewModel(QtCore.QObject):
    """
    Base ViewModel class that provides property change notification
    and thread safety features.
    """
    property_changed = QtCore.pyqtSignal(str)  # Signal for property changes

    def __init__(self):
        super().__init__()

    def set_property(self, name, value, old_value=None):
        """
        Sets a property and emits property_changed if the value changed.
        Returns True if the property was changed, False otherwise.
        """
        if old_value is None:
            old_value = getattr(self, name, None)

        if old_value != value:
            setattr(self, name, value)
            self.property_changed.emit(name)
            return True
        return False

    def run_async(self, func, on_result=None, on_error=None):
        """
        Runs a function asynchronously in a QtThread and calls back with the result.
        """
        worker = AsyncWorker(func)
        thread = QtCore.QThread()
        worker.moveToThread(thread)

        # Connect signals
        thread.started.connect(worker.run)
        worker.finished.connect(lambda result: self._handle_async_result(result, on_result))
        worker.error.connect(lambda error: self._handle_async_error(error, on_error))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Start the thread
        thread.start()
        return thread

    def _handle_async_result(self, result, callback):
        """Handles the result from an async operation on the main thread."""
        if callback:
            callback(result)

    def _handle_async_error(self, error, callback):
        """Handles errors from an async operation on the main thread."""
        if callback:
            callback(error)
        else:
            print(f"Unhandled async error: {error}")
            import traceback
            traceback.print_exc()


class AsyncWorker(QtCore.QObject):
    """
    Worker class that runs a function in a separate thread.
    """
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(Exception)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        """Executes the function and emits signals with the result or error."""
        try:
            result = self.func()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)
