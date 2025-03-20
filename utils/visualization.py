from PyQt5 import QtCore


class VisualizationQueue:
    def __init__(self, optimization_viewmodel):
        self.queue = []
        self.controller = optimization_viewmodel

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(100)

    def append(self, item):
        self.queue.append(item)

    def process_queue(self):
        if not self.queue:
            return

        solution, unassigned = self.queue[-1]
        self.queue.clear()

        try:
            self.controller.update_visualization(solution, unassigned)
        except Exception as e:
            print(f"Error in visualization update: {e}")
