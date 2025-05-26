from PyQt5 import QtCore


class DisruptionGenerationWorker(QtCore.QObject):
    generation_complete = QtCore.pyqtSignal(list)
    generation_failed = QtCore.pyqtSignal(str)
    
    def __init__(self, disruption_service, num_drivers):
        super().__init__()
        self.disruption_service = disruption_service
        self.num_drivers = num_drivers
    
    @QtCore.pyqtSlot()
    def generate_disruptions(self):
        try:
            print(f"DisruptionGenerationWorker: Starting disruption generation for {self.num_drivers} drivers")
            
            new_disruptions = self.disruption_service.generate_disruptions(
                self.num_drivers
            )
            
            print(f"DisruptionGenerationWorker: Generated {len(new_disruptions)} disruptions")
            print(f"DisruptionGenerationWorker: Disruptions list: {[d.id if hasattr(d, 'id') else str(d) for d in new_disruptions]}")
            self.generation_complete.emit(new_disruptions)
            
        except Exception as e:
            print(f"DisruptionGenerationWorker: Error generating disruptions: {e}")
            import traceback
            traceback.print_exc()
            self.generation_failed.emit(str(e)) 