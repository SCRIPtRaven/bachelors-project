from PyQt5 import QtCore


class DisruptionResolutionWorker(QtCore.QObject):
    """Worker class that handles disruption resolution in a separate thread"""
    resolution_complete = QtCore.pyqtSignal(list)
    log_message = QtCore.pyqtSignal(str, str)
    resolution_progress = QtCore.pyqtSignal(str)

    def __init__(self, simulation_controller, resolver, disruption):
        super().__init__()
        self.simulation_controller = simulation_controller
        self.resolver = resolver
        self.disruption = disruption

    @QtCore.pyqtSlot()
    def resolve_disruption(self):
        """Resolve the disruption in the worker thread"""
        try:
            self.log_message.emit(
                f"Starting resolution for {self.disruption.type.value} disruption (ID: {self.disruption.id})",
                "action"
            )

            state = self.simulation_controller.get_current_state()

            should_recalc = self.resolver.should_recalculate(state, self.disruption)
            if not should_recalc:
                self.log_message.emit(
                    f"Skipping resolution - recalculation not needed",
                    "action"
                )
                self.resolution_complete.emit([])
                return

            actions = self.resolver.on_disruption_detected(self.disruption, state)

            if actions:
                self.log_message.emit(
                    f"Found {len(actions)} resolution actions",
                    "action"
                )
            else:
                self.log_message.emit(
                    "No resolution actions needed",
                    "action"
                )

            action_copies = [self._manual_copy_action(action) for action in actions]
            self.resolution_complete.emit(action_copies)

        except Exception as e:
            error_msg = f"Error in disruption resolution: {str(e)}"
            print(error_msg)
            self.log_message.emit(error_msg, "action")
            import traceback
            traceback.print_exc()
            self.resolution_complete.emit([])

    def _manual_copy_action(self, action):
        """Manually copy an action based on its type"""
        from models.resolvers.actions import RerouteAction
        from models.resolvers.actions import RecipientUnavailableAction

        if isinstance(action, RerouteAction):
            return RerouteAction(
                driver_id=action.driver_id,
                new_route=list(action.new_route) if action.new_route else [],
                affected_disruption_id=action.affected_disruption_id,
                rerouted_segment_start=action.rerouted_segment_start,
                rerouted_segment_end=action.rerouted_segment_end,
                next_delivery_index=action.next_delivery_index,
                delivery_indices=list(action.delivery_indices) if action.delivery_indices else []
            )
        elif isinstance(action, RecipientUnavailableAction):
            return RecipientUnavailableAction(
                driver_id=action.driver_id,
                delivery_index=action.delivery_index,
                disruption_id=action.disruption_id,
                duration=action.duration
            )
        else:
            print(f"Unknown action type: {type(action).__name__}")
            return action
