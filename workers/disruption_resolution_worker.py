from PyQt5 import QtCore


class DisruptionResolutionWorker(QtCore.QObject):
    """Worker class that handles disruption resolution in a separate thread"""
    resolution_complete = QtCore.pyqtSignal(list)  # Emits list of actions
    log_message = QtCore.pyqtSignal(str, str)  # For logging
    resolution_progress = QtCore.pyqtSignal(str)  # For progress updates

    def __init__(self, simulation_controller, resolver, disruption):
        super().__init__()
        self.simulation_controller = simulation_controller
        self.resolver = resolver
        self.disruption = disruption

    @QtCore.pyqtSlot()
    def resolve_disruption(self):
        """Resolve the disruption in the worker thread"""
        try:
            # Log start of resolution
            self.log_message.emit(
                f"Starting resolution for {self.disruption.type.value} disruption (ID: {self.disruption.id})",
                "action"
            )

            # Get current state safely
            state = self.simulation_controller.get_current_state()

            # Check if resolution is needed
            should_recalc = self.resolver.should_recalculate(state, self.disruption)
            if not should_recalc:
                self.log_message.emit(
                    f"Skipping resolution - recalculation not needed",
                    "action"
                )
                self.resolution_complete.emit([])
                return

            # Let resolver determine actions
            actions = self.resolver.on_disruption_detected(self.disruption, state)

            # Log resolution results
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

            # Emit results back to main thread
            action_copies = [self._create_action_copy(action) for action in actions]
            self.resolution_complete.emit(action_copies)

        except Exception as e:
            error_msg = f"Error in disruption resolution: {str(e)}"
            print(error_msg)
            self.log_message.emit(error_msg, "action")
            import traceback
            traceback.print_exc()
            # Emit empty list if something goes wrong
            self.resolution_complete.emit([])

    def _create_action_copy(self, action):
        """Create a safe copy of an action using the action's own serialization methods"""
        from models.rl.actions import DisruptionAction

        try:
            # Use the action's own serialization mechanism
            if hasattr(action, 'to_dict'):
                action_dict = action.to_dict()
                return DisruptionAction.from_dict(action_dict)
            else:
                # If serialization isn't available, do a type-specific copy
                return self._manual_copy_action(action)
        except Exception as e:
            print(f"Error copying action: {e}")
            # Return the original as a last resort
            return action

    def _manual_copy_action(self, action):
        """Manually copy an action based on its type"""
        from models.rl.actions import RerouteAction, ReassignDeliveriesAction, WaitAction
        from models.rl.actions import SkipDeliveryAction, PrioritizeDeliveryAction, NoAction

        if isinstance(action, RerouteAction):
            return RerouteAction(
                driver_id=action.driver_id,
                new_route=list(action.new_route) if action.new_route else [],
                affected_disruption_id=action.affected_disruption_id
            )
        elif isinstance(action, ReassignDeliveriesAction):
            return ReassignDeliveriesAction(
                from_driver_id=action.from_driver_id,
                to_driver_id=action.to_driver_id,
                delivery_indices=list(action.delivery_indices) if action.delivery_indices else []
            )
        elif isinstance(action, WaitAction):
            return WaitAction(
                driver_id=action.driver_id,
                wait_time=action.wait_time,
                disruption_id=action.disruption_id
            )
        elif isinstance(action, SkipDeliveryAction):
            return SkipDeliveryAction(
                driver_id=action.driver_id,
                delivery_index=action.delivery_index
            )
        elif isinstance(action, PrioritizeDeliveryAction):
            return PrioritizeDeliveryAction(
                driver_id=action.driver_id,
                delivery_indices=list(action.delivery_indices) if action.delivery_indices else []
            )
        elif isinstance(action, NoAction):
            return NoAction()
        else:
            print(f"Unknown action type: {type(action).__name__}")
            return action
