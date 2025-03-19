def bind_property(view_widget, property_name, view_model, vm_property_name):
    """
    Binds a property of a view widget to a property of a view model.

    Args:
        view_widget: The widget to bind
        property_name: The name of the property on the widget
        view_model: The ViewModel instance
        vm_property_name: The name of the property on the ViewModel
    """
    # Initial update from ViewModel to View
    if hasattr(view_model, vm_property_name):
        value = getattr(view_model, vm_property_name)
        if hasattr(view_widget, 'set' + property_name[0].upper() + property_name[1:]):
            # Use setter method if available (e.g., setText for text property)
            setter = getattr(view_widget, 'set' + property_name[0].upper() + property_name[1:])
            setter(value)
        else:
            # Otherwise set property directly
            setattr(view_widget, property_name, value)

    # Connect ViewModel property_changed signal to update View
    def update_view(changed_property):
        if changed_property == vm_property_name:
            value = getattr(view_model, vm_property_name)
            if hasattr(view_widget, 'set' + property_name[0].upper() + property_name[1:]):
                setter = getattr(view_widget, 'set' + property_name[0].upper() + property_name[1:])
                setter(value)
            else:
                setattr(view_widget, property_name, value)

    view_model.property_changed.connect(update_view)

    # If the widget is an input widget, connect its signals to update ViewModel
    if hasattr(view_widget, 'textChanged'):
        view_widget.textChanged.connect(
            lambda text: setattr(view_model, vm_property_name, text))
    elif hasattr(view_widget, 'valueChanged'):
        view_widget.valueChanged.connect(
            lambda value: setattr(view_model, vm_property_name, value))
    elif hasattr(view_widget, 'toggled'):
        view_widget.toggled.connect(
            lambda checked: setattr(view_model, vm_property_name, checked))
