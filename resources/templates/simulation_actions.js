// Global variables for resolver actions
let pendingActions = {};
let currentSimulationPaused = false;

// Initialize action handling
function initActionHandling() {
    console.log("Initializing action handling");

    // Set up periodic check for pending actions
    setInterval(checkPendingActions, 200);
}

// Check for pending actions from the Python resolver
function checkPendingActions() {
    if (!simulationRunning || currentSimulationPaused) return;

    // For each active driver, check if there are pending actions
    simulationDrivers.forEach(driver => {
        if (driver.isActive) {
            checkDriverActions(driver);
        }
    });
}

// Check if a specific driver has pending actions
function checkDriverActions(driver) {
    if (!window.simInterface) return;

    // Call into Python to get any pending actions
    const response = JSON.parse(window.simInterface.handleEvent(JSON.stringify({
        type: 'get_actions',
        data: {driver_id: driver.id}
    })));

    if (response.success && response.data && response.data.actions) {
        const actions = response.data.actions;
        if (actions.length > 0) {
            executeDriverActions(driver, actions);
        }
    }
}

// Execute actions for a driver
function executeDriverActions(driver, actions) {
    console.log(`Executing ${actions.length} actions for driver ${driver.id}:`, actions);

    actions.forEach(action => {
        console.log(`Executing action for driver ${driver.id}:`, action);

        switch (action.action_type) {
            case 'REROUTE':
                handleRerouteAction(driver, action);
                break;

            case 'WAIT':
                handleWaitAction(driver, action);
                break;

            case 'SKIP_DELIVERY':
                handleSkipDeliveryAction(driver, action);
                break;

            case 'PRIORITIZE_DELIVERY':
                handlePriorityAction(driver, action);
                break;

            default:
                console.warn(`Unknown action type: ${action.action_type}`);
        }
    });
}

function checkDriverNearDisruptions(driver) {
    if (!disruptionsEnabled || !activeDisruptions) return;

    const driverPosition = driver.marker.getLatLng();

    activeDisruptions.forEach(disruption => {
        // Skip if already active
        if (disruption._wasActive) return;

        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );

        const distance = driverPosition.distanceTo(disruptionPos);

        // Default activation distance is 300m if not specified
        const activationDistance = disruption.activation_distance || 300;

        // Activate if within range
        if (distance <= activationDistance) {
            notifyDisruptionActivated(disruption);
        }
    });
}


function notifyDisruptionActivated(disruption) {
    if (!window.simInterface) return;

    window.simInterface.handleEvent(JSON.stringify({
        type: 'disruption_activated',
        data: {
            disruption_id: disruption.id
        }
    }));

    console.log(`Notified Python about disruption activation: ID ${disruption.id}`);
}

// Handle a reroute action
function handleRerouteAction(driver, action) {
    // Pause the driver
    const wasActive = driver.isActive;
    driver.isActive = false;

    // Store the original route and create a visual representation
    const originalPath = [...driver.path];

    // Draw the original route in faded color
    if (originalPath.length > 1) {
        const originalPolyline = L.polyline(originalPath, {
            color: driver.color,
            weight: 3,
            opacity: 0.4,
            dashArray: '5, 5'
        }).addTo(layers.routes);

        // Store the original route visualization
        if (!driver.originalRoutes) driver.originalRoutes = [];
        driver.originalRoutes.push(originalPolyline);

        // Remove old original routes after a limit (to avoid clutter)
        if (driver.originalRoutes.length > 3) {
            const oldRoute = driver.originalRoutes.shift();
            if (oldRoute && layers.routes) {
                layers.routes.removeLayer(oldRoute);
            }
        }
    }

    // Update the driver's route for movement
    driver.path = action.new_route;

    // Create a visual representation of the new route
    if (action.new_route && action.new_route.length > 1) {
        // Create a brighter color for the new route
        const newColor = getBrightVariant(driver.color);

        // Draw the new route
        const newPolyline = L.polyline(action.new_route, {
            color: newColor,
            weight: 4,
            opacity: 0.9
        }).addTo(layers.routes);

        // Store new route visualization
        driver.currentRoutePolyline = newPolyline;
    }

    // Reset driver progress on the current segment
    driver.currentIndex = 0;
    driver.elapsedOnSegment = 0;

    // Resume the driver
    if (wasActive) {
        driver.isActive = true;
    }

    // Notify Python the route has been updated
    notifyDriverPositionUpdate(driver);

    // Show visual feedback
    showActionFeedback(driver, 'Route Recalculated', 'blue');
}

// Helper function to get a brighter variant of a color
function getBrightVariant(color) {
    // Default to a bright color if we can't parse the input
    if (!color || typeof color !== 'string') {
        return '#ff5500';  // Bright orange as fallback
    }

    // Handle different color formats
    let hexColor;
    if (color.startsWith('#')) {
        hexColor = color;
    } else if (color.startsWith('rgb')) {
        // Parse rgb() format
        const rgbMatch = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
        if (rgbMatch) {
            hexColor = '#' +
                parseInt(rgbMatch[1]).toString(16).padStart(2, '0') +
                parseInt(rgbMatch[2]).toString(16).padStart(2, '0') +
                parseInt(rgbMatch[3]).toString(16).padStart(2, '0');
        } else {
            return '#ff5500';  // Fallback
        }
    } else {
        // Can't parse, use fallback
        return '#ff5500';
    }

    // Brighten the color to make it distinct
    try {
        // Remove # if present
        let hex = hexColor.replace('#', '');

        // Parse the hex values
        let r = parseInt(hex.substr(0, 2), 16);
        let g = parseInt(hex.substr(2, 2), 16);
        let b = parseInt(hex.substr(4, 2), 16);

        // Create a significantly different but related color
        // Shift the hue by rotating the color components
        let temp = r;
        r = Math.min(255, g + 50);
        g = Math.min(255, b + 50);
        b = Math.min(255, temp + 50);

        // Convert back to hex
        return '#' +
            r.toString(16).padStart(2, '0') +
            g.toString(16).padStart(2, '0') +
            b.toString(16).padStart(2, '0');
    } catch (e) {
        console.error("Error creating color variant:", e);
        return '#ff5500';  // Fallback
    }
}

// Handle a wait action
function handleWaitAction(driver, action) {
    // Pause the driver for the specified time
    const wasActive = driver.isActive;
    driver.isActive = false;

    // Show visual feedback
    const minutes = Math.round(action.wait_time / 60);
    showActionFeedback(driver, `Waiting ${minutes} minutes`, 'orange');

    // Schedule driver to resume after the wait time
    const simulationWaitTime = action.wait_time / simulationSpeed;
    setTimeout(() => {
        if (wasActive && simulationRunning) {
            driver.isActive = true;
            showActionFeedback(driver, 'Resuming', 'green');
        }
    }, simulationWaitTime * 1000);
}

// Handle a skip delivery action
function handleSkipDeliveryAction(driver, action) {
    // Find the delivery index in the path
    const deliveryIndex = action.delivery_index;

    // Mark as skipped in the driver's visited list
    if (!driver.skipped) driver.skipped = [];
    driver.skipped.push(deliveryIndex);

    // Show visual feedback
    showActionFeedback(driver, 'Skipping Delivery', 'red');

    // Notify Python
    notifyDeliveryFailed(driver, deliveryIndex);
}

// Handle a priority change action
function handlePriorityAction(driver, action) {
    // This would reorder the driver's delivery sequence
    // For now just show visual feedback
    showActionFeedback(driver, 'Delivery Order Changed', 'purple');
}

// Show visual feedback for an action
function showActionFeedback(driver, message, color) {
    // Create a popup at the driver's position
    const position = driver.marker.getLatLng();

    const popup = L.popup({
        autoClose: true,
        closeOnClick: true,
        autoPan: false,
        closeButton: false
    })
        .setLatLng(position)
        .setContent(`<div style="text-align:center; color:${color}"><strong>Driver ${driver.id}</strong><br>${message}</div>`)
        .openOn(map);

    // Auto-close after 3 seconds
    setTimeout(() => {
        map.closePopup(popup);
    }, 3000);
}

// Notify Python about driver position update
function notifyDriverPositionUpdate(driver) {
    if (!window.simInterface) return;

    const position = driver.marker.getLatLng();

    window.simInterface.handleEvent(JSON.stringify({
        type: 'driver_position_updated',
        data: {
            driver_id: driver.id,
            lat: position.lat,
            lon: position.lng
        }
    }));
}

// Notify Python about delivery failure
function notifyDeliveryFailed(driver, deliveryIndex) {
    if (!window.simInterface) return;

    window.simInterface.handleEvent(JSON.stringify({
        type: 'delivery_failed',
        data: {
            driver_id: driver.id,
            delivery_index: deliveryIndex
        }
    }));
}

// Notify Python about disruption activation
function notifyDisruptionActivated(disruption) {
    if (!window.simInterface) return;

    // Mark as active locally
    disruption._wasActive = true;

    // Notify Python
    window.simInterface.handleEvent(JSON.stringify({
        type: 'disruption_activated',
        data: {
            disruption_id: disruption.id
        }
    }));

    console.log(`Notified Python about disruption activation: ID ${disruption.id}`);
}

// Pause the simulation
function pauseSimulation() {
    if (!simulationRunning) return;

    currentSimulationPaused = true;

    // Notify Python
    if (window.simInterface) {
        window.simInterface.handleEvent(JSON.stringify({
            type: 'simulation_paused',
            data: {}
        }));
    }
}

// Resume the simulation
function resumeSimulation() {
    if (!simulationRunning || !currentSimulationPaused) return;

    currentSimulationPaused = false;

    // Notify Python
    if (window.simInterface) {
        window.simInterface.handleEvent(JSON.stringify({
            type: 'simulation_resumed',
            data: {}
        }));
    }
}

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', function () {
    // Add initialization to the global map init
    const oldInitMap = window.initMap;

    window.initMap = function (centerLat, centerLon, zoomLevel) {
        oldInitMap(centerLat, centerLon, zoomLevel);
        initActionHandling();
    };

    // Hook into the updateSimulation function to handle pausing
    const oldUpdateSimulation = window.updateSimulation;

    window.updateSimulation = function () {
        // Skip updates if paused
        if (currentSimulationPaused) return;

        // Call the original function
        oldUpdateSimulation();
    };

    // Hook into disruption activation
    const oldUpdateDisruptionVisibility = window.updateDisruptionVisibility;

    window.updateDisruptionVisibility = function (simulationTime) {
        const previouslyActive = activeDisruptions.map(d => d.id);

        // Call the original function
        oldUpdateDisruptionVisibility(simulationTime);

        // Check for newly activated disruptions
        activeDisruptions.forEach(disruption => {
            const isActive = disruption.start_time <= simulationTime &&
                (disruption.start_time + disruption.duration) >= simulationTime;

            if (isActive && !previouslyActive.includes(disruption.id)) {
                // This disruption was just activated
                notifyDisruptionActivated(disruption);
            }
        });
    };
});