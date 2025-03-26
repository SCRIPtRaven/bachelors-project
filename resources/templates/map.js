let map;
let mapInitialized = false;
let layers = {
    warehouse: L.layerGroup(),
    deliveries: L.layerGroup(),
    routes: L.layerGroup(),
    unassigned: L.layerGroup(),
    drivers: L.layerGroup(),
    disruptions: L.layerGroup()
};

let simulationRunning = false;
let simulationDrivers = [];
let simulationInterval = null;
let simulationSpeed = 5.0;

let simulationTime = 8 * 60 * 60; // Start at 8:00 AM in seconds
let clockElement = null;
let lastClockUpdateTime = Date.now();

let totalExpectedTravelTime = 0;
let initialExpectedCompletionTime = 0;

let targetSimulationTime = 8 * 60 * 60;
let timeSmoothing = true;

let disruptionsEnabled = true;
let disruptionData = [];

function initMap(centerLat, centerLon, zoomLevel) {
    map = L.map('map').setView([centerLat, centerLon], zoomLevel);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    mapInitialized = true;

    for (const key in layers) {
        layers[key].addTo(map);
    }

    new QWebChannel(qt.webChannelTransport, function (channel) {
        window.mapHandler = channel.objects.mapHandler;
        mapHandler.handleEvent("map_initialized");
    });
}

function updateClock() {
    if (!clockElement || !clockElement._container) return;

    const hours = Math.floor(simulationTime / 3600) % 24;
    const minutes = Math.floor((simulationTime % 3600) / 60);
    const seconds = Math.floor(simulationTime % 60);

    const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    clockElement._container.innerHTML = `<strong>Simulation Time:</strong> ${timeString}`;
}

function loadDisruptions(data) {
    disruptionData = data;
    if (typeof window.loadDisruptions === 'function') {
        window.loadDisruptions(data);
    }
}

function showLoadingIndicator() {
    document.getElementById('loading-indicator').style.display = 'block';
    map.dragging.disable();
    map.doubleClickZoom.disable();
    map.scrollWheelZoom.disable();
}

function hideLoadingIndicator() {
    document.getElementById('loading-indicator').style.display = 'none';
    map.dragging.enable();
    map.doubleClickZoom.enable();
    map.scrollWheelZoom.enable();
}

function getDriverDelayFactor(driver) {
    if (!disruptionsEnabled || !activeDisruptions.length)
        return 1.0;

    const driverPosition = driver.marker.getLatLng();
    let slowestFactor = 1.0;

    activeDisruptions.forEach(disruption => {
        // Skip disruptions that aren't active at current simulation time
        if (disruption.start_time > simulationTime ||
            disruption.start_time + disruption.duration < simulationTime)
            return;

        // Calculate distance between driver and disruption center
        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );
        const distance = driverPosition.distanceTo(disruptionPos);

        // Check if driver is within the affected radius
        if (distance <= disruption.radius) {
            // Different disruption types have different effects on speed
            let factor = 1.0;
            switch (disruption.type) {
                case 'traffic_jam':
                    // Traffic jams slow vehicles proportional to severity
                    factor = Math.max(0.2, 1.0 - disruption.severity);
                    break;
                case 'road_closure':
                    // Road closures almost completely block movement
                    factor = 0.1;
                    break;
                case 'vehicle_breakdown':
                    // Breakdowns only affect the specific driver
                    if (disruption.affected_driver_ids &&
                        disruption.affected_driver_ids.includes(driver.id)) {
                        factor = 0.2; // Almost stopped
                    }
                    break;
                default:
                    // Other disruption types have a moderate effect
                    factor = Math.max(0.5, 1.0 - disruption.severity);
            }

            // Multiple disruptions stack by using the most severe effect
            if (factor < slowestFactor) {
                slowestFactor = factor;
            }
        }
    });

    return slowestFactor;
}


function addDeliveryData(data) {
    data.forEach(delivery => {
        const pointColor = delivery.color || 'orange';
        const pointOpacity = delivery.opacity !== undefined ? delivery.opacity : 0.7;

        L.circleMarker([delivery.lat, delivery.lng], {
            radius: 6,
            color: pointColor,
            fillColor: pointColor,
            fillOpacity: pointOpacity,
            opacity: pointOpacity,
            weight: 2
        })
            .bindPopup(delivery.popup)
            .addTo(layers.deliveries);
    });
}

function updateLayer(layerName, data) {
    if (!mapInitialized || !layers[layerName]) return;

    clearLayer(layerName);

    if (layerName === 'warehouse') {
        addWarehouseData(data);
    } else if (layerName === 'deliveries') {
        addDeliveryData(data);
    } else if (layerName === 'routes') {
        addRouteData(data);
    } else if (layerName === 'unassigned') {
        addUnassignedData(data);
    } else if (layerName === 'drivers') {
        addDriverData(data);
    }
}

function clearLayer(layerName) {
    if (!mapInitialized || !layers[layerName]) return;
    layers[layerName].clearLayers();
}

function addWarehouseData(data) {
    data.forEach(warehouse => {
        const warehouseIcon = L.divIcon({
            html: '<div style="background-color: #e74c3c; width: 14px; height: 14px; border-radius: 7px; border: 3px solid white;"></div>',
            className: 'warehouse-icon',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        L.marker([warehouse.lat, warehouse.lng], {
            icon: warehouseIcon
        })
            .bindPopup(warehouse.popup)
            .addTo(layers.warehouse);

        console.log("Added warehouse at:", warehouse.lat, warehouse.lng);
    });
}

function addRouteData(data) {
    data.forEach(route => {
        const path = route.path.map(point => [point[0], point[1]]);

        L.polyline(path, {
            color: route.style.color,
            weight: route.style.weight,
            opacity: route.style.opacity,
            dashArray: route.style.dash_array
        })
            .bindPopup(route.popup)
            .addTo(layers.routes);
    });
}

function addUnassignedData(data) {
    data.forEach(delivery => {
        L.circleMarker([delivery.lat, delivery.lng], {
            radius: 6,
            color: 'black',
            fillColor: 'black',
            fillOpacity: 0.7
        })
            .bindPopup(delivery.popup)
            .addTo(layers.unassigned);
    });
}

function addDriverData(data) {
    data.forEach(driver => {
        const driverIcon = L.divIcon({
            className: 'driver-marker',
            html: `<div style="background-color:${driver.color}">${driver.id}</div>`,
            iconSize: [24, 24]
        });

        L.marker([driver.lat, driver.lng], {icon: driverIcon})
            .bindPopup(`Driver ${driver.id}`)
            .addTo(layers.drivers);
    });
}

function startSimulation(simulationData) {
    stopSimulation();
    clearLayer('drivers');
    simulationDrivers = [];

    totalExpectedTravelTime = 0;
    let maxDriverTravelTime = 0;

    simulationData.forEach(route => {
        let driverTravelTime = 0;
        if (route.travelTimes && route.travelTimes.length > 0) {
            driverTravelTime = route.travelTimes.reduce((sum, time) => sum + time, 0);
            totalExpectedTravelTime += driverTravelTime;

            if (driverTravelTime > maxDriverTravelTime) {
                maxDriverTravelTime = driverTravelTime;
            }
        }
    });

    simulationTime = 8 * 60 * 60;
    initialExpectedCompletionTime = simulationTime + maxDriverTravelTime;

    console.log("Expected total travel time across all drivers (seconds):", totalExpectedTravelTime);
    console.log("Expected completion time:", formatTimeHMS(initialExpectedCompletionTime));

    if (!clockElement) {
        clockElement = L.control({position: 'topright'});

        clockElement.onAdd = function () {
            const div = L.DomUtil.create('div', 'info clock-container');
            div.innerHTML = '<strong>Simulation Time:</strong> 08:00:00';
            div.style.padding = '10px';
            div.style.background = 'rgba(255, 255, 255, 0.8)';
            div.style.borderRadius = '5px';
            div.style.boxShadow = '0 0 5px rgba(0,0,0,0.2)';
            div.style.fontSize = '14px';
            return div;
        };
    }

    clockElement.addTo(map);
    updateClock();

    lastClockUpdateTime = Date.now();

    simulationData.forEach(route => {
        if (route.path && route.path.length > 0) {
            const driverIcon = L.divIcon({
                className: 'driver-marker',
                html: `<div style="background-color:${route.style.color}">${route.driverId}</div>`,
                iconSize: [24, 24]
            });

            const marker = L.marker(route.path[0], {icon: driverIcon})
                .bindPopup(`Driver ${route.driverId}`)
                .addTo(layers.drivers);

            let expectedTravelTime = 0;
            if (route.travelTimes && route.travelTimes.length > 0) {
                expectedTravelTime = route.travelTimes.reduce((sum, time) => sum + time, 0);
            }

            simulationDrivers.push({
                id: route.driverId,
                marker: marker,
                path: route.path,
                travelTimes: route.travelTimes || [],
                deliveryIndices: route.deliveryIndices || [],
                currentIndex: 0,
                lastTime: Date.now(),
                visited: [],
                color: route.style.color,
                elapsedOnSegment: 0,
                completedDistance: 0,
                expectedTravelTime: expectedTravelTime,
                isActive: true
            });
        }
    });

    if (simulationDrivers.length > 0) {
        simulationRunning = true;
        simulationInterval = setInterval(updateSimulation, 50);
    }
}

function formatTimeHMS(timeInSeconds) {
    const hours = Math.floor(timeInSeconds / 3600) % 24;
    const minutes = Math.floor((timeInSeconds % 3600) / 60);
    const seconds = Math.floor(timeInSeconds % 60);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

function stopSimulation() {
    simulationRunning = false;
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
}

function setSimulationSpeed(speed) {
    simulationSpeed = speed;
}

function updateSimulation() {
    try {
        if (!simulationRunning) return;

        const now = Date.now();

        const realElapsedSeconds = (now - lastClockUpdateTime) / 1000;
        lastClockUpdateTime = now;

        const simulationTimeIncrement = realElapsedSeconds * simulationSpeed;
        simulationTime += simulationTimeIncrement;

        if (disruptionsEnabled && typeof updateDisruptionVisibility === 'function') {
            updateDisruptionVisibility(simulationTime);
        }

        let allComplete = true;

        const cappedElapsed = Math.min(realElapsedSeconds, 0.5);

        simulationDrivers.forEach(driver => {
            try {
                if (!driver.isActive) return;

                if (driver.currentIndex >= driver.path.length - 1) {
                    driver.isActive = false;
                    return;
                }

                allComplete = false;

                // Calculate how much this driver is affected by nearby disruptions
                const delayFactor = getDriverDelayFactor(driver);

                // Update visual indicator on driver icon based on delay
                updateDriverDisruptionIndicator(driver, delayFactor);

                // Apply delay factor to reduce speed when in disruption zones
                driver.elapsedOnSegment += cappedElapsed * simulationSpeed * delayFactor;

                let segmentTime = 10;
                if (driver.travelTimes && driver.travelTimes.length > driver.currentIndex) {
                    segmentTime = driver.travelTimes[driver.currentIndex];

                    if (segmentTime <= 0) {
                        console.warn(`Driver ${driver.id}, segment ${driver.currentIndex}: Invalid segmentTime ${segmentTime}. Using fallback.`);
                        segmentTime = 10;
                    }
                }

                const progress = driver.elapsedOnSegment / segmentTime;

                if (progress >= 1.0) {
                    driver.completedDistance += segmentTime;
                    driver.currentIndex++;
                    driver.elapsedOnSegment = 0;

                    if (driver.currentIndex >= driver.path.length - 1) {
                        driver.isActive = false; // Mark as inactive
                        driver.marker.setLatLng(driver.path[driver.path.length - 1]); // Snap to exact end
                        console.log(`Driver ${driver.id} finished at simulation time: ${formatTimeHMS(simulationTime)}`);
                        // Don't return yet, let the allComplete check handle stopping
                    } else {
                        // Still active, snap to the start of the new segment
                        driver.marker.setLatLng(driver.path[driver.currentIndex]);
                    }

                } else if (driver.currentIndex < driver.path.length - 1) {
                    // Interpolate position on the current segment
                    const startPoint = driver.path[driver.currentIndex];
                    const endPoint = driver.path[driver.currentIndex + 1];
                    const newLat = startPoint[0] + (endPoint[0] - startPoint[0]) * progress;
                    const newLng = startPoint[1] + (endPoint[1] - startPoint[1]) * progress;
                    driver.marker.setLatLng([newLat, newLng]);
                }

                if (driver.isActive && // Only trigger if still active
                    driver.deliveryIndices &&
                    driver.deliveryIndices.includes(driver.currentIndex) &&
                    !driver.visited.includes(driver.currentIndex)) {
                    triggerDeliveryAnimation(driver, driver.currentIndex, simulationTime); // Pass simulationTime
                }
            } catch (driverError) {
                console.error(`Error updating driver ${driver.id}:`, driverError, driverError.stack);
            }
        });

        updateClock();

        if (allComplete) {
            console.log("All drivers completed their routes");
            console.log("Final time:", formatTimeHMS(simulationTime));

            stopSimulation();
        }
    } catch (error) {
        console.error("Error in updateSimulation:", error);
        try {
            stopSimulation();
        } catch (e) {
            console.error("Error stopping simulation:", e);
        }
    }
}

function updateDriverDisruptionIndicator(driver, delayFactor) {
    // Don't update if driver isn't active
    if (!driver.isActive) return;

    // Driver is significantly affected if moving less than 90% of normal speed
    const isAffected = delayFactor < 0.9;

    // Only update the icon if the affected status has changed
    if (isAffected !== driver.isDisrupted) {
        // Add warning symbol to affected drivers
        const warningSymbol = isAffected ? "⚠️" : "";

        const iconDiv = new L.DivIcon({
            className: 'driver-marker',
            html: `<div style="background-color:${driver.color}">${driver.id}${warningSymbol}</div>`,
            iconSize: [24, 24]
        });

        driver.marker.setIcon(iconDiv);
        driver.isDisrupted = isAffected;

        // Show a popup explaining the slowdown when a driver becomes affected
        if (isAffected) {
            const slowdownPct = Math.round((1 - delayFactor) * 100);
            driver.marker.bindPopup(`Driver ${driver.id} slowed by ${slowdownPct}%`).openPopup();

            // Close the popup after 2 seconds
            setTimeout(() => {
                if (driver.marker) {
                    driver.marker.closePopup();
                }
            }, 2000);
        }
    }
}

function calculateDriverProgress(driver) {
    if (!driver.isActive) return 1.0;

    const totalExpectedTime = driver.expectedTravelTime || 1;

    let currentCompletedTime = driver.completedDistance;

    if (driver.currentIndex < driver.path.length - 1 && driver.travelTimes.length > driver.currentIndex) {
        const segmentTime = driver.travelTimes[driver.currentIndex];
        const progress = driver.elapsedOnSegment / segmentTime;
        currentCompletedTime += segmentTime * progress;
    }

    return Math.min(currentCompletedTime / totalExpectedTime, 1.0);
}

function setTimeSmoothing(enabled) {
    timeSmoothing = enabled;
    console.log("Time smoothing:", enabled);
}

function triggerDeliveryAnimation(driver, pointIndex) {
    const deliveryPoint = driver.path[pointIndex];

    // Check if there's an active recipient unavailable disruption at this point
    const isRecipientUnavailable = checkForRecipientUnavailable(deliveryPoint, currentSimulationTime);

    if (isRecipientUnavailable) {
        // Don't mark as visited - will try again later
        showDeliveryFailure(driver, deliveryPoint);
        return false;
    }

    // Regular delivery process
    driver.visited.push(pointIndex);

    const pulseCircle = L.circleMarker(deliveryPoint, {
        radius: 15,
        color: driver.color,
        fillColor: driver.color,
        fillOpacity: 0.5,
        weight: 3
    }).addTo(layers.drivers);

    let size = 15;
    let growing = false;
    const pulseAnimation = setInterval(() => {
        if (!pulseCircle._map) { // Stop if circle removed early
            clearInterval(pulseAnimation);
            return;
        }
        if (growing) {
            size += 1;
            if (size >= 25) growing = false;
        } else {
            size -= 1;
            if (size <= 15) growing = true;
        }
        pulseCircle.setRadius(size);
    }, 50);

    let tempPopup = L.popup({closeButton: false, autoClose: false, closeOnClick: false})
        .setLatLng(deliveryPoint)
        .setContent(`<div style="text-align:center"><strong>Driver ${driver.id}</strong><br>Package delivered! 📦</div>`)
        .openOn(map);

    setTimeout(() => {
        clearInterval(pulseAnimation);
        if (tempPopup._map) { // Check if popup still exists
            map.closePopup(tempPopup);
        }
        if (pulseCircle._map) { // Check if circle still exists
            layers.drivers.removeLayer(pulseCircle); // Or remove from its dedicated layer
        }

        // Optional: Add a permanent marker indicating delivery completion
        L.circleMarker(deliveryPoint, {
            radius: 8,
            color: driver.color,
            fillColor: '#ffffff', // White fill to indicate completed
            fillOpacity: 1,
            weight: 3
        }).addTo(layers.drivers); // Add to drivers layer or 'completed' layer

    }, 1500); // Duration of animation + popup

    return true; // Indicate delivery successful
}


function distanceBetween(p1, p2) {
    const lat1 = p1[0];
    const lon1 = p1[1];
    const lat2 = p2[0];
    const lon2 = p2[1];

    return Math.sqrt(Math.pow(lat2 - lat1, 2) + Math.pow(lon2 - lon1, 2));
}

function checkForRecipientUnavailable(point) {
    // Only check if we have disruptions enabled
    if (!disruptionsEnabled || !activeDisruptions || !activeDisruptions.length)
        return false;

    const pointLatLng = L.latLng(point[0], point[1]);

    for (const disruption of activeDisruptions) {
        // Skip if not active AT THE GIVEN TIME or not recipient unavailable type
        if (disruption.start_time > currentSimulationTime || // Check start time
            (disruption.start_time + disruption.duration) < currentSimulationTime || // Check end time
            disruption.type !== 'recipient_unavailable')
            continue;

        // Check if this disruption is at the specific delivery point location
        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );

        // Use a small threshold (e.g., 5 meters) to consider it the same location
        const distance = pointLatLng.distanceTo(disruptionPos);
        if (distance < 5) { // Is the disruption located *at* the delivery point?
            // Check if this specific delivery point index matches metadata, if available
            // This assumes metadata contains the original index from snapped_delivery_points
            // if (disruption.metadata && disruption.metadata.delivery_point_index !== undefined) {
            //      Need a way to map 'point' back to its original index here... complex.
            //      For now, location match is sufficient.
            // }
            return true; // Found an active recipient unavailable disruption at this location and time
        }
    }

    return false; // No matching disruption found
}

function showDeliveryFailure(driver, deliveryPoint) {
    // Create a red X marker for failed delivery
    const failMarker = L.divIcon({
        html: '<div style="color:red; font-size:20px; font-weight:bold;">✕</div>',
        className: 'delivery-fail-marker',
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    const marker = L.marker(deliveryPoint, {icon: failMarker})
        .addTo(layers.drivers);

    // Show failure popup
    let popup = L.popup()
        .setLatLng(deliveryPoint)
        .setContent(`<div style="text-align:center"><strong>Driver ${driver.id}</strong><br>❌ Recipient not available! ❌<br>Delivery failed.</div>`)
        .openOn(map);

    // Remove after 2 seconds
    setTimeout(() => {
        map.closePopup(popup);
        layers.drivers.removeLayer(marker);
    }, 2000);
}