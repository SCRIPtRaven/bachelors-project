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
        window.simInterface = channel.objects.simInterface;
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
        if (!disruption._wasActive || disruption._resolved)
            return;

        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );
        const distance = driverPosition.distanceTo(disruptionPos);

        if (distance <= disruption.radius) {
            let factor = 1.0;
            switch (disruption.type) {
                case 'traffic_jam':
                    factor = Math.max(0.2, 1.0 - disruption.severity);
                    break;
                case 'road_closure':
                    factor = 0.1;
                    break;
                default:
                    factor = Math.max(0.5, 1.0 - disruption.severity);
            }

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
            html: `<div class="driver-icon" style="background-color:${driver.color}">${driver.id}</div>`,
            iconSize: [32, 32]
        });

        L.marker([driver.lat, driver.lng], {icon: driverIcon})
            .bindPopup(`Driver ${driver.id}`)
            .addTo(layers.drivers);
    });
}

function startSimulation(simulationData) {
    if (window.simInterface && typeof window.simInterface.isSimulationControllerAvailable === 'function') {
        const controllerAvailable = window.simInterface.isSimulationControllerAvailable();
        if (!controllerAvailable) {
            console.error("Cannot start simulation: Simulation controller not available");
            alert("Cannot start simulation: Controller not ready. Please try again.");
            return;
        }
    }

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
                html: `<div class="driver-icon" style="background-color:${route.style.color}">${route.driverId}</div>`,
                iconSize: [32, 32]
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

function updateDisruptionVisibility(simulationTime) {
    currentSimulationTime = simulationTime;

    const previouslyActive = activeDisruptions.map(d => d._wasActive || false);

    activeDisruptions.forEach((disruption, index) => {
        if (disruption.duration && !disruption._resolved) {
            if (!disruption.end_time) {
                disruption.end_time = disruption._activationTime + disruption.duration;
            }

            if (disruption._wasActive && simulationTime >= disruption.end_time) {
                disruption._resolved = true;

                if (window.simInterface) {
                    window.simInterface.handleEvent(JSON.stringify({
                        type: 'disruption_resolved',
                        data: {
                            disruption_id: disruption.id
                        }
                    }));
                }
            }
        }

        if (disruption.id in disruptionMarkers) {
            const marker = disruptionMarkers[disruption.id].marker;
            const circle = disruptionMarkers[disruption.id].circle;

            if (disruption._resolved) {
                circle.setStyle({
                    fillOpacity: 0.2,
                    opacity: 0.5,
                    dashArray: "5, 10",
                    color: "#777777",
                    fillColor: "#777777"
                });

                marker.bindPopup(createDisruptionPopup(disruption, false, true));
            } else if (disruption._wasActive) {
                circle.setStyle({
                    fillOpacity: 0.6 * disruption.severity,
                    opacity: 0.9,
                    dashArray: null,
                    color: getDisruptionColor(disruption.type),
                    fillColor: getDisruptionColor(disruption.type)
                });

                marker.bindPopup(createDisruptionPopup(disruption, true));
            } else {
                circle.setStyle({
                    fillOpacity: 0.3,
                    opacity: 0.6,
                    dashArray: "5, 10",
                    color: "#777777",
                    fillColor: "#777777"
                });

                marker.bindPopup(createDisruptionPopup(disruption, false));
            }
        }

        if (disruption._wasActive && !previouslyActive[index]) {
            disruption._activationTime = simulationTime;
            console.log(`Disruption ${disruption.id} activated at ${simulationTime}`);
        }
    });
}


function updateSimulation() {
    try {
        if (!simulationRunning) return;

        const now = Date.now();
        const realElapsedSeconds = (now - lastClockUpdateTime) / 1000;
        lastClockUpdateTime = now;

        const simulationTimeIncrement = realElapsedSeconds * simulationSpeed;
        simulationTime += simulationTimeIncrement;

        checkPendingDeliveries();

        if (window.simInterface) {
            window.simInterface.handleEvent(JSON.stringify({
                type: 'simulation_time_updated',
                data: {
                    current_time: simulationTime
                }
            }));
        }

        if (disruptionsEnabled && typeof updateDisruptionVisibility === 'function') {
            updateDisruptionVisibility(simulationTime);
        }

        if (typeof updateDisruptionStatus === 'function') {
            updateDisruptionStatus(simulationTime);
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

                if (driver.id && (!driver._lastActionCheck || now - driver._lastActionCheck > 500)) {
                    driver._lastActionCheck = now;
                    if (typeof checkDriverActions === 'function') {
                        try {
                            checkDriverActions(driver);
                        } catch (e) {
                            console.error("Error checking actions:", e);
                        }
                    }
                }

                checkDriverNearDisruptions(driver);

                const delayFactor = getDriverDelayFactor(driver);

                updateDriverDisruptionIndicator(driver, delayFactor);

                driver.elapsedOnSegment += cappedElapsed * simulationSpeed * delayFactor;

                let segmentTime = 10;
                if (driver.travelTimes && driver.travelTimes.length > driver.currentIndex) {
                    segmentTime = driver.travelTimes[driver.currentIndex];
                    if (segmentTime <= 0) {
                        segmentTime = 10;
                    }
                }

                const progress = driver.elapsedOnSegment / segmentTime;

                if (progress >= 1.0) {
                    driver.completedDistance += segmentTime;
                    driver.currentIndex++;
                    driver.elapsedOnSegment = 0;

                    if (driver.currentIndex >= driver.path.length - 1) {
                        driver.isActive = false;
                        driver.marker.setLatLng(driver.path[driver.path.length - 1]);
                    } else {
                        driver.marker.setLatLng(driver.path[driver.currentIndex]);
                    }
                } else if (driver.currentIndex < driver.path.length - 1) {
                    const startPoint = driver.path[driver.currentIndex];
                    const endPoint = driver.path[driver.currentIndex + 1];
                    const newLat = startPoint[0] + (endPoint[0] - startPoint[0]) * progress;
                    const newLng = startPoint[1] + (endPoint[1] - startPoint[1]) * progress;
                    const newPosition = [newLat, newLng];
                    driver.marker.setLatLng(newPosition);

                    if (window.simInterface) {
                        window.simInterface.handleEvent(JSON.stringify({
                            type: 'driver_position_updated',
                            data: {
                                driver_id: driver.id,
                                lat: newLat,
                                lon: newLng
                            }
                        }));
                    }
                }

                if (driver.isActive &&
                    driver.deliveryIndices &&
                    driver.deliveryIndices.includes(driver.currentIndex) &&
                    !driver.visited.includes(driver.currentIndex)) {
                    triggerDeliveryAnimation(driver, driver.currentIndex, simulationTime);
                }
            } catch (driverError) {
                console.error(`Error updating driver ${driver.id}:`, driverError);
            }
        });

        updateClock();

        if (allComplete) {
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
    if (!driver.isActive) return;

    const isAffected = delayFactor < 0.7;

    if (isAffected !== driver.isDisrupted) {
        let iconHtml;

        if (isAffected) {
            const slowdownPct = Math.round((1 - delayFactor) * 100);
            iconHtml = `<div class="driver-icon" style="background-color:${driver.color}">
                    ${driver.id}
                    <span style="position: absolute; top: -5px; right: -5px; background-color: #ff4500; border-radius: 50%; width: 14px; height: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid white; font-size: 10px;">⚠️</span>
                </div>`;

            driver.marker.bindPopup(`Driver ${driver.id} slowed by ${slowdownPct}%`).openPopup();

            setTimeout(() => {
                if (driver.marker) {
                    driver.marker.closePopup();
                }
            }, 2000);
        } else {
            iconHtml = `<div class="driver-icon" style="background-color:${driver.color}">${driver.id}</div>`;
        }

        const iconDiv = new L.DivIcon({
            className: 'driver-marker',
            html: iconHtml,
            iconSize: [32, 32]
        });

        driver.marker.setIcon(iconDiv);
        driver.isDisrupted = isAffected;
    }
}

function triggerDeliveryAnimation(driver, pointIndex) {
    const deliveryPoint = driver.path[pointIndex];

    const wasRerouted = driver.recentlyRerouted && driver.recentlyRerouted.includes(pointIndex);

    const isRecipientUnavailable = !wasRerouted &&
        checkForRecipientUnavailable(deliveryPoint, currentSimulationTime);

    if (isRecipientUnavailable) {
        showDeliveryFailure(driver, deliveryPoint);
        return false;
    }

    driver.visited.push(pointIndex);

    if (wasRerouted) {
        driver.recentlyRerouted = driver.recentlyRerouted.filter(idx => idx !== pointIndex);
    }

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
        if (!pulseCircle._map) {
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
        if (tempPopup._map) {
            map.closePopup(tempPopup);
        }
        if (pulseCircle._map) {
            layers.drivers.removeLayer(pulseCircle);
        }

        L.circleMarker(deliveryPoint, {
            radius: 8,
            color: driver.color,
            fillColor: '#ffffff',
            fillOpacity: 1,
            weight: 3
        }).addTo(layers.drivers);

    }, 1500);

    return true;
}


function checkForRecipientUnavailable(point) {
    if (!disruptionsEnabled || !activeDisruptions || !activeDisruptions.length)
        return false;

    const pointLatLng = L.latLng(point[0], point[1]);

    for (const disruption of activeDisruptions) {
        if (disruption.start_time > currentSimulationTime ||
            (disruption.start_time + disruption.duration) < currentSimulationTime ||
            disruption.type !== 'recipient_unavailable')
            continue;

        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );

        const distance = pointLatLng.distanceTo(disruptionPos);
        if (distance < 5) {
            return true;
        }
    }

    return false;
}

function showDeliveryFailure(driver, deliveryPoint) {
    const failMarker = L.divIcon({
        html: '<div style="color:red; font-size:20px; font-weight:bold;">✕</div>',
        className: 'delivery-fail-marker',
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    const marker = L.marker(deliveryPoint, {icon: failMarker})
        .addTo(layers.drivers);

    let popup = L.popup()
        .setLatLng(deliveryPoint)
        .setContent(`<div style="text-align:center"><strong>Driver ${driver.id}</strong><br>❌ Recipient not available! ❌<br>Delivery failed.</div>`)
        .openOn(map);

    setTimeout(() => {
        map.closePopup(popup);
        layers.drivers.removeLayer(marker);
    }, 2000);
}