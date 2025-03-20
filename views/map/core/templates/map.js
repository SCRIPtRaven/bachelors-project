let map;
let mapInitialized = false;
let layers = {
    warehouse: L.layerGroup(),
    deliveries: L.layerGroup(),
    routes: L.layerGroup(),
    unassigned: L.layerGroup(),
    drivers: L.layerGroup()
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

function initMap(centerLat, centerLon, zoomLevel) {
    map = L.map('map').setView([centerLat, centerLon], zoomLevel);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    for (const key in layers) {
        layers[key].addTo(map);
    }

    mapInitialized = true;

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
        let allComplete = true;
        let totalProgressFraction = 0;
        let activeDrivers = 0;

        const realElapsedSeconds = (now - lastClockUpdateTime) / 1000;
        lastClockUpdateTime = now;

        simulationDrivers.forEach(driver => {
            try {
                if (!driver.isActive) return;

                if (driver.currentIndex >= driver.path.length - 1) {
                    driver.isActive = false;
                    console.log(`Driver ${driver.id} completed route`);
                    return;
                }

                allComplete = false;
                activeDrivers++;

                const cappedElapsed = Math.min(realElapsedSeconds, 0.5);
                driver.elapsedOnSegment += cappedElapsed * simulationSpeed;

                let segmentTime = 10;
                if (driver.travelTimes && driver.travelTimes.length > driver.currentIndex) {
                    segmentTime = driver.travelTimes[driver.currentIndex];
                }

                const progress = driver.elapsedOnSegment / segmentTime;

                if (progress >= 1.0) {
                    driver.completedDistance += segmentTime;
                    driver.currentIndex++;
                    driver.elapsedOnSegment = 0;

                    if (driver.currentIndex < driver.path.length) {
                        driver.marker.setLatLng(driver.path[driver.currentIndex]);
                    }
                } else {
                    const startPoint = driver.path[driver.currentIndex];
                    const endPoint = driver.path[driver.currentIndex + 1];
                    const newLat = startPoint[0] + (endPoint[0] - startPoint[0]) * progress;
                    const newLng = startPoint[1] + (endPoint[1] - startPoint[1]) * progress;
                    driver.marker.setLatLng([newLat, newLng]);
                }

                const totalExpectedTime = driver.expectedTravelTime || 1;
                const currentCompletedTime = driver.completedDistance +
                    (driver.currentIndex < driver.path.length - 1 ?
                        (driver.travelTimes[driver.currentIndex] * progress) : 0);
                const driverProgressFraction = Math.min(currentCompletedTime / totalExpectedTime, 1.0);

                totalProgressFraction += driverProgressFraction;

                if (driver.deliveryIndices &&
                    driver.deliveryIndices.includes(driver.currentIndex) &&
                    !driver.visited.includes(driver.currentIndex)) {
                    triggerDeliveryAnimation(driver, driver.currentIndex);
                }
            } catch (driverError) {
                console.error("Error updating driver:", driverError);
            }
        });

        if (activeDrivers > 0) {
            const avgProgressFraction = totalProgressFraction / activeDrivers;

            targetSimulationTime = 8 * 60 * 60 +
                ((initialExpectedCompletionTime - (8 * 60 * 60)) * avgProgressFraction);

            if (timeSmoothing) {
                simulationTime = simulationTime + (targetSimulationTime - simulationTime) * 0.05;
            } else {
                simulationTime = targetSimulationTime;
            }

            updateClock();
        }

        if (allComplete) {
            simulationTime = initialExpectedCompletionTime;
            updateClock();

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
    driver.visited.push(pointIndex);

    const deliveryPoint = driver.path[pointIndex];

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
        if (growing) {
            size += 1;
            if (size >= 25) growing = false;
        } else {
            size -= 1;
            if (size <= 15) growing = true;
        }
        pulseCircle.setRadius(size);
    }, 50);

    let popup = L.popup()
        .setLatLng(deliveryPoint)
        .setContent(`<div style="text-align:center"><strong>Driver ${driver.id}</strong><br>Package delivered! 📦</div>`)
        .openOn(map);

    setTimeout(() => {
        clearInterval(pulseAnimation);
        map.closePopup(popup);
        layers.drivers.removeLayer(pulseCircle);

        L.circleMarker(deliveryPoint, {
            radius: 8,
            color: driver.color,
            fillColor: '#ffffff',
            fillOpacity: 1,
            weight: 3
        }).addTo(layers.drivers);
    }, 1500);
}

function distanceBetween(p1, p2) {
    const lat1 = p1[0];
    const lon1 = p1[1];
    const lat2 = p2[0];
    const lon2 = p2[1];

    return Math.sqrt(Math.pow(lat2 - lat1, 2) + Math.pow(lon2 - lon1, 2));
}