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
                elapsedOnSegment: 0
            });
        }
    });

    if (simulationDrivers.length > 0) {
        simulationRunning = true;
        simulationInterval = setInterval(updateSimulation, 50);
    }
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

        simulationDrivers.forEach(driver => {
            try {
                if (driver.currentIndex >= driver.path.length - 1) return;

                allComplete = false;
                const elapsed = (now - driver.lastTime) / 1000;
                driver.lastTime = now;

                const cappedElapsed = Math.min(elapsed, 0.5);

                driver.elapsedOnSegment += cappedElapsed * simulationSpeed;

                let segmentTime = 10;
                if (driver.travelTimes && driver.travelTimes.length > driver.currentIndex) {
                    segmentTime = driver.travelTimes[driver.currentIndex];
                }

                const progress = driver.elapsedOnSegment / segmentTime;

                if (progress >= 1.0) {
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

                if (driver.deliveryIndices && driver.deliveryIndices.includes(driver.currentIndex) &&
                    !driver.visited.includes(driver.currentIndex)) {
                    triggerDeliveryAnimation(driver, driver.currentIndex);
                }
            } catch (driverError) {
                console.error("Error updating driver:", driverError);
            }
        });

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