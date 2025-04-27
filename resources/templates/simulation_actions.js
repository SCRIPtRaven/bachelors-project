let pendingActions = {};
let currentSimulationPaused = false;
let processedDisruptionIds = new Set();

function initActionHandling() {
    setInterval(checkPendingActions, 200);
}

function checkPendingActions() {
    if (!simulationRunning || currentSimulationPaused) return;

    simulationDrivers.forEach(driver => {
        if (driver.isActive) {
            checkDriverActions(driver);
        }
    });
}

function checkDriverActions(driver) {
    if (!window.simInterface || !driver.isActive) return;

    try {
        const responseStr = window.simInterface.handleEvent(JSON.stringify({
            type: 'get_actions',
            data: {driver_id: driver.id}
        }));

        let response;
        if (typeof responseStr === 'string') {
            try {
                response = JSON.parse(responseStr);
            } catch (e) {
                console.error("Invalid JSON response:", responseStr);
                return;
            }
        } else {
            response = responseStr;
        }

        if (response && response.success && response.data &&
            response.data.actions && response.data.actions.length > 0) {
            executeDriverActions(driver, response.data.actions);
        }
    } catch (error) {
        console.error(`Error checking actions for driver ${driver.id}:`, error);
    }
}

function executeDriverActions(driver, actions) {

    actions.forEach(action => {

        switch (action.action_type) {
            case 'REROUTE_BASIC':
                handleRerouteAction(map, driver, action);
                break;

            case 'RECIPIENT_UNAVAILABLE':
                handleRecipientUnavailableAction(driver, action);
                break;

            default:
                console.warn(`JS: Unknown action type: ${action.action_type}`);
        }
    });
}

function handleRouteUpdate(driver, newRoutePoints, rerouted_segment_start, rerouted_segment_end, affectedDeliveryIndex) {
    if (!driver || !newRoutePoints || newRoutePoints.length < 2) {
        console.error("Invalid route update data");
        return false;
    }

    console.log(`Handling route update for driver ${driver.id}: ${newRoutePoints.length} points`);
    console.log(`Rerouted segment: ${rerouted_segment_start}-${rerouted_segment_end}`);

    const currentPosition = driver.marker.getLatLng();
    driver.path = newRoutePoints;

    let closestSegmentIndex = 0;
    let minDistance = Infinity;
    let progressOnClosestSegment = 0;

    for (let i = 0; i < newRoutePoints.length - 1; i++) {
        const segmentStart = L.latLng(newRoutePoints[i][0], newRoutePoints[i][1]);
        const segmentEnd = L.latLng(newRoutePoints[i + 1][0], newRoutePoints[i + 1][1]);

        const closestPoint = L.GeometryUtil.closest(map, [segmentStart, segmentEnd], currentPosition);
        const distance = currentPosition.distanceTo(closestPoint);

        if (distance < minDistance) {
            minDistance = distance;
            closestSegmentIndex = i;
            const segmentLength = segmentStart.distanceTo(segmentEnd);
            if (segmentLength > 0) {
                progressOnClosestSegment = segmentStart.distanceTo(closestPoint) / segmentLength;
            } else {
                progressOnClosestSegment = 0;
            }
        }
    }

    driver.currentIndex = closestSegmentIndex;

    const segmentStart = L.latLng(newRoutePoints[closestSegmentIndex][0], newRoutePoints[closestSegmentIndex][1]);
    const segmentEnd = L.latLng(newRoutePoints[closestSegmentIndex + 1][0], newRoutePoints[closestSegmentIndex + 1][1]);
    const segmentDistance = segmentStart.distanceTo(segmentEnd);
    const segmentTime = segmentDistance / 8.33;
    driver.elapsedOnSegment = progressOnClosestSegment * (segmentTime > 0 ? segmentTime : 0.001);



    if (driver.routeLines) {
        driver.routeLines.forEach(line => {
            if (line && typeof line.remove === 'function') {
                layers.routes.removeLayer(line);
            }
        });
    }

    if (driver.reroutedLines) {
        driver.reroutedLines.forEach(line => {
            if (line && typeof line.remove === 'function') {
                layers.routes.removeLayer(line);
            }
        });
    }

    if (driver.deliveryMarkers) {
        driver.deliveryMarkers.forEach(marker => {
            if (marker && typeof marker.remove === 'function') {
                layers.drivers.removeLayer(marker);
            }
        });
    }

    driver.routeLines = [];
    driver.reroutedLines = [];
    driver.deliveryMarkers = [];

    const driverColor = driver.color || '#4285F4';

    const routeLine = L.polyline(newRoutePoints, {
        color: driverColor,
        weight: 4,
        opacity: 0.9
    }).addTo(layers.routes);
    driver.routeLines.push(routeLine);

    if (rerouted_segment_start !== undefined && rerouted_segment_end !== undefined &&
        rerouted_segment_start >= 0 && rerouted_segment_end >= rerouted_segment_start &&
        rerouted_segment_end < newRoutePoints.length) {

        const rerouteSegment = newRoutePoints.slice(rerouted_segment_start, rerouted_segment_end + 1);

        if (rerouteSegment.length >= 2) {
            console.log(`Drawing rerouted segment with ${rerouteSegment.length} points`);

            const rerouteLine = L.polyline(rerouteSegment, {
                color: '#000000',
                weight: 5,
                opacity: 0.9,
                dashArray: '8, 12'
            }).addTo(layers.routes);

            rerouteLine.bindPopup(`<div style="text-align:center"><b>Driver ${driver.id}</b><br>Rerouted path</div>`);
            driver.reroutedLines.push(rerouteLine);
        }
    }

    if (driver.deliveryIndices) {
        driver.deliveryIndices.forEach(idx => {
            if (idx >= 0 && idx < newRoutePoints.length) {
                const point = newRoutePoints[idx];

                const marker = L.circleMarker(point, {
                    radius: 6,
                    color: driverColor,
                    fillColor: '#ffffff',
                    fillOpacity: 0.8,
                    weight: 3
                }).addTo(layers.drivers);

                driver.deliveryMarkers.push(marker);
            }
        });
    }

    if (driver.pendingMarkers && affectedDeliveryIndex !== undefined) {
        const remainingMarkers = [];

        for (const item of driver.pendingMarkers) {
            if (item.delivery_index === affectedDeliveryIndex) {
                if (item.marker && typeof item.marker.remove === 'function') {
                    layers.drivers.removeLayer(item.marker);
                }
            } else {
                remainingMarkers.push(item);
            }
        }

        driver.pendingMarkers = remainingMarkers;
    }

    if (affectedDeliveryIndex !== undefined) {
        if (driver.pendingDeliveries) {
            driver.pendingDeliveries = driver.pendingDeliveries.filter(
                pd => pd.index !== affectedDeliveryIndex
            );
        }

        if (driver.skipped) {
            driver.skipped = driver.skipped.filter(idx => idx !== affectedDeliveryIndex);
        }

        if (driver.visited) {
            driver.visited = driver.visited.filter(idx => idx !== affectedDeliveryIndex);
        }

        driver.recentlyRerouted = driver.recentlyRerouted || [];
        driver.recentlyRerouted.push(affectedDeliveryIndex);
    }

    showActionFeedback(driver, '↩️ Route Updated', '#4CAF50');

    console.log(`Route update complete for driver ${driver.id}. New index: ${driver.currentIndex}, Elapsed: ${driver.elapsedOnSegment.toFixed(2)}s`);
    return true;
}

function handleRecipientUnavailableAction(driver, action) {
    const deliveryIndex = action.delivery_index;

    showActionFeedback(driver, 'Recipient Unavailable - Will Try Later', 'orange');

    if (!driver.pendingDeliveries) driver.pendingDeliveries = [];
    driver.pendingDeliveries.push({
        index: deliveryIndex,
        disruption_id: action.disruption_id,
        end_time: currentSimulationTime + action.duration
    });

    if (deliveryIndex < driver.path.length) {
        const deliveryPoint = driver.path[deliveryIndex];

        const pendingMarker = L.circleMarker(deliveryPoint, {
            radius: 8,
            color: 'orange',
            fillColor: 'white',
            fillOpacity: 0.8,
            weight: 3,
            dashArray: '3, 3'
        }).addTo(layers.drivers);

        pendingMarker.bindPopup(
            `<div style="text-align:center">
                <strong>Driver ${driver.id}</strong><br>
                Recipient unavailable - will try again at 
                ${formatTimeHMS(currentSimulationTime + action.duration)}
            </div>`
        );

        if (!driver.pendingMarkers) driver.pendingMarkers = [];
        driver.pendingMarkers.push({
            marker: pendingMarker,
            delivery_index: deliveryIndex
        });
    }
}

function notifyDisruptionResolved(disruption) {
    if (!window.simInterface) return;

    window.simInterface.handleEvent(JSON.stringify({
        type: 'disruption_resolved',
        data: {
            disruption_id: disruption.id
        }
    }));

    console.log(`Notified Python that disruption ${disruption.id} has been resolved`);
}

function checkDriverNearDisruptions(driver) {
    if (!disruptionsEnabled || !activeDisruptions) return;

    const driverPosition = driver.marker.getLatLng();

    activeDisruptions.forEach(disruption => {
        if (disruption._wasActive) return;

        const disruptionPos = L.latLng(
            disruption.location.lat,
            disruption.location.lng
        );

        const distance = driverPosition.distanceTo(disruptionPos);

        const activationDistance = disruption.activation_distance || 300;

        if (distance <= activationDistance) {
            notifyDisruptionActivated(disruption);
        }
    });
}

function handleRerouteAction(map, driver, action) {
    if (action.new_route && action.new_route.length >= 2 && action.times) {

        const newRoutePoints = action.new_route;
        const newTimes = action.times;
        const newDeliveryIndices = action.delivery_indices || [];
        const reroutedSegmentStart = action.rerouted_segment_start;
        const reroutedSegmentEnd = action.rerouted_segment_end;
        const driverColor = driver.color || '#4285F4';

        console.log(`Handling route update for driver ${driver.id}: ${newRoutePoints.length} points, segment ${reroutedSegmentStart}-${reroutedSegmentEnd}`);
        console.log(` Received ${newTimes.length} time segments, ${newDeliveryIndices.length} delivery indices.`);

        driver.path = newRoutePoints;
        driver.times = newTimes;
        driver.deliveryIndices = newDeliveryIndices;

        if (driver.routeLines) {
            driver.routeLines.forEach(line => layers.routes.removeLayer(line));
        }
        if (driver.reroutedLines) {
            driver.reroutedLines.forEach(line => layers.routes.removeLayer(line));
        }
        if (driver.deliveryMarkers) {
             driver.deliveryMarkers.forEach(marker => layers.drivers.removeLayer(marker));
        }
        driver.routeLines = [];
        driver.reroutedLines = [];
        driver.deliveryMarkers = [];

        try {
             if (driver.path.length >= 2) {
                  const newRouteLine = L.polyline(driver.path, {
                       color: driverColor,
                       weight: 4,
                       opacity: 0.9
                  }).addTo(layers.routes);
                  driver.routeLines.push(newRouteLine);
             } else {
                  console.error(`JS handleRerouteAction - ERROR: Path length is ${driver.path.length} before drawing main polyline for driver ${driver.id}.`);
             }
        } catch (e) {
             console.error(`JS handleRerouteAction - ERROR drawing main polyline for driver ${driver.id}:`, e);
             console.error(`  Path length was: ${driver.path ? driver.path.length : 'N/A'}`);
        }

        if (reroutedSegmentStart !== undefined && reroutedSegmentEnd !== undefined &&
            reroutedSegmentStart >= 0 && reroutedSegmentEnd >= reroutedSegmentStart &&
            reroutedSegmentEnd < driver.path.length) {

            const detourSegmentForDrawing = driver.path.slice(reroutedSegmentStart, reroutedSegmentEnd + 1);
            if (detourSegmentForDrawing.length >= 2) {
                 console.log(`Drawing rerouted segment with ${detourSegmentForDrawing.length} points`);
                 try {
                      const reroutedLine = L.polyline(detourSegmentForDrawing, {
                           color: '#000000',
                           weight: 5,
                           opacity: 1.0,
                           dashArray: '8, 12'
                      }).addTo(layers.routes);
                      reroutedLine.bindPopup(`<div style="text-align:center"><b>Driver ${driver.id}</b><br>Rerouted path</div>`);
                      driver.reroutedLines.push(reroutedLine);
                 } catch (e) {
                      console.error(`JS handleRerouteAction - ERROR drawing detour polyline for driver ${driver.id}:`, e);
                      console.error(`  Detour segment length was: ${detourSegmentForDrawing.length}`);
                 }
            } else {
                 console.warn(`JS handleRerouteAction - Skipping detour drawing for driver ${driver.id}, segment length is ${detourSegmentForDrawing.length}.`);
            }
        }

        if (driver.deliveryIndices) {
            driver.deliveryIndices.forEach(idx => {
                if (idx >= 0 && idx < driver.path.length) {
                    const deliveryPoint = driver.path[idx];
                    try {
                        const deliveryMarker = L.circleMarker(deliveryPoint, {
                            radius: 6,
                            color: driverColor,
                            fillColor: '#ffffff',
                            fillOpacity: 0.8,
                            weight: 3
                        }).addTo(layers.drivers);
                        driver.deliveryMarkers.push(deliveryMarker);
                    } catch(e) {
                        console.error(`Error drawing delivery marker at index ${idx} for driver ${driver.id}:`, e);
                    }
                } else {
                    console.warn(`Delivery index ${idx} out of bounds for path length ${driver.path.length} for driver ${driver.id}`);
                }
            });
        }

        console.log(`Main part of handleRerouteAction complete for driver ${driver.id}.`);

        showActionFeedback(driver, '↩️ ROUTE UPDATED', '#000');

        const startIndex = (typeof reroutedSegmentStart === 'number' && reroutedSegmentStart >= 0 && reroutedSegmentStart < driver.path.length -1)
                           ? reroutedSegmentStart
                           : 0;
        driver.currentIndex = startIndex;
        driver.elapsedOnSegment = 0;
        console.log(`Driver ${driver.id} position reset to index: ${driver.currentIndex}`);

    } else {
         console.error(`JS: Invalid data received for REROUTE_BASIC action for driver ${driver.id}:`, action);
    }
}

function showActionFeedback(driver, message, color) {
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

    setTimeout(() => {
        map.closePopup(popup);
    }, 3000);
}

function checkPendingDeliveries() {
    if (!simulationRunning || currentSimulationPaused) return;

    simulationDrivers.forEach(driver => {
        if (!driver.pendingDeliveries) return;

        const now = currentSimulationTime;
        for (let i = driver.pendingDeliveries.length - 1; i >= 0; i--) {
            const delivery = driver.pendingDeliveries[i];
            if (delivery.end_time <= now) {
                console.log(`Delivery ${delivery.index} for driver ${driver.id} is now available (time: ${now}, end_time: ${delivery.end_time})`);

                driver.pendingDeliveries.splice(i, 1);
                handleRecipientAvailable(driver, delivery.index, delivery.disruption_id);
            }
        }
    });
}

function handleRecipientAvailable(driver, deliveryIndex, disruptionId) {
    console.log(`Recipient for delivery ${deliveryIndex} is now available for driver ${driver.id}`);

    if (driver.pendingMarkers) {
        for (let i = 0; i < driver.pendingMarkers.length; i++) {
            if (driver.pendingMarkers[i].delivery_index === deliveryIndex) {
                if (driver.pendingMarkers[i].marker) {
                    layers.drivers.removeLayer(driver.pendingMarkers[i].marker);
                }
                driver.pendingMarkers.splice(i, 1);
                break;
            }
        }
    }

    if (driver.pendingDeliveries) {
        driver.pendingDeliveries = driver.pendingDeliveries.filter(
            pd => pd.index !== deliveryIndex
        );
    }

    if (driver.skipped) {
        driver.skipped = driver.skipped.filter(idx => idx !== deliveryIndex);
    }

    if (driver.visited) {
        driver.visited = driver.visited.filter(idx => idx !== deliveryIndex);
    }

    showActionFeedback(driver, 'Recipient now available! Rerouting...', '#4CAF50');

    if (window.simInterface) {
        window.simInterface.handleEvent(JSON.stringify({
            type: 'recipient_available',
            data: {
                driver_id: driver.id,
                delivery_index: deliveryIndex,
                disruption_id: disruptionId
            }
        }));
    }
}

function notifyDisruptionActivated(disruption) {
    if (processedDisruptionIds.has(disruption.id)) {
        return;
    }

    processedDisruptionIds.add(disruption.id);
    disruption._wasActive = true;
    disruption._activationTime = currentSimulationTime;

    if (disruption.duration) {
        disruption.end_time = currentSimulationTime + disruption.duration;
    }

    if (!window.simInterface) return;

    window.simInterface.handleEvent(JSON.stringify({
        type: 'disruption_activated',
        data: {
            disruption_id: disruption.id
        }
    }));
}

document.addEventListener('DOMContentLoaded', function () {
    const oldInitMap = window.initMap;

    window.initMap = function (centerLat, centerLon, zoomLevel) {
        oldInitMap(centerLat, centerLon, zoomLevel);
        initActionHandling();
    };

    const oldUpdateSimulation = window.updateSimulation;

    window.updateSimulation = function () {
        if (currentSimulationPaused) return;
        oldUpdateSimulation();
    };

    const oldUpdateDisruptionVisibility = window.updateDisruptionVisibility;

    window.updateDisruptionVisibility = function (simulationTime) {
        const previouslyActive = activeDisruptions.map(d => d.id);

        oldUpdateDisruptionVisibility(simulationTime);

        activeDisruptions.forEach(disruption => {
            const isActive = disruption.isActive;

            if (isActive && !previouslyActive.includes(disruption.id)) {
                notifyDisruptionActivated(disruption);
            }
        });
    };
});