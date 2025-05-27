let pendingActions = {};
let currentSimulationPaused = false;
let processedDisruptionIds = new Set();

function initActionHandling() {
  setInterval(checkPendingActions, 200);
}

function checkPendingActions() {
  if (!simulationRunning || currentSimulationPaused) return;

  simulationDrivers.forEach((driver) => {
    if (driver.isActive) {
      checkDriverActions(driver);
    }
  });
}

function checkDriverActions(driver) {
  if (!window.simInterface || !driver.isActive) return;

  try {
    const responseStr = window.simInterface.handleEvent(
      JSON.stringify({
        type: "get_actions",
        data: { driver_id: driver.id },
      })
    );

    let response;
    if (typeof responseStr === "string") {
      try {
        response = JSON.parse(responseStr);
      } catch (e) {
        console.error("Invalid JSON response:", responseStr);
        return;
      }
    } else {
      response = responseStr;
    }

    if (response && response.success && response.data && response.data.actions && response.data.actions.length > 0) {
      executeDriverActions(driver, response.data.actions);
    }
  } catch (error) {
    console.error(`Error checking actions for driver ${driver.id}:`, error);
  }
}

function executeDriverActions(driver, actions) {
  actions.forEach((action) => {
    switch (action.action_type) {
      case "REROUTE_BASIC":
        handleRerouteAction(map, driver, action);
        break;

      default:
        console.warn(`JS: Unknown action type: ${action.action_type}`);
    }
  });
}

function handleRouteUpdate(
  driver,
  newRoutePoints,
  rerouted_segment_start,
  rerouted_segment_end,
  affectedDeliveryIndex
) {
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
    driver.routeLines.forEach((line) => {
      if (line && typeof line.remove === "function") {
        layers.routes.removeLayer(line);
      }
    });
  }

  if (driver.reroutedLines) {
    driver.reroutedLines.forEach((line) => {
      if (line && typeof line.remove === "function") {
        layers.routes.removeLayer(line);
      }
    });
  }

  if (driver.deliveryMarkers) {
    driver.deliveryMarkers.forEach((marker) => {
      if (marker && typeof marker.remove === "function") {
        layers.drivers.removeLayer(marker);
      }
    });
  }

  driver.routeLines = [];
  driver.reroutedLines = [];
  driver.reroutedSegments = [];
  driver.deliveryMarkers = [];

  const driverColor = driver.color || "#4285F4";

  const routeLine = L.polyline(newRoutePoints, {
    color: driverColor,
    weight: 4,
    opacity: 0.9,
  }).addTo(layers.routes);
  driver.routeLines.push(routeLine);

  if (
    rerouted_segment_start !== undefined &&
    rerouted_segment_end !== undefined &&
    rerouted_segment_start >= 0 &&
    rerouted_segment_end >= rerouted_segment_start &&
    rerouted_segment_end < newRoutePoints.length
  ) {
    const rerouteSegment = newRoutePoints.slice(rerouted_segment_start, rerouted_segment_end + 1);

    if (rerouteSegment.length >= 2) {
      console.log(`Drawing rerouted segment with ${rerouteSegment.length} points`);

      const rerouteLine = L.polyline(rerouteSegment, {
        color: "#000000",
        weight: 5,
        opacity: 0.9,
        dashArray: "8, 12",
      }).addTo(layers.routes);

      rerouteLine.bindPopup(`<div style="text-align:center"><b>Driver ${driver.id}</b><br>Rerouted path</div>`);
      driver.reroutedLines.push(rerouteLine);
    }
  }

  if (driver.deliveryIndices) {
    driver.deliveryIndices.forEach((idx) => {
      if (idx >= 0 && idx < newRoutePoints.length) {
        const point = newRoutePoints[idx];

        const marker = L.circleMarker(point, {
          radius: 6,
          color: driverColor,
          fillColor: "#ffffff",
          fillOpacity: 0.8,
          weight: 3,
        }).addTo(layers.drivers);

        driver.deliveryMarkers.push(marker);
      }
    });
  }

  if (driver.pendingMarkers && affectedDeliveryIndex !== undefined) {
    const remainingMarkers = [];

    for (const item of driver.pendingMarkers) {
      if (item.delivery_index === affectedDeliveryIndex) {
        if (item.marker && typeof item.marker.remove === "function") {
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
      driver.pendingDeliveries = driver.pendingDeliveries.filter((pd) => pd.index !== affectedDeliveryIndex);
    }

    if (driver.skipped) {
      driver.skipped = driver.skipped.filter((idx) => idx !== affectedDeliveryIndex);
    }

    if (driver.visited) {
      driver.visited = driver.visited.filter((idx) => idx !== affectedDeliveryIndex);
    }

    driver.recentlyRerouted = driver.recentlyRerouted || [];
    driver.recentlyRerouted.push(affectedDeliveryIndex);
  }

  showActionFeedback(driver, "↩️ Route Updated", "#4CAF50");

  console.log(
    `Route update complete for driver ${driver.id}. New index: ${
      driver.currentIndex
    }, Elapsed: ${driver.elapsedOnSegment.toFixed(2)}s`
  );
  return true;
}

function notifyDisruptionResolved(disruption) {
  if (!window.simInterface) return;

  window.simInterface.handleEvent(
    JSON.stringify({
      type: "disruption_resolved",
      data: {
        disruption_id: disruption.id,
      },
    })
  );

  console.log(`Notified Python that disruption ${disruption.id} has been resolved`);
}

function checkDriverNearDisruptions(driver) {
  if (!disruptionsEnabled || !activeDisruptions) return;

  const driverPosition = driver.marker.getLatLng();

  activeDisruptions.forEach((disruption) => {
    if (disruption._wasActive) return;

    console.log(
      `Checking disruption ${disruption.id} for driver ${driver.id} - Type: ${disruption.type}, Manual: ${
        disruption.manually_placed || false
      }`
    );

    if (disruption.owning_driver_id !== undefined && disruption.owning_driver_id !== null) {
      if (driver.id !== disruption.owning_driver_id) {
        console.log(`Disruption ${disruption.id} not for this driver (assigned to ${disruption.owning_driver_id})`);
        return;
      }

      if (disruption.tripwire_location) {
        if (checkTripwireCrossing(driver, disruption)) {
          console.log(`Tripwire crossed for disruption ${disruption.id}`);
          notifyDisruptionActivated(disruption);
        }
        return;
      }
    }

    if (
      !disruption.location ||
      typeof disruption.location.lat !== "number" ||
      typeof disruption.location.lng !== "number" ||
      isNaN(disruption.location.lat) ||
      isNaN(disruption.location.lng)
    ) {
      console.warn("Invalid disruption location for disruption", disruption.id, ":", disruption.location);
      return;
    }

    const disruptionPos = L.latLng(disruption.location.lat, disruption.location.lng);

    const distance = driverPosition.distanceTo(disruptionPos);
    const activationDistance = disruption.activation_distance || 400;

    if (distance <= activationDistance) {
      notifyDisruptionActivated(disruption);
    }
  });
}

function checkTripwireCrossing(driver, disruption) {
  if (!disruption.tripwire_location) return false;

  const currentPosition = driver.marker.getLatLng();

  if (!disruption._driverPreviousPositions) {
    disruption._driverPreviousPositions = {};
  }

  const previousPosition = disruption._driverPreviousPositions[driver.id];
  disruption._driverPreviousPositions[driver.id] = currentPosition;

  if (!previousPosition) {
    return false;
  }

  if (
    !Array.isArray(disruption.tripwire_location) ||
    disruption.tripwire_location.length !== 2 ||
    typeof disruption.tripwire_location[0] !== "number" ||
    typeof disruption.tripwire_location[1] !== "number" ||
    isNaN(disruption.tripwire_location[0]) ||
    isNaN(disruption.tripwire_location[1])
  ) {
    console.warn("Invalid tripwire location for disruption", disruption.id, ":", disruption.tripwire_location);
    return false;
  }

  const tripwirePos = L.latLng(disruption.tripwire_location[0], disruption.tripwire_location[1]);

  const tripwireRadius = 25;

  const currentDistance = currentPosition.distanceTo(tripwirePos);
  if (currentDistance <= tripwireRadius) {
    return true;
  }

  const distanceToMovementLine = getDistanceToLineSegment(tripwirePos, previousPosition, currentPosition);
  if (distanceToMovementLine <= tripwireRadius) {
    return true;
  }

  return false;
}

function getDistanceToLineSegment(point, lineStart, lineEnd) {
  const A = point.lat - lineStart.lat;
  const B = point.lng - lineStart.lng;
  const C = lineEnd.lat - lineStart.lat;
  const D = lineEnd.lng - lineStart.lng;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;

  if (lenSq === 0) {
    return point.distanceTo(lineStart);
  }

  let param = dot / lenSq;

  let xx, yy;
  if (param < 0) {
    xx = lineStart.lat;
    yy = lineStart.lng;
  } else if (param > 1) {
    xx = lineEnd.lat;
    yy = lineEnd.lng;
  } else {
    xx = lineStart.lat + param * C;
    yy = lineStart.lng + param * D;
  }

  const dx = point.lat - xx;
  const dy = point.lng - yy;
  return Math.sqrt(dx * dx + dy * dy) * 111000;
}

function lineSegmentsIntersect(p1, p2, p3, p4) {
  function ccw(A, B, C) {
    return (C.lng - A.lng) * (B.lat - A.lat) > (B.lng - A.lng) * (C.lat - A.lat);
  }

  return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
}

function handleRerouteAction(map, driver, action) {
  if (action.new_route && action.new_route.length >= 2 && action.times) {
    const newRoutePoints = action.new_route;
    const newTimes = action.times;
    const newDeliveryIndices = action.delivery_indices || [];
    const reroutedSegmentStart = action.rerouted_segment_start;
    const reroutedSegmentEnd = action.rerouted_segment_end;
    const driverColor = driver.color || "#4285F4";

    console.log(
      `Handling route update for driver ${driver.id}: ${newRoutePoints.length} points, segment ${reroutedSegmentStart}-${reroutedSegmentEnd}`
    );
    console.log(` Received ${newTimes.length} time segments, ${newDeliveryIndices.length} delivery indices.`);

    driver.path = newRoutePoints;
    driver.times = newTimes;
    driver.deliveryIndices = newDeliveryIndices;

    if (driver.routeLines) {
      driver.routeLines.forEach((line) => layers.routes.removeLayer(line));
    }
    if (driver.deliveryMarkers) {
      driver.deliveryMarkers.forEach((marker) => layers.drivers.removeLayer(marker));
    }
    driver.routeLines = [];
    if (!driver.reroutedLines) {
      driver.reroutedLines = [];
    }
    if (!driver.reroutedSegments) {
      driver.reroutedSegments = [];
    }
    driver.deliveryMarkers = [];

    const originalStyle = driver.originalStyle || {
      weight: 4,
      opacity: 0.9,
      dashArray: null,
    };

    try {
      if (driver.path.length >= 2) {
        const allReroutedSegments = [];

        if (driver.reroutedSegments) {
          allReroutedSegments.push(...driver.reroutedSegments);
        } else {
          driver.reroutedSegments = [];
        }

        if (
          reroutedSegmentStart !== undefined &&
          reroutedSegmentEnd !== undefined &&
          reroutedSegmentStart >= 0 &&
          reroutedSegmentEnd >= reroutedSegmentStart &&
          reroutedSegmentEnd < driver.path.length
        ) {
          const newSegment = { start: reroutedSegmentStart, end: reroutedSegmentEnd };
          allReroutedSegments.push(newSegment);
          driver.reroutedSegments.push(newSegment);
        }

        if (allReroutedSegments.length === 0) {
          const fullRouteLine = L.polyline(driver.path, {
            color: driverColor,
            weight: originalStyle.weight,
            opacity: originalStyle.opacity,
            dashArray: originalStyle.dashArray,
          }).addTo(layers.routes);
          driver.routeLines.push(fullRouteLine);
        } else {
          let currentIndex = 0;

          const sortedSegments = [...allReroutedSegments].sort((a, b) => a.start - b.start);

          for (const segment of sortedSegments) {
            if (currentIndex < segment.start) {
              const beforeSegment = driver.path.slice(currentIndex, segment.start + 1);
              if (beforeSegment.length >= 2) {
                const beforeLine = L.polyline(beforeSegment, {
                  color: driverColor,
                  weight: originalStyle.weight,
                  opacity: originalStyle.opacity,
                  dashArray: originalStyle.dashArray,
                }).addTo(layers.routes);
                driver.routeLines.push(beforeLine);
              }
            }
            currentIndex = Math.max(currentIndex, segment.end);
          }

          if (currentIndex < driver.path.length - 1) {
            const afterSegment = driver.path.slice(currentIndex);
            if (afterSegment.length >= 2) {
              const afterLine = L.polyline(afterSegment, {
                color: driverColor,
                weight: originalStyle.weight,
                opacity: originalStyle.opacity,
                dashArray: originalStyle.dashArray,
              }).addTo(layers.routes);
              driver.routeLines.push(afterLine);
            }
          }
        }

        for (let i = 0; i < allReroutedSegments.length; i++) {
          const segment = allReroutedSegments[i];
          const detourSegmentForDrawing = driver.path.slice(segment.start, segment.end + 1);

          if (detourSegmentForDrawing.length >= 2) {
            console.log(`Drawing rerouted segment ${i + 1} with ${detourSegmentForDrawing.length} points`);
            try {
              const reroutedLine = L.polyline(detourSegmentForDrawing, {
                color: "#000000",
                weight: 5,
                opacity: 1.0,
                dashArray: "8, 12",
              }).addTo(layers.routes);
              reroutedLine.bindPopup(
                `<div style="text-align:center"><b>Driver ${driver.id}</b><br>Rerouted segment #${i + 1}</div>`
              );

              if (i >= driver.reroutedLines.length) {
                driver.reroutedLines.push(reroutedLine);
              }
            } catch (e) {
              console.error(`JS handleRerouteAction - ERROR drawing detour polyline for driver ${driver.id}:`, e);
              console.error(`  Detour segment length was: ${detourSegmentForDrawing.length}`);
            }
          }
        }
      } else {
        console.error(
          `JS handleRerouteAction - ERROR: Path length is ${driver.path.length} before drawing main polyline for driver ${driver.id}.`
        );
      }
    } catch (e) {
      console.error(`JS handleRerouteAction - ERROR drawing route for driver ${driver.id}:`, e);
      console.error(`  Path length was: ${driver.path ? driver.path.length : "N/A"}`);
    }

    if (driver.deliveryIndices) {
      driver.deliveryIndices.forEach((idx) => {
        if (idx >= 0 && idx < driver.path.length) {
          const deliveryPoint = driver.path[idx];
          try {
            const deliveryMarker = L.circleMarker(deliveryPoint, {
              radius: 6,
              color: driverColor,
              fillColor: "#ffffff",
              fillOpacity: 0.8,
              weight: 3,
            }).addTo(layers.drivers);
            driver.deliveryMarkers.push(deliveryMarker);
          } catch (e) {
            console.error(`Error drawing delivery marker at index ${idx} for driver ${driver.id}:`, e);
          }
        } else {
          console.warn(
            `Delivery index ${idx} out of bounds for path length ${driver.path.length} for driver ${driver.id}`
          );
        }
      });
    }

    console.log(`Main part of handleRerouteAction complete for driver ${driver.id}.`);

    showActionFeedback(driver, "↩️ ROUTE UPDATED", "#000");

    const startIndex =
      typeof reroutedSegmentStart === "number" &&
      reroutedSegmentStart >= 0 &&
      reroutedSegmentStart < driver.path.length - 1
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
    closeButton: false,
  })
    .setLatLng(position)
    .setContent(
      `<div style="text-align:center; color:${color}"><strong>Driver ${driver.id}</strong><br>${message}</div>`
    );

  map.addLayer(popup);

  setTimeout(() => {
    map.closePopup(popup);
  }, 3000);
}

function notifyDisruptionActivated(disruption) {
  if (processedDisruptionIds.has(disruption.id)) {
    console.log(`Disruption ${disruption.id} already processed, skipping activation`);
    return;
  }

  console.log(`Activating disruption ${disruption.id} of type ${disruption.type}`);
  processedDisruptionIds.add(disruption.id);
  disruption._wasActive = true;
  disruption._activationTime = currentSimulationTime;

  if (disruption.duration) {
    disruption.end_time = currentSimulationTime + disruption.duration;
  }

  if (!window.simInterface) {
    console.warn("No simInterface available for disruption activation");
    return;
  }

  console.log(`Sending disruption_activated event for disruption ${disruption.id}`);
  window.simInterface.handleEvent(
    JSON.stringify({
      type: "disruption_activated",
      data: {
        disruption_id: disruption.id,
      },
    })
  );
}

document.addEventListener("DOMContentLoaded", function () {
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
    const previouslyActive = activeDisruptions.map((d) => d.id);

    oldUpdateDisruptionVisibility(simulationTime);

    activeDisruptions.forEach((disruption) => {
      const isActive = disruption.isActive;

      if (isActive && !previouslyActive.includes(disruption.id)) {
        notifyDisruptionActivated(disruption);
      }
    });
  };
});
