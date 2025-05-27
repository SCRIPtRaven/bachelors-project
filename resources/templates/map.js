let map;
let mapInitialized = false;
let layers = {
  warehouse: L.layerGroup(),
  deliveries: L.layerGroup(),
  routes: L.layerGroup(),
  unassigned: L.layerGroup(),
  drivers: L.layerGroup(),
  disruptions: L.layerGroup(),
};

let simulationRunning = false;
let simulationDrivers = [];
let simulationInterval = null;
let simulationSpeed = 5.0;

let simulationTime = 8 * 60 * 60;
let clockElement = null;
let lastClockUpdateTime = Date.now();

let totalExpectedTravelTime = 0;
let initialExpectedCompletionTime = 0;

let disruptionsEnabled = true;
let disruptionData = [];

let manualDisruptionMode = false;
let manualDisruptions = [];
let optimizationCompleted = false;

function initMap(centerLat, centerLon, zoomLevel) {
  map = L.map("map").setView([centerLat, centerLon], zoomLevel);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
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

  const timeString = `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}`;
  clockElement._container.innerHTML = `<strong>Simulation Time:</strong> ${timeString}`;
}

function loadDisruptions(data) {
  disruptionData = data;
  if (typeof window.loadDisruptions === "function") {
    window.loadDisruptions(data);
  }
}

function showLoadingIndicator(message) {
  if (typeof message === "undefined") {
    message = "Optimizing Routes...";
  }
  console.log("showLoadingIndicator called with message:", message);
  const loadingIndicator = document.getElementById("loading-indicator");
  console.log("loadingIndicator element:", loadingIndicator);
  if (loadingIndicator) {
    const messageDiv = loadingIndicator.querySelector("div:last-child");
    console.log("messageDiv element:", messageDiv);
    if (messageDiv) {
      messageDiv.textContent = message;
      console.log("Set message text to:", message);
    }
    loadingIndicator.style.display = "block";
    console.log("Set display to block");
    if (map) {
      map.dragging.disable();
      map.doubleClickZoom.disable();
      map.scrollWheelZoom.disable();
      console.log("Disabled map interactions");
    }
  } else {
    console.error("Loading indicator element not found!");
  }
}

function hideLoadingIndicator() {
  console.log("hideLoadingIndicator called");
  const loadingIndicator = document.getElementById("loading-indicator");
  if (loadingIndicator) {
    loadingIndicator.style.display = "none";
    console.log("Set display to none");
  } else {
    console.error("Loading indicator element not found in hide function!");
  }
  if (map) {
    map.dragging.enable();
    map.doubleClickZoom.enable();
    map.scrollWheelZoom.enable();
    console.log("Enabled map interactions");
  }
}

function getDriverDelayFactor(driver) {
  if (!disruptionsEnabled || !activeDisruptions.length) {
    return 1.0;
  }

  const driverPosition = driver.marker.getLatLng();
  let slowestFactor = 1.0;
  let isInAnyDisruption = false;

  activeDisruptions.forEach((disruption) => {
    if (!disruption._wasActive || disruption._resolved) return;

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

    if (distance <= disruption.radius) {
      isInAnyDisruption = true;
      let factor = 1.0;
      switch (disruption.type) {
        case "traffic_jam":
          factor = Math.max(0.2, 1.0 - disruption.severity);
          break;
        case "road_closure":
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

  if (!isInAnyDisruption) {
    slowestFactor = 1.0;
  }

  return slowestFactor;
}

function addDeliveryData(data) {
  data.forEach((delivery) => {
    const pointColor = delivery.color || "orange";
    const pointOpacity = delivery.opacity !== undefined ? delivery.opacity : 0.7;

    L.circleMarker([delivery.lat, delivery.lng], {
      radius: 6,
      color: pointColor,
      fillColor: pointColor,
      fillOpacity: pointOpacity,
      opacity: pointOpacity,
      weight: 2,
    })
      .bindPopup(delivery.popup)
      .addTo(layers.deliveries);
  });
}

function updateLayer(layerName, data) {
  if (!mapInitialized || !layers[layerName]) return;

  clearLayer(layerName);

  if (layerName === "warehouse") {
    addWarehouseData(data);
  } else if (layerName === "deliveries") {
    addDeliveryData(data);
  } else if (layerName === "routes") {
    addRouteData(data);
  } else if (layerName === "unassigned") {
    addUnassignedData(data);
  } else if (layerName === "drivers") {
    addDriverData(data);
  }
}

function clearLayer(layerName) {
  if (!mapInitialized || !layers[layerName]) return;
  layers[layerName].clearLayers();
}

function addWarehouseData(data) {
  data.forEach((warehouse) => {
    const warehouseIcon = L.divIcon({
      html: '<div style="background-color: #e74c3c; width: 14px; height: 14px; border-radius: 7px; border: 3px solid white;"></div>',
      className: "warehouse-icon",
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });

    L.marker([warehouse.lat, warehouse.lng], {
      icon: warehouseIcon,
    })
      .bindPopup(warehouse.popup)
      .addTo(layers.warehouse);
  });
}

function addRouteData(data) {
  data.forEach((route) => {
    const path = route.path.map((point) => [point[0], point[1]]);

    const polyline = L.polyline(path, {
      color: route.style.color,
      weight: route.style.weight,
      opacity: route.style.opacity,
      dashArray: route.style.dash_array,
      routeData: {
        driverId: route.driverId,
        path: path,
        id: route.id,
      },
    })
      .bindPopup(route.popup)
      .addTo(layers.routes);
  });
}

function addUnassignedData(data) {
  data.forEach((delivery) => {
    L.circleMarker([delivery.lat, delivery.lng], {
      radius: 6,
      color: "black",
      fillColor: "black",
      fillOpacity: 0.7,
    })
      .bindPopup(delivery.popup)
      .addTo(layers.unassigned);
  });
}

function addDriverData(data) {
  data.forEach((driver) => {
    const driverIcon = L.divIcon({
      className: "driver-marker",
      html: `<div class="driver-icon" style="background-color:${driver.color}">${driver.id}</div>`,
      iconSize: [32, 32],
    });

    L.marker([driver.lat, driver.lng], { icon: driverIcon }).bindPopup(`Driver ${driver.id}`).addTo(layers.drivers);
  });
}

function startSimulation(simulationData) {
  if (window.simInterface && typeof window.simInterface.isSimulationControllerAvailable === "function") {
    const controllerAvailable = window.simInterface.isSimulationControllerAvailable();
    if (!controllerAvailable) {
      console.error("Cannot start simulation: Simulation controller not available");
      alert("Cannot start simulation: Controller not ready. Please try again.");
      return;
    }
  }

  stopSimulation();
  clearLayer("drivers");
  simulationDrivers = [];

  totalExpectedTravelTime = 0;
  let maxDriverTravelTime = 0;

  simulationData.forEach((route) => {
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
    clockElement = L.control({ position: "topright" });

    clockElement.onAdd = function () {
      const div = L.DomUtil.create("div", "info clock-container");
      div.innerHTML = "<strong>Simulation Time:</strong> 08:00:00";
      div.style.padding = "10px";
      div.style.background = "rgba(255, 255, 255, 0.8)";
      div.style.borderRadius = "5px";
      div.style.boxShadow = "0 0 5px rgba(0,0,0,0.2)";
      div.style.fontSize = "14px";
      return div;
    };
  }

  clockElement.addTo(map);
  updateClock();

  lastClockUpdateTime = Date.now();

  simulationData.forEach((route) => {
    if (route.path && route.path.length > 1) {
      const driverIcon = L.divIcon({
        className: "driver-marker",
        html: `<div class="driver-icon" style="background-color:${route.style.color}">${route.driverId}</div>`,
        iconSize: [32, 32],
      });

      const marker = L.marker(route.path[0], { icon: driverIcon })
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
        isActive: true,
        originalStyle: {
          weight: route.style.weight || 4,
          opacity: route.style.opacity || 0.9,
          dashArray: route.style.dash_array || null,
        },
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

  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}`;
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

  const previouslyActive = activeDisruptions.map((d) => d._wasActive || false);

  activeDisruptions.forEach((disruption, index) => {
    if (disruption.duration && !disruption._resolved) {
      if (!disruption.end_time) {
        disruption.end_time = disruption._activationTime + disruption.duration;
      }

      if (disruption._wasActive && simulationTime >= disruption.end_time) {
        disruption._resolved = true;

        if (window.simInterface) {
          window.simInterface.handleEvent(
            JSON.stringify({
              type: "disruption_resolved",
              data: {
                disruption_id: disruption.id,
              },
            })
          );
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
          fillColor: "#777777",
        });

        marker.bindPopup(createDisruptionPopup(disruption, false, true));
      } else if (disruption._wasActive) {
        circle.setStyle({
          fillOpacity: 0.6 * disruption.severity,
          opacity: 0.9,
          dashArray: null,
          color: getDisruptionColor(disruption.type),
          fillColor: getDisruptionColor(disruption.type),
        });

        marker.bindPopup(createDisruptionPopup(disruption, true));
      } else {
        circle.setStyle({
          fillOpacity: 0.3,
          opacity: 0.6,
          dashArray: "5, 10",
          color: "#777777",
          fillColor: "#777777",
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
    let timeDelta = (now - lastClockUpdateTime) / 1000;
    lastClockUpdateTime = now;

    timeDelta = Math.min(timeDelta, 0.5);

    const simulationTimeIncrement = timeDelta * simulationSpeed;
    simulationTime += simulationTimeIncrement;

    if (window.simInterface) {
      window.simInterface.handleEvent(
        JSON.stringify({
          type: "simulation_time_updated",
          data: {
            current_time: simulationTime,
          },
        })
      );
    }

    if (disruptionsEnabled && typeof updateDisruptionVisibility === "function") {
      updateDisruptionVisibility(simulationTime);
    }

    if (typeof updateDisruptionStatus === "function") {
      updateDisruptionStatus(simulationTime);
    }

    let allComplete = true;
    let activeDriversCount = 0;

    simulationDrivers.forEach((driver) => {
      try {
        if (!driver.isActive || !driver.path || driver.path.length < 2) return;

        if (driver.currentIndex >= driver.path.length - 1) {
          driver.isActive = false;
          return;
        }

        allComplete = false;
        activeDriversCount++;

        if (driver.id && (!driver._lastActionCheck || now - driver._lastActionCheck > 500)) {
          driver._lastActionCheck = now;
          if (typeof checkDriverActions === "function") {
            try {
              checkDriverActions(driver);
            } catch (e) {
              console.error("Error checking actions:", e);
            }
          }
        }

        checkDriverNearDisruptions(driver);

        let remainingDelta = timeDelta;

        while (remainingDelta > 0 && driver.isActive && driver.currentIndex < driver.path.length - 1) {
          const delayFactor = getDriverDelayFactor(driver);

          updateDriverDisruptionIndicator(driver, delayFactor);

          let segmentDuration = 10;
          if (driver.travelTimes && driver.travelTimes.length > driver.currentIndex) {
            segmentDuration = driver.travelTimes[driver.currentIndex];
            if (segmentDuration <= 0) {
              console.warn(`Driver ${driver.id} has invalid segment duration ${segmentDuration}, using 0.01s`);
              segmentDuration = 0.01;
            }
          }

          const speedMultiplier = simulationSpeed * delayFactor;
          const effectiveSegmentTime = segmentDuration / (speedMultiplier > 0 ? speedMultiplier : 0.001);

          const timeToFinishSegment = effectiveSegmentTime - driver.elapsedOnSegment;

          const timeInSegment = Math.min(remainingDelta, timeToFinishSegment);

          driver.elapsedOnSegment += timeInSegment;

          const progress = driver.elapsedOnSegment / effectiveSegmentTime;

          const startPoint = driver.path[driver.currentIndex];
          const endPoint = driver.path[driver.currentIndex + 1];
          const newLat = startPoint[0] + (endPoint[0] - startPoint[0]) * progress;
          const newLng = startPoint[1] + (endPoint[1] - startPoint[1]) * progress;
          driver.marker.setLatLng([newLat, newLng]);

          if (window.simInterface) {
            window.simInterface.handleEvent(
              JSON.stringify({
                type: "driver_position_updated",
                data: {
                  driver_id: driver.id,
                  lat: newLat,
                  lon: newLng,
                },
              })
            );
          }

          if (driver.elapsedOnSegment >= effectiveSegmentTime - 0.0001) {
            driver.currentIndex++;
            driver.elapsedOnSegment = 0;

            if (driver.currentIndex < driver.path.length) {
              if (
                driver.deliveryIndices &&
                driver.deliveryIndices.includes(driver.currentIndex) &&
                !driver.visited.includes(driver.currentIndex)
              ) {
                triggerDeliveryAnimation(driver, driver.currentIndex);
              }
              if (driver.currentIndex === driver.path.length - 1) {
                console.log(`Driver ${driver.id} returned to warehouse.`);
              }
            }

            if (driver.currentIndex >= driver.path.length - 1) {
              driver.isActive = false;
              break;
            }
          }

          remainingDelta -= timeInSegment;
        }
      } catch (driverError) {
        console.error(`Error updating driver ${driver.id}:`, driverError);
        driver.isActive = false;
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

  const isActuallyInDisruptionArea = isDriverInActiveDisruption(driver);
  const isCurrentlySlowedDown = delayFactor < 1.0;

  const shouldBeMarkedAsAffected = isActuallyInDisruptionArea && isCurrentlySlowedDown;

  if (shouldBeMarkedAsAffected !== driver.isDisrupted) {
    let iconHtml;
    if (shouldBeMarkedAsAffected) {
      const slowdownPct = Math.round((1 - delayFactor) * 100);
      iconHtml = `<div class="driver-icon" style="background-color:${driver.color}">
                    ${driver.id}
                    <span style="position: absolute; top: -5px; right: -5px; background-color: #ff4500; border-radius: 50%; width: 14px; height: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid white; font-size: 10px;">⚠️</span>
                </div>`;

      console.log(
        `Driver ${
          driver.id
        } icon: adding warning. InArea: ${isActuallyInDisruptionArea}, Slowed: ${isCurrentlySlowedDown} (Factor: ${delayFactor.toFixed(
          2
        )}). Slowdown: ${slowdownPct}%`
      );

      if (driver.marker) {
        driver.marker.bindPopup(`Driver ${driver.id} slowed by ${slowdownPct}%`).openPopup();
        setTimeout(() => {
          if (driver.marker && driver.marker.getPopup() && driver.marker.getPopup().isOpen()) {
            driver.marker.closePopup();
          }
        }, 2000);
      }
    } else {
      iconHtml = `<div class="driver-icon" style="background-color:${driver.color}">${driver.id}</div>`;
      console.log(
        `Driver ${
          driver.id
        } icon: removing warning. InArea: ${isActuallyInDisruptionArea}, Slowed: ${isCurrentlySlowedDown} (Factor: ${delayFactor.toFixed(
          2
        )})`
      );
    }

    const iconDiv = new L.DivIcon({
      className: "driver-marker",
      html: iconHtml,
      iconSize: [32, 32],
    });

    if (driver.marker) {
      driver.marker.setIcon(iconDiv);
    }
  }

  driver.isDisrupted = shouldBeMarkedAsAffected;
}

function isDriverInActiveDisruption(driver) {
  if (!disruptionsEnabled || !activeDisruptions.length) {
    return false;
  }

  const driverPosition = driver.marker.getLatLng();

  for (const disruption of activeDisruptions) {
    if (!disruption._wasActive || disruption._resolved) continue;

    if (
      !disruption.location ||
      typeof disruption.location.lat !== "number" ||
      typeof disruption.location.lng !== "number" ||
      isNaN(disruption.location.lat) ||
      isNaN(disruption.location.lng)
    ) {
      console.warn("Invalid disruption location for disruption", disruption.id, ":", disruption.location);
      continue;
    }

    const disruptionPos = L.latLng(disruption.location.lat, disruption.location.lng);
    const distance = driverPosition.distanceTo(disruptionPos);

    if (distance <= disruption.radius) {
      return true;
    }
  }

  return false;
}

function triggerDeliveryAnimation(driver, pointIndex) {
  const deliveryPoint = driver.path[pointIndex];

  const wasRerouted = driver.recentlyRerouted && driver.recentlyRerouted.includes(pointIndex);

  driver.visited.push(pointIndex);

  if (wasRerouted) {
    driver.recentlyRerouted = driver.recentlyRerouted.filter((idx) => idx !== pointIndex);
  }

  const pulseCircle = L.circleMarker(deliveryPoint, {
    radius: 15,
    color: driver.color,
    fillColor: driver.color,
    fillOpacity: 0.5,
    weight: 3,
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

  let tempPopup = L.popup({ closeButton: false, autoClose: false, closeOnClick: false, autoPan: false })
    .setLatLng(deliveryPoint)
    .setContent(`<div style="text-align:center"><strong>Driver ${driver.id}</strong><br>Package delivered! 📦</div>`);

  map.addLayer(tempPopup);

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
      fillColor: "#ffffff",
      fillOpacity: 1,
      weight: 3,
    }).addTo(layers.drivers);
  }, 1500);

  return true;
}

function initManualDisruptionPlacement() {
  const toggleButton = document.getElementById("manual-disruption-toggle");
  const infoDiv = document.getElementById("manual-disruption-info");

  if (!toggleButton || !infoDiv) {
    console.warn("Manual disruption controls not found in DOM");
    return;
  }

  toggleButton.addEventListener("click", function () {
    manualDisruptionMode = !manualDisruptionMode;

    if (manualDisruptionMode) {
      toggleButton.textContent = "✅ Finish Placing";
      toggleButton.classList.add("active");
      infoDiv.classList.add("active");

      if (map && map.getContainer()) {
        map.getContainer().classList.add("manual-placement-cursor");
      }

      map.on("click", handleManualDisruptionClick);

      console.log("Manual disruption placement mode enabled");
    } else {
      toggleButton.textContent = "📍 Place Disruptions";
      toggleButton.classList.remove("active");
      infoDiv.classList.remove("active");

      if (map && map.getContainer()) {
        map.getContainer().classList.remove("manual-placement-cursor");
      }

      map.off("click", handleManualDisruptionClick);

      console.log(`Manual disruption placement finished. Placed ${manualDisruptions.length} disruptions.`);

      if (typeof window.clearAllDisruptions === "function") {
        window.clearAllDisruptions();
      }

      if (manualDisruptions.length > 0 && window.mapHandler) {
        const cleanDisruptions = manualDisruptions.map((disruption) => ({
          id: disruption.id,
          type: disruption.type,
          location: disruption.location,
          severity: disruption.severity,
          radius: disruption.radius,
          affected_area_radius: disruption.affected_area_radius,
          duration: disruption.duration,
          activation_distance: disruption.activation_distance,
          tripwire_location: disruption.tripwire_location,
          is_active: disruption.is_active,
          manually_placed: disruption.manually_placed,
          owning_driver_id: disruption.owning_driver_id,
          metadata: disruption.metadata,
        }));

        window.mapHandler.handleEvent(
          JSON.stringify({
            type: "manual_disruptions_placed",
            data: {
              disruptions: cleanDisruptions,
            },
          })
        );
      }
    }
  });
}

function handleManualDisruptionClick(e) {
  if (!manualDisruptionMode) return;

  const clickedLocation = [e.latlng.lat, e.latlng.lng];

  const snapResult = snapToClosestRoute(clickedLocation);
  if (!snapResult) {
    console.warn("No routes available for snapping disruption");
    return;
  }

  const snappedLocation = snapResult.location;
  const assignedDriverId = snapResult.driverId;

  const isTrafficJam = Math.random() < 0.7;
  const disruptionType = isTrafficJam ? "traffic_jam" : "road_closure";

  let severity, radius, duration, activationDistance;
  if (isTrafficJam) {
    severity = 0.3 + Math.random() * 0.6;
    radius = 30 + Math.random() * 50;
    duration = 1800 + Math.random() * 3600;
    activationDistance = 100 + Math.random() * 150;
  } else {
    severity = 0.7 + Math.random() * 0.3;
    radius = 20 + Math.random() * 40;
    duration = 3600 + Math.random() * 10800;
    activationDistance = 80 + Math.random() * 120;
  }

  const tripwireDistance = 400;
  const tripwireLocation = calculateTripwireLocation(snappedLocation, snapResult.routeSegment, tripwireDistance);

  if (
    !snappedLocation ||
    snappedLocation.length !== 2 ||
    isNaN(snappedLocation[0]) ||
    isNaN(snappedLocation[1]) ||
    !tripwireLocation ||
    tripwireLocation.length !== 2 ||
    isNaN(tripwireLocation[0]) ||
    isNaN(tripwireLocation[1])
  ) {
    console.error("Invalid coordinates for disruption creation:", {
      snappedLocation: snappedLocation,
      tripwireLocation: tripwireLocation,
    });
    return;
  }

  const disruption = {
    id: manualDisruptions.length + 1,
    type: disruptionType,
    location: {
      lat: snappedLocation[0],
      lng: snappedLocation[1],
    },
    severity: severity,
    affected_area_radius: radius,
    radius: radius,
    duration: duration,
    activation_distance: activationDistance,
    tripwire_location: tripwireLocation,
    is_active: false,
    manually_placed: true,
    owning_driver_id: assignedDriverId,
    metadata: {
      description: `Manually placed ${disruptionType.replace("_", " ")} (Driver ${assignedDriverId})`,
      manually_placed: true,
    },
  };

  manualDisruptions.push(disruption);

  if (typeof window.loadDisruptions === "function") {
    disruptionData = [...manualDisruptions];
    window.loadDisruptions(disruptionData);
  }

  if (snapResult.distance > 50) {
    showSnapFeedback(clickedLocation, snappedLocation);
  }

  console.log(
    `Placed ${disruptionType} disruption at ${snappedLocation[0].toFixed(6)}, ${snappedLocation[1].toFixed(
      6
    )} (snapped ${Math.round(snapResult.distance)}m from click) assigned to Driver ${assignedDriverId}`
  );
}

document.addEventListener("DOMContentLoaded", function () {
  setTimeout(initManualDisruptionPlacement, 1000);
});

function hasManualDisruptions() {
  return manualDisruptions.length > 0;
}

function clearAllDriverDisruptionStates() {
  simulationDrivers.forEach((driver) => {
    if (driver.isDisrupted) {
      console.log(`Force clearing disruption state for driver ${driver.id}`);
      driver.isDisrupted = false;

      const iconHtml = `<div class="driver-icon" style="background-color:${driver.color}">${driver.id}</div>`;
      const iconDiv = new L.DivIcon({
        className: "driver-marker",
        html: iconHtml,
        iconSize: [32, 32],
      });
      driver.marker.setIcon(iconDiv);
    }
  });
}

function clearManualDisruptions() {
  manualDisruptions.forEach((disruption) => {
    if (disruption._marker) {
      layers.disruptions.removeLayer(disruption._marker);
    }
    if (disruption._iconMarker) {
      layers.disruptions.removeLayer(disruption._iconMarker);
    }
  });
  manualDisruptions = [];

  if (typeof window.clearAllDisruptions === "function") {
    window.clearAllDisruptions();
  } else {
    activeDisruptions = [];
    disruptionData = [];

    layers.disruptions.clearLayers();

    if (typeof processedDisruptionIds !== "undefined") {
      processedDisruptionIds.clear();
    }
  }

  clearAllDriverDisruptionStates();

  console.log("Manual disruptions and all disruption data cleared");
}

function showManualDisruptionControls() {
  optimizationCompleted = true;
  const controls = document.getElementById("manual-disruption-controls");
  if (controls) {
    controls.style.display = "block";
    console.log("Manual disruption controls are now visible");
  }
}

function hideManualDisruptionControls() {
  optimizationCompleted = false;
  const controls = document.getElementById("manual-disruption-controls");
  if (controls) {
    controls.style.display = "none";
  }
}

function calculateDistance(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function closestPointOnSegment(point, segmentStart, segmentEnd) {
  const A = point[0] - segmentStart[0];
  const B = point[1] - segmentStart[1];
  const C = segmentEnd[0] - segmentStart[0];
  const D = segmentEnd[1] - segmentStart[1];

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;

  if (lenSq === 0) return segmentStart;

  let param = dot / lenSq;
  param = Math.max(0, Math.min(1, param));

  return [segmentStart[0] + param * C, segmentStart[1] + param * D];
}

function snapToClosestRoute(clickLocation) {
  if (!layers.routes || layers.routes.getLayers().length === 0) {
    console.warn("No routes available for snapping");
    return null;
  }

  let closestDistance = Infinity;
  let closestPoint = null;
  let closestDriverId = null;
  let closestSegment = null;

  const routeLayers = layers.routes.getLayers();

  routeLayers.forEach((routeLayer) => {
    if (routeLayer.options && routeLayer.options.routeData) {
      const routeData = routeLayer.options.routeData;
      const driverId = routeData.driverId;
      const path = routeData.path;

      if (path && path.length > 1) {
        for (let i = 0; i < path.length - 1; i++) {
          const segmentStart = path[i];
          const segmentEnd = path[i + 1];

          const closestOnSegment = closestPointOnSegment(clickLocation, segmentStart, segmentEnd);
          const distance = calculateDistance(
            clickLocation[0],
            clickLocation[1],
            closestOnSegment[0],
            closestOnSegment[1]
          );

          if (distance < closestDistance) {
            closestDistance = distance;
            closestPoint = closestOnSegment;
            closestDriverId = driverId;
            closestSegment = { start: segmentStart, end: segmentEnd, index: i };
          }
        }
      }
    }
  });

  if (closestPoint && closestDriverId !== null) {
    return {
      location: closestPoint,
      driverId: closestDriverId,
      distance: closestDistance,
      routeSegment: closestSegment,
    };
  }

  return null;
}

function calculateTripwireLocation(disruptionLocation, routeSegment, activationDistance) {
  if (
    !disruptionLocation ||
    disruptionLocation.length !== 2 ||
    typeof disruptionLocation[0] !== "number" ||
    typeof disruptionLocation[1] !== "number" ||
    isNaN(disruptionLocation[0]) ||
    isNaN(disruptionLocation[1])
  ) {
    console.warn("Invalid disruption location for tripwire calculation:", disruptionLocation);
    return [0, 0];
  }

  const routeLayers = layers.routes.getLayers();
  let targetRoute = null;
  let disruptionSegmentIndex = -1;

  for (const routeLayer of routeLayers) {
    if (routeLayer.options && routeLayer.options.routeData) {
      const routeData = routeLayer.options.routeData;
      const path = routeData.path;

      if (path && path.length > 1) {
        for (let i = 0; i < path.length - 1; i++) {
          const segStart = path[i];
          const segEnd = path[i + 1];

          if (
            routeSegment &&
            routeSegment.start &&
            routeSegment.end &&
            Math.abs(segStart[0] - routeSegment.start[0]) < 0.0001 &&
            Math.abs(segStart[1] - routeSegment.start[1]) < 0.0001 &&
            Math.abs(segEnd[0] - routeSegment.end[0]) < 0.0001 &&
            Math.abs(segEnd[1] - routeSegment.end[1]) < 0.0001
          ) {
            targetRoute = path;
            disruptionSegmentIndex = i;
            break;
          }
        }
        if (targetRoute) break;
      }
    }
  }

  if (!targetRoute || disruptionSegmentIndex === -1) {
    console.warn("Could not find route for tripwire calculation, using fallback");
    return calculateFallbackTripwire(disruptionLocation, routeSegment, activationDistance);
  }

  console.log(
    `Found route for tripwire calculation: ${targetRoute.length} points, disruption at segment ${disruptionSegmentIndex}`
  );

  let remainingDistance = activationDistance;
  let currentSegmentIndex = disruptionSegmentIndex;
  let currentPosition = [...disruptionLocation];

  while (remainingDistance > 0 && currentSegmentIndex >= 0) {
    const segmentStart = targetRoute[currentSegmentIndex];
    const segmentEnd = targetRoute[currentSegmentIndex + 1];

    const segmentLength = calculateDistance(segmentStart[0], segmentStart[1], segmentEnd[0], segmentEnd[1]);

    if (segmentLength >= remainingDistance) {
      const ratio = remainingDistance / segmentLength;

      const tripwireLocation = [
        currentPosition[0] - (segmentEnd[0] - segmentStart[0]) * ratio,
        currentPosition[1] - (segmentEnd[1] - segmentStart[1]) * ratio,
      ];

      if (isNaN(tripwireLocation[0]) || isNaN(tripwireLocation[1])) {
        console.warn("Invalid tripwire calculation result, using fallback");
        return calculateFallbackTripwire(disruptionLocation, routeSegment, activationDistance);
      }

      console.log(
        `Tripwire placed ${activationDistance}m back along route at segment ${currentSegmentIndex}, ratio ${ratio.toFixed(
          3
        )}`
      );
      return tripwireLocation;
    } else {
      remainingDistance -= segmentLength;
      currentPosition = [...segmentStart];
      currentSegmentIndex--;
    }
  }

  if (currentSegmentIndex < 0 && targetRoute.length > 0) {
    console.log("Tripwire distance exceeds route length, placing at route start");
    return [...targetRoute[0]];
  }

  console.warn("Could not calculate tripwire along route, using fallback");
  return calculateFallbackTripwire(disruptionLocation, routeSegment, activationDistance);
}

function calculateFallbackTripwire(disruptionLocation, routeSegment, activationDistance) {
  if (!routeSegment || !routeSegment.start || !routeSegment.end) {
    const tripwireOffset = activationDistance / 111320;
    const angle = Math.random() * 2 * Math.PI;
    const result = [
      disruptionLocation[0] + Math.cos(angle) * tripwireOffset,
      disruptionLocation[1] + Math.sin(angle) * tripwireOffset,
    ];

    if (isNaN(result[0]) || isNaN(result[1])) {
      console.warn("Invalid tripwire calculation result, using small offset");
      return [disruptionLocation[0] + 0.001, disruptionLocation[1] + 0.001];
    }
    return result;
  }

  if (
    !Array.isArray(routeSegment.start) ||
    !Array.isArray(routeSegment.end) ||
    routeSegment.start.length !== 2 ||
    routeSegment.end.length !== 2 ||
    isNaN(routeSegment.start[0]) ||
    isNaN(routeSegment.start[1]) ||
    isNaN(routeSegment.end[0]) ||
    isNaN(routeSegment.end[1])
  ) {
    console.warn("Invalid route segment for tripwire calculation:", routeSegment);
    return [disruptionLocation[0] + 0.001, disruptionLocation[1] + 0.001];
  }

  const segmentVector = [routeSegment.end[0] - routeSegment.start[0], routeSegment.end[1] - routeSegment.start[1]];

  const segmentLength = Math.sqrt(segmentVector[0] * segmentVector[0] + segmentVector[1] * segmentVector[1]);
  if (segmentLength === 0) {
    return [disruptionLocation[0] + 0.001, disruptionLocation[1] + 0.001];
  }

  const normalizedVector = [segmentVector[0] / segmentLength, segmentVector[1] / segmentLength];

  const offsetInDegrees = activationDistance / 111320;

  const tripwireLocation = [
    disruptionLocation[0] - normalizedVector[0] * offsetInDegrees,
    disruptionLocation[1] - normalizedVector[1] * offsetInDegrees,
  ];

  if (isNaN(tripwireLocation[0]) || isNaN(tripwireLocation[1])) {
    console.warn("Invalid tripwire location calculated, using fallback");
    return [disruptionLocation[0] + 0.001, disruptionLocation[1] + 0.001];
  }

  return tripwireLocation;
}

function showSnapFeedback(originalLocation, snappedLocation) {
  const snapLine = L.polyline([originalLocation, snappedLocation], {
    color: "#ff6b6b",
    weight: 3,
    opacity: 0.8,
    dashArray: "5, 10",
  }).addTo(map);

  const originalMarker = L.circleMarker(originalLocation, {
    radius: 6,
    fillColor: "#ff6b6b",
    color: "#fff",
    weight: 2,
    opacity: 1,
    fillOpacity: 0.8,
  }).addTo(map);

  const snappedMarker = L.circleMarker(snappedLocation, {
    radius: 8,
    fillColor: "#4ecdc4",
    color: "#fff",
    weight: 2,
    opacity: 1,
    fillOpacity: 0.9,
  }).addTo(map);

  const distance = Math.round(
    calculateDistance(originalLocation[0], originalLocation[1], snappedLocation[0], snappedLocation[1])
  );

  const popup = L.popup({ closeButton: false, autoClose: false })
    .setLatLng(snappedLocation)
    .setContent(`<div style="text-align:center"><strong>Snapped to Route</strong><br>Distance: ${distance}m</div>`)
    .openOn(map);

  setTimeout(() => {
    map.removeLayer(snapLine);
    map.removeLayer(originalMarker);
    map.removeLayer(snappedMarker);
    map.closePopup(popup);
  }, 3000);
}
