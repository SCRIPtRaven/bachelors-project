let activeDisruptions = [];
let disruptionMarkers = {};
let currentSimulationTime = 8 * 60 * 60;

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

function updateDisruptionStatus(simulationTime) {
  activeDisruptions.forEach((disruption) => {
    if (disruption._wasActive && disruption.end_time && simulationTime >= disruption.end_time) {
      if (!disruption._resolved) {
        disruption._resolved = true;
        notifyDisruptionResolved(disruption);

        if (disruption.id in disruptionMarkers) {
          const circle = disruptionMarkers[disruption.id].circle;
          circle.setStyle({
            fillOpacity: 0.2,
            opacity: 0.4,
            dashArray: "5, 10",
            color: "#444444",
            fillColor: "#444444",
          });

          const marker = disruptionMarkers[disruption.id].marker;
          marker.bindPopup(createDisruptionPopup(disruption, false, true));

          const tripwireMarker = disruptionMarkers[disruption.id].tripwireMarker;
          if (tripwireMarker) {
            if (tripwireMarker instanceof L.Polyline) {
              tripwireMarker.setStyle({
                color: "#888888",
                opacity: 0.4,
              });
            } else {
              tripwireMarker.setStyle({
                color: "#888888",
                fillColor: "#888888",
                fillOpacity: 0.4,
              });
            }
          }
        }
      }
    }
  });
}

function clearAllDisruptions() {
  Object.values(disruptionMarkers).forEach((markerData) => {
    if (markerData.marker) {
      layers.disruptions.removeLayer(markerData.marker);
    }
    if (markerData.circle) {
      layers.disruptions.removeLayer(markerData.circle);
    }
    if (markerData.tripwireMarker) {
      layers.disruptions.removeLayer(markerData.tripwireMarker);
    }
  });

  activeDisruptions = [];
  disruptionMarkers = {};

  if (typeof processedDisruptionIds !== "undefined") {
    processedDisruptionIds.clear();
  }

  console.log("All disruptions cleared from JavaScript");
}

function loadDisruptions(disruptions) {
  const existingIds = activeDisruptions.map((d) => d.id);

  disruptions.forEach((disruption) => {
    if (!existingIds.includes(disruption.id)) {
      if (!disruption.id || !disruption.type || !disruption.location) {
        console.warn("Invalid disruption object, skipping:", disruption);
        return;
      }

      disruption._wasActive = disruption.is_active || false;

      if (disruption.duration) {
        disruption.end_time = currentSimulationTime + disruption.duration;
      }

      activeDisruptions.push(disruption);

      const icon = getDisruptionIcon(disruption.type);
      const color = getDisruptionColor(disruption.type);

      const marker = L.marker([disruption.location.lat, disruption.location.lng], {
        icon: icon,
        zIndexOffset: 1000,
      }).bindPopup(createDisruptionPopup(disruption));

      const circle = L.circle([disruption.location.lat, disruption.location.lng], {
        radius: disruption.radius,
        color: color,
        fillColor: color,
        fillOpacity: 0.3 * disruption.severity,
        weight: 2,
        opacity: 0.7,
      });

      marker.addTo(layers.disruptions);
      circle.addTo(layers.disruptions);

      let tripwireMarker = null;
      if (disruption.tripwire_location) {
        const tripwireIcon = L.divIcon({
          html: '<div style="color: black; font-size: 16px; font-weight: bold; text-align: center; line-height: 16px;">✕</div>',
          className: "tripwire-marker",
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        const tripwireDistance = Math.round(
          calculateDistance(
            disruption.tripwire_location[0],
            disruption.tripwire_location[1],
            disruption.location.lat,
            disruption.location.lng
          )
        );

        tripwireMarker = L.marker([disruption.tripwire_location[0], disruption.tripwire_location[1]], {
          icon: tripwireIcon,
        }).bindPopup(
          `<div><strong>Tripwire for Disruption ${disruption.id}</strong><br>Driver ${
            disruption.owning_driver_id || "Unknown"
          }<br>Trigger point on route<br><strong>Distance to disruption:</strong> ${tripwireDistance}m</div>`
        );

        tripwireMarker.addTo(layers.disruptions);
      }

      disruptionMarkers[disruption.id] = { marker, circle, tripwireMarker };
    }
  });

  updateDisruptionVisibility(currentSimulationTime);
}

function getDisruptionIcon(type) {
  let iconHtml = "";

  switch (type) {
    case "traffic_jam":
      iconHtml = '<i class="fas fa-traffic-light" style="color: #FF5733;"></i>';
      break;
    case "road_closure":
      iconHtml = '<i class="fas fa-road" style="color: #900C3F;"></i>';
      break;
    default:
      iconHtml = '<i class="fas fa-exclamation-triangle" style="color: #FFC300;"></i>';
  }

  return L.divIcon({
    html: `<div class="disruption-icon">${iconHtml}</div>`,
    className: "disruption-marker",
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
}

function getDisruptionColor(type) {
  switch (type) {
    case "traffic_jam":
      return "#FF5733";
    case "road_closure":
      return "#900C3F";
    default:
      return "#FFC300";
  }
}

function createDisruptionPopup(disruption, isActive, isResolved = false) {
  let severityText = "Low";
  if (disruption.severity > 0.3) severityText = "Moderate";
  if (disruption.severity > 0.6) severityText = "High";
  if (disruption.severity > 0.9) severityText = "Extreme";

  let statusClass = isActive ? "active" : isResolved ? "inactive" : "future";
  let statusText = isActive ? "ACTIVE" : isResolved ? "RESOLVED" : "INACTIVE";

  let durationInfo = "";
  if (disruption.duration) {
    const minutes = Math.floor(disruption.duration / 60);
    durationInfo = `<p><strong>Duration:</strong> ${minutes} minutes</p>`;

    if (isActive && disruption.end_time) {
      const endHours = Math.floor(disruption.end_time / 3600) % 24;
      const endMinutes = Math.floor((disruption.end_time % 3600) / 60);
      const endTimeStr = `${endHours.toString().padStart(2, "0")}:${endMinutes.toString().padStart(2, "0")}`;

      durationInfo += `<p><strong>Expected End:</strong> ${endTimeStr}</p>`;
    }
  }

  let ownerInfo = "";
  if (disruption.owning_driver_id !== undefined && disruption.owning_driver_id !== null) {
    ownerInfo = `<p><strong>Assigned Driver:</strong> ${disruption.owning_driver_id}</p>`;
  }

  let tripwireInfo = "";
  if (disruption.tripwire_location) {
    const actualTripwireDistance = Math.round(
      calculateDistance(
        disruption.tripwire_location[0],
        disruption.tripwire_location[1],
        disruption.location.lat,
        disruption.location.lng
      )
    );
    tripwireInfo = `<p><strong>Activation:</strong> Tripwire (${actualTripwireDistance}m ahead)</p>`;
  } else {
    tripwireInfo = `<p><strong>Activation:</strong> Proximity (${Math.round(
      disruption.activation_distance
    )}m radius)</p>`;
  }

  let content = `
        <div class="disruption-popup">
            <h3>${disruption.description}</h3>
            <p class="status ${statusClass}"><strong>Status: ${statusText}</strong></p>
            <p><strong>Type:</strong> ${disruption.type.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase())}</p>
            <p><strong>Severity:</strong> ${severityText} (${Math.round(disruption.severity * 100)}%)</p>
            <p><strong>Affected Area:</strong> ${Math.round(disruption.radius)} meters</p>
            ${ownerInfo}
            ${tripwireInfo}
            ${durationInfo}
        </div>
    `;

  return content;
}
