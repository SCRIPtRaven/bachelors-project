let activeDisruptions = [];
let disruptionMarkers = {};
let disruptionInterval = null;
let currentSimulationTime = 8 * 60 * 60; // Start at 8:00 AM in seconds

function clearDisruptions() {
    if (!layers || !layers.disruptions) return;
    layers.disruptions.clearLayers();
    activeDisruptions = [];
    disruptionMarkers = {};

    if (disruptionInterval) {
        clearInterval(disruptionInterval);
        disruptionInterval = null;
    }
}

function updateDisruptionVisibility(simulationTime) {
    currentSimulationTime = simulationTime;

    activeDisruptions.forEach(disruption => {
        if (disruption.id in disruptionMarkers) {
            const marker = disruptionMarkers[disruption.id].marker;
            const circle = disruptionMarkers[disruption.id].circle;

            // Style based on activation state (_wasActive tracks JS-side activation)
            if (disruption._wasActive) {
                // Active disruption styling
                circle.setStyle({
                    fillOpacity: 0.6 * disruption.severity,
                    opacity: 0.9,
                    dashArray: null,
                    color: getDisruptionColor(disruption.type),
                    fillColor: getDisruptionColor(disruption.type)
                });

                // Update popup content
                marker.bindPopup(createDisruptionPopup(disruption, true, false));
            } else {
                // Inactive disruption styling
                circle.setStyle({
                    fillOpacity: 0.3,
                    opacity: 0.6,
                    dashArray: "5, 10",
                    color: "#777777",
                    fillColor: "#777777"
                });

                // Update popup content
                marker.bindPopup(createDisruptionPopup(disruption, false, false));
            }
        }
    });
}

function loadDisruptions(disruptions) {
    const existingIds = activeDisruptions.map(d => d.id);

    disruptions.forEach(disruption => {
        if (!existingIds.includes(disruption.id)) {
            // Initialize default active state
            disruption._wasActive = disruption.is_active || false;

            activeDisruptions.push(disruption);

            // Create marker and circle for this disruption
            const icon = getDisruptionIcon(disruption.type);
            const color = getDisruptionColor(disruption.type);

            const marker = L.marker([disruption.location.lat, disruption.location.lng], {
                icon: icon,
                zIndexOffset: 1000
            }).bindPopup(createDisruptionPopup(disruption));

            const circle = L.circle([disruption.location.lat, disruption.location.lng], {
                radius: disruption.radius,
                color: color,
                fillColor: color,
                fillOpacity: 0.3 * disruption.severity,
                weight: 2,
                opacity: 0.7
            });

            // Add markers to the map - THIS WAS MISSING
            marker.addTo(layers.disruptions);
            circle.addTo(layers.disruptions);

            disruptionMarkers[disruption.id] = {marker, circle};
        }
    });

    updateDisruptionVisibility(currentSimulationTime);
}

function getDisruptionIcon(type) {
    let iconHtml = '';

    switch (type) {
        case 'traffic_jam':
            iconHtml = '<i class="fas fa-traffic-light" style="color: #FF5733;"></i>';
            break;
        case 'recipient_unavailable':
            iconHtml = '<i class="fas fa-user-times" style="color: #C70039;"></i>';
            break;
        case 'road_closure':
            iconHtml = '<i class="fas fa-road" style="color: #900C3F;"></i>';
            break;
        case 'vehicle_breakdown':
            iconHtml = '<i class="fas fa-car-crash" style="color: #581845;"></i>';
            break;
        default:
            iconHtml = '<i class="fas fa-exclamation-triangle" style="color: #FFC300;"></i>';
    }

    return L.divIcon({
        html: `<div class="disruption-icon">${iconHtml}</div>`,
        className: 'disruption-marker',
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });
}

function getDisruptionColor(type) {
    switch (type) {
        case 'traffic_jam':
            return '#FF5733';
        case 'recipient_unavailable':
            return '#C70039';
        case 'road_closure':
            return '#900C3F';
        case 'vehicle_breakdown':
            return '#581845';
        default:
            return '#FFC300';
    }
}

function createDisruptionPopup(disruption, isActive, isFuture) {
    const startTimeStr = formatTimeHMS(disruption.start_time);
    const endTimeStr = formatTimeHMS(disruption.start_time + disruption.duration);
    const durationMinutes = Math.floor(disruption.duration / 60);

    let severityText = 'Low';
    if (disruption.severity > 0.3) severityText = 'Moderate';
    if (disruption.severity > 0.6) severityText = 'High';
    if (disruption.severity > 0.9) severityText = 'Extreme';

    let statusClass = isActive ? 'active' : (isFuture ? 'future' : 'inactive');
    let statusText = isActive ? 'ACTIVE' : (isFuture ? 'UPCOMING' : 'RESOLVED');

    let content = `
        <div class="disruption-popup">
            <h3>${disruption.description}</h3>
            <p class="status ${statusClass}"><strong>Status: ${statusText}</strong></p>
            <p><strong>Type:</strong> ${disruption.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
            <p><strong>Time:</strong> ${startTimeStr} - ${endTimeStr}</p>
            <p><strong>Duration:</strong> ${durationMinutes} minutes</p>
            <p><strong>Severity:</strong> ${severityText} (${Math.round(disruption.severity * 100)}%)</p>
            <p><strong>Affected Area:</strong> ${Math.round(disruption.radius)} meters</p>
        </div>
    `;

    return content;
}

function updateDisruptionSimulationTime(time) {
    currentSimulationTime = time;
    updateDisruptionVisibility(time);
}