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

function updateDisruptionStatus(simulationTime) {
    activeDisruptions.forEach(disruption => {
        // Check if disruption has expired
        if (disruption._wasActive && disruption.end_time && simulationTime >= disruption.end_time) {
            if (!disruption._resolved) {
                disruption._resolved = true;
                notifyDisruptionResolved(disruption);

                // Update circle visualization to show resolved state
                if (disruption.id in disruptionMarkers) {
                    const circle = disruptionMarkers[disruption.id].circle;
                    circle.setStyle({
                        fillOpacity: 0.2,
                        opacity: 0.4,
                        dashArray: "5, 10",
                        color: "#444444",
                        fillColor: "#444444"
                    });

                    const marker = disruptionMarkers[disruption.id].marker;
                    marker.bindPopup(createDisruptionPopup(disruption, false, true));
                }
            }
        }
    });
}

function loadDisruptions(disruptions) {
    const existingIds = activeDisruptions.map(d => d.id);

    disruptions.forEach(disruption => {
        if (!existingIds.includes(disruption.id)) {
            disruption._wasActive = disruption.is_active || false;

            if (disruption.duration) {
                disruption.end_time = currentSimulationTime + disruption.duration;
            }

            activeDisruptions.push(disruption);

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
        default:
            return '#FFC300';
    }
}

function createDisruptionPopup(disruption, isActive, isResolved = false) {
    let severityText = 'Low';
    if (disruption.severity > 0.3) severityText = 'Moderate';
    if (disruption.severity > 0.6) severityText = 'High';
    if (disruption.severity > 0.9) severityText = 'Extreme';

    let statusClass = isActive ? 'active' : (isResolved ? 'inactive' : 'future');
    let statusText = isActive ? 'ACTIVE' : (isResolved ? 'RESOLVED' : 'INACTIVE');

    let durationInfo = '';
    if (disruption.duration) {
        const minutes = Math.floor(disruption.duration / 60);
        durationInfo = `<p><strong>Duration:</strong> ${minutes} minutes</p>`;

        if (isActive && disruption.end_time) {
            const endHours = Math.floor(disruption.end_time / 3600) % 24;
            const endMinutes = Math.floor((disruption.end_time % 3600) / 60);
            const endTimeStr = `${endHours.toString().padStart(2, '0')}:${endMinutes.toString().padStart(2, '0')}`;

            durationInfo += `<p><strong>Expected End:</strong> ${endTimeStr}</p>`;
        }
    }

    let content = `
        <div class="disruption-popup">
            <h3>${disruption.description}</h3>
            <p class="status ${statusClass}"><strong>Status: ${statusText}</strong></p>
            <p><strong>Type:</strong> ${disruption.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
            <p><strong>Severity:</strong> ${severityText} (${Math.round(disruption.severity * 100)}%)</p>
            <p><strong>Affected Area:</strong> ${Math.round(disruption.radius)} meters</p>
            ${durationInfo}
        </div>
    `;

    return content;
}