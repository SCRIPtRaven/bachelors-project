# Map center coordinates (Kaunas)
DEFAULT_CENTER = (54.8985, 23.9036)
DEFAULT_ZOOM = 12

# Route settings
DEFAULT_NETWORK_TYPE = 'drive'
ROUTE_COLORS = {
    'normal': 'blue',
    'tsp': 'purple'
}

# TSP settings
INNER_POINTS_RATIO = 0.75  # Ratio of points to generate in inner area

PACKAGE_CONSTRAINTS = {
    'weight': {  # in kg
        'min': 0.1,
        'max': 30.0
    },
    'volume': {  # in cubic meters (m³)
        'min': 0.001,  # 1 liter
        'max': 0.5  # 500 liters
    }
}

DRIVER_CONSTRAINTS = {
    'weight_capacity': {  # in kg
        'min': 500.0,
        'max': 2000.0
    },
    'volume_capacity': {  # in cubic meters
        'min': 5.0,
        'max': 20.0
    }
}
