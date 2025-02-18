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
INNER_POINTS_RATIO = 0.70  # Ratio of points to generate in inner area

PACKAGE_CONSTRAINTS = {
    'weight': {  # in kg
        'min': 2,
        'max': 25.0,
        'step': 0.5
    },
    'volume': {  # in cubic meters (m³)
        'min': 0.01,
        'max': 0.5,
        'step': 0.01
    }
}

DRIVER_CONSTRAINTS = {
    'weight_capacity': {  # in kg
        'min': 1000.0,
        'max': 2000.0,
        'step': 100.0
    },
    'volume_capacity': {  # in cubic meters (m³)
        'min': 50.0,
        'max': 100.0,
        'step': 0.5
    }
}

OPTIMIZATION_SETTINGS = {
    'VISUALIZE_PROCESS': True,
    'INITIAL_TEMPERATURE': 1000.0,
    'COOLING_RATE': 0.99,
    'MIN_TEMPERATURE': 0.075,
    'ITERATIONS_PER_TEMPERATURE': 100
}
