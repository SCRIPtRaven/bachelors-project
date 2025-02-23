PACKAGE_CONSTRAINTS = {
    'weight': {  # in kg
        'min': 0.5,     # Small package/envelope
        'max': 30.0,    # Heavy box
        'step': 0.5
    },
    'volume': {  # in cubic meters (m³)
        'min': 0.001,   # ~ 10x10x10 cm box
        'max': 0.125,   # ~ 50x50x50 cm box
        'step': 0.001
    }
}

DRIVER_CONSTRAINTS = {
    'weight_capacity': {  # in kg
        'min': 800.0,    # Smaller delivery van
        'max': 1500.0,   # Larger delivery van
        'step': 100.0
    },
    'volume_capacity': {  # in cubic meters (m³)
        'min': 8.0,      # Typical cargo van
        'max': 15.0,     # Large cargo van
        'step': 0.5
    }
}

INNER_POINTS_RATIO = 0.70  # Ratio of points to generate in inner area