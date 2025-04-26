from pathlib import Path


class DeliveryConfig:
    """Configuration for delivery packages and drivers"""

    PACKAGE_CONSTRAINTS = {
        'weight': {  # in kg
            'min': 0.5,
            'max': 30.0,
            'step': 0.5
        },
        'volume': {  # in cubic meters (m³)
            'min': 0.001,
            'max': 0.125,
            'step': 0.001
        }
    }

    DRIVER_CONSTRAINTS = {
        'weight_capacity': {  # in kg
            'min': 800.0,
            'max': 1500.0,
            'step': 100.0
        },
        'volume_capacity': {  # in cubic meters (m³)
            'min': 8.0,
            'max': 15.0,
            'step': 0.5
        }
    }

    # Ratio of points to generate in inner area
    INNER_POINTS_RATIO = 0.70


class OptimizationConfig:
    """Configuration for optimization algorithms"""

    # Simulated Annealing settings
    SETTINGS = {
        'VISUALIZE_PROCESS': True,
        'INITIAL_TEMPERATURE': 250,
        'COOLING_RATE': 0.99,
        'MIN_TEMPERATURE': 0.1,
        'ITERATIONS_PER_TEMPERATURE': 100,
    }


class PathsConfig:
    """Configuration for file paths"""

    BASE_DIR = Path(__file__).parent.parent
    RESOURCES_DIR = BASE_DIR / 'resources'
    MAP_HTML = RESOURCES_DIR / 'generated' / 'map.html'

    @staticmethod
    def get_graph_file_path(city_name: str) -> Path:
        """
        Generates the appropriate graph file path for a given city.
        Example: "Kaunas, Lithuania" -> "resources/kaunas.graphml"
        """
        city_filename = city_name.split(',')[0].strip().lower() + ".graphml"
        return PathsConfig.RESOURCES_DIR / 'data' / 'graphs' / city_filename

    @staticmethod
    def get_travel_times_path(city_name: str) -> Path:
        """
        Generates the path for city-specific travel times CSV.
        Example: "Kaunas, Lithuania" -> "resources/kaunas_travel_times.csv"
        """
        city_filename = city_name.split(',')[0].strip().lower() + "_travel_times.csv"
        return PathsConfig.RESOURCES_DIR / 'data' / 'travel_times' / city_filename


class DisruptionConfig:
    """Configuration for disruption system"""

    ENABLED_TYPES = [
        'traffic_jam',
        'road_closure',
        # 'recipient_unavailable',
    ]


class SimulationConfig:
    """Configuration for simulation settings"""

    # Simulation speed multiplier (higher = faster simulation)
    SPEED = 20


class RouteConfig:
    """Configuration for route visualization"""

    GOLDEN_RATIO = (1 + 5 ** 0.5) / 2
    BASE_HUES = [
        0.0,  # Red
        0.33,  # Green
        0.66,  # Blue
        0.15,  # Yellow-Orange
        0.45,  # Turquoise
        0.75,  # Purple
        0.05,  # Orange
        0.25,  # Yellow-Green
        0.55,  # Sky Blue
        0.85  # Magenta
    ]
    SATURATION_LEVELS = [1.0, 0.85, 0.7]
    BRIGHTNESS_LEVELS = [0.95, 0.75, 0.55]

    LINE_PATTERNS = [
        {'type': 'solid', 'weight': 4, 'dash_array': None},
        {'type': 'dashed', 'weight': 4, 'dash_array': '10, 10'},
        {'type': 'dotted', 'weight': 4, 'dash_array': '3, 7'}
    ]
