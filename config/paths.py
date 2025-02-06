from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESOURCES_DIR = BASE_DIR / 'resources'
MAP_HTML = RESOURCES_DIR / 'map.html'

def get_graph_file_path(city_name):
    """
    Generates the appropriate graph file path for a given city.
    Example: "Kaunas, Lithuania" -> "resources/kaunas.graphml"
    """
    city_filename = city_name.split(',')[0].strip().lower() + ".graphml"
    return RESOURCES_DIR / city_filename

def get_travel_times_path(city_name):
    """
    Generates the path for city-specific travel times CSV.
    Example: "Kaunas, Lithuania" -> "resources/kaunas_travel_times.csv"
    """
    city_filename = city_name.split(',')[0].strip().lower() + "_travel_times.csv"
    return RESOURCES_DIR / city_filename

DEFAULT_CITY = "Kaunas, Lithuania"
DATA_FILENAME = get_graph_file_path(DEFAULT_CITY)
TRAVEL_TIMES_CSV = get_travel_times_path(DEFAULT_CITY)