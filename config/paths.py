from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESOURCES_DIR = BASE_DIR / 'resources'
DATA_FILENAME = RESOURCES_DIR / 'kaunas.graphml'
TRAVEL_TIMES_CSV = RESOURCES_DIR / 'adjusted_travel_times.csv'
MAP_HTML = RESOURCES_DIR / 'map.html'
