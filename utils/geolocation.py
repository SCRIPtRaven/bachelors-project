from functools import lru_cache

import osmnx as ox


@lru_cache(maxsize=32)
def get_city_coordinates(city_name):
    """
    Get the center coordinates for a city using OpenStreetMap's geocoding service.

    This function uses caching to avoid repeated API calls for the same city name.
    The cache stores up to 32 different city results.

    Args:
        city_name (str): Full city name with country, e.g. "Kaunas, Lithuania"

    Returns:
        tuple: A pair of ((latitude, longitude), zoom_level)
            The coordinates are floats representing the city center
            The zoom level is automatically calculated based on city size

    Example:
        coords, zoom = get_city_coordinates("Kaunas, Lithuania")
        # Returns something like ((54.8985, 23.9036), 12)
    """
    try:
        location = ox.geocode(city_name)
        zoom = calculate_zoom_level(city_name)
        return (location[0], location[1]), zoom
    except Exception as e:
        print(f"Warning: Could not geocode {city_name}. Using default coordinates. Error: {e}")
        return (54.8985, 23.9036), 12


def calculate_zoom_level(city_name):
    """
    Calculate an appropriate zoom level for a city based on its geographical size.

    This function analyzes the city's boundaries to determine an appropriate
    zoom level for map display. Larger cities get a lower zoom level (more zoomed out)
    while smaller cities get a higher zoom level (more zoomed in).

    Args:
        city_name (str): Full city name with country, e.g. "Kaunas, Lithuania"

    Returns:
        int: A zoom level value between 11 and 13
            11 - Large cities
            12 - Medium cities
            13 - Small cities
    """
    try:
        gdf = ox.geocode_to_gdf(city_name)

        bounds = gdf.total_bounds
        width = abs(bounds[2] - bounds[0])

        if width > 0.5:  # Large city
            return 11
        elif width > 0.2:  # Medium city
            return 12
        else:  # Small city
            return 13
    except Exception as e:
        print(f"Warning: Could not calculate zoom level for {city_name}. Using default. Error: {e}")
        return 12
