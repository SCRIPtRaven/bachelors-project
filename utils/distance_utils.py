import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.

    The Haversine formula determines the great-circle distance between two points on a sphere
    given their latitudes and longitudes. This is particularly useful for calculating
    distances between geographical coordinates on Earth.

    Args:
        lat1, lon1: Latitude and longitude of the first point in degrees
        lat2, lon2: Latitude and longitude of the second point in degrees

    Returns:
        Distance between the points in meters
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371000

    return c * r
