"""Pure-Python geographic utilities.

All functions operate on WGS-84 coordinates (latitude / longitude in degrees).
No external GIS library is required.
"""

import math
from typing import Any


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in **metres** between two points.

    Uses the Haversine formula with an Earth radius of 6 371 000 m.
    """
    r_earth = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_earth * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the initial compass bearing in degrees (0–360) from point 1 to point 2.

    0° = North, 90° = East, 180° = South, 270° = West.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)

    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing_rad = math.atan2(x, y)
    return (math.degrees(bearing_rad) + 360) % 360


def point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    """Determine whether a point lies inside a polygon using the ray-casting algorithm.

    *polygon* is a list of ``(lat, lon)`` vertices (closed or open — the
    algorithm implicitly closes the shape).
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_circle(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> bool:
    """Return ``True`` if the point is within *radius_m* metres of the centre."""
    return haversine_distance(lat, lon, center_lat, center_lon) <= radius_m


def buffer_zone(lat: float, lon: float, radius_m: float, num_points: int = 32) -> dict[str, Any]:
    """Create a circular GeoJSON *Polygon* centred on the given point.

    Returns a GeoJSON geometry dict with ``type`` and ``coordinates``.
    """
    r_earth = 6_371_000.0
    coords: list[list[float]] = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dlat = (radius_m / r_earth) * math.cos(angle)
        dlon = (radius_m / (r_earth * math.cos(math.radians(lat)))) * math.sin(angle)
        coords.append([round(lon + math.degrees(dlon), 6), round(lat + math.degrees(dlat), 6)])
    coords.append(coords[0])  # close the ring
    return {"type": "Polygon", "coordinates": [coords]}


def simplify_route(
    route: list[tuple[float, float]],
    tolerance_m: float = 10.0,
) -> list[tuple[float, float]]:
    """Simplify a route using the Douglas-Peucker algorithm.

    *tolerance_m* is the maximum perpendicular distance (in metres) a point
    may deviate from the simplified path.
    """
    if len(route) <= 2:
        return list(route)

    def _point_line_distance_m(
        pt: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        """Approximate perpendicular distance from *pt* to line *a–b* in metres."""
        lat, lon = pt
        lat_a, lon_a = a
        lat_b, lon_b = b
        dx = (lon_b - lon_a) * math.cos(math.radians(lat))
        dy = lat_b - lat_a
        seg_sq = dx * dx + dy * dy
        if seg_sq == 0:
            return haversine_distance(lat, lon, lat_a, lon_a)
        t = max(0.0, min(1.0, ((lon - lon_a) * dx + (lat - lat_a) * dy) / seg_sq))
        proj_lon = lon_a + t * (lon_b - lon_a)
        proj_lat = lat_a + t * (lat_b - lat_a)
        return haversine_distance(lat, lon, proj_lat, proj_lon)

    dmax = 0.0
    index = 0
    end = len(route) - 1
    for i in range(1, end):
        d = _point_line_distance_m(route[i], route[0], route[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > tolerance_m:
        left = simplify_route(route[: index + 1], tolerance_m)
        right = simplify_route(route[index:], tolerance_m)
        return left[:-1] + right
    return [route[0], route[end]]


def total_route_distance(route: list[tuple[float, float]]) -> float:
    """Return the total distance of a route in **kilometres**."""
    if len(route) < 2:
        return 0.0
    total_m = 0.0
    for i in range(1, len(route)):
        total_m += haversine_distance(route[i - 1][0], route[i - 1][1], route[i][0], route[i][1])
    return total_m / 1000.0
