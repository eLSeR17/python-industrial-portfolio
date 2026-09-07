"""Tests for geographic utility functions."""

import math

from src.utils.geo_utils import (
    buffer_zone,
    calculate_bearing,
    haversine_distance,
    point_in_circle,
    point_in_polygon,
    simplify_route,
    total_route_distance,
)


class TestHaversineDistance:
    """Verify the Haversine implementation against known distances."""

    def test_nyc_to_la(self) -> None:
        """NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) ≈ 3 944 km."""
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < d < 4_000_000

    def test_same_point(self) -> None:
        """Distance from a point to itself must be zero."""
        assert haversine_distance(10.0, 20.0, 10.0, 20.0) == 0.0

    def test_symmetry(self) -> None:
        """Distance must be symmetric."""
        a = haversine_distance(51.5, -0.1, 48.8, 2.3)
        b = haversine_distance(48.8, 2.3, 51.5, -0.1)
        assert abs(a - b) < 1e-6

    def test_known_short_distance(self) -> None:
        """1 degree latitude at equator ≈ 111 km."""
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < d < 112_000


class TestCalculateBearing:
    """Verify compass bearing calculations."""

    def test_north(self) -> None:
        """Going due north → bearing ≈ 0."""
        b = calculate_bearing(40.0, -74.0, 41.0, -74.0)
        assert abs(b) < 0.01 or abs(b - 360) < 0.01

    def test_east(self) -> None:
        """Going due east → bearing ≈ 90."""
        b = calculate_bearing(40.0, -74.0, 40.0, -73.0)
        assert 89 < b < 91

    def test_south(self) -> None:
        """Going due south → bearing ≈ 180."""
        b = calculate_bearing(41.0, -74.0, 40.0, -74.0)
        assert 179 < b < 181

    def test_west(self) -> None:
        """Going due west → bearing ≈ 270."""
        b = calculate_bearing(40.0, -74.0, 40.0, -75.0)
        assert 269 < b < 271


class TestPointInPolygon:
    """Ray-casting algorithm tests."""

    def test_point_inside_square(self) -> None:
        """Point clearly inside a unit square."""
        square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        assert point_in_polygon(0.5, 0.5, square) is True

    def test_point_outside_square(self) -> None:
        """Point clearly outside the square."""
        square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        assert point_in_polygon(2.0, 2.0, square) is False

    def test_point_on_edge(self) -> None:
        """Point on the boundary — algorithm-dependent but must not crash."""
        square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        result = point_in_polygon(0.0, 0.5, square)
        assert isinstance(result, bool)

    def test_triangle(self) -> None:
        """Point inside a triangle."""
        triangle = [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0)]
        assert point_in_polygon(0.3, 0.3, triangle) is True

    def test_small_polygon(self) -> None:
        """Fewer than 3 vertices → always outside."""
        tiny = [(0.0, 0.0), (1.0, 1.0)]
        assert point_in_polygon(0.5, 0.5, tiny) is False


class TestPointInCircle:
    """Circular containment tests."""

    def test_inside(self) -> None:
        assert point_in_circle(0.001, 0.001, 0.0, 0.0, 200.0) is True

    def test_outside(self) -> None:
        assert point_in_circle(1.0, 1.0, 0.0, 0.0, 100.0) is False

    def test_on_boundary(self) -> None:
        """Exactly on the radius — should be inside (≤)."""
        d = haversine_distance(0.0, 0.0, 0.001, 0.0)
        assert point_in_circle(0.0, 0.0, 0.001, 0.0, d) is True


class TestBufferZone:
    """Verify circular GeoJSON generation."""

    def test_output_structure(self) -> None:
        geojson = buffer_zone(40.0, -74.0, 100.0, num_points=16)
        assert geojson["type"] == "Polygon"
        coords = geojson["coordinates"][0]
        assert len(coords) == 17  # 16 + closing point

    def test_radius_approximation(self) -> None:
        """Outer points should be roughly *radius_m* from the centre."""
        geojson = buffer_zone(40.0, -74.0, 500.0, num_points=32)
        for pt in geojson["coordinates"][0][:-1]:
            d = haversine_distance(40.0, -74.0, pt[1], pt[0])
            assert abs(d - 500.0) < 20.0  # small numerical error OK


class TestSimplifyRoute:
    """Douglas-Peucker simplification."""

    def test_straight_line(self) -> None:
        """A perfectly straight route should collapse to endpoints."""
        route = [(0.0, float(i)) for i in range(100)]
        simplified = simplify_route(route, tolerance_m=1.0)
        assert len(simplified) == 2

    def test_preserves_ends(self) -> None:
        route = [(0.0, 0.0), (0.001, 0.001), (0.0, 0.002)]
        simplified = simplify_route(route, tolerance_m=10.0)
        assert simplified[0] == route[0]
        assert simplified[-1] == route[-1]

    def test_two_points(self) -> None:
        """Two-point route can't be simplified further."""
        route = [(0.0, 0.0), (1.0, 1.0)]
        assert simplify_route(route, tolerance_m=1.0) == route


class TestTotalRouteDistance:
    """Sum of segment distances."""

    def test_known_route(self) -> None:
        """Two points ~111 km apart (1° lat)."""
        d = total_route_distance([(0.0, 0.0), (1.0, 0.0)])
        assert 110 < d < 112

    def test_empty_route(self) -> None:
        assert total_route_distance([]) == 0.0

    def test_single_point(self) -> None:
        assert total_route_distance([(1.0, 1.0)]) == 0.0

    def test_multi_segment(self) -> None:
        """Sum of individual segments equals total."""
        route = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        total = total_route_distance(route)
        seg1 = haversine_distance(0, 0, 1, 0) / 1000
        seg2 = haversine_distance(1, 0, 1, 1) / 1000
        assert abs(total - (seg1 + seg2)) < 0.001
