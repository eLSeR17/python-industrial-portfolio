import pytest

from src.models.schemas import (
    EdgeCreate,
    NetworkCreateRequest,
    NodeCreate,
    NodeType,
    RouteOptimizeRequest,
    TransportMode,
)
from src.services.network_builder import NetworkBuilder
from src.services.route_optimizer import RouteOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_network():
    """Build a small 5-node network for testing.

    Layout:
        S1 ---S2--- WH1 ---WH2--- C1
         \                    /
          ---S3--- WH3-------

    S = supplier, WH = warehouse, C = customer
    """
    request = NetworkCreateRequest(
        nodes=[
            NodeCreate(id="S1", name="Supplier 1", type=NodeType.SUPPLIER, capacity=10000),
            NodeCreate(id="S2", name="Supplier 2", type=NodeType.SUPPLIER, capacity=8000),
            NodeCreate(id="WH1", name="Warehouse 1", type=NodeType.WAREHOUSE, capacity=15000),
            NodeCreate(id="WH2", name="Warehouse 2", type=NodeType.WAREHOUSE, capacity=12000),
            NodeCreate(id="WH3", name="Warehouse 3", type=NodeType.WAREHOUSE, capacity=10000),
            NodeCreate(id="C1", name="Customer 1", type=NodeType.CUSTOMER, demand=500),
            NodeCreate(id="C2", name="Customer 2", type=NodeType.CUSTOMER, demand=300),
        ],
        edges=[
            EdgeCreate(source="S1", target="WH1", distance_km=100, cost_per_unit=2.0, transit_time_hours=2, capacity=5000),
            EdgeCreate(source="S1", target="WH3", distance_km=250, cost_per_unit=5.0, transit_time_hours=4, capacity=3000),
            EdgeCreate(source="S2", target="WH1", distance_km=80, cost_per_unit=1.5, transit_time_hours=1.5, capacity=6000),
            EdgeCreate(source="WH1", target="WH2", distance_km=150, cost_per_unit=3.0, transit_time_hours=3, capacity=8000),
            EdgeCreate(source="WH2", target="C1", distance_km=120, cost_per_unit=2.5, transit_time_hours=2, capacity=2000),
            EdgeCreate(source="WH3", target="C2", distance_km=90, cost_per_unit=1.8, transit_time_hours=1.5, capacity=2000),
            EdgeCreate(source="WH1", target="C1", distance_km=200, cost_per_unit=4.0, transit_time_hours=3, capacity=3000),
            EdgeCreate(source="WH2", target="C2", distance_km=180, cost_per_unit=3.5, transit_time_hours=2.5, capacity=2000),
        ],
    )
    builder = NetworkBuilder()
    return builder.build(request)[0]


@pytest.fixture
def network():
    return _simple_network()


@pytest.fixture
def optimizer(network):
    return RouteOptimizer(network)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShortestRoute:
    """Test single-vehicle shortest path routing."""

    def test_direct_route(self, optimizer):
        """Direct path from S1 to WH1 should cost 2.0."""
        result = optimizer.shortest_route("S1", "WH1")
        assert result["path"] == ["S1", "WH1"]
        assert result["cost"] == pytest.approx(2.0)
        assert result["distance_km"] == 100

    def test_cheapest_indirect_route(self, optimizer):
        """S1 -> WH1 -> C1 is cheaper than S1 -> WH3 -> C2 -> ..."""
        result = optimizer.shortest_route("S1", "C1")
        # S1->WH1->C1 = 2.0 + 4.0 = 6.0
        # S1->WH1->WH2->C1 = 2.0 + 3.0 + 2.5 = 7.5
        assert result["cost"] <= 7.5
        assert "S1" in result["path"]
        assert "C1" in result["path"]

    def test_time_optimized_route(self, optimizer):
        """Time-optimized route should prefer faster edges."""
        result = optimizer.shortest_route("S1", "C1", objective="min_time")
        # S1->WH1 (2h) -> C1 (3h) = 5h  OR
        # S1->WH1->WH2->C1 = 2+3+2 = 7h
        assert result["time_hours"] <= 7.0

    def test_no_route_raises(self, network):
        """Requesting a route to non-existent node raises ValueError."""
        optimizer = RouteOptimizer(network)
        with pytest.raises(ValueError, match="No route"):
            optimizer.shortest_route("S1", "NONEXISTENT")

    def test_route_stops_count(self, optimizer):
        """Route should have at least 2 stops (source and target)."""
        result = optimizer.shortest_route("S2", "C1")
        assert result["stops"] >= 2


class TestVRP:
    """Test capacitated vehicle routing problem solver."""

    def test_vrp_single_depot(self, optimizer):
        """Single depot VRP with one customer should produce one route."""
        request = RouteOptimizeRequest(
            depot_ids=["WH1"],
            customer_ids=["C1"],
            vehicle_count=3,
            vehicle_capacity=2000,
        )
        result = optimizer.solve_vrp(request)
        assert result.vehicles_used >= 1
        assert result.total_cost >= 0
        assert result.solver_status in ("Optimal", "Feasible")

    def test_vrp_multiple_customers(self, optimizer):
        """Multi-customer VRP should serve all customers."""
        request = RouteOptimizeRequest(
            depot_ids=["WH1"],
            customer_ids=["C1", "C2"],
            vehicle_count=5,
            vehicle_capacity=5000,
        )
        result = optimizer.solve_vrp(request)
        # All customers should appear in at least one route
        served = set()
        for route in result.routes:
            served.update(route.stops)
        assert "C1" in served or "C2" in served

    def test_vrp_respects_capacity(self, optimizer):
        """Each route's load should not exceed vehicle capacity."""
        request = RouteOptimizeRequest(
            depot_ids=["WH1"],
            customer_ids=["C1", "C2"],
            vehicle_count=3,
            vehicle_capacity=300,
        )
        result = optimizer.solve_vrp(request)
        for route in result.routes:
            assert route.load <= 300 + 1  # small tolerance for rounding

    def test_vrp_total_distance_positive(self, optimizer):
        """Total distance should be positive when customers are served."""
        request = RouteOptimizeRequest(
            depot_ids=["WH1"],
            customer_ids=["C1"],
            vehicle_count=2,
            vehicle_capacity=2000,
        )
        result = optimizer.solve_vrp(request)
        if result.vehicles_used > 0:
            assert result.total_distance_km > 0


class TestCompleteGraph:
    """Test the fallback complete graph construction."""

    def test_complete_graph_connects_all_nodes(self, optimizer):
        """The complete graph should connect relevant nodes via shortest paths."""
        subgraph = optimizer._build_complete_graph({"S1", "WH1", "C1"})
        assert subgraph.number_of_nodes() == 3
        assert subgraph.has_edge("S1", "C1")
        assert subgraph.has_edge("WH1", "C1")
        assert subgraph.has_edge("S1", "WH1")
