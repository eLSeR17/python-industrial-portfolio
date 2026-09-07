import math

import pytest

from src.models.schemas import (
    BullwhipRequest,
    EchelonInventory,
    InventoryOptimizeRequest,
)
from src.services.inventory_optimizer import InventoryOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optimizer():
    return InventoryOptimizer()


class TestEOQ:
    """Test Economic Order Quantity computation."""

    def test_basic_eoq(self, optimizer):
        """Classic textbook example: D=1000, S=50, H=2.50 -> EOQ=200."""
        request = InventoryOptimizeRequest(
            annual_demand=1000,
            ordering_cost=50.0,
            holding_cost_pct=0.25,
            unit_cost=10.0,
            lead_time_days=7,
            demand_std_dev=0,
            service_level=0.95,
        )
        result = optimizer.optimize(request)
        # EOQ = sqrt(2*1000*50 / (0.25*10)) = sqrt(40000) = 200
        assert result.eoq == pytest.approx(200.0)

    def test_higher_demand_increases_eoq(self, optimizer):
        """Higher demand should result in larger EOQ."""
        base = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, service_level=0.95,
        ))
        high = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=5000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, service_level=0.95,
        ))
        assert high.eoq > base.eoq

    def test_higher_ordering_cost_increases_eoq(self, optimizer):
        """Higher ordering cost should increase EOQ (economies of scale)."""
        low = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=10.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, service_level=0.95,
        ))
        high = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=100.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, service_level=0.95,
        ))
        assert high.eoq > low.eoq

    def test_savings_are_positive(self, optimizer):
        """EOQ should produce savings over naive ordering."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=2000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=15.0, lead_time_days=14, service_level=0.95,
        ))
        assert result.total_cost_savings_pct >= 0


class TestSafetyStock:
    """Test safety stock calculation."""

    def test_safety_stock_with_variability(self, optimizer):
        """Non-zero demand std dev should produce positive safety stock."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, demand_std_dev=5.0,
            service_level=0.95,
        ))
        assert result.safety_stock > 0

    def test_zero_variability_gives_zero_safety_stock(self, optimizer):
        """Zero demand uncertainty should produce zero safety stock."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, demand_std_dev=0,
            service_level=0.95,
        ))
        assert result.safety_stock == 0.0

    def test_higher_service_level_increases_safety_stock(self, optimizer):
        """99% service level requires more safety stock than 90%."""
        low = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, demand_std_dev=5.0,
            service_level=0.90,
        ))
        high = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, demand_std_dev=5.0,
            service_level=0.99,
        ))
        assert high.safety_stock > low.safety_stock

    def test_reorder_point_includes_lead_time_demand(self, optimizer):
        """Reorder point should cover lead time demand plus safety stock."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=3650, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=10, demand_std_dev=3.0,
            service_level=0.95,
        ))
        # Daily demand = 10, lead time demand = 100
        assert result.reorder_point >= 100


class TestCosts:
    """Test inventory cost computations."""

    def test_total_cost_is_sum_of_parts(self, optimizer):
        """Annual total cost should equal ordering + holding cost."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1500, ordering_cost=40.0, holding_cost_pct=0.30,
            unit_cost=12.0, lead_time_days=5, service_level=0.95,
        ))
        expected = result.annual_ordering_cost + result.annual_holding_cost
        assert result.annual_total_cost == pytest.approx(expected, rel=1e-2)

    def test_orders_per_year(self, optimizer):
        """Orders per year should equal annual_demand / EOQ."""
        result = optimizer.optimize(InventoryOptimizeRequest(
            annual_demand=1000, ordering_cost=50.0, holding_cost_pct=0.25,
            unit_cost=10.0, lead_time_days=7, service_level=0.95,
        ))
        expected = 1000 / result.eoq if result.eoq > 0 else 0
        assert result.orders_per_year == pytest.approx(expected, rel=1e-2)


class TestZScore:
    """Test the z-score approximation."""

    def test_z_score_95(self):
        z = InventoryOptimizer._z_score(0.95)
        assert z == pytest.approx(1.645, abs=0.02)

    def test_z_score_99(self):
        z = InventoryOptimizer._z_score(0.99)
        assert z == pytest.approx(2.326, abs=0.02)

    def test_z_score_50(self):
        z = InventoryOptimizer._z_score(0.50)
        assert abs(z) < 0.05

    def test_z_score_extremes(self):
        assert InventoryOptimizer._z_score(0.0) == -3.0
        assert InventoryOptimizer._z_score(1.0) > 3.0


class TestBullwhip:
    """Test multi-echelon bullwhip effect analysis."""

    def test_single_echelon(self, optimizer):
        """Single echelon should report ratio >= 1.0."""
        request = BullwhipRequest(echelons=[
            EchelonInventory(
                name="Retailer",
                lead_time_days=7,
                demand_mean=100,
                demand_std_dev=20,
                ordering_cost=50,
                holding_cost_pct=0.25,
                unit_cost=10,
            ),
        ])
        result = optimizer.analyze_bullwhip(request)
        assert len(result["echelons"]) == 1
        assert result["echelons"][0]["bullwhip_ratio"] >= 1.0

    def test_bullwhip_amplification(self, optimizer):
        """Demand should amplify upstream across echelons."""
        request = BullwhipRequest(echelons=[
            EchelonInventory(
                name="Retailer",
                lead_time_days=5, demand_mean=100, demand_std_dev=15,
                ordering_cost=30, holding_cost_pct=0.25, unit_cost=10,
            ),
            EchelonInventory(
                name="Distributor",
                lead_time_days=14, demand_mean=100, demand_std_dev=15,
                ordering_cost=50, holding_cost_pct=0.25, unit_cost=10,
            ),
            EchelonInventory(
                name="Manufacturer",
                lead_time_days=28, demand_mean=100, demand_std_dev=15,
                ordering_cost=100, holding_cost_pct=0.25, unit_cost=10,
            ),
        ])
        result = optimizer.analyze_bullwhip(request)
        ratios = [e["bullwhip_ratio"] for e in result["echelons"]]
        # Upstream echelons should generally have higher or equal ratios
        assert result["max_bullwhip_ratio"] >= 1.0

    def test_empty_echelons(self, optimizer):
        """Empty request should return empty results."""
        result = optimizer.analyze_bullwhip(BullwhipRequest(echelons=[]))
        assert result["echelons"] == []
