import pytest

from src.models.schemas import (
    CostBreakdownRequest,
    EdgeCreate,
    NetworkCreateRequest,
    NodeCreate,
    NodeType,
    SupplierAnalyzeRequest,
    SupplierMetrics,
    SupplierWeights,
)
from src.services.cost_analyzer import CostAnalyzer
from src.services.network_builder import NetworkBuilder
from src.services.supplier_scorer import SupplierScorer
from src.utils.cost_models import (
    cost_optimization_report,
    holding_cost_model,
    sensitivity_analysis,
    total_landed_cost,
    transportation_cost_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_test_network():
    """Build a minimal network for cost analysis tests."""
    request = NetworkCreateRequest(
        nodes=[
            NodeCreate(id="S1", name="Supplier A", type=NodeType.SUPPLIER,
                       capacity=5000, variable_cost=3.0, fixed_cost=1000),
            NodeCreate(id="WH1", name="Warehouse Central", type=NodeType.WAREHOUSE,
                       capacity=8000, fixed_cost=5000),
            NodeCreate(id="C1", name="Customer East", type=NodeType.CUSTOMER, demand=1000),
            NodeCreate(id="C2", name="Customer West", type=NodeType.CUSTOMER, demand=800),
        ],
        edges=[
            EdgeCreate(source="S1", target="WH1", distance_km=200, cost_per_unit=4.0,
                       transit_time_hours=3, capacity=5000),
            EdgeCreate(source="WH1", target="C1", distance_km=150, cost_per_unit=3.0,
                       transit_time_hours=2, capacity=3000),
            EdgeCreate(source="WH1", target="C2", distance_km=300, cost_per_unit=5.5,
                       transit_time_hours=5, capacity=3000),
        ],
    )
    builder = NetworkBuilder()
    return builder.build(request)[0]


@pytest.fixture
def network():
    return _build_test_network()


@pytest.fixture
def analyzer(network):
    return CostAnalyzer(network)


# ---------------------------------------------------------------------------
# Landed cost tests
# ---------------------------------------------------------------------------

class TestLandedCost:
    """Test landed cost computation between nodes."""

    def test_basic_landed_cost(self, analyzer):
        """Landed cost from S1 to C1 should include material + transport + handling."""
        result = analyzer.compute_landed_costs("S1", "C1", quantity=100)
        assert result["total_per_unit"] > 0
        assert result["material_cost_per_unit"] == pytest.approx(3.0)
        # Shortest cost path S1->WH1->C1 = 4.0 + 3.0 = 7.0
        assert result["transport_cost_per_unit"] == pytest.approx(7.0)
        assert result["total_cost"] == pytest.approx(
            result["total_per_unit"] * 100
        )

    def test_landed_cost_scales_with_quantity(self, analyzer):
        """Total cost should scale linearly with quantity."""
        r1 = analyzer.compute_landed_costs("S1", "C1", quantity=100)
        r2 = analyzer.compute_landed_costs("S1", "C1", quantity=500)
        assert r2["total_cost"] == pytest.approx(r1["total_cost"] * 5, rel=1e-3)

    def test_no_path_raises(self, network):
        """Requesting landed cost for disconnected nodes should raise."""
        analyzer = CostAnalyzer(network)
        with pytest.raises(ValueError, match="No path"):
            analyzer.compute_landed_costs("S1", "NONEXISTENT", quantity=100)


# ---------------------------------------------------------------------------
# Cost breakdown tests
# ---------------------------------------------------------------------------

class TestCostBreakdown:
    """Test full cost decomposition."""

    def test_breakdown_sums_correctly(self, analyzer):
        """Component costs should sum to total landed cost."""
        request = CostBreakdownRequest(
            material_cost=1000,
            transport_cost=500,
            duty_cost=100,
            handling_cost=75,
            storage_cost=50,
            insurance_cost=30,
            quantity=100,
            markup_pct=0.0,
        )
        result = analyzer.cost_breakdown(request)
        expected = 1000 + 500 + 100 + 75 + 50 + 30
        assert result.total_landed_cost == pytest.approx(expected)

    def test_markup_applied(self, analyzer):
        """Markup should increase the final cost."""
        base = CostBreakdownRequest(
            material_cost=1000, transport_cost=500, quantity=100, markup_pct=0.0,
        )
        marked = CostBreakdownRequest(
            material_cost=1000, transport_cost=500, quantity=100, markup_pct=0.15,
        )
        r1 = analyzer.cost_breakdown(base)
        r2 = analyzer.cost_breakdown(marked)
        assert r2.total_with_markup > r1.total_with_markup

    def test_per_unit_cost(self, analyzer):
        """Per-unit cost should equal total with markup / quantity."""
        request = CostBreakdownRequest(
            material_cost=500, transport_cost=250, quantity=50, markup_pct=0.10,
        )
        result = analyzer.cost_breakdown(request)
        assert result.per_unit_cost == pytest.approx(
            result.total_with_markup / 50, rel=1e-4
        )

    def test_percentages_sum_to_100(self, analyzer):
        """Component percentages should approximately sum to 100%."""
        request = CostBreakdownRequest(
            material_cost=800, transport_cost=400, duty_cost=200,
            handling_cost=100, storage_cost=50, insurance_cost=50,
            quantity=200,
        )
        result = analyzer.cost_breakdown(request)
        total_pct = sum(result.component_pct.values())
        assert total_pct == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# Network-wide cost tests
# ---------------------------------------------------------------------------

class TestNetworkCost:
    """Test total network cost computation."""

    def test_network_cost_positive(self, analyzer):
        """Total network cost should be positive."""
        result = analyzer.total_network_cost()
        assert result["total_cost"] > 0

    def test_cost_categories(self, analyzer):
        """Cost breakdown should include fixed, variable, and transport."""
        result = analyzer.total_network_cost()
        assert "fixed_costs" in result
        assert "variable_costs" in result
        assert "transport_costs" in result
        assert result["fixed_costs"]["total"] > 0

    def test_cost_by_node_type(self, analyzer):
        """Costs should be attributable to node types."""
        result = analyzer.total_network_cost()
        by_type = result["cost_by_node_type"]
        assert "supplier" in by_type
        assert "warehouse" in by_type
        assert "customer" in by_type


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    """Test sensitivity analysis on cost parameters."""

    def test_sensitivity_returns_all_params(self, analyzer):
        """Each parameter should have a sweep result."""
        results = analyzer.sensitivity_analysis(
            base_cost=10000,
            parameters={"fuel": (0.8, 1.2), "labor": (0.9, 1.1)},
        )
        assert "fuel" in results
        assert "labor" in results

    def test_sensitivity_midpoint_matches_base(self, analyzer):
        """At factor=1.0, cost should equal base cost."""
        results = analyzer.sensitivity_analysis(
            base_cost=10000,
            parameters={"param1": (0.8, 1.2)},
        )
        sweep = results["param1"]
        mid = sweep[len(sweep) // 2]  # factor=1.0
        assert mid["cost"] == pytest.approx(10000.0)
        assert mid["delta_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Supplier scoring tests
# ---------------------------------------------------------------------------

class TestSupplierScoring:
    """Test multi-criteria supplier analysis."""

    @pytest.fixture
    def sample_suppliers(self):
        return [
            SupplierMetrics(
                supplier_id="SUP-A", name="Alpha Corp",
                unit_price=10.0, defect_rate_ppm=50, lead_time_days=7,
                on_time_delivery_pct=95.0, esg_risk_score=20,
            ),
            SupplierMetrics(
                supplier_id="SUP-B", name="Beta Inc",
                unit_price=8.5, defect_rate_ppm=200, lead_time_days=10,
                on_time_delivery_pct=88.0, esg_risk_score=45,
            ),
            SupplierMetrics(
                supplier_id="SUP-C", name="Gamma LLC",
                unit_price=12.0, defect_rate_ppm=10, lead_time_days=5,
                on_time_delivery_pct=98.0, esg_risk_score=15,
            ),
        ]

    def test_all_suppliers_scored(self, sample_suppliers):
        """Every supplier should receive a score."""
        scorer = SupplierScorer()
        request = SupplierAnalyzeRequest(suppliers=sample_suppliers)
        result = scorer.analyze(request)
        assert len(result.scores) == 3

    def test_scores_ranked(self, sample_suppliers):
        """Scores should be ranked 1 to N."""
        scorer = SupplierScorer()
        request = SupplierAnalyzeRequest(suppliers=sample_suppliers)
        result = scorer.analyze(request)
        ranks = [s.rank for s in result.scores]
        assert sorted(ranks) == [1, 2, 3]

    def test_recommended_is_top_scored(self, sample_suppliers):
        """Recommended supplier should match the top-ranked one."""
        scorer = SupplierScorer()
        request = SupplierAnalyzeRequest(suppliers=sample_suppliers)
        result = scorer.analyze(request)
        top = result.scores[0]
        assert result.recommended_supplier == top.supplier_id

    def test_weights_affect_ranking(self, sample_suppliers):
        """Different weights should potentially change the winner."""
        scorer = SupplierScorer()

        price_heavy = SupplierAnalyzeRequest(
            suppliers=sample_suppliers,
            weights=SupplierWeights(price=0.70, quality=0.10, lead_time=0.05,
                                    reliability=0.10, esg_risk=0.05),
        )
        quality_heavy = SupplierAnalyzeRequest(
            suppliers=sample_suppliers,
            weights=SupplierWeights(price=0.10, quality=0.70, lead_time=0.05,
                                    reliability=0.10, esg_risk=0.05),
        )
        r1 = scorer.analyze(price_heavy)
        r2 = scorer.analyze(quality_heavy)
        # At least one should have different top supplier
        # (not guaranteed but very likely with these weights)
        assert r1.scores[0].weighted_score != r2.scores[0].weighted_score

    def test_sensitivity_provided(self, sample_suppliers):
        """Sensitivity analysis should be returned for each criterion."""
        scorer = SupplierScorer()
        request = SupplierAnalyzeRequest(
            suppliers=sample_suppliers,
            sensitivity_perturbation=0.10,
        )
        result = scorer.analyze(request)
        assert "price" in result.sensitivity
        assert "quality" in result.sensitivity

    def test_strengths_weaknesses_identified(self, sample_suppliers):
        """Suppliers should have strengths and weaknesses listed."""
        scorer = SupplierScorer()
        request = SupplierAnalyzeRequest(suppliers=sample_suppliers)
        result = scorer.analyze(request)
        for score in result.scores:
            assert isinstance(score.strengths, list)
            assert isinstance(score.weaknesses, list)


# ---------------------------------------------------------------------------
# Cost model utility tests
# ---------------------------------------------------------------------------

class TestCostModels:
    """Test standalone cost model functions."""

    def test_total_landed_cost(self):
        result = total_landed_cost(
            material_cost=5000, transport_cost=1000, duty_cost=250,
            handling_cost=100, storage_cost=50, insurance_cost=100,
            quantity=500,
        )
        assert result["total_landed_cost"] == 6500
        assert result["per_unit_cost"] == pytest.approx(13.0)

    def test_transportation_cost(self):
        result = transportation_cost_model(
            distance_km=500, weight_tons=10,
            cost_per_km_per_ton=0.35,
        )
        assert result["total_transport_cost"] == pytest.approx(1750.0)

    def test_transportation_cost_minimum_charge(self):
        result = transportation_cost_model(
            distance_km=10, weight_tons=0.5,
            cost_per_km_per_ton=0.35,
            minimum_charge=50.0,
        )
        assert result["total_transport_cost"] == 50.0

    def test_holding_cost_model(self):
        result = holding_cost_model(
            avg_inventory_units=1000, unit_cost=10.0,
        )
        assert result["inventory_value"] == 10000.0
        assert result["total_annual_holding_cost"] == pytest.approx(2500.0)

    def test_optimization_report_savings(self):
        result = cost_optimization_report(
            current_cost=100000, optimized_cost=85000,
            implementation_cost=10000,
            implementation_time_months=3,
        )
        assert result["annual_savings"] == 15000
        assert result["savings_pct"] == pytest.approx(15.0)
        assert result["impact_classification"] == "transformative"
        assert result["payback_months"] == pytest.approx(8.0, abs=0.5)

    def test_optimization_report_no_savings(self):
        result = cost_optimization_report(
            current_cost=100000, optimized_cost=100000,
        )
        assert result["savings_pct"] == 0.0
        assert result["recommendation"] == "Savings may not justify implementation cost"
