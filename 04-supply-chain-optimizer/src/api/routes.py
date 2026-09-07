"""FastAPI route definitions for the supply chain optimizer API."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.models.network import SupplyChainNetwork
from src.models.schemas import (
    BullwhipRequest,
    CostBreakdownRequest,
    CostBreakdownResponse,
    InventoryOptimizeRequest,
    InventoryOptimizeResponse,
    NetworkCreateRequest,
    NetworkCreateResponse,
    RouteOptimizeRequest,
    RouteOptimizeResponse,
    SupplierAnalyzeRequest,
    SupplierAnalyzeResponse,
    WhatIfRequest,
    WhatIfResponse,
    WhatIfParameter,
    WhatIfScenario,
)
from src.services.cost_analyzer import CostAnalyzer
from src.services.demand_forecaster import DemandForecaster
from src.services.inventory_optimizer import InventoryOptimizer
from src.services.network_builder import NetworkBuilder
from src.services.route_optimizer import RouteOptimizer
from src.services.supplier_scorer import SupplierScorer

router = APIRouter()

# In-memory network store (replace with Redis/DB in production)
_networks: dict[str, Any] = {}
_builder = NetworkBuilder()


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

@router.post(
    "/network/create",
    response_model=NetworkCreateResponse,
    summary="Create supply chain network",
)
async def create_network(request: NetworkCreateRequest) -> NetworkCreateResponse:
    """Build a supply chain graph from nodes and edges.

    Validates connectivity, capacity feasibility, and returns network
    summary with centrality metrics.
    """
    try:
        network, response = _builder.build(request)
        _networks[response.network_id] = network
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get(
    "/networks",
    summary="List all networks",
)
async def list_networks() -> list[dict[str, Any]]:
    """Return summaries of all loaded networks."""
    return [
        {
            "network_id": nid,
            "nodes": net.graph.number_of_nodes(),
            "edges": net.graph.number_of_edges(),
        }
        for nid, net in _networks.items()
    ]


# --------------------------------------------------------------------------
# Route optimization
# --------------------------------------------------------------------------

@router.get(
    "/optimize/route",
    response_model=RouteOptimizeResponse,
    summary="Optimize delivery routes",
)
async def optimize_route(
    network_id: str,
    depot_ids: str,
    customer_ids: str,
    vehicle_count: int = 5,
    vehicle_capacity: float = 1000.0,
) -> RouteOptimizeResponse:
    """Solve the capacitated vehicle routing problem.

    `depot_ids` and `customer_ids` are comma-separated node IDs.
    """
    net = _get_network(network_id)
    optimizer = RouteOptimizer(net)

    depots = [d.strip() for d in depot_ids.split(",")]
    customers = [c.strip() for c in customer_ids.split(",")]

    request = RouteOptimizeRequest(
        depot_ids=depots,
        customer_ids=customers,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
    )

    try:
        return optimizer.solve_vrp(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


# --------------------------------------------------------------------------
# Inventory optimization
# --------------------------------------------------------------------------

@router.get(
    "/optimize/inventory",
    response_model=InventoryOptimizeResponse,
    summary="Optimize inventory levels",
)
async def optimize_inventory(
    annual_demand: float,
    unit_cost: float = 10.0,
    ordering_cost: float = 50.0,
    holding_cost_pct: float = 0.25,
    lead_time_days: float = 7.0,
    demand_std_dev: float = 0.0,
    service_level: float = 0.95,
) -> InventoryOptimizeResponse:
    """Compute EOQ, safety stock, and reorder point."""
    optimizer = InventoryOptimizer()
    request = InventoryOptimizeRequest(
        annual_demand=annual_demand,
        ordering_cost=ordering_cost,
        holding_cost_pct=holding_cost_pct,
        unit_cost=unit_cost,
        lead_time_days=lead_time_days,
        demand_std_dev=demand_std_dev,
        service_level=service_level,
    )
    return optimizer.optimize(request)


@router.post(
    "/optimize/bullwhip",
    summary="Analyze bullwhip effect",
)
async def analyze_bullwhip(request: BullwhipRequest) -> dict[str, Any]:
    """Quantify demand amplification across supply chain echelons."""
    optimizer = InventoryOptimizer()
    return optimizer.analyze_bullwhip(request)


# --------------------------------------------------------------------------
# Supplier scoring
# --------------------------------------------------------------------------

@router.post(
    "/analyze/supplier",
    response_model=SupplierAnalyzeResponse,
    summary="Score and rank suppliers",
)
async def analyze_suppliers(request: SupplierAnalyzeRequest) -> SupplierAnalyzeResponse:
    """Multi-criteria supplier evaluation with sensitivity analysis."""
    scorer = SupplierScorer()

    # Validate weights sum to ~1.0
    w = request.weights
    total = w.price + w.quality + w.lead_time + w.reliability + w.esg_risk
    if abs(total - 1.0) > 0.01:
        raise HTTPException(
            status_code=422,
            detail=f"Weights must sum to 1.0, got {total:.3f}",
        )

    return scorer.analyze(request)


# --------------------------------------------------------------------------
# Cost breakdown
# --------------------------------------------------------------------------

@router.post(
    "/cost-breakdown",
    response_model=CostBreakdownResponse,
    summary="Full landed cost breakdown",
)
async def cost_breakdown(request: CostBreakdownRequest) -> CostBreakdownResponse:
    """Decompose total cost into material, transport, duties, handling, etc."""
    analyzer = CostAnalyzer.__new__(CostAnalyzer)
    return analyzer.cost_breakdown(request)


@router.get(
    "/cost/network",
    summary="Total network cost analysis",
)
async def network_cost(network_id: str) -> dict[str, Any]:
    """Compute total cost across the entire supply chain."""
    net = _get_network(network_id)
    analyzer = CostAnalyzer(net)
    return analyzer.total_network_cost()


# --------------------------------------------------------------------------
# What-if simulation
# --------------------------------------------------------------------------

@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    summary="Scenario simulation",
)
async def what_if_analysis(request: WhatIfRequest) -> WhatIfResponse:
    """Simulate cost impact of parameter changes.

    Each parameter is multiplied by its factor (1.0 = no change, 1.2 = +20%).
    Returns per-scenario cost impact and a recommendation.
    """
    if not request.baseline_cost:
        # Try to compute from a network if available
        if _networks:
            net = next(iter(_networks.values()))
            analyzer = CostAnalyzer(net)
            baseline = analyzer.total_network_cost()["total_cost"]
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide baseline_cost or create a network first",
            )
    else:
        baseline = request.baseline_cost

    # Run single combined scenario
    total_factor = 1.0
    impact_breakdown: dict[str, float] = {}
    for p in request.parameters:
        total_factor *= p.factor
        impact_breakdown[p.parameter] = round((p.factor - 1.0) * baseline, 2)

    scenario_cost = baseline * total_factor
    delta = scenario_cost - baseline

    scenario = WhatIfScenario(
        parameters=request.parameters,
        baseline_cost=round(baseline, 2),
        scenario_cost=round(scenario_cost, 2),
        delta=round(delta, 2),
        delta_pct=round((delta / baseline * 100) if baseline else 0, 2),
        impact_breakdown=impact_breakdown,
    )

    # Identify most impactful parameter
    if impact_breakdown:
        most_impactful = max(impact_breakdown, key=lambda k: abs(impact_breakdown[k]))
    else:
        most_impactful = "none"

    # Generate recommendation
    if delta > 0:
        rec = (
            f"Cost increase of {abs(scenario.delta_pct):.1f}% predicted. "
            f"Primary driver: {most_impactful}. Consider mitigation strategies."
        )
    elif delta < 0:
        rec = (
            f"Cost reduction of {abs(scenario.delta_pct):.1f}% expected. "
            f"Primary benefit from: {most_impactful}."
        )
    else:
        rec = "No net cost impact from proposed changes."

    return WhatIfResponse(
        scenarios=[scenario],
        most_impactful=most_impactful,
        recommendation=rec,
    )


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

@router.post(
    "/forecast/demand",
    summary="Demand forecast",
)
async def forecast_demand(
    historical: list[float],
    method: str = "auto",
    periods_ahead: int = 6,
    seasonality_period: int | None = None,
) -> dict[str, Any]:
    """Run demand forecasting on historical data.

    Methods: auto, ses, holt, moving_average, seasonal.
    """
    if len(historical) < 3:
        raise HTTPException(status_code=422, detail="Need at least 3 data points")

    forecaster = DemandForecaster()

    if method == "auto":
        return forecaster.auto_forecast(historical, periods_ahead, seasonality_period)
    elif method == "ses":
        return forecaster.exponential_smoothing(historical, periods_ahead=periods_ahead)
    elif method == "holt":
        return forecaster.holt_smoothing(historical, periods_ahead=periods_ahead)
    elif method == "moving_average":
        return forecaster.moving_average(historical, periods_ahead=periods_ahead)
    elif method == "seasonal":
        if not seasonality_period:
            raise HTTPException(
                status_code=422,
                detail="seasonality_period required for seasonal method",
            )
        return forecaster.seasonal_decompose(
            historical, period=seasonality_period, periods_ahead=periods_ahead
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown method: {method}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get_network(network_id: str) -> SupplyChainNetwork:
    """Retrieve a network by ID or raise 404."""
    net = _networks.get(network_id)
    if net is None:
        raise HTTPException(
            status_code=404,
            detail=f"Network '{network_id}' not found. Create one first.",
        )
    return net
