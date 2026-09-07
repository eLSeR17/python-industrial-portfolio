"""Pydantic request/response schemas for the supply chain optimizer API."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    SUPPLIER = "supplier"
    WAREHOUSE = "warehouse"
    DISTRIBUTION_CENTER = "distribution_center"
    CUSTOMER = "customer"
    PORT = "port"
    MANUFACTURING = "manufacturing"


class TransportMode(str, Enum):
    TRUCK = "truck"
    RAIL = "rail"
    SEA = "sea"
    AIR = "air"
    INTERMODAL = "intermodal"


class OptimizationObjective(str, Enum):
    MIN_COST = "min_cost"
    MIN_TIME = "min_time"
    BALANCED = "balanced"


# ---------------------------------------------------------------------------
# Network creation
# ---------------------------------------------------------------------------

class NodeCreate(BaseModel):
    """A single node in the supply chain network."""
    id: str = Field(..., description="Unique node identifier", examples=["SUP-001"])
    name: str = Field(..., description="Human-readable name")
    type: NodeType
    latitude: float = Field(0.0, ge=-90.0, le=90.0)
    longitude: float = Field(0.0, ge=-180.0, le=180.0)
    capacity: float = Field(0.0, ge=0.0, description="Throughput capacity (units/period)")
    fixed_cost: float = Field(0.0, ge=0.0, description="Fixed operating cost per period")
    variable_cost: float = Field(0.0, ge=0.0, description="Variable cost per unit handled")
    demand: float = Field(0.0, ge=0.0, description="Demand (only for customer nodes)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeCreate(BaseModel):
    """A transport route between two nodes."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    transport_mode: TransportMode = TransportMode.TRUCK
    distance_km: float = Field(..., gt=0.0, description="Distance in kilometers")
    cost_per_unit: float = Field(..., ge=0.0, description="Transport cost per unit")
    transit_time_hours: float = Field(..., gt=0.0, description="Transit time in hours")
    capacity: float = Field(..., gt=0.0, description="Maximum flow capacity (units/period)")
    reliability: float = Field(1.0, ge=0.0, le=1.0, description="On-time delivery probability")
    metadata: dict[str, Any] = Field(default_factory=dict)


class NetworkCreateRequest(BaseModel):
    """Full supply chain network definition."""
    nodes: list[NodeCreate] = Field(..., min_length=1)
    edges: list[EdgeCreate] = Field(default_factory=list)


class NetworkCreateResponse(BaseModel):
    """Response after building the supply chain graph."""
    network_id: str
    node_count: int
    edge_count: int
    connected_components: int
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class RouteOptimizeRequest(BaseModel):
    """Parameters for vehicle routing optimization."""
    depot_ids: list[str] = Field(..., min_length=1, description="Depot node IDs")
    customer_ids: list[str] = Field(..., min_length=1, description="Customer node IDs")
    vehicle_count: int = Field(5, gt=0)
    vehicle_capacity: float = Field(1000.0, gt=0.0)
    objective: OptimizationObjective = OptimizationObjective.MIN_COST


class RouteOptimizeResponse(BaseModel):
    """Optimized routes for a fleet."""
    routes: list[RouteResult]
    total_distance_km: float
    total_cost: float
    total_time_hours: float
    vehicles_used: int
    solver_status: str


class RouteResult(BaseModel):
    """A single vehicle route."""
    vehicle_id: int
    stops: list[str]
    distance_km: float
    cost: float
    time_hours: float
    load: float


class InventoryOptimizeRequest(BaseModel):
    """Parameters for inventory optimization."""
    annual_demand: float = Field(..., gt=0.0, description="Annual demand in units")
    ordering_cost: float = Field(50.0, gt=0.0, description="Fixed cost per order")
    holding_cost_pct: float = Field(0.25, gt=0.0, le=1.0, description="Annual holding cost as fraction of unit cost")
    unit_cost: float = Field(10.0, gt=0.0, description="Cost per unit")
    lead_time_days: float = Field(7.0, ge=0.0, description="Supplier lead time in days")
    demand_std_dev: float = Field(0.0, ge=0.0, description="Daily demand standard deviation")
    service_level: float = Field(0.95, gt=0.0, le=1.0, description="Target cycle service level")
    days_per_year: int = Field(365, gt=0)


class InventoryOptimizeResponse(BaseModel):
    """Optimal inventory parameters."""
    eoq: float
    reorder_point: float
    safety_stock: float
    max_inventory: float
    average_inventory: float
    orders_per_year: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_total_cost: float
    total_cost_savings_pct: float
    bullwhip_ratio: float | None = None


class BullwhipRequest(BaseModel):
    """Multi-echelon inventory parameters for bullwhip analysis."""
    echelons: list[EchelonInventory]


class EchelonInventory(BaseModel):
    """Inventory parameters for a single supply chain tier."""
    name: str
    lead_time_days: float
    demand_mean: float
    demand_std_dev: float
    ordering_cost: float
    holding_cost_pct: float
    unit_cost: float


# ---------------------------------------------------------------------------
# Supplier scoring
# ---------------------------------------------------------------------------

class SupplierMetrics(BaseModel):
    """Measurable attributes for a single supplier."""
    supplier_id: str
    name: str
    unit_price: float = Field(..., gt=0.0)
    defect_rate_ppm: float = Field(..., ge=0.0, description="Defects per million units")
    lead_time_days: float = Field(..., gt=0.0)
    on_time_delivery_pct: float = Field(..., ge=0.0, le=100.0)
    esg_risk_score: float = Field(0.0, ge=0.0, le=100.0, description="ESG risk (0=low, 100=high)")
    capacity_utilization_pct: float = Field(50.0, ge=0.0, le=100.0)
    financial_stability_score: float = Field(50.0, ge=0.0, le=100.0)


class SupplierWeights(BaseModel):
    """Weights for multi-criteria supplier scoring (must sum to 1.0)."""
    price: float = 0.30
    quality: float = 0.25
    lead_time: float = 0.15
    reliability: float = 0.20
    esg_risk: float = 0.10


class SupplierAnalyzeRequest(BaseModel):
    """Request to score and rank suppliers."""
    suppliers: list[SupplierMetrics] = Field(..., min_length=2)
    weights: SupplierWeights = Field(default_factory=SupplierWeights)
    sensitivity_perturbation: float = Field(
        0.10, ge=0.0, le=0.50,
        description="Fraction to perturb each weight for sensitivity analysis",
    )


class SupplierScore(BaseModel):
    """Computed score for a single supplier."""
    supplier_id: str
    name: str
    weighted_score: float
    rank: int
    criterion_scores: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]


class SupplierAnalyzeResponse(BaseModel):
    """Full supplier analysis with scoring and sensitivity."""
    scores: list[SupplierScore]
    sensitivity: dict[str, list[dict[str, Any]]]
    recommended_supplier: str


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

class CostBreakdownRequest(BaseModel):
    """Request full landed cost decomposition."""
    material_cost: float = Field(..., ge=0.0)
    transport_cost: float = Field(..., ge=0.0)
    duty_cost: float = Field(0.0, ge=0.0)
    handling_cost: float = Field(0.0, ge=0.0)
    storage_cost: float = Field(0.0, ge=0.0)
    insurance_cost: float = Field(0.0, ge=0.0)
    quantity: float = Field(..., gt=0.0)
    markup_pct: float = Field(0.0, ge=0.0, le=1.0)
    overhead_allocations: dict[str, float] = Field(default_factory=dict)


class CostBreakdownResponse(BaseModel):
    """Itemized cost breakdown with per-unit and total."""
    total_landed_cost: float
    per_unit_cost: float
    components: dict[str, float]
    component_pct: dict[str, float]
    total_with_markup: float


# ---------------------------------------------------------------------------
# What-if simulation
# ---------------------------------------------------------------------------

class WhatIfParameter(BaseModel):
    """A single parameter change for simulation."""
    parameter: str = Field(..., description="Dot-notation path, e.g. 'fuel_cost_per_km'")
    factor: float = Field(..., gt=0.0, description="Multiplier (1.0 = no change, 1.2 = +20%)")
    description: str = ""


class WhatIfRequest(BaseModel):
    """Scenario simulation with multiple parameter perturbations."""
    parameters: list[WhatIfParameter] = Field(..., min_length=1)
    baseline_cost: float = Field(0.0, ge=0.0, description="Known baseline total cost")


class WhatIfScenario(BaseModel):
    """Result of a single what-if scenario."""
    parameters: list[WhatIfParameter]
    baseline_cost: float
    scenario_cost: float
    delta: float
    delta_pct: float
    impact_breakdown: dict[str, float]


class WhatIfResponse(BaseModel):
    """Aggregated what-if analysis results."""
    scenarios: list[WhatIfScenario]
    most_impactful: str
    recommendation: str
