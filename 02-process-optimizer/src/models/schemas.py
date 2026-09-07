"""Pydantic schemas for API request/response validation and serialization.

All schemas use Pydantic v2 with strict type validation. Field constraints
mirror physical limits of industrial process equipment.
"""

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ProcessType(str, Enum):
    """Categories of industrial processes this system supports."""

    CHEMICAL_REACTOR = "chemical_reactor"
    HEAT_EXCHANGER = "heat_exchanger"
    DISTILLATION_COLUMN = "distillation_column"
    MIXING_TANK = "mixing_tank"
    CONVEYOR_OVEN = "conveyor_oven"
    ROLLING_MILL = "rolling_mill"


class AlarmSeverity(str, Enum):
    """SPC alarm severity levels aligned with ISA-18.2."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class OptimizationMethod(str, Enum):
    """Supported gradient-free optimization algorithms."""

    NELDER_MEAD = "nelder_mead"
    COORDINATE_DESCENT = "coordinate_descent"
    BAYESIAN = "bayesian"


# ---------------------------------------------------------------------------
# Sensor & Process Data
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """A single measurement from an IoT sensor or PLC."""

    sensor_id: str = Field(..., min_length=1, max_length=64)
    process_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float = Field(..., description="Measured value in engineering units")
    unit: str = Field(default="", max_length=16)
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Signal quality 0-1")


class ProcessUpdateRequest(BaseModel):
    """Payload for POST /process/update — a batch of sensor readings."""

    process_id: str = Field(..., min_length=1, max_length=64)
    process_type: ProcessType
    readings: list[SensorReading] = Field(..., min_length=1, max_length=100)
    setpoints: dict[str, float] = Field(default_factory=dict, description="Current setpoints {variable: value}")


class ProcessVariable(BaseModel):
    """A single tracked variable within a process."""

    name: str
    value: float
    unit: str = ""
    min_limit: float = Field(default=-1e9, description="Lower safe operating limit")
    max_limit: float = Field(default=1e9, description="Upper safe operating limit")
    setpoint: float | None = None


# ---------------------------------------------------------------------------
# Process State
# ---------------------------------------------------------------------------

class ProcessState(BaseModel):
    """Complete snapshot of a process at a point in time."""

    process_id: str
    process_type: ProcessType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    variables: list[ProcessVariable] = Field(default_factory=list)
    is_running: bool = True
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class OptimizationConstraints(BaseModel):
    """Physical and operational constraints for the optimizer."""

    variable_limits: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Per-variable (min, max) bounds",
    )
    rate_of_change_limits: dict[str, float] = Field(
        default_factory=dict,
        description="Max allowed change per optimization cycle",
    )
    power_limit_kw: float | None = Field(default=None, description="Total power budget")
    penalty_weight: float = Field(
        default=100.0,
        description="Quadratic penalty weight for constraint violations",
    )


class OptimizationRequest(BaseModel):
    """Request body for triggering an optimization run."""

    process_id: str
    method: OptimizationMethod = OptimizationMethod.NELDER_MEAD
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    max_iterations: int = Field(default=200, ge=1, le=10000)
    objective: str = Field(
        default="minimize_waste",
        description="Objective function identifier",
    )


class OptimizationResult(BaseModel):
    """Outcome of a single optimization run."""

    result_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    process_id: str
    method: OptimizationMethod
    recommended_setpoints: dict[str, float]
    predicted_improvement_pct: float = Field(ge=0.0, le=100.0)
    iterations_used: int
    convergence_achieved: bool
    objective_value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    constraints_satisfied: bool = True


# ---------------------------------------------------------------------------
# SPC (Statistical Process Control)
# ---------------------------------------------------------------------------

class SPCAlarm(BaseModel):
    """A single SPC alarm raised by the analysis engine."""

    alarm_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    rule: str = Field(..., description="Western Electric or CUSUM rule name")
    severity: AlarmSeverity
    variable: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float | None = None
    limit: float | None = None


class SPCChart(BaseModel):
    """X-bar or R-chart data for a single variable."""

    variable: str
    chart_type: str = "xbar"
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    data_points: list[float] = Field(default_factory=list)
    timestamps: list[datetime] = Field(default_factory=list)
    violations: list[int] = Field(default_factory=list, description="Indices of out-of-control points")


class SPCResult(BaseModel):
    """Complete SPC analysis result for a process."""

    process_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    charts: list[SPCChart] = Field(default_factory=list)
    alarms: list[SPCAlarm] = Field(default_factory=list)
    cusum_state: dict[str, float] = Field(default_factory=dict, description="Per-variable CUSUM accumulators")
    process_capability: dict[str, float] = Field(default_factory=dict, description="Cp, Cpk per variable")


# ---------------------------------------------------------------------------
# OEE (Overall Equipment Effectiveness)
# ---------------------------------------------------------------------------

class OEEComponents(BaseModel):
    """The three factors of OEE."""

    availability: float = Field(ge=0.0, le=1.0, description="Uptime / planned production time")
    performance: float = Field(ge=0.0, le=1.0, description="Actual throughput / theoretical max")
    quality: float = Field(ge=0.0, le=1.0, description="Good units / total units produced")

    @property
    def oee(self) -> float:
        """OEE = Availability × Performance × Quality."""
        return self.availability * self.performance * self.quality


class BottleneckInfo(BaseModel):
    """Identifies the slowest stage in a multi-stage process."""

    stage_name: str
    cycle_time_seconds: float
    utilization_pct: float
    is_bottleneck: bool = False


# ---------------------------------------------------------------------------
# Dashboard & WebSocket
# ---------------------------------------------------------------------------

class DashboardData(BaseModel):
    """Full payload for the operator dashboard."""

    processes: list[ProcessState] = Field(default_factory=list)
    latest_optimizations: dict[str, OptimizationResult] = Field(default_factory=dict)
    spc_results: dict[str, SPCResult] = Field(default_factory=dict)
    oee: dict[str, OEEComponents] = Field(default_factory=dict)
    active_alarms: list[SPCAlarm] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSMessage(BaseModel):
    """WebSocket message envelope."""

    channel: str
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
