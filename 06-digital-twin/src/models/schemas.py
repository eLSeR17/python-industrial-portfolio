"""Pydantic schemas for simulation configuration and results."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScheduleType(str, Enum):
    """Supported job scheduling strategies."""

    FIFO = "FIFO"
    SPT = "SPT"
    EDD = "EDD"
    CRITICAL_RATIO = "CRITICAL_RATIO"


class MachineStatus(str, Enum):
    """Possible states for a machine during simulation."""

    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    REPAIRING = "REPAIRING"
    BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# Input configuration models
# ---------------------------------------------------------------------------

class MachineConfig(BaseModel):
    """Configuration for a single machine / workstation."""

    id: str = Field(..., description="Unique machine identifier")
    name: str = Field(..., description="Human-readable name")
    processing_time_mean: float = Field(
        ..., gt=0, description="Mean processing time (sim-minutes)"
    )
    processing_time_std: float = Field(
        ..., ge=0, description="Std-dev of processing time"
    )
    failure_rate: float = Field(
        default=0.0, ge=0, le=1, description="Per-minute failure probability"
    )
    repair_time_mean: float = Field(
        default=0.0, ge=0, description="Mean repair time (sim-minutes)"
    )
    repair_time_std: float = Field(
        default=0.0, ge=0, description="Std-dev of repair time"
    )
    capacity: int = Field(default=1, ge=1, description="Parallel processing slots")


class BufferConfig(BaseModel):
    """Configuration for an inventory buffer between stations."""

    id: str = Field(..., description="Unique buffer identifier")
    name: str = Field(..., description="Human-readable name")
    capacity: int = Field(..., gt=0, description="Maximum items the buffer can hold")
    initial_level: int = Field(default=0, ge=0, description="Starting inventory")


class ConveyorConfig(BaseModel):
    """Configuration for a transport conveyor between two stations."""

    id: str = Field(..., description="Unique conveyor identifier")
    name: str = Field(..., description="Human-readable name")
    from_station: str = Field(..., description="Source machine / buffer id")
    to_station: str = Field(..., description="Destination machine / buffer id")
    speed: float = Field(default=1.0, gt=0, description="Travel time multiplier")
    capacity: int = Field(default=1, ge=1, description="Items in transit at once")


class SimulationConfig(BaseModel):
    """Top-level configuration for a single simulation run."""

    id: str = Field(..., description="Unique simulation identifier")
    name: str = Field("", description="Human-readable simulation name")
    duration: float = Field(default=1000.0, gt=0, description="Simulation length")
    warmup_period: float = Field(
        default=100.0, ge=0, description="Warm-up period excluded from stats"
    )
    machines: list[MachineConfig] = Field(default_factory=list)
    buffers: list[BufferConfig] = Field(default_factory=list)
    conveyors: list[ConveyorConfig] = Field(default_factory=list)
    schedule_type: ScheduleType = Field(default=ScheduleType.FIFO)
    random_seed: int = Field(default=42, description="Base random seed")


# ---------------------------------------------------------------------------
# Output / result models
# ---------------------------------------------------------------------------

class SimulationResult(BaseModel):
    """Full result of a simulation run (or average of replications)."""

    id: str
    config_id: str
    duration: float
    throughput: float = Field(description="Units produced per sim-minute")
    avg_cycle_time: float = Field(description="Average time a unit spends in system")
    avg_wip: float = Field(description="Average work-in-progress")
    oee: dict[str, float] = Field(default_factory=dict, description="OEE per machine")
    utilization: dict[str, float] = Field(
        default_factory=dict, description="Utilization per machine"
    )
    bottleneck_id: str = Field(default="", description="Machine id with lowest throughput")
    total_produced: int = 0
    total_failed: int = 0
    replication_results: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioComparison(BaseModel):
    """Result of comparing multiple simulation scenarios."""

    scenarios: list[SimulationResult] = Field(default_factory=list)
    best_throughput: str = Field(default="", description="Config id with best throughput")
    best_oee: str = Field(default="", description="Config id with best OEE")
    recommendations: list[str] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    """Point-in-time snapshot of a single machine's state."""

    timestamp: float
    machine_id: str
    status: MachineStatus
    queue_length: int = 0
    utilization_current: float = 0.0
