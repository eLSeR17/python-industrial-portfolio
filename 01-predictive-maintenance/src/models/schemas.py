"""Pydantic v2 schemas for API request/response validation.

Defines the data contracts for sensor ingestion, health assessment,
failure prediction, alerts, and maintenance scheduling. All models use
strict validation to prevent malformed data from entering the pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────


class AssetType(str, Enum):
    """Supported industrial equipment types.

    Each class has distinct vibration signatures and failure modes:
    - MOTOR: bearing wear, rotor imbalance, electrical faults
    - PUMP: cavitation, seal failure, impeller damage
    - COMPRESSOR: valve leakage, refrigerant issues, piston ring wear
    """

    MOTOR = "motor"
    PUMP = "pump"
    COMPRESSOR = "compressor"


class Severity(str, Enum):
    """Alert severity levels aligned with ISA-18.2 alarm management."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Lifecycle states for an alert instance."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# ── Sensor Ingestion ─────────────────────────────────────────────────────


class SensorReading(BaseModel):
    """A single multi-sensor measurement from an industrial asset.

    Sensor channels:
        vibration_x/y/z: Triaxial accelerometer data (mm/s RMS)
        temperature: Surface or winding temperature (°C)
        pressure: Process or hydraulic pressure (bar)
        current: Motor current draw (amps)
        rpm: Rotational speed (revolutions per minute)

    Validation ensures physical plausibility of all readings.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_id: Annotated[str, Field(min_length=1, max_length=64, examples=["MOTOR-001"])]
    asset_type: AssetType = AssetType.MOTOR
    timestamp: datetime
    vibration_x: Annotated[float, Field(ge=-50.0, le=50.0)] = 0.0
    vibration_y: Annotated[float, Field(ge=-50.0, le=50.0)] = 0.0
    vibration_z: Annotated[float, Field(ge=-50.0, le=50.0)] = 0.0
    temperature: Annotated[float, Field(ge=-40.0, le=300.0)] = 25.0
    pressure: Annotated[float, Field(ge=0.0, le=100.0)] = 1.0
    current: Annotated[float, Field(ge=0.0, le=500.0)] = 0.0
    rpm: Annotated[float, Field(ge=0.0, le=50000.0)] = 0.0

    @field_validator("asset_id")
    @classmethod
    def asset_id_uppercase(cls, v: str) -> str:
        """Normalize asset identifiers to uppercase for consistent lookups."""
        return v.upper().strip()


class IngestBatchRequest(BaseModel):
    """Batch of sensor readings for high-throughput ingestion.

    Accepting batches reduces HTTP overhead when collecting from
    edge gateways that aggregate multiple sensor polls.
    """

    readings: Annotated[list[SensorReading], Field(min_length=1, max_length=10000)]


class IngestResponse(BaseModel):
    """Response after successful sensor data ingestion."""

    accepted: int
    rejected: int = 0
    processing_time_ms: float


# ── Health Assessment ────────────────────────────────────────────────────


class HealthScore(BaseModel):
    """Composite health score for an asset (0.0 = failed, 1.0 = perfect).

    The score aggregates:
    - Vibration health (30%): based on RMS vs. ISO 10816 thresholds
    - Thermal health (25%): temperature vs. rated limits
    - Operational health (25%): current draw and RPM stability
    - Anomaly health (20%): Isolation Forest anomaly score
    """

    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    timestamp: datetime
    overall_score: Annotated[float, Field(ge=0.0, le=1.0)]
    vibration_health: Annotated[float, Field(ge=0.0, le=1.0)]
    thermal_health: Annotated[float, Field(ge=0.0, le=1.0)]
    operational_health: Annotated[float, Field(ge=0.0, le=1.0)]
    anomaly_health: Annotated[float, Field(ge=0.0, le=1.0)]
    trend: Literal["improving", "stable", "degrading"]


# ── Failure Prediction ───────────────────────────────────────────────────


class FailurePrediction(BaseModel):
    """ML-based failure prediction output.

    Combines:
    - probability: Random Forest classifier output (0–1)
    - remaining_useful_life_hours: Regression estimate
    - confidence: Based on model certainty and data recency
    - recommended_action: Textual maintenance recommendation
    """

    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    timestamp: datetime
    failure_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    remaining_useful_life_hours: Annotated[float, Field(ge=0.0)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    failure_mode: str
    recommended_action: str
    severity: Severity


# ── Alerts ───────────────────────────────────────────────────────────────


class Alert(BaseModel):
    """Maintenance alert generated by threshold or ML rules.

    Follows ISA-18.2 alarm management standards with four severity tiers.
    """

    model_config = ConfigDict(from_attributes=True)

    alert_id: str
    asset_id: str
    timestamp: datetime
    severity: Severity
    status: AlertStatus = AlertStatus.ACTIVE
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    recommended_action: str


# ── Maintenance Schedule ────────────────────────────────────────────────


class MaintenanceWindow(BaseModel):
    """Optimal maintenance window computed from prediction horizons.

    Groups assets by priority to minimize production impact while
    ensuring critical maintenance is performed before predicted failure.
    """

    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    priority: Annotated[int, Field(ge=1, le=5)]
    earliest_start: datetime
    latest_end: datetime
    estimated_duration_hours: float
    maintenance_type: str
    estimated_cost: float
    failure_risk_if_deferred: Annotated[float, Field(ge=0.0, le=1.0)]


class MaintenanceSchedule(BaseModel):
    """Complete maintenance schedule response."""

    generated_at: datetime
    planning_horizon_days: int
    windows: list[MaintenanceWindow]
    total_estimated_cost: float
    potential_savings: float
