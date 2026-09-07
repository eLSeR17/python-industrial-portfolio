"""SQLAlchemy ORM models for TimescaleDB time-series storage.

Defines the hypertable schema for sensor readings, health scores,
alerts, and predictions. Uses TimescaleDB-specific features for
efficient time-range queries and continuous aggregation.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class SensorReadingRow(Base):
    """Raw sensor reading stored in a TimescaleDB hypertable.

    Partitioned by time with a chunk interval of 1 day. Retention policy
    drops raw data older than 90 days, keeping only the 1-hour aggregates
    produced by the continuous aggregation policy.

    Indexes:
        - (asset_id, time): primary query pattern for time-range fetches
        - (time): for global time-range scans
    """

    __tablename__ = "sensor_readings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    vibration_x: Mapped[float] = mapped_column(Float, default=0.0)
    vibration_y: Mapped[float] = mapped_column(Float, default=0.0)
    vibration_z: Mapped[float] = mapped_column(Float, default=0.0)
    temperature: Mapped[float] = mapped_column(Float, default=25.0)
    pressure: Mapped[float] = mapped_column(Float, default=1.0)
    current: Mapped[float] = mapped_column(Float, default=0.0)
    rpm: Mapped[float] = mapped_column(Float, default=0.0)


class HealthScoreRow(Base):
    """Computed health scores stored for historical trending.

    Written after each feature engineering + scoring cycle.
    Used by the trend analysis in the health endpoint.
    """

    __tablename__ = "health_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    vibration_health: Mapped[float] = mapped_column(Float, default=1.0)
    thermal_health: Mapped[float] = mapped_column(Float, default=1.0)
    operational_health: Mapped[float] = mapped_column(Float, default=1.0)
    anomaly_health: Mapped[float] = mapped_column(Float, default=1.0)
    trend: Mapped[str] = mapped_column(String(16), default="stable")


class AlertRow(Base):
    """Alert records for audit trail and active alert management.

    Alerts are created by both threshold rules and ML-based detection.
    Status transitions: ACTIVE → ACKNOWLEDGED → RESOLVED.
    """

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, default="")


class PredictionRow(Base):
    """Failure prediction results persisted for validation and retraining.

    Ground truth (actual_failure, actual_time_to_failure) is populated
    retrospectively to evaluate model accuracy over time.
    """

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_useful_life_hours: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    failure_mode: Mapped[str] = mapped_column(String(64), default="unknown")
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(16), default="info")
    # Ground truth columns for model validation
    actual_failure: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    actual_time_to_failure_hours: Mapped[float | None] = mapped_column(Float, nullable=True)


class AssetMetadataRow(Base):
    """Static metadata for each monitored asset.

    Includes rated parameters used as baselines in health scoring
    and failure mode classification.
    """

    __tablename__ = "asset_metadata"

    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    rated_power_kw: Mapped[float] = mapped_column(Float, default=0.0)
    rated_rpm: Mapped[float] = mapped_column(Float, default=1500.0)
    max_temperature: Mapped[float] = mapped_column(Float, default=90.0)
    max_vibration: Mapped[float] = mapped_column(Float, default=7.1)
    max_pressure: Mapped[float] = mapped_column(Float, default=6.0)
    installed_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    location: Mapped[str] = mapped_column(String(128), default="Unknown")
