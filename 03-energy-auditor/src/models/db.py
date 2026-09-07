"""SQLAlchemy ORM models for time-series energy data."""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Facility(Base):
    """Physical facility being audited (factory, plant, warehouse)."""

    __tablename__ = "facilities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    address: Mapped[str | None] = mapped_column(Text)
    facility_type: Mapped[str] = mapped_column(String(50), default="manufacturing")
    contract_demand_kva: Mapped[float] = mapped_column(Float, nullable=False, comment="Contracted demand in kVA")
    tariff_profile: Mapped[str] = mapped_column(String(50), default="tou_general", comment="Tariff code")
    timezone: Mapped[str] = mapped_column(String(40), default="UTC")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MeterReading(Base):
    """Individual smart-meter readings ingested at regular intervals."""

    __tablename__ = "meter_readings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    facility_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    meter_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="Smart meter serial")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active_energy_kwh: Mapped[float] = mapped_column(Float, nullable=False, comment="Active energy in kWh")
    reactive_energy_kvarh: Mapped[float] = mapped_column(Float, default=0.0, comment="Reactive energy in kVARh")
    apparent_energy_kvah: Mapped[float] = mapped_column(Float, default=0.0, comment="Apparent energy in kVAh")
    demand_kw: Mapped[float] = mapped_column(Float, default=0.0, comment="Instantaneous active demand in kW")
    demand_kva: Mapped[float] = mapped_column(Float, default=0.0, comment="Instantaneous apparent demand in kVA")
    power_factor: Mapped[float] = mapped_column(Float, default=1.0, comment="Power factor (0-1)")
    voltage_avg: Mapped[float | None] = mapped_column(Float, comment="Average line voltage V")
    frequency_hz: Mapped[float | None] = mapped_column(Float, comment="Grid frequency Hz")
    current_a: Mapped[float | None] = mapped_column(Float, comment="Average current A")
    thd_voltage_pct: Mapped[float | None] = mapped_column(Float, comment="Voltage THD %")
    temperature_c: Mapped[float | None] = mapped_column(Float, comment="Ambient temperature °C")

    __table_args__ = (
        Index("ix_meter_readings_facility_ts", "facility_id", "timestamp", unique=True),
        Index("ix_meter_readings_meter_ts", "meter_id", "timestamp"),
    )


class EquipmentProfile(Base):
    """Sub-metered equipment or circuit for disaggregation analysis."""

    __tablename__ = "equipment_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    facility_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    equipment_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="e.g. hvac, motor, furnace, lighting")
    rated_power_kw: Mapped[float] = mapped_column(Float, nullable=False)
    meter_id: Mapped[str | None] = mapped_column(String(50), comment="Associated sub-meter if any")
    schedule_profile: Mapped[str] = mapped_column(String(50), default="continuous", comment="continuous | shift_based | intermittent")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditRecommendation(Base):
    """Generated recommendation from an energy audit run."""

    __tablename__ = "audit_recommendations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    facility_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    audit_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, comment="power_factor | load_shifting | equipment | hvac | lighting")
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_savings_usd: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_savings_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    priority: Mapped[str] = mapped_column(String(20), default="medium", comment="critical | high | medium | low")
    implementation_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    payback_months: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
