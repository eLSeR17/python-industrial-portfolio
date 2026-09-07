"""SQLAlchemy ORM models for the Fleet Tracker.

Uses standard SQLAlchemy columns (no PostGIS) so the application can run
on any SQLite / PostgreSQL database.  Spatial queries are performed in
Python via :mod:`src.utils.geo_utils`.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class Asset(Base):
    """A tracked industrial asset (forklift, AGV, truck …)."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    department: Mapped[str | None] = mapped_column(String(64))
    manufacturer: Mapped[str | None] = mapped_column(String(128))
    model_year: Mapped[int | None] = mapped_column(Integer)
    max_speed_kmh: Mapped[float] = mapped_column(Float, default=50.0)
    maintenance_interval_hours: Mapped[float] = mapped_column(Float, default=500.0)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")
    total_hours_used: Mapped[float] = mapped_column(Float, default=0.0)
    distance_traveled_km: Mapped[float] = mapped_column(Float, default=0.0)
    last_latitude: Mapped[float | None] = mapped_column(Float)
    last_longitude: Mapped[float | None] = mapped_column(Float)
    last_update: Mapped[datetime | None] = mapped_column(DateTime)

    locations: Mapped[list["LocationHistory"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    maintenance_records: Mapped[list["MaintenanceRecord"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class LocationHistory(Base):
    """GPS position sample for an asset."""

    __tablename__ = "location_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(64), ForeignKey("assets.id"), index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    speed_kmh: Mapped[float] = mapped_column(Float, default=0.0)
    heading: Mapped[float] = mapped_column(Float, default=0.0)
    is_moving: Mapped[bool] = mapped_column(Boolean, default=False)

    asset: Mapped["Asset"] = relationship(back_populates="locations")


class Geofence(Base):
    """A geofence zone (polygon or circle)."""

    __tablename__ = "geofences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    fence_type: Mapped[str] = mapped_column(String(16), nullable=False)
    coordinates_json: Mapped[str] = mapped_column(Text, nullable=False)
    zone_type: Mapped[str] = mapped_column(String(32), nullable=False)
    alert_on_entry: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_on_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class GeofenceEvent(Base):
    """Record of an asset entering or exiting a geofence."""

    __tablename__ = "geofence_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    geofence_id: Mapped[int] = mapped_column(Integer, ForeignKey("geofences.id"))
    event_type: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)

    geofence: Mapped["Geofence"] = relationship()


class MaintenanceRecord(Base):
    """Service history entry for an asset."""

    __tablename__ = "maintenance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(64), ForeignKey("assets.id"), index=True)
    service_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scheduled_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_date: Mapped[datetime | None] = mapped_column(DateTime)
    hours_at_service: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")

    asset: Mapped["Asset"] = relationship(back_populates="maintenance_records")
