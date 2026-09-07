"""Pydantic schemas for request / response validation."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    """Type of industrial asset."""

    FORKLIFT = "FORKLIFT"
    AGV = "AGV"
    TRUCK = "TRUCK"
    CRANE = "CRANE"
    CONVEYOR = "CONVEYOR"
    OTHER = "OTHER"


class AssetStatus(str, Enum):
    """Operational status of an asset."""

    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"


class FenceType(str, Enum):
    """Geofence shape."""

    POLYGON = "POLYGON"
    CIRCLE = "CIRCLE"


class ZoneType(str, Enum):
    """Semantic zone classification."""

    WAREHOUSE = "WAREHOUSE"
    RESTRICTED = "RESTRICTED"
    CHARGING = "CHARGING"
    PARKING = "PARKING"


class EventType(str, Enum):
    """Geofence crossing event type."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


class MaintenancePriority(str, Enum):
    """Priority level for maintenance scheduling."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------

class AssetCreate(BaseModel):
    """Payload to register a new asset."""

    id: str = Field(..., max_length=64)
    name: str = Field(..., max_length=128)
    asset_type: AssetType
    department: str | None = None
    manufacturer: str | None = None
    model: str | None = Field(None, alias="model")
    year: int | None = None
    max_speed_kmh: float = 50.0
    maintenance_interval_hours: float = 500.0


class AssetResponse(BaseModel):
    """Full representation of an asset."""

    id: str
    name: str
    asset_type: AssetType
    department: str | None = None
    manufacturer: str | None = None
    model_year: int | None = None
    max_speed_kmh: float = 50.0
    maintenance_interval_hours: float = 500.0
    status: AssetStatus = AssetStatus.ACTIVE
    last_location: dict | None = None
    last_update: datetime | None = None
    total_hours_used: float = 0.0
    distance_traveled_km: float = 0.0

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# GPS / Location
# ---------------------------------------------------------------------------

class GPSUpdate(BaseModel):
    """Incoming GPS data-point from a vehicle tracker."""

    asset_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime | None = None
    speed_kmh: float = 0.0
    heading_degrees: float = 0.0
    battery_level: float | None = None


class LocationRecord(BaseModel):
    """Persisted location sample with derived metrics."""

    id: int
    asset_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    speed_kmh: float = 0.0
    heading: float = 0.0
    distance_from_prev: float = 0.0
    is_moving: bool = False

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Geofence
# ---------------------------------------------------------------------------

class GeofenceCreate(BaseModel):
    """Payload to create a new geofence."""

    name: str = Field(..., max_length=128)
    fence_type: FenceType
    coordinates: list
    zone_type: ZoneType
    alert_on_entry: bool = False
    alert_on_exit: bool = False


class GeofenceResponse(BaseModel):
    """Full representation of a geofence."""

    id: int
    name: str
    fence_type: FenceType
    coordinates: list
    zone_type: ZoneType
    alert_on_entry: bool = False
    alert_on_exit: bool = False
    is_active: bool = True
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class GeofenceEventResponse(BaseModel):
    """A recorded geofence crossing event."""

    id: int
    asset_id: str
    geofence_id: int
    event_type: EventType
    timestamp: datetime
    latitude: float
    longitude: float

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Utilization
# ---------------------------------------------------------------------------

class UtilizationReport(BaseModel):
    """Utilization breakdown for one asset in a time window."""

    asset_id: str
    asset_name: str
    period_start: datetime
    period_end: datetime
    total_hours: float
    active_hours: float
    idle_hours: float
    maintenance_hours: float
    utilization_pct: float
    idle_pct: float


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

class MaintenanceSchedule(BaseModel):
    """Next predicted maintenance window for an asset."""

    asset_id: str
    asset_name: str
    next_service_date: datetime
    hours_until_service: float
    distance_until_service: float
    priority: MaintenancePriority
    estimated_downtime_hours: float


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

class Deviation(BaseModel):
    """A detected deviation from the planned route."""

    latitude: float
    longitude: float
    distance_from_planned_m: float
    index: int


class RouteRecord(BaseModel):
    """A recorded route with planned vs actual comparison."""

    id: int
    asset_id: str
    start_time: datetime
    end_time: datetime
    planned_route: list[dict] = []
    actual_route: list[dict] = []
    distance_planned_km: float = 0.0
    distance_actual_km: float = 0.0
    efficiency_pct: float = 0.0
    deviations: list[Deviation] = []

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    """Aggregate fleet KPIs."""

    total_assets: int = 0
    active_assets: int = 0
    idle_assets: int = 0
    avg_utilization: float = 0.0
    alerts_count: int = 0
    maintenance_due_count: int = 0
