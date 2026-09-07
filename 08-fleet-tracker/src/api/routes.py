"""REST API routes and WebSocket endpoint for the Fleet Tracker."""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from src.db import get_session
from src.models.db import Asset
from src.models.schemas import (
    AssetCreate,
    AssetResponse,
    DashboardStats,
    GeofenceCreate,
    GeofenceEventResponse,
    GeofenceResponse,
    GPSUpdate,
    LocationRecord,
    MaintenanceSchedule,
    RouteRecord,
    UtilizationReport,
)
from src.services.geofence_service import GeofenceService
from src.services.location_service import LocationService
from src.services.maintenance_scheduler import MaintenanceScheduler
from src.services.route_analyzer import RouteAnalyzer
from src.services.utilization_analyzer import UtilizationAnalyzer
from src.websocket.manager import ConnectionManager

router = APIRouter(prefix="/api/v1", tags=["fleet"])

# Service singletons — wired in ``init_routes``.
_location_service: LocationService | None = None
_geofence_service: GeofenceService | None = None
_utilization_service: UtilizationAnalyzer | None = None
_maintenance_service: MaintenanceScheduler | None = None
_route_service: RouteAnalyzer | None = None
_ws_manager: ConnectionManager | None = None


def init_routes(
    location_service: LocationService,
    geofence_service: GeofenceService,
    utilization_service: UtilizationAnalyzer,
    maintenance_service: MaintenanceScheduler,
    route_service: RouteAnalyzer,
    ws_manager: ConnectionManager,
) -> None:
    """Wire service instances into the router module."""
    global _location_service, _geofence_service, _utilization_service  # noqa: PLW0603
    global _maintenance_service, _route_service, _ws_manager  # noqa: PLW0603
    _location_service = location_service
    _geofence_service = geofence_service
    _utilization_service = utilization_service
    _maintenance_service = maintenance_service
    _route_service = route_service
    _ws_manager = ws_manager


def _svc() -> LocationService:
    assert _location_service is not None, "Routes not initialised"
    return _location_service


def _geo() -> GeofenceService:
    assert _geofence_service is not None, "Routes not initialised"
    return _geofence_service


def _util() -> UtilizationAnalyzer:
    assert _utilization_service is not None, "Routes not initialised"
    return _utilization_service


def _maint() -> MaintenanceScheduler:
    assert _maintenance_service is not None, "Routes not initialised"
    return _maintenance_service


def _route() -> RouteAnalyzer:
    assert _route_service is not None, "Routes not initialised"
    return _route_service


def _ws() -> ConnectionManager:
    assert _ws_manager is not None, "Routes not initialised"
    return _ws_manager


# ---------------------------------------------------------------------------
# Asset endpoints
# ---------------------------------------------------------------------------

@router.post("/assets", response_model=AssetResponse, status_code=201)
async def create_asset(payload: AssetCreate) -> AssetResponse:
    """Register a new tracked asset."""
    async with get_session() as session:
        existing = await session.get(Asset, payload.id)
        if existing:
            raise HTTPException(status_code=409, detail=f"Asset {payload.id!r} already exists")
        asset = Asset(
            id=payload.id,
            name=payload.name,
            asset_type=payload.asset_type.value,
            department=payload.department,
            manufacturer=payload.manufacturer,
            model_year=payload.year,
            max_speed_kmh=payload.max_speed_kmh,
            maintenance_interval_hours=payload.maintenance_interval_hours,
        )
        session.add(asset)
        await session.commit()

    return AssetResponse(
        id=asset.id,
        name=asset.name,
        asset_type=asset.asset_type,
        department=asset.department,
        manufacturer=asset.manufacturer,
        model_year=asset.model_year,
        max_speed_kmh=asset.max_speed_kmh,
        maintenance_interval_hours=asset.maintenance_interval_hours,
        status=asset.status,
    )


@router.get("/assets", response_model=list[AssetResponse])
async def list_assets() -> list[AssetResponse]:
    """List all assets with their last known position."""
    rows = await _svc().get_all_active_assets()
    return [AssetResponse(**r) for r in rows]


@router.get("/assets/{asset_id}", response_model=AssetResponse)
async def get_asset(asset_id: str) -> AssetResponse:
    """Get details of a single asset."""
    rows = await _svc().get_all_active_assets()
    for r in rows:
        if r["id"] == asset_id:
            return AssetResponse(**r)
    raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} not found")


@router.get("/assets/{asset_id}/history", response_model=list[LocationRecord])
async def get_asset_history(
    asset_id: str,
    hours: int = Query(24, ge=1, le=720),
) -> list[LocationRecord]:
    """Get GPS history for an asset."""
    return await _svc().get_asset_history(asset_id, hours)


# ---------------------------------------------------------------------------
# Location ingestion
# ---------------------------------------------------------------------------

@router.post("/location", response_model=LocationRecord, status_code=201)
async def ingest_location(payload: GPSUpdate) -> LocationRecord:
    """Ingest a single GPS update."""
    try:
        record = await _svc().ingest_location(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Check geofences
    await _geo().check_asset_in_geofences(
        payload.asset_id, payload.latitude, payload.longitude,
    )

    # Broadcast via WebSocket
    await _ws().broadcast_to_asset(payload.asset_id, {
        "type": "location_update",
        "asset_id": record.asset_id,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "speed_kmh": record.speed_kmh,
        "heading": record.heading,
        "is_moving": record.is_moving,
        "timestamp": record.timestamp.isoformat(),
    })

    return record


@router.post("/location/batch", response_model=list[LocationRecord], status_code=201)
async def ingest_batch(payload: list[GPSUpdate]) -> list[LocationRecord]:
    """Ingest multiple GPS updates in one call."""
    records: list[LocationRecord] = []
    for update in payload:
        try:
            r = await _svc().ingest_location(update)
            records.append(r)
        except ValueError:
            continue
    return records


# ---------------------------------------------------------------------------
# Geofences
# ---------------------------------------------------------------------------

@router.post("/geofences", response_model=GeofenceResponse, status_code=201)
async def create_geofence(payload: GeofenceCreate) -> GeofenceResponse:
    """Create a new geofence zone."""
    return await _geo().create_geofence(payload)


@router.get("/geofences", response_model=list[GeofenceResponse])
async def list_geofences() -> list[GeofenceResponse]:
    """List all registered geofences."""
    return await _geo().get_all_geofences()


@router.get("/geofences/events", response_model=list[GeofenceEventResponse])
async def get_geofence_events(
    asset_id: str | None = None,
    hours: int = Query(24, ge=1, le=720),
) -> list[GeofenceEventResponse]:
    """Retrieve geofence crossing events."""
    return await _geo().get_geofence_events(asset_id, hours)


# ---------------------------------------------------------------------------
# Utilisation
# ---------------------------------------------------------------------------

@router.get("/utilization", response_model=list[UtilizationReport])
async def get_fleet_utilization(
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[UtilizationReport]:
    """Fleet-wide utilisation report."""
    now = datetime.now(timezone.utc)
    end = end or now
    start = start or (now - timedelta(hours=24))
    return await _util().calculate_fleet_utilization(start, end)


@router.get("/utilization/{asset_id}", response_model=UtilizationReport)
async def get_asset_utilization(
    asset_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> UtilizationReport:
    """Utilisation report for a single asset."""
    now = datetime.now(timezone.utc)
    end = end or now
    start = start or (now - timedelta(hours=24))
    try:
        return await _util().calculate_utilization(asset_id, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

@router.get("/maintenance", response_model=list[MaintenanceSchedule])
async def get_maintenance_schedule() -> list[MaintenanceSchedule]:
    """Fleet maintenance schedule."""
    return await _maint().get_fleet_maintenance_schedule()


@router.post("/maintenance/{asset_id}/service", status_code=204)
async def record_service(
    asset_id: str,
    service_type: str = Query(...),
    notes: str = Query(""),
    hours: float = Query(0.0),
) -> None:
    """Record a completed maintenance event."""
    try:
        await _maint().record_service(asset_id, service_type, hours, notes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/routes/{asset_id}", response_model=list[RouteRecord])
async def get_route_history(
    asset_id: str,
    days: int = Query(7, ge=1, le=90),
) -> list[RouteRecord]:
    """Historical route records for an asset."""
    return await _route().get_route_history(asset_id, days)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard() -> DashboardStats:
    """Aggregate fleet KPIs."""
    return await _util().get_dashboard_stats()


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

ws_router = APIRouter()


@ws_router.websocket("/ws/tracking")
async def ws_tracking(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time asset tracking.

    The first message from the client may be a JSON object
    ``{"subscribe": ["asset-1", "asset-2"]}`` to filter updates.
    """
    client_id = str(uuid.uuid4())
    try:
        await _ws().connect(websocket, client_id)
        # Attempt to read initial subscription
        try:
            init_msg = await websocket.receive_json()
            if isinstance(init_msg, dict) and "subscribe" in init_msg:
                _ws().subscribe_to_assets(client_id, init_msg["subscribe"])
        except Exception:
            _ws().subscribe_to_assets(client_id, [])

        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and "subscribe" in data:
                _ws().subscribe_to_assets(client_id, data["subscribe"])
    except WebSocketDisconnect:
        pass
    finally:
        _ws().disconnect(client_id)
