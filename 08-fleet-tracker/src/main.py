"""Fleet & Asset Tracker — FastAPI application entry point.

Creates the app, initialises services, and exposes REST + WebSocket
endpoints for real-time industrial fleet management.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from src.api.routes import init_routes, router, ws_router
from src.db import get_session_factory, init_db
from src.services.geofence_service import GeofenceService
from src.services.location_service import LocationService
from src.services.maintenance_scheduler import MaintenanceScheduler
from src.services.route_analyzer import RouteAnalyzer
from src.services.utilization_analyzer import UtilizationAnalyzer
from src.websocket.manager import ConnectionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle — initialise DB, services, and seed data."""
    settings = get_settings()

    # Create tables
    await init_db()

    # Shared session factory
    session_factory = get_session_factory()

    # Build services
    location_svc = LocationService(session_factory)
    geofence_svc = GeofenceService(session_factory)
    utilization_svc = UtilizationAnalyzer(session_factory)
    maintenance_svc = MaintenanceScheduler(session_factory, settings.maintenance_warning_days)
    route_svc = RouteAnalyzer(session_factory)
    ws_mgr = ConnectionManager()

    # Wire into routes
    init_routes(location_svc, geofence_svc, utilization_svc, maintenance_svc, route_svc, ws_mgr)

    # Seed sample geofences
    await _seed_geofences(geofence_svc)

    yield  # app is running


async def _seed_geofences(geofence_svc: GeofenceService) -> None:
    """Load sample geofences from ``data/sample_geofences.json`` if empty."""
    import json
    from pathlib import Path

    existing = await geofence_svc.get_all_geofences()
    if existing:
        return

    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_geofences.json"
    if not sample_path.exists():
        return

    data = json.loads(sample_path.read_text())
    from src.models.schemas import FenceType, GeofenceCreate, ZoneType
    for item in data:
        await geofence_svc.create_geofence(
            GeofenceCreate(
                name=item["name"],
                fence_type=FenceType(item["fence_type"]),
                coordinates=item["coordinates"],
                zone_type=ZoneType(item["zone_type"]),
                alert_on_entry=item.get("alert_on_entry", False),
                alert_on_exit=item.get("alert_on_exit", False),
            )
        )


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Real-time fleet & asset tracking for industrial environments",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(ws_router)

    # Serve static files (Leaflet demo)
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
