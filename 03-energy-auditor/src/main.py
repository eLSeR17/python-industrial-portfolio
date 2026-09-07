"""Application entry point – FastAPI + Dash dashboard.

Starts the FastAPI server with all API routes and mounts the Plotly Dash
dashboard as a sub-application. Includes CORS middleware, startup events
for data seeding, and health check endpoint.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config.settings import get_settings
from src.api.routes import router
from src.dashboard.plots import dash_app, update_data
from src.services.meter_reader import generate_synthetic_readings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: seed demo data. Shutdown: cleanup."""
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Seed demo facility + data for immediate dashboard use
    _seed_demo_data()

    yield

    logger.info("Shutting down %s", settings.app_name)


def _seed_demo_data() -> None:
    """Create a demo facility with 30 days of synthetic readings."""
    import uuid
    from datetime import timedelta, timezone
    from datetime import datetime

    from src.api.routes import _facilities, _readings

    fid = str(uuid.UUID("10000000-0000-0000-0000-000000000001"))
    _facilities[fid] = {
        "id": uuid.UUID(fid),
        "name": "Acme Manufacturing Plant",
        "code": "ACME-01",
        "address": "123 Industrial Blvd, Manufacturing District",
        "facility_type": "manufacturing",
        "contract_demand_kva": 600.0,
        "tariff_profile": "tou_general",
        "timezone": "UTC",
    }

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    df = generate_synthetic_readings(
        facility_id=uuid.UUID(fid),
        meter_id="MTR-MAIN",
        start=start,
        end=end,
        baseload_kw=150.0,
        peak_kw=480.0,
    )
    _readings[fid] = df
    update_data(fid, df, _facilities[fid])
    logger.info("Seeded demo facility %s with %d readings", fid, len(df))


# ── FastAPI app ────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Industrial Energy Auditor – monitors, analyzes, and optimizes "
        "energy consumption across industrial facilities to reduce costs "
        "by 10-15%."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page with links to API docs and dashboard."""
    return """
    <html>
    <head><title>Industrial Energy Auditor</title></head>
    <body style="font-family:system-ui; max-width:800px; margin:2rem auto; padding:0 1rem;">
        <h1>Industrial Energy Auditor</h1>
        <p>Monitor, analyze, and optimize energy consumption across industrial facilities.</p>
        <ul>
            <li><a href="/docs">API Documentation (Swagger)</a></li>
            <li><a href="/redoc">API Documentation (ReDoc)</a></li>
            <li><a href="/dashboard/">Interactive Dashboard</a></li>
        </ul>
        <h2>Quick Start</h2>
        <ol>
            <li>POST <code>/api/v1/readings/generate/{facility_id}</code> to create sample data</li>
            <li>GET <code>/api/v1/dashboard/{facility_id}</code> for load profile</li>
            <li>GET <code>/api/v1/savings-report/{facility_id}</code> for recommendations</li>
            <li>GET <code>/api/v1/audit/{facility_id}</code> for full audit report</li>
        </ol>
        <p>Demo facility ID: <code>10000000-0000-0000-0000-000000000001</code></p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "healthy", "version": settings.app_version}


# Mount Dash at /dashboard/
from fastapi.staticfiles import StaticFile

_dash_server = dash_app.server
app.mount("/dashboard/", StaticFiles(directory="src/dashboard", html=True), name="dash-static")
