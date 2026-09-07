"""FastAPI application entry point.

Creates the app, registers routers, and manages startup/shutdown lifecycle
for the Kafka stream processor and Redis connection.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router as api_router
from src.api.websocket import router as ws_router
from src.services.stream_processor import stream_processor

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    On startup:
        - Connect to Kafka (consumer + producer).
    On shutdown:
        - Stop the Kafka stream processor gracefully.
    """
    logger.info("Starting %s", settings.app_name)
    try:
        await stream_processor.start()
        logger.info("Stream processor connected to Kafka")
    except Exception:
        logger.warning(
            "Kafka not available — running without stream processor. "
            "API endpoints are still functional with manual /process/update calls."
        )

    yield

    logger.info("Shutting down %s", settings.app_name)
    await stream_processor.stop()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Real-Time Process Optimizer for industrial manufacturing. "
        "Reduces waste from 20-30% to 5-10% through continuous setpoint "
        "optimization, statistical process control, and OEE analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the dashboard (any origin in development, restrict in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers.
app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
async def root() -> dict:
    """Root endpoint — basic API info."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "process_update": "POST /api/v1/process/update",
            "optimize": "GET /api/v1/optimize/{process_id}",
            "spc": "GET /api/v1/spc/{process_id}",
            "dashboard": "GET /api/v1/dashboard-data",
            "websocket": "ws://host/ws/live",
            "health": "GET /api/v1/health",
        },
    }
