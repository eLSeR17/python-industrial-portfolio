"""FastAPI application entry point.

Creates the app, mounts routers, and configures CORS middleware.
The lifespan context manager handles startup/shutdown resources.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialise resources on startup."""
    logger.info("Starting %s v1.0", settings.app_name)
    logger.info("Model path: %s", settings.model_path or "(none — using classical CV)")
    logger.info("Confidence threshold: %.2f", settings.confidence_threshold)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="Computer Vision Quality Inspector for manufacturing lines",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()
