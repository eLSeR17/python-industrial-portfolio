"""FastAPI application entry-point.

Creates and configures the ASGI application, mounts routes, and
optionally initialises the database tables on startup.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import logging

from fastapi import FastAPI

from config.settings import get_settings
from src.api.routes import router

settings = get_settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks.

    On startup, creates all database tables if they do not exist.
    """
    from sqlalchemy import create_engine
    from src.models.db import Base
    engine = create_engine(settings.database_url.replace("+asyncpg", "+psycopg2"))
    Base.metadata.create_all(engine)
    engine.dispose()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Automated document compliance analysis for industrial safety.",
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
