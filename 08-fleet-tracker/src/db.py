"""SQLAlchemy async session factory and engine setup."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings

_engine = None
_session_factory = None


def get_engine():
    """Return or create the async engine singleton."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = create_async_engine(
            get_settings().database_url,
            echo=get_settings().debug,
            future=True,
        )
    return _engine


def get_session_factory():
    """Return or create the session factory singleton."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session scope."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables (SQLite auto-creates; Postgres needs migrations)."""
    from src.models.db import Base
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
