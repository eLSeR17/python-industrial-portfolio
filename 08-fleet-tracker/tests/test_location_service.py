"""Tests for the LocationService using an in-memory SQLite database."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models.db import Asset, Base
from src.models.schemas import AssetStatus, GPSUpdate
from src.services.location_service import LocationService


@pytest_asyncio.fixture
async def engine():
    """Create an in-memory async engine."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    """Async session factory."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def service(session_factory) -> LocationService:
    """LocationService wired to the in-memory DB."""
    return LocationService(session_factory)


async def _create_asset(session_factory, asset_id="FL-001"):
    """Helper to insert an asset row."""
    async with session_factory() as session:
        asset = Asset(id=asset_id, name="Forklift 1", asset_type="FORKLIFT")
        session.add(asset)
        await session.commit()


async def test_ingest_first_update(service, session_factory):
    """First GPS fix should store a record with distance_from_prev == 0."""
    await _create_asset(session_factory)
    update = GPSUpdate(
        asset_id="FL-001",
        latitude=40.7128,
        longitude=-74.0060,
        speed_kmh=5.0,
        heading_degrees=90.0,
    )
    record = await service.ingest_location(update)
    assert record.asset_id == "FL-001"
    assert record.latitude == 40.7128
    assert record.distance_from_prev == 0.0
    assert record.is_moving is True


async def test_ingest_second_update_computes_distance(service, session_factory):
    """Second fix should produce a non-zero distance."""
    await _create_asset(session_factory)
    u1 = GPSUpdate(asset_id="FL-001", latitude=40.7128, longitude=-74.0060, speed_kmh=0.0)
    await service.ingest_location(u1)

    u2 = GPSUpdate(asset_id="FL-001", latitude=40.7138, longitude=-74.0060, speed_kmh=8.0)
    record = await service.ingest_location(u2)
    assert record.distance_from_prev > 0


async def test_is_moving_detection(service, session_factory):
    """Speed <= 0.5 km/h should be idle; above should be moving."""
    await _create_asset(session_factory)

    u_slow = GPSUpdate(asset_id="FL-001", latitude=40.7128, longitude=-74.0060, speed_kmh=0.1)
    r1 = await service.ingest_location(u_slow)
    assert r1.is_moving is False

    u_fast = GPSUpdate(asset_id="FL-001", latitude=40.7130, longitude=-74.0060, speed_kmh=3.0)
    r2 = await service.ingest_location(u_fast)
    assert r2.is_moving is True


async def test_get_asset_history(service, session_factory):
    """History should return records in order."""
    await _create_asset(session_factory)
    for i in range(5):
        u = GPSUpdate(
            asset_id="FL-001",
            latitude=40.7128 + i * 0.001,
            longitude=-74.0060,
            speed_kmh=5.0,
        )
        await service.ingest_location(u)

    history = await service.get_asset_history("FL-001", hours=1)
    assert len(history) == 5
    assert history[0].latitude < history[-1].latitude


async def test_get_asset_current(service, session_factory):
    """Current position should match the latest ingest."""
    await _create_asset(session_factory)
    u = GPSUpdate(asset_id="FL-001", latitude=40.999, longitude=-74.999, speed_kmh=0.0)
    await service.ingest_location(u)

    current = await service.get_asset_current("FL-001")
    assert current is not None
    assert current.latitude == pytest.approx(40.999)


async def test_get_all_active_assets(service, session_factory):
    """Should return all registered assets."""
    await _create_asset(session_factory, "A1")
    await _create_asset(session_factory, "A2")

    u = GPSUpdate(asset_id="A1", latitude=40.0, longitude=-74.0, speed_kmh=0.0)
    await service.ingest_location(u)

    assets = await service.get_all_active_assets()
    assert len(assets) == 2
    ids = {a["id"] for a in assets}
    assert ids == {"A1", "A2"}


async def test_ingest_unknown_asset_raises(service):
    """Ingesting GPS for a non-existent asset should raise ValueError."""
    u = GPSUpdate(asset_id="GHOST", latitude=0.0, longitude=0.0, speed_kmh=0.0)
    with pytest.raises(ValueError, match="not found"):
        await service.ingest_location(u)


def test_determine_status_offline(service):
    """Old timestamp should map to OFFLINE."""
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert service._determine_status(5.0, old) == AssetStatus.OFFLINE


def test_determine_status_idle(service):
    """Recent timestamp with zero speed -> IDLE."""
    now = datetime.now(timezone.utc)
    assert service._determine_status(0.0, now) == AssetStatus.IDLE


def test_determine_status_active(service):
    """Recent timestamp with speed -> ACTIVE."""
    now = datetime.now(timezone.utc)
    assert service._determine_status(10.0, now) == AssetStatus.ACTIVE
