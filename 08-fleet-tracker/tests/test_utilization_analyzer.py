"""Tests for the UtilizationAnalyzer using an in-memory SQLite database."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import update as sql_update

from src.models.db import Asset, Base, LocationHistory
from src.services.utilization_analyzer import UtilizationAnalyzer


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def analyzer(session_factory) -> UtilizationAnalyzer:
    return UtilizationAnalyzer(session_factory)


async def _seed_asset(session_factory, asset_id="FL-001", interval=500.0):
    """Insert an asset and 8 hours of location data (alternating active/idle)."""
    async with session_factory() as session:
        asset = Asset(
            id=asset_id,
            name="Forklift 1",
            asset_type="FORKLIFT",
            maintenance_interval_hours=interval,
        )
        session.add(asset)

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=8)

        for i in range(60):
            ts = start + timedelta(minutes=i * 8)
            is_moving = i % 2 == 0
            speed = 5.0 if is_moving else 0.0
            lh = LocationHistory(
                asset_id=asset_id,
                latitude=40.7128 + i * 0.0001,
                longitude=-74.006,
                timestamp=ts,
                speed_kmh=speed,
                heading=0.0,
                is_moving=is_moving,
            )
            session.add(lh)
        await session.commit()


async def test_calculate_utilization(session_factory, analyzer):
    """Utilisation % should be between 0 and 100."""
    await _seed_asset(session_factory)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=8)

    report = await analyzer.calculate_utilization("FL-001", start, now)
    assert report.asset_id == "FL-001"
    assert report.total_hours == pytest.approx(8.0, abs=0.05)
    assert 0 <= report.utilization_pct <= 100
    assert 0 <= report.idle_pct <= 100


async def test_active_hours_nonzero(session_factory, analyzer):
    """Half the points are moving -> active_hours > 0."""
    await _seed_asset(session_factory)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=8)

    report = await analyzer.calculate_utilization("FL-001", start, now)
    assert report.active_hours > 0
    assert report.idle_hours > 0


async def test_fleet_utilization(session_factory, analyzer):
    """Fleet report should cover all assets."""
    await _seed_asset(session_factory, "A1")
    await _seed_asset(session_factory, "A2")
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=8)

    reports = await analyzer.calculate_fleet_utilization(start, now)
    assert len(reports) == 2
    ids = {r.asset_id for r in reports}
    assert ids == {"A1", "A2"}


async def test_get_dashboard_stats(session_factory, analyzer):
    """Dashboard should report total and active assets."""
    await _seed_asset(session_factory, "A1")
    await _seed_asset(session_factory, "A2")
    async with session_factory() as session:
        await session.execute(
            sql_update(Asset).where(Asset.id == "A2").values(status="IDLE")
        )
        await session.commit()

    stats = await analyzer.get_dashboard_stats()
    assert stats.total_assets == 2
    assert stats.active_assets >= 1
    assert stats.idle_assets >= 1


async def test_unknown_asset_raises(session_factory, analyzer):
    """Requesting utilisation for a missing asset should raise ValueError."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="not found"):
        await analyzer.calculate_utilization("GHOST", now - timedelta(hours=1), now)


def test_shift_hours(analyzer):
    """Shift breakdown should sum to total hours."""
    start = datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 18, 0, tzinfo=timezone.utc)
    result = analyzer._get_shift_hours(start, end)
    total = result["day_hours"] + result["night_hours"]
    assert abs(total - 12.0) < 0.1


async def test_classify_periods(session_factory, analyzer):
    """Internal classifier should produce non-empty period list."""
    await _seed_asset(session_factory)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=8)

    result = analyzer._classify_periods(
        [{"timestamp": start, "is_moving": True},
         {"timestamp": start + timedelta(hours=1), "is_moving": False}],
        threshold_minutes=5,
    )
    assert "periods" in result
    assert len(result["periods"]) >= 2
