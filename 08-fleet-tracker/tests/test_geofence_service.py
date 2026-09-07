"""Tests for the GeofenceService using an in-memory SQLite database."""

import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models.db import Base
from src.models.schemas import EventType, FenceType, GeofenceCreate, ZoneType
from src.services.geofence_service import GeofenceService


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
async def service(session_factory) -> GeofenceService:
    return GeofenceService(session_factory)


async def test_create_polygon_geofence(service):
    """Create a polygon geofence and verify response fields."""
    gc = GeofenceCreate(
        name="Zone A",
        fence_type=FenceType.POLYGON,
        coordinates=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        zone_type=ZoneType.WAREHOUSE,
        alert_on_entry=True,
        alert_on_exit=False,
    )
    resp = await service.create_geofence(gc)
    assert resp.id > 0
    assert resp.name == "Zone A"
    assert resp.fence_type == FenceType.POLYGON
    assert resp.is_active is True


async def test_create_circle_geofence(service):
    """Create a circle geofence."""
    gc = GeofenceCreate(
        name="Charging",
        fence_type=FenceType.CIRCLE,
        coordinates=[{"lat": 40.7125, "lon": -74.006, "radius_m": 15}],
        zone_type=ZoneType.CHARGING,
    )
    resp = await service.create_geofence(gc)
    assert resp.name == "Charging"
    coords = json.loads(json.dumps(resp.coordinates))
    assert coords[0]["radius_m"] == 15


async def test_get_all_geofences(service):
    """List should return all created geofences."""
    for i in range(3):
        await service.create_geofence(
            GeofenceCreate(
                name=f"Zone {i}",
                fence_type=FenceType.POLYGON,
                coordinates=[[float(i), float(i)], [float(i), float(i + 1)],
                             [float(i + 1), float(i + 1)], [float(i + 1), float(i)]],
                zone_type=ZoneType.WAREHOUSE,
            )
        )
    fences = await service.get_all_geofences()
    assert len(fences) == 3


async def test_check_entry_event(service):
    """Asset entering a polygon should generate ENTRY event."""
    await service.create_geofence(
        GeofenceCreate(
            name="Box",
            fence_type=FenceType.POLYGON,
            coordinates=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            zone_type=ZoneType.WAREHOUSE,
            alert_on_entry=True,
            alert_on_exit=True,
        )
    )
    events = await service.check_asset_in_geofences("FL-001", 0.5, 0.5)
    assert len(events) == 1
    assert events[0].event_type == EventType.ENTRY


async def test_no_event_when_outside(service):
    """Point outside with no prior entry -> no events."""
    await service.create_geofence(
        GeofenceCreate(
            name="Box",
            fence_type=FenceType.POLYGON,
            coordinates=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            zone_type=ZoneType.WAREHOUSE,
            alert_on_entry=True,
            alert_on_exit=True,
        )
    )
    events = await service.check_asset_in_geofences("FL-001", 5.0, 5.0)
    assert len(events) == 0


async def test_circle_containment(service):
    """Point inside circle with alert_on_entry -> ENTRY event."""
    await service.create_geofence(
        GeofenceCreate(
            name="Circle",
            fence_type=FenceType.CIRCLE,
            coordinates=[{"lat": 0.0, "lon": 0.0, "radius_m": 200}],
            zone_type=ZoneType.CHARGING,
            alert_on_entry=True,
            alert_on_exit=False,
        )
    )
    events = await service.check_asset_in_geofences("FL-001", 0.001, 0.001)
    assert len(events) == 1
    assert events[0].event_type == EventType.ENTRY


async def test_get_geofence_events(service):
    """Query events within the time window."""
    await service.create_geofence(
        GeofenceCreate(
            name="Box",
            fence_type=FenceType.POLYGON,
            coordinates=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            zone_type=ZoneType.WAREHOUSE,
            alert_on_entry=True,
            alert_on_exit=True,
        )
    )
    await service.check_asset_in_geofences("FL-001", 0.5, 0.5)
    events = await service.get_geofence_events("FL-001", hours=1)
    assert len(events) >= 1


def test_point_in_polygon_method(service):
    """Ray-casting via internal method."""
    polygon = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    assert service._point_in_polygon(5.0, 5.0, polygon) is True
    assert service._point_in_polygon(20.0, 20.0, polygon) is False


def test_point_in_circle_method(service):
    """Distance check via internal method."""
    assert service._point_in_circle(0.001, 0.001, 0.0, 0.0, 200.0) is True
    assert service._point_in_circle(5.0, 5.0, 0.0, 0.0, 100.0) is False
