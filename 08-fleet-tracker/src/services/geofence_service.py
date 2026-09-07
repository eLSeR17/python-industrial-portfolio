"""Geofence management and breach detection.

Handles geofence CRUD, point-in-polygon/circle tests, and ENTRY/EXIT
event generation.
"""

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.db import Geofence as GeofenceDB
from src.models.db import GeofenceEvent
from src.models.schemas import (
    EventType,
    FenceType,
    GeofenceCreate,
    GeofenceEventResponse,
    GeofenceResponse,
    ZoneType,
)
from src.utils.geo_utils import haversine_distance, point_in_circle, point_in_polygon


class GeofenceService:
    """Create and evaluate geofences against asset positions."""

    def __init__(self, db_session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = db_session_factory

    async def create_geofence(self, geofence: GeofenceCreate) -> GeofenceResponse:
        """Persist a new geofence and return its representation."""
        coords_json = json.dumps(geofence.coordinates)
        async with self._session_factory() as session:
            db_fence = GeofenceDB(
                name=geofence.name,
                fence_type=geofence.fence_type.value,
                coordinates_json=coords_json,
                zone_type=geofence.zone_type.value,
                alert_on_entry=geofence.alert_on_entry,
                alert_on_exit=geofence.alert_on_exit,
            )
            session.add(db_fence)
            await session.commit()
            await session.refresh(db_fence)

        return GeofenceResponse(
            id=db_fence.id,
            name=db_fence.name,
            fence_type=FenceType(db_fence.fence_type),
            coordinates=json.loads(db_fence.coordinates_json),
            zone_type=ZoneType(db_fence.zone_type),
            alert_on_entry=db_fence.alert_on_entry,
            alert_on_exit=db_fence.alert_on_exit,
            is_active=db_fence.is_active,
            created_at=db_fence.created_at,
        )

    async def check_asset_in_geofences(
        self,
        asset_id: str,
        lat: float,
        lon: float,
    ) -> list[GeofenceEventResponse]:
        """Check the point against every active geofence and generate ENTRY/EXIT events."""
        now = datetime.now(timezone.utc)
        events: list[GeofenceEventResponse] = []

        async with self._session_factory() as session:
            stmt = select(GeofenceDB).where(GeofenceDB.is_active.is_(True))
            fences = (await session.execute(stmt)).scalars().all()

            for fence in fences:
                coords = json.loads(fence.coordinates_json)
                inside = self._test_inside(fence.fence_type, lat, lon, coords)

                # Determine previous state via latest event
                prev_event_stmt = (
                    select(GeofenceEvent)
                    .where(GeofenceEvent.asset_id == asset_id)
                    .where(GeofenceEvent.geofence_id == fence.id)
                    .order_by(GeofenceEvent.timestamp.desc())
                    .limit(1)
                )
                prev = (await session.execute(prev_event_stmt)).scalar_one_or_none()
                was_inside = prev.event_type == EventType.ENTRY.value if prev else False

                event_type: str | None = None
                if inside and not was_inside and fence.alert_on_entry:
                    event_type = EventType.ENTRY.value
                elif not inside and was_inside and fence.alert_on_exit:
                    event_type = EventType.EXIT.value

                if event_type:
                    event = GeofenceEvent(
                        asset_id=asset_id,
                        geofence_id=fence.id,
                        event_type=event_type,
                        timestamp=now,
                        latitude=lat,
                        longitude=lon,
                    )
                    session.add(event)
                    await session.commit()
                    await session.refresh(event)

                    events.append(GeofenceEventResponse(
                        id=event.id,
                        asset_id=event.asset_id,
                        geofence_id=event.geofence_id,
                        event_type=EventType(event.event_type),
                        timestamp=event.timestamp,
                        latitude=event.latitude,
                        longitude=event.longitude,
                    ))

        return events

    async def get_geofence_events(
        self,
        asset_id: str | None = None,
        hours: int = 24,
    ) -> list[GeofenceEventResponse]:
        """Return geofence events within the specified time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self._session_factory() as session:
            stmt = (
                select(GeofenceEvent)
                .where(GeofenceEvent.timestamp >= cutoff)
                .order_by(GeofenceEvent.timestamp.desc())
            )
            if asset_id:
                stmt = stmt.where(GeofenceEvent.asset_id == asset_id)
            rows = (await session.execute(stmt)).scalars().all()

        return [
            GeofenceEventResponse(
                id=r.id,
                asset_id=r.asset_id,
                geofence_id=r.geofence_id,
                event_type=EventType(r.event_type),
                timestamp=r.timestamp,
                latitude=r.latitude,
                longitude=r.longitude,
            )
            for r in rows
        ]

    async def get_all_geofences(self) -> list[GeofenceResponse]:
        """Return every registered geofence."""
        async with self._session_factory() as session:
            stmt = select(GeofenceDB).order_by(GeofenceDB.id)
            rows = (await session.execute(stmt)).scalars().all()

        return [
            GeofenceResponse(
                id=r.id,
                name=r.name,
                fence_type=FenceType(r.fence_type),
                coordinates=json.loads(r.coordinates_json),
                zone_type=ZoneType(r.zone_type),
                alert_on_entry=r.alert_on_entry,
                alert_on_exit=r.alert_on_exit,
                is_active=r.is_active,
                created_at=r.created_at,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _test_inside(
        fence_type: str,
        lat: float,
        lon: float,
        coords: list,
    ) -> bool:
        """Dispatch to the correct containment test."""
        if fence_type == FenceType.POLYGON.value:
            polygon = [(c[0], c[1]) for c in coords]
            return point_in_polygon(lat, lon, polygon)
        if fence_type == FenceType.CIRCLE.value:
            c = coords[0]
            return point_in_circle(lat, lon, c["lat"], c["lon"], c["radius_m"])
        return False

    @staticmethod
    def _point_in_polygon(
        lat: float, lon: float, polygon: list[tuple[float, float]]
    ) -> bool:
        """Ray-casting point-in-polygon test."""
        return point_in_polygon(lat, lon, polygon)

    @staticmethod
    def _point_in_circle(
        lat: float,
        lon: float,
        center_lat: float,
        center_lon: float,
        radius_m: float,
    ) -> bool:
        """Distance-based containment test."""
        return haversine_distance(lat, lon, center_lat, center_lon) <= radius_m
