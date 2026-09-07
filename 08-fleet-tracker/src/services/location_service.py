"""Location tracking service.

Handles GPS data ingestion, distance computation, speed / bearing
derivation, and historical position queries.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.db import Asset, LocationHistory
from src.models.schemas import AssetStatus, GPSUpdate, LocationRecord
from src.utils.geo_utils import calculate_bearing, haversine_distance


class LocationService:
    """Ingest and query GPS positions for tracked assets."""

    def __init__(self, db_session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = db_session_factory

    async def ingest_location(self, update: GPSUpdate) -> LocationRecord:
        """Persist a new GPS fix, compute distance/bearing from previous fix, and update the asset's last-known position."""
        ts = update.timestamp or datetime.now(timezone.utc)

        async with self._session_factory() as session:
            # Fetch asset
            asset = await session.get(Asset, update.asset_id)
            if asset is None:
                raise ValueError(f"Asset {update.asset_id!r} not found")

            # Calculate distance from last known position
            dist_m = 0.0
            bearing = update.heading_degrees
            if asset.last_latitude is not None and asset.last_longitude is not None:
                dist_m = haversine_distance(
                    asset.last_latitude,
                    asset.last_longitude,
                    update.latitude,
                    update.longitude,
                )
                bearing = calculate_bearing(
                    asset.last_latitude,
                    asset.last_longitude,
                    update.latitude,
                    update.longitude,
                )

            is_moving = update.speed_kmh > 0.5

            record = LocationHistory(
                asset_id=update.asset_id,
                latitude=update.latitude,
                longitude=update.longitude,
                timestamp=ts,
                speed_kmh=update.speed_kmh,
                heading=bearing,
                is_moving=is_moving,
            )
            session.add(record)

            # Update asset position + stats
            asset.last_latitude = update.latitude
            asset.last_longitude = update.longitude
            asset.last_update = ts
            asset.distance_traveled_km += dist_m / 1000.0

            # Determine new status
            asset.status = self._determine_status(update.speed_kmh, ts).value

            await session.commit()
            await session.refresh(record)

            return LocationRecord(
                id=record.id,
                asset_id=record.asset_id,
                latitude=record.latitude,
                longitude=record.longitude,
                timestamp=record.timestamp,
                speed_kmh=record.speed_kmh,
                heading=record.heading,
                distance_from_prev=round(dist_m, 2),
                is_moving=record.is_moving,
            )

    async def get_asset_history(self, asset_id: str, hours: int = 24) -> list[LocationRecord]:
        """Return GPS history for *asset_id* over the last *hours* hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self._session_factory() as session:
            stmt = (
                select(LocationHistory)
                .where(LocationHistory.asset_id == asset_id)
                .where(LocationHistory.timestamp >= cutoff)
                .order_by(LocationHistory.timestamp)
            )
            rows = (await session.execute(stmt)).scalars().all()

        records: list[LocationRecord] = []
        prev: LocationRecord | None = None
        for r in rows:
            rec = LocationRecord(
                id=r.id,
                asset_id=r.asset_id,
                latitude=r.latitude,
                longitude=r.longitude,
                timestamp=r.timestamp,
                speed_kmh=r.speed_kmh,
                heading=r.heading,
                distance_from_prev=0.0,
                is_moving=r.is_moving,
            )
            if prev is not None:
                rec.distance_from_prev = round(
                    haversine_distance(prev.latitude, prev.longitude, rec.latitude, rec.longitude),
                    2,
                )
            records.append(rec)
            prev = rec
        return records

    async def get_asset_current(self, asset_id: str) -> LocationRecord | None:
        """Return the most recent GPS fix for *asset_id*."""
        async with self._session_factory() as session:
            stmt = (
                select(LocationHistory)
                .where(LocationHistory.asset_id == asset_id)
                .order_by(LocationHistory.timestamp.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return LocationRecord(
                id=row.id,
                asset_id=row.asset_id,
                latitude=row.latitude,
                longitude=row.longitude,
                timestamp=row.timestamp,
                speed_kmh=row.speed_kmh,
                heading=row.heading,
                is_moving=row.is_moving,
            )

    async def get_all_active_assets(self) -> list[dict]:
        """Return all assets with their last known position."""
        async with self._session_factory() as session:
            result = await session.execute(select(Asset))
            assets = result.scalars().all()

        output: list[dict] = []
        for a in assets:
            output.append({
                "id": a.id,
                "name": a.name,
                "asset_type": a.asset_type,
                "status": a.status,
                "latitude": a.last_latitude,
                "longitude": a.last_longitude,
                "last_update": a.last_update.isoformat() if a.last_update else None,
                "total_hours_used": a.total_hours_used,
                "distance_traveled_km": a.distance_traveled_km,
            })
        return output

    async def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Haversine wrapper returning metres."""
        return haversine_distance(lat1, lon1, lat2, lon2)

    @staticmethod
    def _determine_status(speed: float, last_update: datetime) -> AssetStatus:
        """Derive asset status from the latest GPS data."""
        now = datetime.now(timezone.utc)
        age = (now - last_update.replace(tzinfo=timezone.utc)).total_seconds()
        if age > 3600:
            return AssetStatus.OFFLINE
        if speed > 0.5:
            return AssetStatus.ACTIVE
        return AssetStatus.IDLE
