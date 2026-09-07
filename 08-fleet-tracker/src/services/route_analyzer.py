"""Route recording and deviation analysis.

Builds routes from GPS history, compares planned vs actual trajectories,
and detects significant deviations.
"""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.db import Asset, LocationHistory
from src.models.schemas import Deviation, RouteRecord
from src.utils.geo_utils import haversine_distance, simplify_route, total_route_distance


class RouteAnalyzer:
    """Analyse and compare routes from GPS trace data."""

    def __init__(self, db_session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = db_session_factory

    async def record_route(
        self,
        asset_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> RouteRecord:
        """Build a :class:`RouteRecord` from the location history within the given window."""
        async with self._session_factory() as session:
            stmt = (
                select(LocationHistory)
                .where(LocationHistory.asset_id == asset_id)
                .where(LocationHistory.timestamp >= start_time)
                .where(LocationHistory.timestamp <= end_time)
                .order_by(LocationHistory.timestamp)
            )
            rows = (await session.execute(stmt)).scalars().all()

        if not rows:
            raise ValueError(f"No location data for asset {asset_id!r} in the specified window")

        actual_route = [(r.latitude, r.longitude) for r in rows]
        simplified = simplify_route(actual_route, tolerance_m=20.0)
        actual_distance = total_route_distance(actual_route)

        # Straight-line distance as baseline
        straight = haversine_distance(
            actual_route[0][0], actual_route[0][1],
            actual_route[-1][0], actual_route[-1][1],
        ) / 1000.0
        efficiency = self._calculate_efficiency(actual_distance, straight)

        actual_geojson = [{"latitude": p[0], "longitude": p[1]} for p in simplified]

        return RouteRecord(
            id=hash((asset_id, start_time.isoformat())) % (2**31),
            asset_id=asset_id,
            start_time=start_time,
            end_time=end_time,
            planned_route=[],
            actual_route=actual_geojson,
            distance_planned_km=round(straight, 3),
            distance_actual_km=round(actual_distance, 3),
            efficiency_pct=round(efficiency, 2),
            deviations=[],
        )

    async def compare_routes(
        self,
        planned: list[tuple[float, float]],
        actual: list[tuple[float, float]],
    ) -> dict:
        """Compare planned vs actual and return a summary dict with deviation list."""
        planned_distance = total_route_distance(planned)
        actual_distance = total_route_distance(actual)
        deviations = self._detect_deviations(actual, planned, threshold_m=50.0)
        efficiency = self._calculate_efficiency(actual_distance, planned_distance)
        return {
            "planned_distance_km": round(planned_distance, 3),
            "actual_distance_km": round(actual_distance, 3),
            "efficiency_pct": round(efficiency, 2),
            "deviation_count": len(deviations),
            "deviations": deviations,
        }

    async def get_route_history(
        self,
        asset_id: str,
        days: int = 7,
    ) -> list[RouteRecord]:
        """Return route records for the last *days* days, segmented per day."""
        now = datetime.now(timezone.utc)
        routes: list[RouteRecord] = []
        for d in range(days):
            day_start = (now - timedelta(days=d + 1)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            try:
                route = await self.record_route(asset_id, day_start, day_end)
                routes.append(route)
            except ValueError:
                continue
        return routes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_deviations(
        actual_route: list[tuple[float, float]],
        planned_route: list[tuple[float, float]],
        threshold_m: float = 50.0,
    ) -> list[dict]:
        """Find points in *actual_route* that are further than *threshold_m* from the nearest planned point."""
        deviations: list[dict] = []
        for idx, (lat, lon) in enumerate(actual_route):
            min_dist = float("inf")
            for plat, plon in planned_route:
                d = haversine_distance(lat, lon, plat, plon)
                if d < min_dist:
                    min_dist = d
            if min_dist > threshold_m:
                deviations.append({
                    "latitude": lat,
                    "longitude": lon,
                    "distance_from_planned_m": round(min_dist, 2),
                    "index": idx,
                })
        return deviations

    @staticmethod
    def _calculate_efficiency(actual_distance: float, reference_distance: float) -> float:
        """Route efficiency as a percentage (100 % = perfectly straight)."""
        if reference_distance <= 0:
            return 0.0
        return min(100.0, (reference_distance / actual_distance) * 100.0) if actual_distance > 0 else 0.0
