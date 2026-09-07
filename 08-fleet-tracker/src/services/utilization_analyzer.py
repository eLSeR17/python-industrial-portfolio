"""Fleet utilisation analysis.

Computes active / idle / maintenance time for individual assets and the
whole fleet, and produces dashboard KPIs.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.settings import get_settings
from src.models.db import Asset, LocationHistory, MaintenanceRecord
from src.models.schemas import AssetStatus, DashboardStats, UtilizationReport
from src.utils.time_utils import classify_time_periods


class UtilizationAnalyzer:
    """Derive utilisation metrics from location and maintenance data."""

    def __init__(self, db_session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = db_session_factory

    async def calculate_utilization(
        self,
        asset_id: str,
        start: datetime,
        end: datetime,
    ) -> UtilizationReport:
        """Produce a utilisation report for one asset within a time window."""
        async with self._session_factory() as session:
            asset = await session.get(Asset, asset_id)
            if asset is None:
                raise ValueError(f"Asset {asset_id!r} not found")

            stmt = (
                select(LocationHistory)
                .where(LocationHistory.asset_id == asset_id)
                .where(LocationHistory.timestamp >= start)
                .where(LocationHistory.timestamp <= end)
                .order_by(LocationHistory.timestamp)
            )
            locations = (await session.execute(stmt)).scalars().all()

            # Maintenance hours
            m_stmt = (
                select(MaintenanceRecord)
                .where(MaintenanceRecord.asset_id == asset_id)
                .where(MaintenanceRecord.scheduled_date >= start)
                .where(MaintenanceRecord.scheduled_date <= end)
            )
            maint = (await session.execute(m_stmt)).scalars().all()

        maintenance_hours = sum(
            (m.completed_date - m.scheduled_date).total_seconds() / 3600.0
            for m in maint
            if m.completed_date
        )

        total_hours = (end - start).total_seconds() / 3600.0

        # Classify periods
        loc_dicts = [
            {"timestamp": l.timestamp, "is_moving": l.is_moving, "speed_kmh": l.speed_kmh}
            for l in locations
        ]
        periods = classify_time_periods(loc_dicts, get_settings().idle_threshold_minutes)

        active_hours = sum(
            p["duration_minutes"] / 60.0 for p in periods if p["type"] == "active"
        )
        idle_hours = sum(
            p["duration_minutes"] / 60.0 for p in periods if p["type"] == "idle"
        )

        utilization_pct = (active_hours / total_hours * 100) if total_hours > 0 else 0.0
        idle_pct = (idle_hours / total_hours * 100) if total_hours > 0 else 0.0

        return UtilizationReport(
            asset_id=asset_id,
            asset_name=asset.name,
            period_start=start,
            period_end=end,
            total_hours=round(total_hours, 2),
            active_hours=round(active_hours, 2),
            idle_hours=round(idle_hours, 2),
            maintenance_hours=round(maintenance_hours, 2),
            utilization_pct=round(utilization_pct, 2),
            idle_pct=round(idle_pct, 2),
        )

    async def calculate_fleet_utilization(
        self,
        start: datetime,
        end: datetime,
    ) -> list[UtilizationReport]:
        """Produce utilisation reports for every asset."""
        async with self._session_factory() as session:
            result = await session.execute(select(Asset))
            assets = result.scalars().all()

        reports: list[UtilizationReport] = []
        for a in assets:
            try:
                r = await self.calculate_utilization(a.id, start, end)
                reports.append(r)
            except ValueError:
                continue
        return reports

    async def get_dashboard_stats(self) -> DashboardStats:
        """Aggregate fleet-wide KPIs for the dashboard."""
        async with self._session_factory() as session:
            assets = (await session.execute(select(Asset))).scalars().all()

        total = len(assets)
        active = sum(1 for a in assets if a.status == AssetStatus.ACTIVE.value)
        idle = sum(1 for a in assets if a.status == AssetStatus.IDLE.value)

        # Maintenance due: assets where hours_used > (interval - warning_days * daily_hours)
        settings = get_settings()
        daily_hours = 8.0
        warning_threshold = settings.maintenance_warning_days * daily_hours
        maint_due = sum(
            1 for a in assets
            if a.total_hours_used >= a.maintenance_interval_hours - warning_threshold
        )

        # Average utilisation over last 24 h
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(hours=24)
        total_util = 0.0
        count_util = 0
        for a in assets:
            try:
                report = await self.calculate_utilization(a.id, day_ago, now)
                total_util += report.utilization_pct
                count_util += 1
            except ValueError:
                continue
        avg_util = total_util / count_util if count_util else 0.0

        # Alerts: geofence events in last 24 h (simplified count from DB)
        from src.models.db import GeofenceEvent  # noqa: F811
        async with self._session_factory() as session:
            alert_stmt = select(func.count(GeofenceEvent.id)).where(
                GeofenceEvent.timestamp >= day_ago
            )
            alerts = (await session.execute(alert_stmt)).scalar() or 0

        return DashboardStats(
            total_assets=total,
            active_assets=active,
            idle_assets=idle,
            avg_utilization=round(avg_util, 2),
            alerts_count=alerts,
            maintenance_due_count=maint_due,
        )

    @staticmethod
    def _classify_periods(
        locations: list,
        threshold_minutes: int,
    ) -> dict:
        """Classify consecutive location records into active / idle blocks."""
        loc_dicts = [
            {"timestamp": l.timestamp if hasattr(l, "timestamp") else l["timestamp"],
             "is_moving": l.is_moving if hasattr(l, "is_moving") else l.get("is_moving", False)}
            for l in locations
        ]
        periods = classify_time_periods(loc_dicts, threshold_minutes)
        return {"periods": periods}

    @staticmethod
    def _get_shift_hours(
        start: datetime,
        end: datetime,
    ) -> dict[str, float]:
        """Break total time into day / night shift hours."""
        total_h = (end - start).total_seconds() / 3600.0
        day_start = start.replace(hour=6, minute=0, second=0, microsecond=0)
        day_end = start.replace(hour=18, minute=0, second=0, microsecond=0)
        if start.hour >= 6 and start.hour < 18:
            day_h = min(total_h, (day_end - start).total_seconds() / 3600.0)
        else:
            day_h = max(0.0, total_h - (day_end - start).total_seconds() / 3600.0)
        night_h = total_h - day_h
        return {"day_hours": round(max(0.0, day_h), 2), "night_hours": round(max(0.0, night_h), 2)}
