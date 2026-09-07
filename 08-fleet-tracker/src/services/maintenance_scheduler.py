"""Predictive maintenance scheduling.

Extrapolates next service dates from usage-rate data and classifies
urgency by priority.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.db import Asset, MaintenanceRecord
from src.models.schemas import MaintenancePriority, MaintenanceSchedule


class MaintenanceScheduler:
    """Calculate and track preventive-maintenance windows."""

    def __init__(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        warning_days: int = 30,
    ) -> None:
        self._session_factory = db_session_factory
        self._warning_days = warning_days

    async def schedule_maintenance(self, asset_id: str) -> MaintenanceSchedule:
        """Predict the next service date and priority for *asset_id*."""
        async with self._session_factory() as session:
            asset = await session.get(Asset, asset_id)
            if asset is None:
                raise ValueError(f"Asset {asset_id!r} not found")

            next_date = self._predict_maintenance_date(asset)
            hours_until = max(0.0, asset.maintenance_interval_hours - asset.total_hours_used)
            distance_until = max(0.0, hours_until * 10.0)  # rough estimate: 10 km / h
            priority = self._calculate_priority(hours_until, self._warning_days)

            downtime = self._estimate_downtime(asset.asset_type)

            return MaintenanceSchedule(
                asset_id=asset.id,
                asset_name=asset.name,
                next_service_date=next_date,
                hours_until_service=round(hours_until, 2),
                distance_until_service=round(distance_until, 2),
                priority=priority,
                estimated_downtime_hours=downtime,
            )

    async def get_fleet_maintenance_schedule(self) -> list[MaintenanceSchedule]:
        """Return predicted maintenance for every asset, sorted by priority."""
        async with self._session_factory() as session:
            assets = (await session.execute(select(Asset))).scalars().all()

        schedules: list[MaintenanceSchedule] = []
        for a in assets:
            try:
                s = await self.schedule_maintenance(a.id)
                schedules.append(s)
            except ValueError:
                continue

        priority_order = {MaintenancePriority.CRITICAL.value: 0,
                          MaintenancePriority.HIGH.value: 1,
                          MaintenancePriority.MEDIUM.value: 2,
                          MaintenancePriority.LOW.value: 3}
        schedules.sort(key=lambda s: priority_order.get(s.priority.value, 99))
        return schedules

    async def record_service(
        self,
        asset_id: str,
        service_type: str,
        hours: float,
        notes: str,
    ) -> None:
        """Record a completed maintenance event and reset the usage counter."""
        async with self._session_factory() as session:
            asset = await session.get(Asset, asset_id)
            if asset is None:
                raise ValueError(f"Asset {asset_id!r} not found")

            record = MaintenanceRecord(
                asset_id=asset_id,
                service_type=service_type,
                scheduled_date=datetime.now(timezone.utc),
                completed_date=datetime.now(timezone.utc),
                hours_at_service=hours,
                notes=notes,
            )
            session.add(record)
            asset.total_hours_used = 0.0
            await session.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_maintenance_date(self, asset: Asset) -> datetime:
        """Extrapolate the date when hours_used will reach the interval."""
        now = datetime.now(timezone.utc)
        remaining = asset.maintenance_interval_hours - asset.total_hours_used
        if remaining <= 0:
            return now  # overdue

        # Estimate daily usage rate from total_hours_used and creation data
        days_active = max(1, (now - (asset.last_update or now)).days or 1)
        daily_rate = asset.total_hours_used / days_active if days_active else 1.0
        days_remaining = remaining / daily_rate if daily_rate > 0 else 30
        return now + timedelta(days=days_remaining)

    @staticmethod
    def _calculate_priority(hours_until: float, warning_days: int) -> MaintenancePriority:
        """Map remaining hours to a priority level."""
        if hours_until <= 0:
            return MaintenancePriority.CRITICAL
        remaining_pct = hours_until / 500.0  # normalise to a typical interval
        if remaining_pct < 0.2:
            return MaintenancePriority.CRITICAL
        if remaining_pct < 0.5:
            return MaintenancePriority.HIGH
        if remaining_pct < 0.8:
            return MaintenancePriority.MEDIUM
        return MaintenancePriority.LOW

    @staticmethod
    def _estimate_downtime(asset_type: str) -> float:
        """Rough expected downtime in hours by asset type."""
        defaults: dict[str, float] = {
            "FORKLIFT": 4.0,
            "AGV": 6.0,
            "TRUCK": 8.0,
            "CRANE": 12.0,
            "CONVEYOR": 10.0,
            "OTHER": 6.0,
        }
        return defaults.get(asset_type, 6.0)
