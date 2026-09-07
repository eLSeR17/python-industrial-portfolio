"""Time-related helper functions for the Fleet Tracker.

Provides shift calculations, working-hours classification, and period-overlap
utilities used by the utilisation and maintenance services.
"""

from datetime import datetime, timedelta


def get_current_shift(
    dt: datetime | None = None,
    start_hour: int = 6,
    shift_hours: int = 12,
) -> str:
    """Return ``"Day"`` or ``"Night"`` depending on the current hour.

    The day shift starts at *start_hour* and lasts *shift_hours* hours.
    """
    if dt is None:
        dt = datetime.now()
    hour = dt.hour
    if start_hour <= hour < start_hour + shift_hours:
        return "Day"
    return "Night"


def get_shift_boundaries(
    date: datetime,
    start_hour: int = 6,
    shift_hours: int = 12,
) -> tuple[datetime, datetime]:
    """Return ``(shift_start, shift_end)`` for the shift that *date* falls into."""
    d = date.date()
    shift_start = datetime(d.year, d.month, d.day, start_hour)
    shift_end = shift_start + timedelta(hours=shift_hours)

    if date < shift_start:
        shift_start -= timedelta(hours=shift_hours)
        shift_end = datetime(d.year, d.month, d.day, start_hour)
    elif date >= shift_end:
        shift_start = shift_end
        shift_end = shift_start + timedelta(hours=shift_hours)

    return shift_start, shift_end


def is_working_hours(
    dt: datetime,
    work_start: int = 6,
    work_end: int = 22,
) -> bool:
    """Return ``True`` if *dt* falls within working hours."""
    return work_start <= dt.hour < work_end


def time_in_period(
    start: datetime,
    end: datetime,
    period_start: datetime,
    period_end: datetime,
) -> float:
    """Return the number of **hours** that the interval [*start*, *end*] overlaps with [*period_start*, *period_end*]."""
    latest_start = max(start, period_start)
    earliest_end = min(end, period_end)
    delta = earliest_end - latest_start
    if delta.total_seconds() <= 0:
        return 0.0
    return delta.total_seconds() / 3600.0


def classify_time_periods(
    locations: list[dict],
    idle_threshold_minutes: int = 5,
) -> list[dict]:
    """Given a chronologically sorted list of location dicts, classify each consecutive interval as ``"active"`` or ``"idle"``.

    Each dict in *locations* must contain at least ``timestamp`` and ``is_moving``.
    Returns a list of dicts with ``start``, ``end``, ``type`` and ``duration_minutes``.
    """
    if len(locations) < 2:
        return []

    periods: list[dict] = []
    current_type: str = "active" if locations[0].get("is_moving", False) else "idle"
    period_start = locations[0]["timestamp"]

    for i in range(1, len(locations)):
        moving = locations[i].get("is_moving", False)
        this_type = "active" if moving else "idle"

        if this_type != current_type:
            period_end = locations[i]["timestamp"]
            duration = (period_end - period_start).total_seconds() / 60.0
            periods.append({
                "start": period_start,
                "end": period_end,
                "type": current_type,
                "duration_minutes": round(duration, 2),
            })
            current_type = this_type
            period_start = period_end

    # final segment
    period_end = locations[-1]["timestamp"]
    duration = (period_end - period_start).total_seconds() / 60.0
    periods.append({
        "start": period_start,
        "end": period_end,
        "type": current_type,
        "duration_minutes": round(duration, 2),
    })

    return periods
