"""Smart meter data reader – simulates or ingests real meter data.

Supports:
- Generating realistic synthetic meter data with daily/weekly/seasonal patterns
- Ingesting real smart meter readings via API
- Resampling raw readings into standard intervals (15-min, hourly, daily)
"""

import math
import random
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_synthetic_readings(
    facility_id: uuid.UUID,
    meter_id: str = "MTR-001",
    start: datetime | None = None,
    end: datetime | None = None,
    interval_minutes: int = 15,
    baseload_kw: float = 120.0,
    peak_kw: float = 450.0,
    weekend_ratio: float = 0.55,
    pf_nominal: float = 0.92,
    pf_std: float = 0.03,
    noise_pct: float = 0.05,
    shift_pattern: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Generate a DataFrame of synthetic meter readings with realistic patterns.

    The synthetic data exhibits:
    - Diurnal cycle: low overnight, ramp-up at shift start, peak mid-day, ramp-down
    - Weekly cycle: lower consumption on weekends
    - Seasonal variation: higher in summer (cooling) and winter (heating)
    - Random noise simulating process variability
    - Power factor variations (drops during motor-heavy periods)

    Parameters
    ----------
    facility_id : UUID of the facility.
    meter_id : Smart meter serial number.
    start / end : Time range (defaults: last 30 days to now).
    interval_minutes : Reading interval (15, 30, or 60 minutes).
    baseload_kw : Minimum overnight/weekend load.
    peak_kw : Maximum daytime load on weekdays.
    weekend_ratio : Weekday peak as fraction of weekday peak.
    pf_nominal : Nominal power factor.
    pf_std : Standard deviation of power factor.
    noise_pct : Random noise as fraction of load.
    shift_pattern : List of (start_hour, end_hour) tuples defining active shifts.
        Defaults to two shifts: (6,14) and (14,22).

    Returns
    -------
    DataFrame with columns matching the MeterReading schema.
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=30)
    if end is None:
        end = datetime.now(timezone.utc)

    if shift_pattern is None:
        shift_pattern = [(6, 14), (14, 22)]

    intervals = pd.date_range(start, end, freq=f"{interval_minutes}min", tz=timezone.utc)
    n = len(intervals)
    rng = np.random.default_rng(42)

    # --- Base load curve (hourly profile) ---
    hour_profile = np.zeros(24)
    for h in range(24):
        in_shift = any(s <= h < e for s, e in shift_pattern)
        if in_shift:
            # Ramp up first hour, steady, ramp down last hour
            relative_hour = h - min(s for s, e in shift_pattern if s <= h < e)
            shift_len = max(e - s for s, e in shift_pattern if s <= h < e)
            if relative_hour == 0:
                hour_profile[h] = 0.6
            elif relative_hour == 1:
                hour_profile[h] = 0.9
            elif relative_hour >= shift_len - 1:
                hour_profile[h] = 0.7
            else:
                hour_profile[h] = 1.0
        else:
            hour_profile[h] = 0.0  # baseload only

    records = []
    for i, ts in enumerate(intervals):
        hour = ts.hour
        month = ts.month
        is_weekend = ts.weekday() >= 5

        # Load factor from profile
        load_factor = hour_profile[hour]
        max_for_day = peak_kw * (weekend_ratio if is_weekend else 1.0)

        # Seasonal multiplier (summer cooling, mild winter heating)
        seasonal = 1.0 + 0.12 * math.sin(2 * math.pi * (month - 3) / 12)

        # Active power
        target_kw = baseload_kw + (max_for_day - baseload_kw) * load_factor * seasonal
        noise = rng.normal(0, target_kw * noise_pct)
        active_kw = max(baseload_kw * 0.8, target_kw + noise)

        # Power factor: drops during heavy motor loads
        pf_offset = -0.04 * load_factor + rng.normal(0, pf_std)
        pf = max(0.70, min(1.0, pf_nominal + pf_offset))

        # Derived quantities
        apparent_kw = active_kw / pf if pf > 0 else active_kw
        reactive_kw = active_kw * math.tan(math.acos(pf)) if pf > 0 else 0.0

        # Energy for this interval
        hours_in_interval = interval_minutes / 60.0
        active_kwh = active_kw * hours_in_interval
        reactive_kvarh = reactive_kw * hours_in_interval
        apparent_kvah = apparent_kw * hours_in_interval

        # Voltage & frequency (slight variation around nominal)
        voltage = 480.0 + rng.normal(0, 2.0)
        frequency = 60.0 + rng.normal(0, 0.05)

        # Current: I = P / (sqrt(3) * V * PF) for 3-phase
        current = active_kw * 1000 / (math.sqrt(3) * voltage * pf) if pf > 0 else 0

        records.append({
            "facility_id": facility_id,
            "meter_id": meter_id,
            "timestamp": ts.to_pydatetime(),
            "active_energy_kwh": round(active_kwh, 4),
            "reactive_energy_kvarh": round(reactive_kvarh, 4),
            "apparent_energy_kvah": round(apparent_kvah, 4),
            "demand_kw": round(active_kw, 2),
            "demand_kva": round(apparent_kw, 2),
            "power_factor": round(pf, 4),
            "voltage_avg": round(voltage, 1),
            "frequency_hz": round(frequency, 3),
            "current_a": round(current, 1),
            "thd_voltage_pct": round(rng.uniform(1.5, 4.5), 2),
            "temperature_c": round(22 + 8 * math.sin(2 * math.pi * (hour - 6) / 24) + rng.normal(0, 1.5), 1),
        })

    return pd.DataFrame(records)


def resample_readings(
    df: pd.DataFrame,
    freq: str = "1h",
    agg: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resample meter readings to a coarser interval.

    Parameters
    ----------
    df : Raw readings DataFrame (must have 'timestamp' column).
    freq : Pandas frequency string ('15min', '1h', '1D').
    agg : Custom aggregation dict. Defaults to sensible energy/demand aggregations.

    Returns
    -------
    Resampled DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if agg is None:
        agg = {
            "active_energy_kwh": "sum",
            "reactive_energy_kvarh": "sum",
            "apparent_energy_kvah": "sum",
            "demand_kw": "max",
            "demand_kva": "max",
            "power_factor": "mean",
            "voltage_avg": "mean",
            "frequency_hz": "mean",
            "current_a": "mean",
            "thd_voltage_pct": "mean",
            "temperature_c": "mean",
        }

    existing_agg = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(freq).agg(existing_agg).dropna(subset=["active_energy_kwh"])
    return resampled.reset_index()


def compute_rolling_stats(
    df: pd.DataFrame,
    window: str = "168h",
) -> pd.DataFrame:
    """Add rolling statistics columns to a readings DataFrame.

    Adds rolling mean, std, min, max for demand_kw and power_factor.
    Useful for trend analysis and anomaly detection baselines.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    for col in ["demand_kw", "power_factor"]:
        if col in df.columns:
            rm = df[col].rolling(window, min_periods=1)
            df[f"{col}_rmean"] = rm.mean()
            df[f"{col}_rstd"] = rm.std()
            df[f"{col}_rmin"] = rm.min()
            df[f"{col}_rmax"] = rm.max()

    return df.reset_index()
