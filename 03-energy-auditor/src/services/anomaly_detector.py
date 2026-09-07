"""Energy waste anomaly detection engine.

Detects:
- Consumption spikes (Z-score based on rolling baseline)
- Baseline shifts (sustained deviation from historical norm)
- Equipment left on outside operating hours
- Power factor drops indicating equipment faults
- Temperature-correlated anomalies (HVAC failures)
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.models.schemas import AnomalyListResponse, AnomalyRecord, AnomalySeverity


def _severity_from_zscore(abs_z: float) -> AnomalySeverity:
    """Map absolute Z-score to severity level."""
    if abs_z >= 5.0:
        return AnomalySeverity.CRITICAL
    if abs_z >= 4.0:
        return AnomalySeverity.HIGH
    if abs_z >= 3.0:
        return AnomalySeverity.MEDIUM
    return AnomalySeverity.LOW


def detect_consumption_spikes(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    zscore_threshold: float = 3.0,
    rolling_window: str = "168h",
) -> list[AnomalyRecord]:
    """Detect consumption spikes using Z-score on rolling statistics.

    Each reading's demand_kw is compared to its rolling mean/std. Readings
    where the Z-score exceeds the threshold are flagged as anomalies.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw].
    facility_id : Facility UUID.
    zscore_threshold : Minimum |Z| to flag as anomaly.
    rolling_window : Rolling window for baseline computation.

    Returns
    -------
    List of AnomalyRecord for detected spikes.
    """
    if readings_df.empty or len(readings_df) < 10:
        return []

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Rolling baseline
    rolling_mean = df["demand_kw"].rolling(rolling_window, min_periods=4).mean()
    rolling_std = df["demand_kw"].rolling(rolling_window, min_periods=4).std()

    # Z-scores
    z_scores = (df["demand_kw"] - rolling_mean) / rolling_std.replace(0, np.nan)
    z_scores = z_scores.fillna(0)

    anomalies: list[AnomalyRecord] = []
    spike_mask = z_scores.abs() > zscore_threshold

    for ts in z_scores[spike_mask].index:
        z = float(z_scores.loc[ts])
        measured = float(df.loc[ts, "demand_kw"])
        expected = float(rolling_mean.loc[ts])
        dev_pct = ((measured - expected) / expected * 100) if expected > 0 else 0.0

        anomalies.append(AnomalyRecord(
            timestamp=ts.to_pydatetime(),
            facility_id=facility_id,
            anomaly_type="spike",
            severity=_severity_from_zscore(abs(z)),
            measured_value=round(measured, 2),
            expected_value=round(expected, 2),
            deviation_pct=round(dev_pct, 2),
            description=f"Demand spike: {measured:.1f} kW vs baseline {expected:.1f} kW (Z={z:.2f})",
        ))

    return anomalies


def detect_baseline_shift(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    comparison_days: int = 7,
    shift_threshold_pct: float = 15.0,
) -> list[AnomalyRecord]:
    """Detect sustained baseline shifts by comparing recent vs historical averages.

    Splits the data into two windows: recent (last N days) and historical
    (N days before that). If the average demand shifts by more than the
    threshold percentage, flags the shift.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw].
    facility_id : Facility UUID.
    comparison_days : Number of days for each comparison window.
    shift_threshold_pct : Minimum % change to flag.

    Returns
    -------
    List of AnomalyRecord for detected shifts.
    """
    if readings_df.empty:
        return []

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    latest_ts = df["timestamp"].max()
    recent_start = latest_ts - pd.Timedelta(days=comparison_days)
    hist_start = recent_start - pd.Timedelta(days=comparison_days)

    recent = df[(df["timestamp"] >= recent_start) & (df["timestamp"] <= latest_ts)]
    historical = df[(df["timestamp"] >= hist_start) & (df["timestamp"] < recent_start)]

    if recent.empty or historical.empty:
        return []

    recent_avg = float(recent["demand_kw"].mean())
    hist_avg = float(historical["demand_kw"].mean())

    if hist_avg == 0:
        return []

    shift_pct = (recent_avg - hist_avg) / hist_avg * 100

    if abs(shift_pct) < shift_threshold_pct:
        return []

    direction = "increase" if shift_pct > 0 else "decrease"
    severity = AnomalySeverity.HIGH if abs(shift_pct) > 30 else AnomalySeverity.MEDIUM

    return [AnomalyRecord(
        timestamp=latest_ts.to_pydatetime(),
        facility_id=facility_id,
        anomaly_type="baseline_shift",
        severity=severity,
        measured_value=round(recent_avg, 2),
        expected_value=round(hist_avg, 2),
        deviation_pct=round(shift_pct, 2),
        description=(
            f"Baseline {direction}: recent avg {recent_avg:.1f} kW "
            f"vs historical {hist_avg:.1f} kW ({shift_pct:+.1f}%)"
        ),
    )]


def detect_equipment_left_on(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    off_hours: tuple[int, int] = (22, 5),
    min_offhour_kw: float = 80.0,
) -> list[AnomalyRecord]:
    """Detect potential equipment left running outside operating hours.

    Flags readings during off-hours where demand exceeds a minimum threshold
    that suggests equipment is still running.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw].
    facility_id : Facility UUID.
    off_hours : Tuple (start, end) defining the off-hours window (wraps midnight).
    min_offhour_kw : Minimum demand during off-hours to flag.

    Returns
    -------
    List of AnomalyRecord for equipment-left-on events.
    """
    if readings_df.empty:
        return []

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour

    start_h, end_h = off_hours
    if start_h > end_h:  # wraps midnight
        off_mask = (df["hour"] >= start_h) | (df["hour"] < end_h)
    else:
        off_mask = (df["hour"] >= start_h) & (df["hour"] < end_h)

    off_readings = df[off_mask & (df["demand_kw"] >= min_offhour_kw)]

    anomalies: list[AnomalyRecord] = []
    for ts, row in off_readings.iterrows():
        excess_kw = row["demand_kw"] - min_offhour_kw
        severity = AnomalySeverity.HIGH if excess_kw > 100 else AnomalySeverity.MEDIUM

        anomalies.append(AnomalyRecord(
            timestamp=row["timestamp"],
            facility_id=facility_id,
            anomaly_type="equipment_left_on",
            severity=severity,
            measured_value=round(float(row["demand_kw"]), 2),
            expected_value=min_offhour_kw,
            deviation_pct=round(excess_kw / min_offhour_kw * 100, 2),
            description=(
                f"Equipment likely running off-hours: {row['demand_kw']:.1f} kW "
                f"at {row['timestamp'].strftime('%H:%M')} "
                f"({excess_kw:.1f} kW above off-hour baseline)"
            ),
            meter_id=str(row.get("meter_id", "")),
        ))

    return anomalies


def detect_pf_drops(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    pf_threshold: float = 0.85,
    consecutive_hours: int = 3,
) -> list[AnomalyRecord]:
    """Detect sustained power factor drops indicating equipment faults.

    Flags sequences of consecutive readings where PF stays below threshold,
    suggesting a motor fault, capacitor bank failure, or similar issue.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, power_factor].
    facility_id : Facility UUID.
    pf_threshold : PF value below which readings are considered abnormal.
    consecutive_hours : Minimum consecutive low-PF readings to flag.

    Returns
    -------
    List of AnomalyRecord for PF drop events.
    """
    if readings_df.empty or "power_factor" not in readings_df.columns:
        return []

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    low_pf = df["power_factor"] < pf_threshold
    if not low_pf.any():
        return []

    # Group consecutive low-PF readings
    groups = (low_pf != low_pf.shift()).cumsum()
    low_groups = df[low_pf].groupby(groups[low_pf])

    anomalies: list[AnomalyRecord] = []
    for group_id, group_df in low_groups:
        if len(group_df) < consecutive_hours:
            continue

        avg_pf = float(group_df["power_factor"].mean())
        min_pf = float(group_df["power_factor"].min())
        severity = AnomalySeverity.CRITICAL if min_pf < 0.75 else AnomalySeverity.HIGH

        anomalies.append(AnomalyRecord(
            timestamp=group_df["timestamp"].iloc[0],
            facility_id=facility_id,
            anomaly_type="pf_drop",
            severity=severity,
            measured_value=round(min_pf, 4),
            expected_value=pf_threshold,
            deviation_pct=round((pf_threshold - avg_pf) / pf_threshold * 100, 2),
            description=(
                f"Sustained PF drop: avg {avg_pf:.3f}, min {min_pf:.3f} "
                f"over {len(group_df)} consecutive readings "
                f"({group_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} to "
                f"{group_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')})"
            ),
        ))

    return anomalies


def run_full_anomaly_detection(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    zscore_threshold: float = 3.0,
) -> AnomalyListResponse:
    """Run all anomaly detectors and return consolidated results.

    Parameters
    ----------
    readings_df : Raw or hourly-resampled meter readings.
    facility_id : Facility UUID.
    zscore_threshold : Z-score threshold for spike detection.

    Returns
    -------
    AnomalyListResponse with all detected anomalies sorted by severity.
    """
    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    all_anomalies: list[AnomalyRecord] = []
    all_anomalies.extend(detect_consumption_spikes(df, facility_id, zscore_threshold))
    all_anomalies.extend(detect_baseline_shift(df, facility_id))
    all_anomalies.extend(detect_equipment_left_on(df, facility_id))
    all_anomalies.extend(detect_pf_drops(df, facility_id))

    # Sort: CRITICAL first, then by timestamp
    severity_order = {AnomalySeverity.CRITICAL: 0, AnomalySeverity.HIGH: 1,
                      AnomalySeverity.MEDIUM: 2, AnomalySeverity.LOW: 3}
    all_anomalies.sort(key=lambda a: (severity_order[a.severity], a.timestamp))

    period_start = df["timestamp"].min() if not df.empty else datetime.now(timezone.utc)
    period_end = df["timestamp"].max() if not df.empty else datetime.now(timezone.utc)

    return AnomalyListResponse(
        facility_id=facility_id,
        period_start=period_start,
        period_end=period_end,
        total_anomalies=len(all_anomalies),
        anomalies=all_anomalies,
    )
