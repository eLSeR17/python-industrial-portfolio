"""Utility functions for time handling, data validation, and domain helpers.

Provides common operations used across the predictive maintenance pipeline:
- Timestamp normalization and windowing
- Physical plausibility checks for sensor data
- Degradation curve math for simulation
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def utc_now() -> datetime:
    """Return the current UTC time with timezone info.

    All timestamps in the system are UTC to avoid ambiguity
    across distributed components and time-zone-aware clients.
    """
    return datetime.now(timezone.utc)


def compute_rms(signal: NDArray[np.floating]) -> float:
    """Compute root-mean-square of a signal.

    RMS is the standard metric for vibration severity per ISO 10816.
    A healthy motor typically shows 0.5–2.8 mm/s RMS; values above
    4.5 mm/s indicate deteriorating condition.

    Args:
        signal: 1-D array of instantaneous signal values.

    Returns:
        RMS value (same units as input).
    """
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def compute_peak(signal: NDArray[np.floating]) -> float:
    """Compute peak (max absolute) amplitude of a signal.

    Peak amplitude captures transient shock events (e.g., bearing
    impacts) that RMS may average out.
    """
    if signal.size == 0:
        return 0.0
    return float(np.max(np.abs(signal.astype(np.float64))))


def compute_crest_factor(signal: NDArray[np.floating]) -> float:
    """Compute crest factor: peak / RMS.

    Normal machinery has CF ≈ 3. Rising CF indicates impulsive
    faults (e.g., cracked bearing race) even when RMS is still
    within normal limits.
    """
    rms = compute_rms(signal)
    if rms < 1e-12:
        return 0.0
    return compute_peak(signal) / rms


def zero_crossing_rate(signal: NDArray[np.floating]) -> float:
    """Compute the zero-crossing rate of a signal.

    ZCR measures how frequently the signal crosses zero. In vibration
    analysis, increasing ZCR with stable RPM can indicate looseness
    or misalignment.

    Args:
        signal: 1-D array of signal values.

    Returns:
        Fraction of adjacent samples with opposite signs.
    """
    if signal.size < 2:
        return 0.0
    signs = np.sign(signal.astype(np.float64))
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (signal.size - 1))


def degradation_curve(
    time_hours: float,
    start_hour: float = 72.0,
    failure_hour: float = 96.0,
    base_value: float = 1.0,
) -> float:
    """Model exponential degradation of an asset over time.

    Uses an exponential growth model for vibration amplitude:
        v(t) = base * exp(k * (t - start) / (failure - start))

    This approximates real bearing/pump degradation curves observed
    in condition monitoring literature (ISO 13373).

    Args:
        time_hours: Current operating hour.
        start_hour: Hour at which degradation begins.
        failure_hour: Hour at which failure is expected.
        base_value: Baseline value before degradation.

    Returns:
        Degraded value (multiplier) at the given time.
    """
    if time_hours <= start_hour:
        return base_value
    if time_hours >= failure_hour:
        return base_value * math.exp(4.0)
    progress = (time_hours - start_hour) / (failure_hour - start_hour)
    return base_value * math.exp(4.0 * progress)


def validate_sensor_ranges(
    vibration_x: float,
    vibration_y: float,
    vibration_z: float,
    temperature: float,
    pressure: float,
) -> list[str]:
    """Check sensor readings against physically plausible ranges.

    Catches obvious sensor malfunctions (stuck values, electrical
    noise spikes) before they pollute the feature store.

    Returns:
        List of warning messages. Empty if all readings are plausible.
    """
    warnings: list[str] = []
    for name, value in [
        ("vibration_x", vibration_x),
        ("vibration_y", vibration_y),
        ("vibration_z", vibration_z),
    ]:
        if abs(value) > 20.0:
            warnings.append(f"{name}={value:.1f} mm/s exceeds typical range ±20")
    if temperature > 150:
        warnings.append(f"temperature={temperature:.1f}°C exceeds equipment limit")
    if temperature < -20:
        warnings.append(f"temperature={temperature:.1f}°C below ambient range")
    if pressure < 0:
        warnings.append(f"pressure={pressure:.2f} bar is negative (sensor fault?)")
    return warnings


def exponential_moving_average(
    values: NDArray[np.floating], span: int = 10
) -> NDArray[np.float64]:
    """Compute EMA for smoothing noisy sensor signals.

    EMA gives more weight to recent observations, making it suitable
    for real-time trend detection in sensor streams.

    Args:
        values: Input signal array.
        span: Number of periods over which the EMA decays.

    Returns:
        EMA-smoothed array of same length.
    """
    if values.size == 0:
        return np.array([], dtype=np.float64)
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = float(values[0])
    for i in range(1, values.size):
        result[i] = alpha * float(values[i]) + (1 - alpha) * result[i - 1]
    return result


def percentile_rank(values: NDArray[np.floating], value: float) -> float:
    """Compute the percentile rank of a value within a distribution.

    Used for scoring how unusual a current reading is relative to
    historical baseline data for the same asset.
    """
    if values.size == 0:
        return 0.5
    return float(np.sum(values <= value) / values.size)


def format_duration_hours(hours: float) -> str:
    """Format a duration in hours to a human-readable string.

    Examples:
        0.5 → "30 minutes"
        2.3 → "2 hours 18 minutes"
        48.0 → "2 days"
    """
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    days = int(hours // 24)
    remaining = hours - days * 24
    hrs = int(remaining)
    mins = int((remaining - hrs) * 60)
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hrs > 0:
        parts.append(f"{hrs} hour{'s' if hrs != 1 else ''}")
    if mins > 0 and days == 0:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    return " ".join(parts) if parts else "< 1 minute"
