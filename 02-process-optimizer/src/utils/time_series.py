"""Time-series utilities: sliding windows, change detection, and accumulation.

These building blocks are consumed by the SPC engine and the stream processor.
All functions are pure (no side effects) and operate on NumPy arrays.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------

class SlidingWindow:
    """A fixed-size FIFO buffer backed by a pre-allocated NumPy array.

    Using a pre-allocated array avoids the cost of repeated
    ``np.append`` / ``np.concatenate`` calls that create new arrays.

    Example::

        win = SlidingWindow(size=100)
        for reading in sensor_stream:
            win.push(reading.value)
            if win.is_full:
                compute_stats(win.data)
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self._size = size
        self._buffer = np.zeros(size, dtype=np.float64)
        self._count = 0

    def push(self, value: float) -> None:
        """Append a value, overwriting the oldest entry when full."""
        idx = self._count % self._size
        self._buffer[idx] = value
        self._count += 1

    @property
    def data(self) -> NDArray[np.float64]:
        """Return the window contents in chronological order.

        If fewer values than *size* have been pushed, the leading entries
        are zeros and should be discarded via ``self.values``.
        """
        if self._count < self._size:
            return self._buffer[: self._count].copy()
        # Rotate so index 0 is the oldest value.
        offset = self._count % self._size
        return np.roll(self._buffer, -offset)

    @property
    def values(self) -> NDArray[np.float64]:
        """Same as ``data`` — alias for readability."""
        return self.data

    @property
    def is_full(self) -> bool:
        """True once *size* values have been pushed."""
        return self._count >= self._size

    @property
    def count(self) -> int:
        """Total number of values pushed (may exceed *size*)."""
        return self._count

    @property
    def size(self) -> int:
        """Maximum capacity of the window."""
        return self._size

    def reset(self) -> None:
        """Clear all data and start fresh."""
        self._buffer[:] = 0.0
        self._count = 0

    def mean(self) -> float:
        """Return the mean of the current window contents."""
        d = self.data
        return float(np.mean(d)) if len(d) > 0 else 0.0

    def std(self) -> float:
        """Return the standard deviation of the current window contents."""
        d = self.data
        return float(np.std(d, ddof=1)) if len(d) > 1 else 0.0

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# CUSUM (Cumulative Sum) Change Detection
# ---------------------------------------------------------------------------

@dataclass
class CUSUMDetector:
    """Two-sided CUSUM algorithm for detecting small persistent shifts in a
    process mean.

    The CUSUM accumulates deviations from a target mean. When the
    accumulation exceeds a decision threshold, a change point is flagged.

    Reference: Hawkins & Olwell, *Cumulative Sum Charts and Charting for
    Quality Improvement*, Springer, 1998.

    Attributes:
        target: Target (in-control) process mean.
        threshold: Decision threshold *h* — higher = fewer false alarms.
        drift: Allowable slack *k* — typically 0.5 × σ of the process.
    """

    target: float = 0.0
    threshold: float = 5.0
    drift: float = 0.5

    # Internal accumulators (reset on alarm or manual reset).
    _s_pos: float = field(default=0.0, init=False, repr=False)
    _s_neg: float = field(default=0.0, init=False, repr=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def update(self, value: float) -> dict[str, Any]:
        """Feed a new observation and check for a change.

        Args:
            value: Latest process measurement.

        Returns:
            Dict with keys:
                - ``s_pos``: positive-side CUSUM statistic.
                - ``s_neg``: negative-side CUSUM statistic.
                - ``alarm``: True if either side exceeded the threshold.
                - ``direction``: "up" if positive shift detected, "down" if
                  negative, None otherwise.
        """
        deviation = value - self.target
        self._s_pos = max(0.0, self._s_pos + deviation - self.drift)
        self._s_neg = max(0.0, self._s_neg - deviation - self.drift)

        alarm = False
        direction: str | None = None

        if self._s_pos > self.threshold:
            alarm = True
            direction = "up"
        elif self._s_neg > self.threshold:
            alarm = True
            direction = "down"

        result = {
            "s_pos": self._s_pos,
            "s_neg": self._s_neg,
            "alarm": alarm,
            "direction": direction,
        }
        self._history.append(result)
        return result

    def reset(self) -> None:
        """Reset accumulators — called after acknowledging an alarm."""
        self._s_pos = 0.0
        self._s_neg = 0.0

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the full history of CUSUM statistics."""
        return list(self._history)

    @property
    def state(self) -> dict[str, float]:
        """Current accumulator values (for serialization / dashboard)."""
        return {"s_pos": self._s_pos, "s_neg": self._s_neg}


# ---------------------------------------------------------------------------
# Exponential Weighted Moving Average (EWMA) Controller
# ---------------------------------------------------------------------------

class EWMAControl:
    """EWMA chart for detecting small sustained shifts.

    Unlike Shewhart charts that only react to points outside control limits,
    the EWMA incorporates all past data with exponentially decaying weights.
    """

    def __init__(
        self,
        target: float = 0.0,
        sigma: float = 1.0,
        alpha: float = 0.2,
        n_sigma: float = 3.0,
    ) -> None:
        """
        Args:
            target: In-control process mean.
            sigma: Process standard deviation.
            alpha: Smoothing constant (0 < α ≤ 1). Smaller = more sensitive.
            n_sigma: Number of standard deviations for the control limit.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.target = target
        self.sigma = sigma
        self.alpha = alpha
        self.n_sigma = n_sigma
        self._ewma_stat: float = target
        self._in_control_limit = n_sigma * sigma * np.sqrt(alpha / (2.0 - alpha))

    def update(self, value: float) -> dict[str, Any]:
        """Process a new observation and return the EWMA state.

        Args:
            value: Latest measurement.

        Returns:
            Dict with ``ewma``, ``ucl``, ``lcl``, ``alarm``.
        """
        self._ewma_stat = self.alpha * value + (1.0 - self.alpha) * self._ewma_stat
        ucl = self.target + self._in_control_limit
        lcl = self.target - self._in_control_limit
        alarm = self._ewma_stat > ucl or self._ewma_stat < lcl
        return {
            "ewma": self._ewma_stat,
            "ucl": ucl,
            "lcl": lcl,
            "alarm": alarm,
        }

    @property
    def statistic(self) -> float:
        return self._ewma_stat

    def reset(self) -> None:
        self._ewma_stat = self.target


# ---------------------------------------------------------------------------
# Rate of Change Limiter
# ---------------------------------------------------------------------------

def rate_limit(
    new_value: float,
    current_value: float,
    max_rate: float,
    dt: float = 1.0,
) -> float:
    """Clamp the rate of change of a setpoint.

    Prevents the optimizer from commanding a jump that the physical
    actuator cannot follow (e.g., a valve that can only move 5% per
    second).

    Args:
        new_value: Desired new value from the optimizer.
        current_value: Current actuator position.
        max_rate: Maximum allowed change per unit time.
        dt: Time since the last update (default 1.0 → rate is per-unit-time).

    Returns:
        The allowable value, clamped to the rate limit.
    """
    max_delta = max_rate * dt
    delta = new_value - current_value
    if abs(delta) <= max_delta:
        return new_value
    return current_value + np.sign(delta) * max_delta


# ---------------------------------------------------------------------------
# Linear Regression (for trend detection)
# ---------------------------------------------------------------------------

def linear_trend(
    y: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit a linear trend to a 1-D signal via least-squares.

    Args:
        y: 1-D data array.

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    if y.ndim != 1 or len(y) < 2:
        return 0.0, float(y[0]) if len(y) > 0 else 0.0, 0.0

    x = np.arange(len(y), dtype=np.float64)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, intercept, r_squared
