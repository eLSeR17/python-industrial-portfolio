"""NumPy-based signal processing and numerical utilities.

These functions are used throughout the system for filtering noisy sensor
data, computing statistical measures, and preparing arrays for optimization.
All functions operate on NumPy arrays and return NumPy arrays — no Python
loops over data.
"""

import numpy as np
from numpy.typing import NDArray


def exponential_moving_average(
    data: NDArray[np.float64],
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    """Compute the exponential moving average (EMA) of a 1-D signal.

    The EMA gives more weight to recent observations, making it responsive
    to real process changes while smoothing high-frequency sensor noise.

    Formula:
        EMA[0] = data[0]
        EMA[i] = alpha * data[i] + (1 - alpha) * EMA[i-1]

    Args:
        data: Input signal array.
        alpha: Smoothing factor in (0, 1]. Lower = smoother, higher = more responsive.

    Returns:
        EMA array of the same shape as *data*.

    Raises:
        ValueError: If alpha is not in (0, 1].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")

    result = np.empty_like(data, dtype=np.float64)
    result[0] = data[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + one_minus_alpha * result[i - 1]
    return result


def savitzky_golay_filter(
    data: NDArray[np.float64],
    window_length: int = 11,
    polyorder: int = 3,
) -> NDArray[np.float64]:
    """Apply a Savitzky-Golay polynomial smoothing filter.

    Preserves signal shape (peaks, valleys) better than a simple moving
    average. Requires ``scipy``.

    Args:
        data: 1-D input signal.
        window_length: Filter window length (must be odd, >= polyorder + 2).
        polyorder: Polynomial order for the local fit.

    Returns:
        Smoothed signal of the same length as *data*.
    """
    from scipy.signal import savgol_filter

    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 2:
        window_length = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4

    # Extend edges to avoid boundary artifacts.
    pad = window_length // 2
    padded = np.pad(data, pad, mode="edge")
    smoothed = savgol_filter(padded, window_length, polyorder)
    return smoothed[pad : pad + len(data)]


def moving_average(
    data: NDArray[np.float64],
    window: int = 5,
) -> NDArray[np.float64]:
    """Compute a centered simple moving average.

    The first and last ``window // 2`` elements use a truncated window so
    the output length equals the input length.

    Args:
        data: 1-D input signal.
        window: Window size (must be >= 1).

    Returns:
        Smoothed signal, same length as *data*.
    """
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    result = (cumsum[window:] - cumsum[:-window]) / window
    # Pad edges to maintain original length.
    half = window // 2
    left = np.full(half, result[0])
    right = np.full(half, result[-1])
    return np.concatenate([left, result, right])[: len(data)]


def compute_residuals(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Element-wise residual (actual − predicted).

    Args:
        actual: Observed values.
        predicted: Model-predicted values.

    Returns:
        1-D array of residuals, same shape as inputs.

    Raises:
        ValueError: If shapes differ.
    """
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    return actual - predicted


def rmse(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> np.float64:
    """Root mean square error between actual and predicted signals.

    Args:
        actual: Observed values.
        predicted: Model-predicted values.

    Returns:
        Scalar RMSE value.
    """
    residuals = compute_residuals(actual, predicted)
    return np.sqrt(np.mean(residuals**2))


def clip_to_limits(
    values: NDArray[np.float64],
    lower: NDArray[np.float64] | float,
    upper: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Clip each element of *values* to its corresponding [lower, upper] range.

    Args:
        values: Array to clip.
        lower: Lower bound(s) — scalar or array broadcastable to *values*.
        upper: Upper bound(s) — scalar or array broadcastable to *values*.

    Returns:
        Clipped array (new allocation, input is not modified).
    """
    return np.clip(values, lower, upper)


def normalize(
    data: NDArray[np.float64],
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> tuple[NDArray[np.float64], float, float]:
    """Min-max normalize data to [min_val, max_val].

    Args:
        data: Input array.
        min_val: Target minimum after normalization.
        max_val: Target maximum after normalization.

    Returns:
        Tuple of (normalized_data, original_min, original_max).
        The original bounds are needed to denormalize later.
    """
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = data_max - data_min
    if span == 0:
        return np.full_like(data, (min_val + max_val) / 2.0), data_min, data_max
    normalized = (data - data_min) / span * (max_val - min_val) + min_val
    return normalized, data_min, data_max


def denormalize(
    data: NDArray[np.float64],
    original_min: float,
    original_max: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> NDArray[np.float64]:
    """Reverse a min-max normalization.

    Args:
        data: Normalized data.
        original_min: Minimum from the original data (returned by ``normalize``).
        original_max: Maximum from the original data.
        min_val: The target min used during normalization.
        max_val: The target max used during normalization.

    Returns:
        Data in the original scale.
    """
    span = max_val - min_val
    if span == 0:
        return np.full_like(data, original_min)
    return (data - min_val) / span * (original_max - original_min) + original_min


def bandpass_filter(
    data: NDArray[np.float64],
    low_freq: float,
    high_freq: float,
    sample_rate: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Apply a Butterworth bandpass filter to isolate a frequency band.

    Useful for extracting periodic disturbances (e.g., pump vibration)
    from composite sensor signals.

    Args:
        data: 1-D input signal.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        sample_rate: Sampling rate of the signal in Hz.
        order: Filter order (higher = sharper cutoff, more phase delay).

    Returns:
        Filtered signal, same length as input.
    """
    from scipy.signal import butter, filtfilt

    nyquist = sample_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data).astype(np.float64)


def detrend(
    data: NDArray[np.float64],
    degree: int = 1,
) -> NDArray[np.float64]:
    """Remove a polynomial trend from a signal.

    Args:
        data: 1-D input signal.
        degree: Polynomial degree of the trend to remove (1 = linear).

    Returns:
        Detrended signal.
    """
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    x = np.arange(len(data), dtype=np.float64)
    coeffs = np.polyfit(x, data, degree)
    trend = np.polyval(coeffs, x)
    return data - trend


def signal_to_noise_ratio(
    signal: NDArray[np.float64],
) -> float:
    """Estimate the SNR of a signal in dB.

    The signal power is estimated as the variance of the full signal,
    and the noise power as the variance of the detrended residual.

    Args:
        signal: 1-D input signal.

    Returns:
        SNR in decibels (dB).
    """
    if signal.ndim != 1 or len(signal) < 3:
        return 0.0
    detrended = detrend(signal, degree=1)
    signal_power = float(np.var(signal))
    noise_power = float(np.var(detrended))
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
