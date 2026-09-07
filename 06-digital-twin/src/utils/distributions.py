"""Utility functions for statistical distributions used in simulation."""

from typing import Union

import numpy as np
from scipy import stats


def weibull_random(
    shape: float, scale: float, size: int = 1
) -> Union[float, np.ndarray]:
    """Draw samples from a Weibull distribution.

    Args:
        shape: Shape parameter (beta / k) – must be > 0.
        scale: Scale parameter (eta / lambda) – must be > 0.
        size: Number of samples to draw.

    Returns:
        Scalar when *size* == 1, otherwise a numpy array.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    samples = np.random.weibull(shape, size=size) * scale
    return float(samples[0]) if size == 1 else samples


def lognormal_random(
    mean: float, std: float, size: int = 1
) -> Union[float, np.ndarray]:
    """Draw samples from a lognormal distribution parameterised by the
    *underlying* normal's mean and std.

    Args:
        mean: Mean of the underlying normal.
        std: Standard deviation of the underlying normal (> 0).
        size: Number of samples.

    Returns:
        Scalar when *size* == 1, otherwise a numpy array.
    """
    if std < 0:
        raise ValueError(f"std must be >= 0, got {std}")
    if std == 0:
        result = np.exp(np.full(size, mean))
        return float(result[0]) if size == 1 else result
    result = np.random.lognormal(mean=mean, sigma=std, size=size)
    return float(result[0]) if size == 1 else result


def lognormal_from_mean_std(
    desired_mean: float, desired_std: float, size: int = 1
) -> Union[float, np.ndarray]:
    """Draw samples from a lognormal whose *real-world* mean and std match
    the requested values.

    The underlying normal parameters are:
        mu    = ln(mean^2 / sqrt(var + mean^2))
        sigma = sqrt(ln(1 + var / mean^2))

    Args:
        desired_mean: Desired arithmetic mean of the lognormal (> 0).
        desired_std: Desired standard deviation (>= 0).
        size: Number of samples.
    """
    if desired_mean <= 0:
        raise ValueError(f"desired_mean must be > 0, got {desired_mean}")
    if desired_std < 0:
        raise ValueError(f"desired_std must be >= 0, got {desired_std}")
    if desired_std == 0:
        result = np.exp(np.full(size, np.log(desired_mean)))
        return float(result[0]) if size == 1 else result

    var = desired_std ** 2
    mu = np.log(desired_mean ** 2 / np.sqrt(var + desired_mean ** 2))
    sigma = np.sqrt(np.log(1 + var / desired_mean ** 2))
    result = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    return float(result[0]) if size == 1 else result


def exponential_random(rate: float, size: int = 1) -> Union[float, np.ndarray]:
    """Draw samples from an exponential distribution (mean = 1/rate).

    Args:
        rate: Rate parameter (> 0).
        size: Number of samples.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    result = np.random.exponential(scale=1.0 / rate, size=size)
    return float(result[0]) if size == 1 else result


def triangular_random(
    left: float, mode: float, right: float, size: int = 1
) -> Union[float, np.ndarray]:
    """Draw samples from a triangular distribution.

    Args:
        left: Lower limit.
        mode: Peak location (must satisfy left <= mode <= right).
        right: Upper limit.
        size: Number of samples.
    """
    if not (left <= mode <= right):
        raise ValueError(
            f"Must satisfy left <= mode <= right, got {left} <= {mode} <= {right}"
        )
    if left == right:
        result = np.full(size, left)
        return float(result[0]) if size == 1 else result
    result = np.random.triangular(left=left, mode=mode, right=right, size=size)
    return float(result[0]) if size == 1 else result


def fit_weibull_mle(data: np.ndarray) -> tuple[float, float]:
    """Fit a Weibull distribution to data using maximum likelihood estimation.

    Args:
        data: 1-D array of positive lifetime observations.

    Returns:
        (shape, scale) tuple.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("data must be a 1-D array with at least 2 elements.")
    if np.any(arr <= 0):
        raise ValueError("All data points must be positive for Weibull MLE.")
    shape, loc, scale = stats.weibull_min.fit(arr, floc=0)
    return float(shape), float(scale)


def fit_lognormal(data: np.ndarray) -> tuple[float, float]:
    """Fit a lognormal distribution to data using MLE.

    Returns:
        (mu, sigma) of the underlying normal.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("data must be a 1-D array with at least 2 elements.")
    if np.any(arr <= 0):
        raise ValueError("All data points must be positive for lognormal MLE.")
    mu, sigma = stats.lognorm.fit(arr, floc=0)[:2]
    # scipy returns shape (sigma) and loc; we convert to underlying-normal params
    ln_data = np.log(arr)
    return float(np.mean(ln_data)), float(np.std(ln_data, ddof=1))
