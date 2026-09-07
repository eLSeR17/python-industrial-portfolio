"""Weibull-based failure and repair models for machines."""

import math
from typing import Optional

import numpy as np

from src.utils.distributions import fit_weibull_mle, weibull_random, lognormal_from_mean_std


class WeibullFailureModel:
    """Model time-to-failure using a Weibull distribution.

    Args:
        shape_beta: Shape parameter (beta).  beta < 1 → infant mortality,
                    beta == 1 → exponential (memoryless), beta > 1 → wear-out.
        scale_eta: Scale parameter (eta / characteristic life).
        min_life: Minimum time before a failure is possible (burn-in).
    """

    def __init__(self, shape_beta: float, scale_eta: float, min_life: float = 0.0) -> None:
        if shape_beta <= 0:
            raise ValueError(f"shape_beta must be > 0, got {shape_beta}")
        if scale_eta <= 0:
            raise ValueError(f"scale_eta must be > 0, got {scale_eta}")
        if min_life < 0:
            raise ValueError(f"min_life must be >= 0, got {min_life}")
        self.shape_beta = shape_beta
        self.scale_eta = scale_eta
        self.min_life = min_life

    def sample(self) -> float:
        """Draw a random time-to-failure from the Weibull distribution."""
        ttf = weibull_random(self.shape_beta, self.scale_eta)
        return max(ttf, self.min_life)

    def should_fail(self, elapsed_time: float) -> bool:
        """Return ``True`` if a failure should occur at *elapsed_time*.

        Uses the Weibull CDF: P(T <= t) = 1 - exp(-(t / eta)^beta).
        """
        t = elapsed_time - self.min_life
        if t <= 0:
            return False
        cdf = 1.0 - math.exp(-((t / self.scale_eta) ** self.shape_beta))
        return np.random.random() < cdf

    @classmethod
    def from_mtbf_mttr(
        cls, mtbf: float, mttr: float, shape: float = 2.0
    ) -> WeibullFailureModel:
        """Create a failure model from Mean Time Between Failures and MTTR.

        The scale (eta) is derived so that the Weibull mean equals *mtbf*.
        Weibull mean = eta * Gamma(1 + 1/beta).

        Args:
            mtbf: Mean time between failures (> 0).
            mttr: Mean time to repair (> 0, used only for the repair model).
            shape: Weibull shape parameter (beta).
        """
        if mtbf <= 0:
            raise ValueError(f"mtbf must be > 0, got {mtbf}")
        if mttr <= 0:
            raise ValueError(f"mttr must be > 0, got {mttr}")
        gamma_term = math.gamma(1.0 + 1.0 / shape)
        eta = mtbf / gamma_term
        return cls(shape_beta=shape, scale_eta=eta)


class RepairModel:
    """Model repair time using a configurable distribution.

    Args:
        mean: Desired mean repair time (> 0).
        std: Desired standard deviation of repair time (>= 0).
        distribution: ``"lognormal"`` (default) or ``"exponential"``.
    """

    def __init__(
        self, mean: float, std: float, distribution: str = "lognormal"
    ) -> None:
        if mean <= 0:
            raise ValueError(f"mean must be > 0, got {mean}")
        if std < 0:
            raise ValueError(f"std must be >= 0, got {std}")
        if distribution not in ("lognormal", "exponential"):
            raise ValueError(
                f"distribution must be 'lognormal' or 'exponential', got '{distribution}'"
            )
        self.mean = mean
        self.std = std
        self.distribution = distribution

    def sample(self) -> float:
        """Draw a random repair time."""
        if self.distribution == "exponential":
            return float(np.random.exponential(self.mean))
        return max(0.0, float(lognormal_from_mean_std(self.mean, self.std)))


# ---------------------------------------------------------------------------
# Stand-alone helper functions
# ---------------------------------------------------------------------------


def calculate_mtbf(failure_times: list[float]) -> float:
    """Calculate Mean Time Between Failures from a list of inter-failure durations.

    Args:
        failure_times: Sequence of durations between consecutive failures.

    Returns:
        Arithmetic mean of the intervals.
    """
    if not failure_times:
        raise ValueError("failure_times must not be empty.")
    return float(np.mean(failure_times))


def calculate_mttr(repair_times: list[float]) -> float:
    """Calculate Mean Time To Repair from a list of repair durations.

    Args:
        repair_times: Sequence of observed repair durations.

    Returns:
        Arithmetic mean of the repair times.
    """
    if not repair_times:
        raise ValueError("repair_times must not be empty.")
    return float(np.mean(repair_times))


def fit_weibull(data: list[float]) -> tuple[float, float]:
    """Fit a Weibull distribution to observed lifetime data (MLE).

    Args:
        data: List of positive lifetime observations.

    Returns:
        (shape, scale) tuple.
    """
    arr = np.asarray(data, dtype=float)
    return fit_weibull_mle(arr)
