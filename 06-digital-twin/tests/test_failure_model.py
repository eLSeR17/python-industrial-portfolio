"""Tests for Weibull failure model and distribution utilities."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.simulation.failure_model import (
    RepairModel,
    WeibullFailureModel,
    calculate_mtbf,
    calculate_mttr,
    fit_weibull,
)
from src.utils.distributions import fit_weibull_mle, weibull_random, lognormal_from_mean_std


def test_weibull_sampling_mean() -> None:
    """Weibull mean = scale * Gamma(1 + 1/shape).
    shape=2, scale=100 → mean ≈ 100 * Gamma(1.5) ≈ 88.62."""
    shape, scale = 2.0, 100.0
    samples = np.array([weibull_random(shape, scale) for _ in range(5000)])
    expected_mean = scale * math.gamma(1.0 + 1.0 / shape)
    observed_mean = float(np.mean(samples))
    assert abs(observed_mean - expected_mean) / expected_mean < 0.1, (
        f"Mean {observed_mean:.1f} too far from expected {expected_mean:.1f}"
    )


def test_mtbf_mttr_calculation() -> None:
    """calculate_mtbf / calculate_mttr should return the arithmetic mean."""
    failure_intervals = [100.0, 120.0, 80.0, 110.0]
    assert abs(calculate_mtbf(failure_intervals) - 102.5) < 1e-9

    repair_times = [10.0, 20.0, 15.0]
    assert abs(calculate_mttr(repair_times) - 15.0) < 1e-9


def test_weibull_fit_mle() -> None:
    """fit_weibull_mle should recover shape and scale within tolerance."""
    true_shape, true_scale = 2.5, 200.0
    rng = np.random.default_rng(42)
    data = rng.weibull(true_shape, size=2000) * true_scale
    fitted_shape, fitted_scale = fit_weibull(data)

    assert abs(fitted_shape - true_shape) / true_shape < 0.15, (
        f"Fitted shape {fitted_shape:.2f} too far from {true_shape}"
    )
    assert abs(fitted_scale - true_scale) / true_scale < 0.15, (
        f"Fitted scale {fitted_scale:.2f} too far from {true_scale}"
    )


def test_from_mtbf_mttr_factory() -> None:
    """WeibullFailureModel.from_mtbf_mttr should set scale so mean = mtbf."""
    mtbf, mttr, shape = 500.0, 30.0, 2.0
    model = WeibullFailureModel.from_mtbf_mttr(mtbf, mttr, shape)
    expected_eta = mtbf / math.gamma(1.0 + 1.0 / shape)
    assert abs(model.scale_eta - expected_eta) < 1e-6
    assert model.shape_beta == shape


def test_repair_model_lognormal() -> None:
    """RepairModel with lognormal should return positive values near the mean."""
    model = RepairModel(mean=25.0, std=5.0, distribution="lognormal")
    samples = [model.sample() for _ in range(1000)]
    assert all(s > 0 for s in samples), "All repair times must be positive"
    avg = sum(samples) / len(samples)
    assert 15.0 < avg < 40.0, f"Average repair time {avg:.1f} not near 25"


def test_repair_model_exponential() -> None:
    """RepairModel with exponential should have mean ≈ configured mean."""
    model = RepairModel(mean=30.0, std=0.0, distribution="exponential")
    samples = [model.sample() for _ in range(2000)]
    avg = sum(samples) / len(samples)
    assert 25.0 < avg < 35.0, f"Exponential mean {avg:.1f} not near 30"


def test_should_fail_probability() -> None:
    """should_fail should return True roughly proportional to CDF at given time."""
    model = WeibullFailureModel(shape_beta=2.0, scale_eta=100.0)
    # At t=100 (eta), CDF = 1 - exp(-1) ≈ 0.632
    trials = 5000
    successes = sum(1 for _ in range(trials) if model.should_fail(100.0))
    ratio = successes / trials
    assert 0.55 < ratio < 0.72, f"Fail ratio {ratio:.3f} not near CDF 0.632"
