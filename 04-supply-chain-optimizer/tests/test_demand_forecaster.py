"""Tests for classical time-series demand forecasting methods."""

import math

import numpy as np
import pytest

from src.services.demand_forecaster import DemandForecaster


@pytest.fixture
def forecaster():
    return DemandForecaster()


@pytest.fixture
def stable_data():
    """Stationary demand around 100."""
    rng = np.random.default_rng(42)
    return [float(100 + rng.normal(0, 5)) for _ in range(30)]


@pytest.fixture
def trending_data():
    """Demand with linear upward trend."""
    return [float(50 + 3 * i + np.random.default_rng(7).normal(0, 2)) for i in range(30)]


@pytest.fixture
def seasonal_data():
    """Two full seasonal cycles (period=12)."""
    rng = np.random.default_rng(99)
    return [
        float(100 + 20 * math.sin(2 * math.pi * i / 12) + rng.normal(0, 3))
        for i in range(24)
    ]


# -----------------------------------------------------------------------
# moving_average
# -----------------------------------------------------------------------

class TestMovingAverage:

    def test_forecast_length(self, forecaster, stable_data):
        result = forecaster.moving_average(stable_data, window=5, periods_ahead=4)
        assert len(result["forecast"]) == 4

    def test_forecast_near_mean(self, forecaster, stable_data):
        result = forecaster.moving_average(stable_data, window=10, periods_ahead=1)
        assert 80 < result["forecast"][0] < 120

    def test_too_short_raises(self, forecaster, stable_data):
        with pytest.raises(ValueError, match="Need at least"):
            forecaster.moving_average(stable_data, window=50)

    def test_in_sample_mae_non_negative(self, forecaster, stable_data):
        result = forecaster.moving_average(stable_data, window=5)
        assert result["in_sample_mae"] >= 0

    def test_prediction_interval_present(self, forecaster, stable_data):
        result = forecaster.moving_average(stable_data, window=5, periods_ahead=3)
        assert len(result["prediction_interval_95"]) == 3
        for lo, hi in result["prediction_interval_95"]:
            assert lo <= hi

    def test_ma_values_length(self, forecaster, stable_data):
        result = forecaster.moving_average(stable_data, window=5)
        expected_ma_len = len(stable_data) - 5 + 1
        assert len(result["ma_values"]) == expected_ma_len


# -----------------------------------------------------------------------
# weighted_moving_average
# -----------------------------------------------------------------------

class TestWeightedMovingAverage:

    def test_forecast_length(self, forecaster, stable_data):
        result = forecaster.weighted_moving_average(stable_data, periods_ahead=3)
        assert len(result["forecast"]) == 3

    def test_custom_weights(self, forecaster, stable_data):
        n = len(stable_data)
        weights = [1.0] * n
        result = forecaster.weighted_moving_average(stable_data, weights=weights)
        # Equal weights => same as simple average (forecast is rounded to 2 dp)
        assert result["forecast"][0] == pytest.approx(sum(stable_data) / n, abs=0.01)

    def test_weight_length_mismatch_raises(self, forecaster, stable_data):
        with pytest.raises(ValueError, match="Weights length"):
            forecaster.weighted_moving_average(stable_data, weights=[1, 2])

    def test_zero_weights_raises(self, forecaster):
        with pytest.raises(ValueError, match="Weights must sum"):
            forecaster.weighted_moving_average([10, 20, 30], weights=[0, 0, 0])


# -----------------------------------------------------------------------
# exponential_smoothing
# -----------------------------------------------------------------------

class TestExponentialSmoothing:

    def test_forecast_length(self, forecaster, stable_data):
        result = forecaster.exponential_smoothing(stable_data, alpha=0.3, periods_ahead=5)
        assert len(result["forecast"]) == 5

    def test_forecast_near_last_obs(self, forecaster, stable_data):
        result = forecaster.exponential_smoothing(stable_data, alpha=0.9)
        assert result["forecast"][0] == pytest.approx(stable_data[-1], abs=10)

    def test_invalid_alpha_raises(self, forecaster, stable_data):
        with pytest.raises(ValueError, match="alpha must be"):
            forecaster.exponential_smoothing(stable_data, alpha=1.5)

    def test_two_points_minimum(self, forecaster):
        with pytest.raises(ValueError, match="Need at least 2"):
            forecaster.exponential_smoothing([100])

    def test_smoothed_values_length(self, forecaster, stable_data):
        result = forecaster.exponential_smoothing(stable_data)
        assert len(result["smoothed_values"]) == len(stable_data)

    def test_high_alpha_responsive(self, forecaster):
        data = [100] * 10 + [200] * 5
        low = forecaster.exponential_smoothing(data, alpha=0.1)
        high = forecaster.exponential_smoothing(data, alpha=0.9)
        # High alpha should be closer to the last observation (200)
        assert high["forecast"][0] > low["forecast"][0]


# -----------------------------------------------------------------------
# holt_smoothing
# -----------------------------------------------------------------------

class TestHoltSmoothing:

    def test_forecast_length(self, forecaster, trending_data):
        result = forecaster.holt_smoothing(trending_data, periods_ahead=6)
        assert len(result["forecast"]) == 6

    def test_trend_direction_increasing(self, forecaster, trending_data):
        result = forecaster.holt_smoothing(trending_data)
        assert result["trend_direction"] == "increasing"
        assert result["final_trend"] > 0

    def test_forecast_extends_trend(self, forecaster, trending_data):
        result = forecaster.holt_smoothing(trending_data, periods_ahead=3)
        # Forecasts should be increasing for upward trend
        for i in range(1, len(result["forecast"])):
            assert result["forecast"][i] >= result["forecast"][i - 1]

    def test_too_few_points_raises(self, forecaster):
        with pytest.raises(ValueError, match="Need at least 3"):
            forecaster.holt_smoothing([100, 200])

    def test_invalid_params_raises(self, forecaster, trending_data):
        with pytest.raises(ValueError):
            forecaster.holt_smoothing(trending_data, alpha=0.0, beta=0.0)


# -----------------------------------------------------------------------
# seasonal_decompose
# -----------------------------------------------------------------------

class TestSeasonalDecompose:

    def test_forecast_length(self, forecaster, seasonal_data):
        result = forecaster.seasonal_decompose(
            seasonal_data, period=12, periods_ahead=6
        )
        assert len(result["forecast"]) == 6

    def test_seasonal_indices_length(self, forecaster, seasonal_data):
        result = forecaster.seasonal_decompose(seasonal_data, period=12)
        assert len(result["seasonal_indices"]) == 12

    def test_seasonal_indices_sum_near_zero(self, forecaster, seasonal_data):
        result = forecaster.seasonal_decompose(seasonal_data, period=12)
        assert sum(result["seasonal_indices"]) == pytest.approx(0.0, abs=0.01)

    def test_too_short_raises(self, forecaster):
        with pytest.raises(ValueError, match="Need at least"):
            forecaster.seasonal_decompose([1, 2, 3], period=12)

    def test_trend_values_length(self, forecaster, seasonal_data):
        result = forecaster.seasonal_decompose(seasonal_data, period=12)
        assert len(result["trend_values"]) == len(seasonal_data)


# -----------------------------------------------------------------------
# auto_forecast
# -----------------------------------------------------------------------

class TestAutoForecast:

    def test_selects_method(self, forecaster, stable_data):
        result = forecaster.auto_forecast(stable_data, periods_ahead=3)
        assert result["selected_method"] in ("SES", "Holt", "Seasonal")
        assert len(result["forecast"]) == 3

    def test_all_methods_reported(self, forecaster, stable_data):
        result = forecaster.auto_forecast(stable_data)
        assert "all_methods" in result
        assert len(result["all_methods"]) >= 2  # SES + Holt minimum

    def test_best_mae_lowest(self, forecaster, stable_data):
        result = forecaster.auto_forecast(stable_data)
        all_maes = list(result["all_methods"].values())
        assert result["in_sample_mae"] == pytest.approx(min(all_maes))

    def test_seasonal_with_enough_data(self, forecaster, seasonal_data):
        result = forecaster.auto_forecast(
            seasonal_data, periods_ahead=6, seasonality_period=12
        )
        assert len(result["forecast"]) == 6
