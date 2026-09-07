"""Classical time-series demand forecasting methods."""

import math
from typing import Any

import numpy as np


class DemandForecaster:
    """Time series demand forecasting using classical statistical methods.

    Implements multiple forecasting techniques that can be selected based
    on data characteristics:

    - Simple Moving Average (SMA): smoothing for stable demand
    - Weighted Moving Average (WMA): recent periods weighted more heavily
    - Single Exponential Smoothing (SES): for data without trend/seasonality
    - Double Exponential Smoothing (Holt): for data with trend
    - Seasonal Decomposition: additive decomposition for seasonal patterns

    All methods return point forecasts plus prediction intervals estimated
    from in-sample residuals.
    """

    # ------------------------------------------------------------------
    # Simple Moving Average
    # ------------------------------------------------------------------

    def moving_average(
        self,
        historical: list[float],
        window: int = 3,
        periods_ahead: int = 1,
    ) -> dict[str, Any]:
        """Compute a simple moving average forecast.

        Args:
            historical: Chronologically ordered historical demand values.
            window: Number of periods in the moving average window.
            periods_ahead: How many periods to forecast.

        Returns:
            Forecast values, MA values, and in-sample error metrics.
        """
        if len(historical) < window:
            raise ValueError(
                f"Need at least {window} data points, got {len(historical)}"
            )

        n = len(historical)
        ma = []
        residuals = []

        # Compute MA values
        for i in range(window, n + 1):
            avg = sum(historical[i - window : i]) / window
            ma.append(avg)
            if i < n:
                residuals.append(historical[i] - avg)

        # Forecast: last MA value repeated
        last_ma = ma[-1] if ma else historical[-1]
        forecast = [last_ma] * periods_ahead

        # Prediction interval from residual standard deviation
        if residuals:
            residual_std = float(np.std(residuals, ddof=1))
        else:
            residual_std = 0.0

        return {
            "method": "simple_moving_average",
            "window": window,
            "ma_values": [round(v, 2) for v in ma],
            "forecast": [round(v, 2) for v in forecast],
            "forecast_periods": periods_ahead,
            "in_sample_mae": round(
                float(np.mean(np.abs(residuals))) if residuals else 0.0, 2
            ),
            "in_sample_rmse": round(
                float(np.sqrt(np.mean([r**2 for r in residuals]))) if residuals else 0.0, 2
            ),
            "prediction_interval_95": [
                (round(v - 1.96 * residual_std, 2), round(v + 1.96 * residual_std, 2))
                for v in forecast
            ],
        }

    # ------------------------------------------------------------------
    # Weighted Moving Average
    # ------------------------------------------------------------------

    def weighted_moving_average(
        self,
        historical: list[float],
        weights: list[float] | None = None,
        periods_ahead: int = 1,
    ) -> dict[str, Any]:
        """Compute a weighted moving average forecast.

        Args:
            historical: Chronologically ordered historical demand values.
            weights: Weight per period (most recent last). Default: linear.
            periods_ahead: How many periods to forecast.

        Returns:
            Forecast values with weights and error metrics.
        """
        n = len(historical)
        if weights is None:
            # Linear weights: most recent gets highest weight
            weights = [i + 1 for i in range(n)]
        if len(weights) != n:
            raise ValueError("Weights length must equal historical data length")

        w_sum = sum(weights)
        if w_sum == 0:
            raise ValueError("Weights must sum to a positive value")

        # Normalize weights
        norm_weights = [w / w_sum for w in weights]

        # Weighted average of all available data
        forecast_value = sum(h * w for h, w in zip(historical, norm_weights))
        forecast = [forecast_value] * periods_ahead

        # In-sample residuals
        residuals = []
        for i in range(1, n):
            subset = historical[:i]
            sub_w = norm_weights[:i]
            sub_w_sum = sum(sub_w)
            if sub_w_sum > 0:
                pred = sum(h * w for h, w in zip(subset, [w / sub_w_sum for w in sub_w]))
                residuals.append(historical[i] - pred)

        return {
            "method": "weighted_moving_average",
            "weights_used": [round(w, 4) for w in norm_weights],
            "forecast": [round(v, 2) for v in forecast],
            "forecast_periods": periods_ahead,
            "in_sample_mae": round(
                float(np.mean(np.abs(residuals))) if residuals else 0.0, 2
            ),
        }

    # ------------------------------------------------------------------
    # Single Exponential Smoothing (SES)
    # ------------------------------------------------------------------

    def exponential_smoothing(
        self,
        historical: list[float],
        alpha: float = 0.3,
        periods_ahead: int = 1,
    ) -> dict[str, Any]:
        """Single exponential smoothing for level-only time series.

        Args:
            historical: Chronologically ordered demand values.
            alpha: Smoothing parameter (0 < alpha < 1). Higher = more responsive.
            periods_ahead: How many periods to forecast.

        Returns:
            Smoothed values, forecast, and fitted error metrics.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        if len(historical) < 2:
            raise ValueError("Need at least 2 data points")

        n = len(historical)
        smoothed = [historical[0]]
        residuals = []

        for i in range(1, n):
            s = alpha * historical[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s)
            residuals.append(historical[i] - smoothed[-2])

        forecast = [smoothed[-1]] * periods_ahead

        return {
            "method": "exponential_smoothing",
            "alpha": alpha,
            "smoothed_values": [round(v, 2) for v in smoothed],
            "forecast": [round(v, 2) for v in forecast],
            "forecast_periods": periods_ahead,
            "in_sample_mae": round(float(np.mean(np.abs(residuals))), 2),
            "in_sample_rmse": round(
                float(np.sqrt(np.mean([r**2 for r in residuals]))), 2
            ),
        }

    # ------------------------------------------------------------------
    # Double Exponential Smoothing (Holt's Linear Trend)
    # ------------------------------------------------------------------

    def holt_smoothing(
        self,
        historical: list[float],
        alpha: float = 0.3,
        beta: float = 0.1,
        periods_ahead: int = 1,
    ) -> dict[str, Any]:
        """Double exponential smoothing with trend estimation.

        Holt's method decomposes the time series into level and trend
        components, allowing forecasts to capture upward or downward
        trajectories in demand.

        Args:
            historical: Chronologically ordered demand values.
            alpha: Level smoothing parameter (0 < alpha < 1).
            beta: Trend smoothing parameter (0 < beta < 1).
            periods_ahead: How many periods to forecast.

        Returns:
            Level and trend components, forecast with intervals.
        """
        if not (0 < alpha < 1) or not (0 < beta < 1):
            raise ValueError("alpha and beta must be between 0 and 1")
        if len(historical) < 3:
            raise ValueError("Need at least 3 data points for Holt's method")

        n = len(historical)

        # Initialize: level = first value, trend = average of first differences
        level = historical[0]
        trend = (historical[1] - historical[0])

        levels = [level]
        trends = [trend]
        fitted = [level + trend]
        residuals = []

        for i in range(1, n):
            # Update level
            new_level = alpha * historical[i] + (1 - alpha) * (levels[-1] + trends[-1])
            # Update trend
            new_trend = beta * (new_level - levels[-1]) + (1 - beta) * trends[-1]

            levels.append(new_level)
            trends.append(new_trend)

            # Fitted value for previous period
            pred = levels[-2] + trends[-2]
            fitted.append(pred)
            residuals.append(historical[i] - pred)

        # Forecast
        forecast = [
            levels[-1] + (h + 1) * trends[-1]
            for h in range(periods_ahead)
        ]

        # Prediction interval from residual variance
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        return {
            "method": "holt_double_exponential_smoothing",
            "alpha": alpha,
            "beta": beta,
            "final_level": round(levels[-1], 2),
            "final_trend": round(trends[-1], 2),
            "forecast": [round(v, 2) for v in forecast],
            "forecast_periods": periods_ahead,
            "trend_direction": "increasing" if trends[-1] > 0 else "decreasing",
            "in_sample_mae": round(float(np.mean(np.abs(residuals))), 2),
            "prediction_interval_95": [
                (round(v - 1.96 * residual_std, 2), round(v + 1.96 * residual_std, 2))
                for v in forecast
            ],
        }

    # ------------------------------------------------------------------
    # Seasonal Decomposition
    # ------------------------------------------------------------------

    def seasonal_decompose(
        self,
        historical: list[float],
        period: int = 12,
        periods_ahead: int = 12,
    ) -> dict[str, Any]:
        """Additive seasonal decomposition with trend and forecast.

        Decomposes the time series into trend + seasonal + residual
        components. Forecasts extend the trend and repeat the seasonal
        pattern.

        Args:
            historical: Chronologically ordered demand values.
            period: Seasonal period length (e.g. 12 for monthly, 4 for quarterly).
            periods_ahead: How many periods to forecast.

        Returns:
            Decomposition components and seasonal forecast.
        """
        if len(historical) < 2 * period:
            raise ValueError(
                f"Need at least {2 * period} data points for period={period}"
            )

        arr = np.array(historical, dtype=float)
        n = len(arr)

        # --- Trend via centered moving average ---
        trend = np.zeros(n)
        half = period // 2
        for i in range(half, n - half):
            window_start = i - half
            window_end = i + half + (period % 2)
            trend[i] = np.mean(arr[window_start:window_end])

        # Fill edges with nearest computed trend
        first_valid = trend[half] if half < n else arr[0]
        last_valid = trend[n - half - 1] if (n - half - 1) >= 0 else arr[-1]
        for i in range(half):
            trend[i] = first_valid
        for i in range(n - half, n):
            trend[i] = last_valid

        # --- Deseasonalized ---
        deseasonalized = arr - trend

        # --- Seasonal indices ---
        seasonal = np.zeros(period)
        for s in range(period):
            indices = [deseasonalized[i] for i in range(s, n, period) if i < n]
            seasonal[s] = np.mean(indices) if indices else 0.0

        # Normalize seasonal indices to sum to 0
        seasonal -= np.mean(seasonal)

        # --- Residuals ---
        seasonal_full = np.tile(seasonal, (n // period + 1))[:n]
        residual = arr - trend - seasonal_full

        # --- Forecast ---
        last_trend = trend[-1]
        # Estimate trend slope from last full period
        if n >= 2 * period:
            slope = (trend[-1] - trend[-period]) / period
        else:
            slope = 0.0

        forecast = []
        for h in range(1, periods_ahead + 1):
            t = last_trend + slope * h
            s = seasonal[(n + h - 1) % period]
            forecast.append(round(float(t + s), 2))

        residual_std = float(np.std(residual, ddof=1))

        return {
            "method": "seasonal_decomposition_additive",
            "period": period,
            "trend_values": [round(v, 2) for v in trend.tolist()],
            "seasonal_indices": [round(v, 2) for v in seasonal.tolist()],
            "residual_std": round(residual_std, 2),
            "forecast": forecast,
            "forecast_periods": periods_ahead,
            "prediction_interval_95": [
                (round(v - 1.96 * residual_std, 2), round(v + 1.96 * residual_std, 2))
                for v in forecast
            ],
        }

    # ------------------------------------------------------------------
    # Auto-select best method
    # ------------------------------------------------------------------

    def auto_forecast(
        self,
        historical: list[float],
        periods_ahead: int = 6,
        seasonality_period: int | None = None,
    ) -> dict[str, Any]:
        """Automatically select the best forecasting method.

        Runs SES, Holt, and (if data length permits) seasonal decomposition,
        then selects the method with lowest in-sample MAE.

        Args:
            historical: Chronologically ordered demand values.
            periods_ahead: How many periods to forecast.
            seasonality_period: If known, the seasonal period length.

        Returns:
            Forecast from the best-performing method with selection rationale.
        """
        results = []
        n = len(historical)

        # Always try SES
        ses = self.exponential_smoothing(historical, alpha=0.3, periods_ahead=periods_ahead)
        results.append(("SES", ses["in_sample_mae"], ses))

        # Always try Holt
        if n >= 3:
            holt = self.holt_smoothing(historical, alpha=0.3, beta=0.1, periods_ahead=periods_ahead)
            results.append(("Holt", holt["in_sample_mae"], holt))

        # Seasonal if enough data
        if n >= 24 and seasonality_period and seasonality_period >= 2:
            try:
                seasonal = self.seasonal_decompose(
                    historical, period=seasonality_period, periods_ahead=periods_ahead
                )
                # Compute MAE for seasonal
                trend_vals = seasonal["trend_values"][-seasonality_period:]
                seasonal_idx = seasonal["seasonal_indices"]
                reconstructed = [
                    trend_vals[i % len(trend_vals)] + seasonal_idx[i % len(seasonal_idx)]
                    for i in range(len(historical) - len(trend_vals), len(historical))
                ]
                actuals = historical[-len(reconstructed):]
                mae = float(np.mean([abs(a - r) for a, r in zip(actuals, reconstructed)]))
                results.append(("Seasonal", mae, seasonal))
            except (ValueError, Exception):
                pass

        # Select best
        results.sort(key=lambda x: x[1])
        best_name, best_mae, best_result = results[0]

        return {
            "selected_method": best_name,
            "in_sample_mae": round(best_mae, 2),
            "all_methods": {
                name: round(mae, 2) for name, mae, _ in results
            },
            "forecast": best_result["forecast"],
            "forecast_periods": periods_ahead,
            "detail": best_result,
        }
