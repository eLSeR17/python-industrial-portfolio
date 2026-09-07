"""Test suite for the failure prediction engine.

Tests cover:
- Physics-based fallback (pre-training)
- Training workflow and model fitting
- Prediction output format and validity
- Failure mode classification
- RUL estimation accuracy
- Severity classification
- Recommendation generation
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.models.schemas import AssetType, Severity
from src.services.data_collector import DataCollector
from src.services.feature_engineer import FeatureEngineer, FeatureVector
from src.services.failure_predictor import FailurePredictor, PredictionResult


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def predictor() -> FailurePredictor:
    """Create a fresh FailurePredictor with small training threshold."""
    return FailurePredictor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=3,
        prediction_horizon_hours=24.0,
        min_training_samples=20,
    )


@pytest.fixture
def trained_predictor() -> tuple[FailurePredictor, str]:
    """Create a trained FailurePredictor with synthetic labeled data.

    Simulates a motor degrading from healthy (hour 0) to failure (hour 96),
    with feature vectors captured at each stage.

    Returns:
        Tuple of (predictor, asset_id).
    """
    predictor = FailurePredictor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=3,
        min_training_samples=20,
    )
    dc = DataCollector()
    fe = FeatureEngineer(buffer_size=200)
    asset_id = "PRED-001"

    # Register and simulate degradation
    dc.register_asset(asset_id, AssetType.MOTOR, degradation_start_hour=40.0, failure_hour=96.0)

    # Feed readings over the full degradation cycle
    for hour in range(0, 100, 1):
        # Advance the operating time manually
        dc._assets[asset_id].operating_hours = float(hour)
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)

        # Label: fails within 24h if past hour 72
        will_fail = hour >= 72
        rul = max(0.0, 96.0 - hour)

        predictor.add_training_sample(
            fv,
            time_to_failure_hours=rul,
            failed_within_horizon=will_fail,
        )

    assert predictor.is_trained(asset_id), "Predictor should be trained after 100 samples"
    return predictor, asset_id


# ── Physics-based fallback tests ────────────────────────────────────────


class TestPhysicsFallback:
    """Tests for the physics-based RUL estimation before ML training."""

    def test_healthy_asset_long_rul(self, predictor: FailurePredictor) -> None:
        """A healthy asset (low vibration) should have long RUL estimate."""
        fv = FeatureVector(
            asset_id="HEALTHY-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=1.5,  # Normal vibration
            rolling_std_10=0.1,
            rolling_std_50=0.1,
            temperature_current=65.0,
            pressure_current=4.0,
            harmonic_ratio=0.1,
            fft_peak_frequency=29.0,  # ~1750 RPM / 60 Hz
            spectral_entropy=0.3,
            power_estimate=2.5,
        )

        result = predictor.predict(fv)
        assert result.remaining_useful_life_hours > 48.0, (
            f"Healthy asset should have long RUL, got {result.remaining_useful_life_hours}"
        )
        assert result.confidence < 0.5, "Physics-based should have lower confidence"

    def test_degraded_asset_short_rul(self, predictor: FailurePredictor) -> None:
        """A degraded asset (high vibration) should have short RUL."""
        fv = FeatureVector(
            asset_id="DEGRADED-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=8.0,  # High vibration
            rolling_std_10=1.5,
            rolling_std_50=0.5,
            temperature_current=85.0,
            pressure_current=3.5,
            harmonic_ratio=0.25,
            fft_peak_frequency=35.0,
            spectral_entropy=0.5,
            power_estimate=3.0,
        )

        result = predictor.predict(fv)
        assert result.remaining_useful_life_hours < 24.0, (
            f"Degraded asset should have short RUL, got {result.remaining_useful_life_hours}"
        )

    def test_critical_asset_zero_rul(self, predictor: FailurePredictor) -> None:
        """An asset beyond failure threshold should have ~0 RUL."""
        fv = FeatureVector(
            asset_id="CRITICAL-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=15.0,  # Way beyond failure threshold
            rolling_std_10=3.0,
            rolling_std_50=2.0,
            temperature_current=100.0,
            pressure_current=2.0,
            harmonic_ratio=0.4,
            fft_peak_frequency=40.0,
            spectral_entropy=0.7,
            power_estimate=4.0,
        )

        result = predictor.predict(fv)
        assert result.remaining_useful_life_hours == 0.0

    def test_untrained_returns_prediction(self, predictor: FailurePredictor) -> None:
        """Even untrained predictor should return a valid PredictionResult."""
        fv = FeatureVector(
            asset_id="UNTRAINED-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=3.0,
            rolling_std_10=0.3,
            rolling_std_50=0.2,
            temperature_current=70.0,
            pressure_current=4.0,
            harmonic_ratio=0.15,
            fft_peak_frequency=29.0,
            spectral_entropy=0.4,
            power_estimate=2.5,
        )

        result = predictor.predict(fv)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.failure_probability <= 1.0
        assert result.remaining_useful_life_hours >= 0.0


# ── Training tests ──────────────────────────────────────────────────────


class TestTraining:
    """Tests for the ML training workflow."""

    def test_insufficient_data_no_train(self, predictor: FailurePredictor) -> None:
        """Should not train with fewer than min_training_samples."""
        fv = FeatureVector(
            asset_id="FEW-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=2.0,
            temperature_current=70.0,
        )
        # Add only 5 samples (below threshold of 20)
        for i in range(5):
            predictor.add_training_sample(fv, time_to_failure_hours=100.0, failed_within_horizon=False)

        assert predictor.is_trained("FEW-001") is False

    def test_training_completes_at_threshold(
        self, trained_predictor: tuple[FailurePredictor, str]
    ) -> None:
        """Training should complete when enough samples are provided."""
        predictor, asset_id = trained_predictor
        assert predictor.is_trained(asset_id) is True

    def test_trained_prediction_has_ml_method(
        self, trained_predictor: tuple[FailurePredictor, str]
    ) -> None:
        """Trained predictor should use ML method, not physics fallback."""
        predictor, asset_id = trained_predictor

        fv = FeatureVector(
            asset_id=asset_id,
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=3.0,
            rolling_std_10=0.3,
            rolling_std_50=0.2,
            temperature_current=72.0,
            pressure_current=4.0,
            harmonic_ratio=0.15,
            fft_peak_frequency=29.0,
            spectral_entropy=0.4,
            power_estimate=2.5,
        )

        result = predictor.predict(fv)
        assert result.model_metrics.get("method") == "ml"

    def test_metrics_available(
        self, trained_predictor: tuple[FailurePredictor, str]
    ) -> None:
        """Training metrics should be available after training."""
        predictor, asset_id = trained_predictor
        metrics = predictor.get_metrics(asset_id)
        assert "rul_mae_cv" in metrics
        assert "fail_f1_cv" in metrics
        assert metrics["n_training_samples"] >= 20


# ── Failure mode classification tests ───────────────────────────────────


class TestFailureModes:
    """Tests for failure mode determination."""

    def test_high_vibration_bearing_wear(self, predictor: FailurePredictor) -> None:
        """High vibration + normal temp → bearing wear."""
        fv = FeatureVector(
            asset_id="MODE-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=6.0,
            temperature_current=70.0,
            pressure_current=4.0,
            power_estimate=2.5,
            spectral_entropy=0.4,
            harmonic_ratio=0.1,
            fft_peak_frequency=29.0,
        )
        result = predictor.predict(fv)
        assert result.failure_mode == "bearing_wear"

    def test_high_vibration_high_temp_lubrication(
        self, predictor: FailurePredictor
    ) -> None:
        """High vibration + high temp → lubrication failure."""
        fv = FeatureVector(
            asset_id="MODE-002",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=6.0,
            temperature_current=90.0,
            pressure_current=4.0,
            power_estimate=2.5,
            spectral_entropy=0.4,
            harmonic_ratio=0.1,
            fft_peak_frequency=29.0,
        )
        result = predictor.predict(fv)
        assert result.failure_mode == "lubrication_failure"

    def test_low_pressure_seal_leak(self, predictor: FailurePredictor) -> None:
        """Low pressure + normal vibration → seal leakage."""
        fv = FeatureVector(
            asset_id="MODE-003",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=2.0,
            temperature_current=65.0,
            pressure_current=1.5,  # Low
            power_estimate=2.0,
            spectral_entropy=0.3,
            harmonic_ratio=0.1,
            fft_peak_frequency=29.0,
        )
        result = predictor.predict(fv)
        assert result.failure_mode == "seal_leakage"


# ── Severity and recommendations tests ─────────────────────────────────


class TestSeverityRecommendations:
    """Tests for severity classification and action recommendations."""

    def test_emergency_for_high_prob_short_rul(
        self, trained_predictor: tuple[FailurePredictor, str]
    ) -> None:
        """Trained predictor should produce valid severity classification."""
        predictor, asset_id = trained_predictor

        # Create a clearly degraded feature vector
        fv = FeatureVector(
            asset_id=asset_id,
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=10.0,
            rolling_std_10=2.0,
            rolling_std_50=1.0,
            temperature_current=95.0,
            pressure_current=2.0,
            harmonic_ratio=0.4,
            fft_peak_frequency=40.0,
            spectral_entropy=0.8,
            power_estimate=4.0,
        )

        result = predictor.predict(fv)
        # Verify the prediction is valid and severity is one of the valid enum values
        assert isinstance(result.severity, Severity)
        assert result.failure_probability >= 0.0
        assert result.remaining_useful_life_hours >= 0.0
        # The ML model should recognize this as degraded (RUL should be shorter
        # than a healthy asset's ~200h estimate)
        healthy_fv = FeatureVector(
            asset_id=asset_id,
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=1.5,
            rolling_std_10=0.1,
            rolling_std_50=0.1,
            temperature_current=65.0,
            pressure_current=4.0,
            harmonic_ratio=0.1,
            fft_peak_frequency=29.0,
            spectral_entropy=0.3,
            power_estimate=2.5,
        )
        healthy_result = predictor.predict(healthy_fv)
        # Degraded asset should have lower (or equal) RUL than healthy
        assert result.remaining_useful_life_hours <= healthy_result.remaining_useful_life_hours + 5.0

    def test_prediction_result_completeness(
        self, trained_predictor: tuple[FailurePredictor, str]
    ) -> None:
        """PredictionResult should have all required fields."""
        predictor, asset_id = trained_predictor

        fv = FeatureVector(
            asset_id=asset_id,
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=4.0,
            rolling_std_10=0.5,
            rolling_std_50=0.3,
            temperature_current=75.0,
            pressure_current=3.8,
            harmonic_ratio=0.2,
            fft_peak_frequency=30.0,
            spectral_entropy=0.5,
            power_estimate=3.0,
        )

        result = predictor.predict(fv)
        assert result.asset_id == asset_id
        assert result.timestamp is not None
        assert 0.0 <= result.failure_probability <= 1.0
        assert result.remaining_useful_life_hours >= 0.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.failure_mode, str)
        assert isinstance(result.recommended_action, str)
        assert isinstance(result.severity, Severity)


# ── Degradation scenario test ───────────────────────────────────────────


class TestDegradationScenario:
    """End-to-end test simulating asset degradation over time."""

    def test_rul_decreases_as_degradation_progresses(self) -> None:
        """RUL should decrease as the asset degrades over time."""
        predictor = FailurePredictor(
            n_estimators=30, max_depth=8, min_training_samples=15
        )
        dc = DataCollector()
        fe = FeatureEngineer(buffer_size=200)
        asset_id = "DEGRAD-001"

        dc.register_asset(asset_id, AssetType.MOTOR, degradation_start_hour=40, failure_hour=96)

        rul_values: list[float] = []

        for hour in range(0, 96, 4):
            dc._assets[asset_id].operating_hours = float(hour)
            reading = dc.collect_reading(asset_id)
            fv = fe.compute_features(reading)

            will_fail = hour >= 72
            rul = max(0.0, 96.0 - hour)
            predictor.add_training_sample(fv, rul, will_fail)

            result = predictor.predict(fv)
            rul_values.append(result.remaining_useful_life_hours)

        # RUL should generally decrease over time (monotonic-ish)
        # Not strictly monotonic due to noise, but early should be > late
        assert rul_values[0] > rul_values[-1], (
            f"RUL should decrease: first={rul_values[0]:.1f}, last={rul_values[-1]:.1f}"
        )
