"""Test suite for the feature engineering pipeline.

Tests cover:
- Rolling statistics accuracy
- FFT spectral feature computation
- Zero-crossing rate calculation
- Edge cases (empty signals, single sample)
- Feature vector completeness
- Degradation curve behavior
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.models.schemas import SensorReading, AssetType
from src.services.data_collector import DataCollector
from src.services.feature_engineer import FeatureEngineer, FeatureVector
from src.utils.helpers import (
    compute_rms,
    compute_peak,
    compute_crest_factor,
    zero_crossing_rate,
    degradation_curve,
    validate_sensor_ranges,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def collector() -> DataCollector:
    """Create a fresh DataCollector with one registered asset."""
    dc = DataCollector()
    dc.register_asset("TEST-001", AssetType.MOTOR)
    return dc


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create a fresh FeatureEngineer."""
    return FeatureEngineer(buffer_size=200, fft_window=256)


@pytest.fixture
def sample_reading() -> SensorReading:
    """Create a deterministic sensor reading for testing."""
    return SensorReading(
        asset_id="TEST-001",
        asset_type=AssetType.MOTOR,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        vibration_x=2.5,
        vibration_y=1.8,
        vibration_z=3.2,
        temperature=72.0,
        pressure=4.1,
        current=12.5,
        rpm=1750.0,
    )


# ── Time-domain feature tests ───────────────────────────────────────────


class TestTimeDomainFeatures:
    """Tests for basic signal processing utilities."""

    def test_rms_of_sinusoid(self) -> None:
        """RMS of a pure sinusoid should equal amplitude / sqrt(2)."""
        t = np.linspace(0, 2 * np.pi, 1000)
        signal = 3.0 * np.sin(t)  # amplitude = 3.0
        rms = compute_rms(signal)
        expected = 3.0 / math.sqrt(2)
        assert abs(rms - expected) < 0.01, f"Expected ~{expected:.4f}, got {rms:.4f}"

    def test_rms_of_constant(self) -> None:
        """RMS of a constant signal equals the constant."""
        signal = np.full(100, 5.0)
        assert abs(compute_rms(signal) - 5.0) < 1e-10

    def test_rms_empty_signal(self) -> None:
        """RMS of empty signal returns 0."""
        assert compute_rms(np.array([])) == 0.0

    def test_peak_detection(self) -> None:
        """Peak should capture the maximum absolute value."""
        signal = np.array([-3.0, 1.0, 5.0, -2.0, 4.0])
        assert compute_peak(signal) == 5.0

    def test_peak_negative_dominant(self) -> None:
        """Peak should handle signals dominated by negative values."""
        signal = np.array([-10.0, 1.0, 2.0])
        assert compute_peak(signal) == 10.0

    def test_crest_factor_of_sinusoid(self) -> None:
        """Crest factor of a sinusoid is sqrt(2) ≈ 1.414."""
        t = np.linspace(0, 2 * np.pi, 1000)
        signal = np.sin(t)
        cf = compute_crest_factor(signal)
        assert abs(cf - math.sqrt(2)) < 0.05

    def test_crest_factor_impulsive(self) -> None:
        """Impulsive signals (spikes) should have high crest factor."""
        signal = np.zeros(1000)
        signal[500] = 10.0  # Single spike
        cf = compute_crest_factor(signal)
        assert cf > 5.0, f"Expected high CF for impulsive signal, got {cf}"

    def test_zero_crossing_rate_sine(self) -> None:
        """ZCR of a sine wave at known frequency."""
        sr = 1000
        t = np.arange(sr) / sr  # 1 second
        freq = 50  # 50 Hz
        signal = np.sin(2 * np.pi * freq * t)
        zcr = zero_crossing_rate(signal)
        # Expected: ~2 * freq / sr per sample = 100/1000 = 0.1
        assert abs(zcr - 0.1) < 0.02

    def test_zero_crossing_rate_constant(self) -> None:
        """Constant signal has zero ZCR."""
        signal = np.ones(100) * 5.0
        assert zero_crossing_rate(signal) == 0.0

    def test_zero_crossing_rate_empty(self) -> None:
        """Empty signal has zero ZCR."""
        assert zero_crossing_rate(np.array([])) == 0.0


# ── Feature engineering tests ───────────────────────────────────────────


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_feature_vector_dimensionality(
        self, feature_engineer: FeatureEngineer, sample_reading: SensorReading
    ) -> None:
        """Feature vector should have correct number of dimensions."""
        fv = feature_engineer.compute_features(sample_reading)
        arr = fv.to_array()
        # Should have 45+ numeric features
        assert arr.size >= 40, f"Expected 40+ features, got {arr.size}"

    def test_feature_names_count(self) -> None:
        """Feature names should match vector dimensionality."""
        names = FeatureVector.feature_names()
        fv = FeatureVector(asset_id="X", timestamp=None)
        arr = fv.to_array()
        assert len(names) == arr.size, (
            f"Names ({len(names)}) and array ({arr.size}) dimensionality mismatch"
        )

    def test_asset_id_preserved(
        self, feature_engineer: FeatureEngineer, sample_reading: SensorReading
    ) -> None:
        """Asset ID should be preserved in the feature vector."""
        fv = feature_engineer.compute_features(sample_reading)
        assert fv.asset_id == "TEST-001"

    def test_rolling_stats_with_buffer(
        self, feature_engineer: FeatureEngineer, collector: DataCollector
    ) -> None:
        """Rolling statistics should stabilize as more data is added."""
        # Add 60 readings to build up buffer
        for _ in range(60):
            reading = collector.collect_reading("TEST-001")
            fv = feature_engineer.compute_features(reading)

        # After 60 readings, rolling_mean_50 should exist and be reasonable
        assert fv.rolling_mean_50 > 0.0, "Rolling mean should be positive"
        assert fv.rolling_std_50 >= 0.0, "Rolling std should be non-negative"
        assert fv.rolling_max_50 >= fv.rolling_min_50, "Max should be >= min"

    def test_spectral_features_non_negative(
        self, feature_engineer: FeatureEngineer, collector: DataCollector
    ) -> None:
        """Spectral features should be non-negative."""
        for _ in range(10):
            reading = collector.collect_reading("TEST-001")
            fv = feature_engineer.compute_features(reading)

        assert fv.fft_peak_frequency >= 0.0
        assert fv.fft_peak_magnitude >= 0.0
        assert fv.spectral_energy >= 0.0
        assert 0.0 <= fv.spectral_entropy <= 1.0

    def test_temperature_features(
        self, feature_engineer: FeatureEngineer, sample_reading: SensorReading
    ) -> None:
        """Temperature features should reflect the input reading."""
        fv = feature_engineer.compute_features(sample_reading)
        assert fv.temperature_current == 72.0

    def test_pressure_features(
        self, feature_engineer: FeatureEngineer, sample_reading: SensorReading
    ) -> None:
        """Pressure features should reflect the input reading."""
        fv = feature_engineer.compute_features(sample_reading)
        assert fv.pressure_current == 4.1

    def test_cross_sensor_ratios(
        self, feature_engineer: FeatureEngineer, sample_reading: SensorReading
    ) -> None:
        """Cross-sensor ratios should be computed correctly."""
        fv = feature_engineer.compute_features(sample_reading)
        # vib_temp_ratio = composite_rms / temperature
        assert fv.vib_temp_ratio > 0.0
        # pressure_current_ratio = pressure / current
        assert fv.pressure_current_ratio > 0.0

    def test_clear_buffer(self, feature_engineer: FeatureEngineer, collector: DataCollector) -> None:
        """Clearing buffer should reset the asset's stored data."""
        for _ in range(10):
            reading = collector.collect_reading("TEST-001")
            feature_engineer.compute_features(reading)

        assert "TEST-001" in feature_engineer._buffers
        feature_engineer.clear_buffer("TEST-001")
        assert "TEST-001" not in feature_engineer._buffers


# ── Degradation curve tests ─────────────────────────────────────────────


class TestDegradationCurve:
    """Tests for the exponential degradation model."""

    def test_baseline_before_degradation(self) -> None:
        """Before degradation start, value should equal base."""
        assert degradation_curve(50.0, start_hour=72.0, failure_hour=96.0) == 1.0

    def test_exponential_growth(self) -> None:
        """Value should grow exponentially between start and failure."""
        v1 = degradation_curve(80.0, start_hour=72.0, failure_hour=96.0)
        v2 = degradation_curve(90.0, start_hour=72.0, failure_hour=96.0)
        assert v2 > v1, "Degradation should increase over time"

    def test_at_failure(self) -> None:
        """At failure hour, value should be base * e^4 ≈ 54.6."""
        val = degradation_curve(96.0, start_hour=72.0, failure_hour=96.0)
        expected = math.exp(4.0)
        assert abs(val - expected) < 0.01

    def test_past_failure(self) -> None:
        """After failure, value should cap at e^4."""
        val = degradation_curve(120.0, start_hour=72.0, failure_hour=96.0)
        assert abs(val - math.exp(4.0)) < 0.01


# ── Validation tests ────────────────────────────────────────────────────


class TestValidation:
    """Tests for sensor data validation utilities."""

    def test_normal_values_no_warnings(self) -> None:
        """Normal readings should produce no warnings."""
        warnings = validate_sensor_ranges(2.0, 1.5, 2.5, 70.0, 4.0)
        assert warnings == []

    def test_high_vibration_warning(self) -> None:
        """Excessive vibration should trigger a warning."""
        warnings = validate_sensor_ranges(25.0, 1.5, 2.5, 70.0, 4.0)
        assert len(warnings) >= 1
        assert "vibration_x" in warnings[0]

    def test_high_temperature_warning(self) -> None:
        """Temperature above limit should trigger a warning."""
        warnings = validate_sensor_ranges(2.0, 1.5, 2.5, 160.0, 4.0)
        assert len(warnings) >= 1
        assert "temperature" in warnings[0]

    def test_negative_pressure_warning(self) -> None:
        """Negative pressure should trigger a warning."""
        warnings = validate_sensor_ranges(2.0, 1.5, 2.5, 70.0, -1.0)
        assert len(warnings) >= 1
        assert "pressure" in warnings[0]


# ── Data collector integration test ─────────────────────────────────────


class TestDataCollectorIntegration:
    """Integration tests for DataCollector with FeatureEngineer."""

    def test_full_pipeline(
        self, collector: DataCollector, feature_engineer: FeatureEngineer
    ) -> None:
        """Full pipeline: collect → feature-engineer → verify output."""
        # Feed 30 readings
        for _ in range(30):
            reading = collector.collect_reading("TEST-001")
            fv = feature_engineer.compute_features(reading)

        # Verify feature vector is complete and valid
        arr = fv.to_array()
        assert np.all(np.isfinite(arr)), "Feature vector contains non-finite values"
        assert arr.size >= 40

    def test_multiple_assets(self) -> None:
        """DataCollector should handle multiple assets independently."""
        dc = DataCollector()
        dc.register_asset("MOTOR-001", AssetType.MOTOR)
        dc.register_asset("PUMP-001", AssetType.PUMP)

        r1 = dc.collect_reading("MOTOR-001")
        r2 = dc.collect_reading("PUMP-001")

        assert r1.asset_id == "MOTOR-001"
        assert r2.asset_id == "PUMP-001"
        assert r1.asset_type == AssetType.MOTOR
        assert r2.asset_type == AssetType.PUMP

    def test_history_retrieval(self, collector: DataCollector) -> None:
        """History should return readings in order."""
        for _ in range(5):
            collector.collect_reading("TEST-001")

        history = collector.get_history("TEST-001")
        assert len(history) == 5
        # Timestamps should be non-decreasing
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i - 1].timestamp

    def test_unregistered_asset_raises(self) -> None:
        """Collecting from unregistered asset should raise KeyError."""
        dc = DataCollector()
        with pytest.raises(KeyError, match="not registered"):
            dc.collect_reading("NONEXISTENT")

    def test_vibration_array_extraction(self, collector: DataCollector) -> None:
        """Vibration array extraction should return correct data."""
        for _ in range(10):
            collector.collect_reading("TEST-001")

        arr = collector.get_vibration_array("TEST-001", "vibration_x", last_n=10)
        assert arr.size == 10
        assert arr.dtype == np.float64
