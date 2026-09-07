"""Test suite for the anomaly detection engine.

Tests cover:
- Calibration workflow
- Statistical tier detection (Z-score)
- ML tier detection (Isolation Forest)
- Combined tier behavior
- Edge cases (uncalibrated assets, empty data)
- Baseline computation
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.models.schemas import AssetType, SensorReading
from src.services.anomaly_detector import AnomalyDetector, AnomalyResult
from src.services.data_collector import DataCollector
from src.services.feature_engineer import FeatureEngineer, FeatureVector


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def detector() -> AnomalyDetector:
    """Create a fresh AnomalyDetector with small calibration size for tests."""
    return AnomalyDetector(
        contamination=0.05,
        n_estimators=50,  # Fewer trees for faster tests
        z_score_threshold=3.0,
        calibration_size=30,
    )


@pytest.fixture
def calibrated_detector() -> tuple[AnomalyDetector, str]:
    """Create an AnomalyDetector calibrated with healthy data.

    Returns:
        Tuple of (detector, asset_id).
    """
    dc = DataCollector()
    fe = FeatureEngineer(buffer_size=200)
    asset_id = "CAL-001"
    dc.register_asset(asset_id, AssetType.MOTOR)

    detector = AnomalyDetector(
        contamination=0.05,
        n_estimators=50,
        calibration_size=30,
    )

    # Feed 30 healthy readings for calibration
    for _ in range(30):
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)
        detector.add_calibration_sample(fv)

    assert detector.needs_calibration(asset_id) is False, "Should be calibrated"
    return detector, asset_id


@pytest.fixture
def calibrated_pair() -> tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]:
    """Create a fully initialized collector-engineer-detector triple.

    Returns:
        Tuple of (data_collector, feature_engineer, anomaly_detector, asset_id).
    """
    dc = DataCollector()
    fe = FeatureEngineer(buffer_size=200)
    detector = AnomalyDetector(
        contamination=0.05,
        n_estimators=50,
        calibration_size=30,
    )
    asset_id = "TEST-001"
    dc.register_asset(asset_id, AssetType.MOTOR)

    # Calibrate
    for _ in range(30):
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)
        detector.add_calibration_sample(fv)

    return dc, fe, detector, asset_id


# ── Calibration tests ───────────────────────────────────────────────────


class TestCalibration:
    """Tests for the calibration workflow."""

    def test_needs_calibration_initially(self, detector: AnomalyDetector) -> None:
        """New detector should need calibration for all assets."""
        assert detector.needs_calibration("NEW-ASSET") is True

    def test_calibration_progress(
        self, detector: AnomalyDetector
    ) -> None:
        """Calibration should not complete until enough samples."""
        dc = DataCollector()
        fe = FeatureEngineer()
        dc.register_asset("PROG-001", AssetType.MOTOR)

        # Feed fewer than calibration_size samples
        for _ in range(20):
            reading = dc.collect_reading("PROG-001")
            fv = fe.compute_features(reading)
            result = detector.add_calibration_sample(fv)

        assert result is False, "Should not be calibrated yet"
        assert detector.needs_calibration("PROG-001") is True

    def test_calibration_completes(self, detector: AnomalyDetector) -> None:
        """Calibration should complete after enough samples."""
        dc = DataCollector()
        fe = FeatureEngineer()
        dc.register_asset("COMP-001", AssetType.MOTOR)

        calibrated = False
        for _ in range(35):
            reading = dc.collect_reading("COMP-001")
            fv = fe.compute_features(reading)
            calibrated = detector.add_calibration_sample(fv)

        assert calibrated is True
        assert detector.needs_calibration("COMP-001") is False

    def test_baselines_populated(
        self, calibrated_detector: tuple[AnomalyDetector, str]
    ) -> None:
        """Baselines should be computed after calibration."""
        det, asset_id = calibrated_detector
        baselines = det.get_baseline_summary(asset_id)
        assert len(baselines) > 0
        # Check that key features have non-zero baselines
        assert "vib_composite_rms" in baselines
        assert baselines["vib_composite_rms"]["mean"] > 0.0


# ── Detection tests ─────────────────────────────────────────────────────


class TestAnomalyDetection:
    """Tests for anomaly detection on calibrated assets."""

    def test_normal_reading_not_anomalous(
        self, calibrated_pair: tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]
    ) -> None:
        """A normal reading should not be flagged as anomalous."""
        dc, fe, detector, asset_id = calibrated_pair
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)

        result = detector.detect(fv)
        # Normal reading from same distribution should mostly not be anomalous
        # (some false positives expected with Isolation Forest, but shouldn't be extreme)
        assert result.anomaly_score < 0.9, f"Normal reading scored too high: {result.anomaly_score}"

    def test_anomalous_spike_detected(
        self, calibrated_pair: tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]
    ) -> None:
        """An artificially extreme reading should be flagged."""
        dc, fe, detector, asset_id = calibrated_pair

        # Create an extreme sensor reading
        extreme_reading = SensorReading(
            asset_id=asset_id,
            asset_type=AssetType.MOTOR,
            timestamp=datetime.now(timezone.utc),
            vibration_x=25.0,  # Extremely high
            vibration_y=20.0,
            vibration_z=22.0,
            temperature=120.0,  # Over limit
            pressure=8.0,
            current=30.0,
            rpm=1750.0,
        )

        # Feed through feature engineer (needs buffer history)
        for _ in range(5):
            normal = dc.collect_reading(asset_id)
            fe.compute_features(normal)

        fv = fe.compute_features(extreme_reading)
        result = detector.detect(fv)

        # Should be flagged as anomalous
        assert result.is_anomaly is True, "Extreme reading should be detected as anomalous"
        assert result.anomaly_score > 0.3, f"Anomaly score too low: {result.anomaly_score}"

    def test_uncalibrated_asset_returns_safe_result(
        self, detector: AnomalyDetector
    ) -> None:
        """Uncalibrated assets should return a safe 'uncalibrated' result."""
        fv = FeatureVector(
            asset_id="UNCAL-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=2.0,
            temperature_current=70.0,
        )
        result = detector.detect(fv)
        assert result.anomaly_type == "uncalibrated"
        assert result.is_anomaly is False

    def test_result_has_z_scores(
        self, calibrated_pair: tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]
    ) -> None:
        """Detection result should include per-feature Z-scores."""
        dc, fe, detector, asset_id = calibrated_pair
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)

        result = detector.detect(fv)
        assert len(result.z_scores) > 0
        assert isinstance(result.z_scores, dict)

    def test_anomaly_type_classification(
        self, calibrated_pair: tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]
    ) -> None:
        """Anomaly type should be one of the valid categories."""
        dc, fe, detector, asset_id = calibrated_pair
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)

        result = detector.detect(fv)
        valid_types = {"none", "statistical", "ml", "both", "uncalibrated"}
        assert result.anomaly_type in valid_types


# ── Edge case tests ─────────────────────────────────────────────────────


class TestAnomalyEdgeCases:
    """Edge cases for anomaly detection."""

    def test_single_calibration_sample(self, detector: AnomalyDetector) -> None:
        """Single sample shouldn't complete calibration."""
        fv = FeatureVector(
            asset_id="EDGE-001",
            timestamp=datetime.now(timezone.utc),
            vib_composite_rms=2.0,
            temperature_current=70.0,
            pressure_current=4.0,
        )
        result = detector.add_calibration_sample(fv)
        assert result is False

    def test_detect_returns_all_fields(
        self, calibrated_pair: tuple[DataCollector, FeatureEngineer, AnomalyDetector, str]
    ) -> None:
        """AnomalyResult should have all expected fields populated."""
        dc, fe, detector, asset_id = calibrated_pair
        reading = dc.collect_reading(asset_id)
        fv = fe.compute_features(reading)

        result = detector.detect(fv)
        assert result.asset_id == asset_id
        assert result.timestamp is not None
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.anomaly_score, float)
        assert isinstance(result.statistical_anomaly, bool)
        assert isinstance(result.ml_anomaly, bool)
        assert isinstance(result.contributing_features, list)
        assert isinstance(result.details, dict)
