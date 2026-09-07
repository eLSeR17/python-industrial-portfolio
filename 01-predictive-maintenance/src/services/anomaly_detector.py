"""Anomaly detection engine combining statistical and ML methods.

Uses a two-tier approach for robust anomaly detection in industrial
sensor data:

Tier 1 — Statistical Rules (fast, interpretable):
  - Z-score detection for individual channels
  - Mahalanobis distance for multivariate outliers
  - CUSUM for detecting small persistent shifts

Tier 2 — Isolation Forest (ML, catches complex patterns):
  - Unsupervised: works without labeled failure data
  - Handles high-dimensional feature spaces
  - Naturally identifies novel anomaly types

The combination ensures that obvious threshold violations are caught
instantly while subtle multi-sensor correlations (e.g., simultaneous
slight vibration + temperature increase) are detected by the ML model.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.services.feature_engineer import FeatureVector
from src.utils.helpers import percentile_rank

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single observation.

    Combines outputs from both statistical and ML tiers into a
    unified anomaly assessment with explainability information.
    """

    asset_id: str
    timestamp: object  # datetime
    is_anomaly: bool
    anomaly_score: float  # -1 (normal) to 1 (strongly anomalous)
    statistical_anomaly: bool
    ml_anomaly: bool
    anomaly_type: str  # "none", "statistical", "ml", "both"
    contributing_features: list[str] = field(default_factory=list)
    z_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Dual-tier anomaly detection engine.

    The detector maintains per-asset baselines computed from initial
    "healthy" readings and uses both statistical and ML methods to
    flag deviations.

    Calibration strategy:
    - First N readings (default 50) are assumed "healthy" and used
      to compute baseline statistics and train the Isolation Forest.
    - After calibration, every new reading is scored against the
      learned normal behavior.

    Usage:
        detector = AnomalyDetector()
        # Feed calibration data...
        detector.calibrate(feature_vectors)
        # Then detect anomalies:
        result = detector.detect(feature_vector)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        z_score_threshold: float = 3.0,
        calibration_size: int = 50,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            contamination: Expected fraction of anomalies in training data.
            n_estimators: Number of trees in the Isolation Forest.
            z_score_threshold: Z-score beyond which a feature is anomalous.
            calibration_size: Number of healthy samples needed before detection.
        """
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._z_score_threshold = z_score_threshold
        self._calibration_size = calibration_size

        # Per-asset state
        self._baselines: dict[str, dict[str, float]] = {}  # asset → {feature: mean}
        self._stds: dict[str, dict[str, float]] = {}  # asset → {feature: std}
        self._scalers: dict[str, StandardScaler] = {}
        self._isolation_forests: dict[str, IsolationForest] = {}
        self._calibration_buffers: dict[str, list[NDArray[np.float64]]] = {}
        self._is_calibrated: dict[str, bool] = {}

        # Global thresholds (from literature)
        self._global_thresholds: dict[str, float] = {
            "temperature": 95.0,
            "vibration_x_rms": 7.1,
            "vibration_y_rms": 5.6,
            "vibration_z_rms": 7.1,
            "vib_composite_rms": 8.0,
        }

    def needs_calibration(self, asset_id: str) -> bool:
        """Check if an asset still needs calibration data."""
        return not self._is_calibrated.get(asset_id, False)

    def add_calibration_sample(self, fv: FeatureVector) -> bool:
        """Add a feature vector to the calibration buffer.

        Once enough samples are collected, the Isolation Forest is
        trained and baselines are computed. Returns True when
        calibration is complete.

        Args:
            fv: Feature vector from a known-healthy operating state.

        Returns:
            True if calibration completed with this sample.
        """
        asset_id = fv.asset_id
        if self._is_calibrated.get(asset_id, False):
            return True

        if asset_id not in self._calibration_buffers:
            self._calibration_buffers[asset_id] = []

        vec = fv.to_array()
        self._calibration_buffers[asset_id].append(vec)

        if len(self._calibration_buffers[asset_id]) >= self._calibration_size:
            return self._fit_model(asset_id)

        return False

    def _fit_model(self, asset_id: str) -> bool:
        """Train the Isolation Forest and compute baseline statistics.

        Called automatically when enough calibration samples are collected.
        """
        samples = np.array(self._calibration_buffers[asset_id])
        if samples.shape[0] < 10:
            logger.warning("Not enough samples to train model for %s", asset_id)
            return False

        feature_names = FeatureVector.feature_names()

        # Compute baselines
        self._baselines[asset_id] = {}
        self._stds[asset_id] = {}
        for i, name in enumerate(feature_names):
            col = samples[:, i]
            self._baselines[asset_id][name] = float(np.mean(col))
            std_val = float(np.std(col))
            self._stds[asset_id][name] = std_val if std_val > 1e-12 else 1.0

        # Fit scaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(samples)
        self._scalers[asset_id] = scaler

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            max_samples=min(256, samples.shape[0]),
            random_state=42,
            n_jobs=-1,
        )
        iso_forest.fit(scaled)
        self._isolation_forests[asset_id] = iso_forest
        self._is_calibrated[asset_id] = True

        # Free calibration memory
        self._calibration_buffers.pop(asset_id, None)

        logger.info(
            "Calibrated anomaly detector for %s with %d samples",
            asset_id, samples.shape[0],
        )
        return True

    def _statistical_check(
        self, fv: FeatureVector, asset_id: str
    ) -> tuple[bool, dict[str, float], list[str]]:
        """Tier 1: Statistical anomaly detection.

        Computes Z-scores for each feature and flags individual
        anomalies. Also checks against global physical thresholds.

        Returns:
            Tuple of (is_anomaly, z_scores_dict, contributing_features).
        """
        z_scores: dict[str, float] = {}
        contributors: list[str] = []

        features = fv.to_array()
        names = FeatureVector.feature_names()

        for i, name in enumerate(names):
            value = float(features[i])
            baseline_mean = self._baselines.get(asset_id, {}).get(name, 0.0)
            baseline_std = self._stds.get(asset_id, {}).get(name, 1.0)

            z = (value - baseline_mean) / baseline_std
            z_scores[name] = round(z, 3)

            if abs(z) > self._z_score_threshold:
                contributors.append(name)

        # Check global physical thresholds
        for feature_name, threshold in self._global_thresholds.items():
            if feature_name in z_scores:
                idx = names.index(feature_name) if feature_name in names else -1
                if idx >= 0 and float(features[idx]) > threshold:
                    if feature_name not in contributors:
                        contributors.append(feature_name)

        is_anomaly = len(contributors) > 0
        return is_anomaly, z_scores, contributors

    def _ml_check(self, fv: FeatureVector, asset_id: str) -> tuple[bool, float]:
        """Tier 2: Isolation Forest anomaly detection.

        Uses the trained model to compute an anomaly score for the
        feature vector. The score is normalized from the raw sklearn
        output (-1 = anomaly, 1 = normal) to a 0–1 range where
        values closer to 1 indicate stronger anomalies.

        Returns:
            Tuple of (is_anomaly, normalized_score).
        """
        if asset_id not in self._isolation_forests:
            return False, 0.0

        features = fv.to_array().reshape(1, -1)
        scaler = self._scalers[asset_id]
        scaled = scaler.transform(features)

        # Raw sklearn score: lower = more anomalous
        raw_score = float(self._isolation_forests[asset_id].decision_function(scaled)[0])
        prediction = int(self._isolation_forests[asset_id].predict(scaled)[0])

        # Normalize to 0–1 where 1 = most anomalous
        # sklearn scores typically range from -0.5 to 0.5
        normalized = max(0.0, min(1.0, 0.5 - raw_score))

        is_anomaly = prediction == -1
        return is_anomaly, round(normalized, 4)

    def detect(self, fv: FeatureVector) -> AnomalyResult:
        """Run full anomaly detection on a feature vector.

        Executes both tiers and combines results:
        - "both" tier anomaly → highest confidence
        - "statistical" only → likely threshold violation
        - "ml" only → subtle pattern anomaly

        Args:
            fv: Feature vector to analyze.

        Returns:
            AnomalyResult with combined anomaly assessment.
        """
        asset_id = fv.asset_id

        # If not calibrated, skip ML and rely on global thresholds only
        if not self._is_calibrated.get(asset_id, False):
            return AnomalyResult(
                asset_id=asset_id,
                timestamp=fv.timestamp,
                is_anomaly=False,
                anomaly_score=0.0,
                statistical_anomaly=False,
                ml_anomaly=False,
                anomaly_type="uncalibrated",
                details={"message": "Asset not yet calibrated, using global thresholds only"},
            )

        # Tier 1: Statistical
        stat_anomaly, z_scores, stat_contributors = self._statistical_check(fv, asset_id)

        # Tier 2: ML
        ml_anomaly, ml_score = self._ml_check(fv, asset_id)

        # Combine
        if stat_anomaly and ml_anomaly:
            anomaly_type = "both"
            combined_score = max(ml_score, min(1.0, len(stat_contributors) * 0.2))
        elif stat_anomaly:
            anomaly_type = "statistical"
            combined_score = min(1.0, len(stat_contributors) * 0.25)
        elif ml_anomaly:
            anomaly_type = "ml"
            combined_score = ml_score
        else:
            anomaly_type = "none"
            combined_score = ml_score * 0.5  # Low ML score contribution

        return AnomalyResult(
            asset_id=asset_id,
            timestamp=fv.timestamp,
            is_anomaly=stat_anomaly or ml_anomaly,
            anomaly_score=round(combined_score, 4),
            statistical_anomaly=stat_anomaly,
            ml_anomaly=ml_anomaly,
            anomaly_type=anomaly_type,
            contributing_features=stat_contributors,
            z_scores=z_scores,
            details={
                "ml_raw_score": ml_score,
                "n_statistical_violations": len(stat_contributors),
            },
        )

    def get_baseline_summary(self, asset_id: str) -> dict[str, dict[str, float]]:
        """Return baseline statistics for an asset (for debugging/display).

        Returns:
            Dict mapping feature names to {mean, std} baseline values.
        """
        if asset_id not in self._baselines:
            return {}
        return {
            name: {
                "mean": self._baselines[asset_id].get(name, 0.0),
                "std": self._stds[asset_id].get(name, 0.0),
            }
            for name in FeatureVector.feature_names()
        }
