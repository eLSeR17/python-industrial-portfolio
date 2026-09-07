"""Failure prediction engine using Random Forest regression and classification.

Combines two complementary ML models:

1. **Time-to-Failure Regressor** (RandomForestRegressor):
   Predicts remaining useful life (RUL) in hours. Trained on historical
   feature vectors with known failure times. Uses features like vibration
   trend, temperature drift, and spectral entropy degradation.

2. **Failure Classifier** (RandomForestClassifier):
   Predicts probability of failure within the next N hours (configurable,
   default 24h). Provides a binary risk assessment for alert triggering.

Both models use the same feature vector input and share preprocessing
(StandardScaler) but are trained on different target variables:
- Regressor target: hours until failure (continuous)
- Classifier target: will_fail_within_24h (binary)

The pipeline handles cold-start gracefully: before enough data is
collected to train, it falls back to physics-based degradation estimates.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.models.schemas import Severity
from src.services.feature_engineer import FeatureVector
from src.utils.helpers import format_duration_hours

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Combined failure prediction output.

    Merges regression (RUL) and classification (probability) outputs
    into a single actionable prediction with maintenance recommendations.
    """

    asset_id: str
    timestamp: object  # datetime
    failure_probability: float  # 0.0–1.0 from classifier
    remaining_useful_life_hours: float  # from regressor
    confidence: float  # 0.0–1.0 based on model certainty
    failure_mode: str  # Predicted dominant failure mode
    recommended_action: str
    severity: Severity
    model_metrics: dict[str, float] = field(default_factory=dict)


class FailurePredictor:
    """Dual-model failure prediction engine.

    Lifecycle:
    1. Collect training data during normal operation + known failures
    2. Train models once enough labeled data exists (≥50 samples with labels)
    3. Predict on new feature vectors
    4. Retrain periodically as new failure events are labeled

    Before training, falls back to a physics-based RUL estimate using
    the exponential degradation model from ISO 13373.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        min_samples_split: int = 5,
        prediction_horizon_hours: float = 24.0,
        min_training_samples: int = 50,
    ) -> None:
        """Initialize the failure prediction engine.

        Args:
            n_estimators: Trees in each Random Forest.
            max_depth: Maximum tree depth to prevent overfitting.
            min_samples_split: Minimum samples to split a node.
            prediction_horizon_hours: Classification target window.
            min_training_samples: Minimum labeled samples to train.
        """
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._prediction_horizon = prediction_horizon_hours
        self._min_training = min_training_samples

        # Models (one per asset or global)
        self._regressors: dict[str, RandomForestRegressor] = {}
        self._classifiers: dict[str, RandomForestClassifier] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._is_trained: dict[str, bool] = {}

        # Training data accumulators
        self._X_train: dict[str, list[NDArray[np.float64]]] = {}
        self._y_rul: dict[str, list[float]] = {}
        self._y_fail: dict[str, list[int]] = {}

        # Model performance metrics
        self._metrics: dict[str, dict[str, float]] = {}

    def add_training_sample(
        self,
        fv: FeatureVector,
        time_to_failure_hours: float,
        failed_within_horizon: bool,
        asset_id: str | None = None,
    ) -> bool:
        """Add a labeled training sample.

        Args:
            fv: Feature vector from a known point in time.
            time_to_failure_hours: Actual hours until failure (or infinity if healthy).
            failed_within_horizon: Whether failure occurred within prediction horizon.
            asset_id: Asset ID (uses fv.asset_id if None).

        Returns:
            True if models were (re)trained with the new data.
        """
        aid = asset_id or fv.asset_id
        if aid not in self._X_train:
            self._X_train[aid] = []
            self._y_rul[aid] = []
            self._y_fail[aid] = []

        self._X_train[aid].append(fv.to_array())
        # Cap RUL at a reasonable maximum (720 hours = 30 days)
        capped_rul = min(time_to_failure_hours, 720.0)
        self._y_rul[aid].append(capped_rul)
        self._y_fail[aid].append(1 if failed_within_horizon else 0)

        # Auto-train when enough data
        n_samples = len(self._X_train[aid])
        if n_samples >= self._min_training and n_samples % 10 == 0:
            return self.train(aid)

        return False

    def train(self, asset_id: str) -> bool:
        """Train both regressor and classifier for an asset.

        Uses cross-validation to estimate performance and logs metrics.
        Falls back to default models if training fails.

        Args:
            asset_id: Asset to train models for.

        Returns:
            True if training succeeded.
        """
        if asset_id not in self._X_train or len(self._X_train[asset_id]) < self._min_training:
            logger.warning(
                "Insufficient training data for %s: need %d, have %d",
                asset_id, self._min_training,
                len(self._X_train.get(asset_id, [])),
            )
            return False

        X = np.array(self._X_train[asset_id])
        y_rul = np.array(self._y_rul[asset_id])
        y_fail = np.array(self._y_fail[asset_id])

        # Check for degenerate cases
        if len(np.unique(y_fail)) < 2:
            logger.warning(
                "Cannot train classifier for %s: only one class present", asset_id
            )
            return False

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scalers[asset_id] = scaler

        # Train regressor (RUL prediction)
        regressor = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        regressor.fit(X_scaled, y_rul)
        self._regressors[asset_id] = regressor

        # Train classifier (failure probability)
        classifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",  # Handle imbalanced failure data
        )
        classifier.fit(X_scaled, y_fail)
        self._classifiers[asset_id] = classifier

        # Cross-validation metrics
        cv_folds = min(5, len(X) // 5) if len(X) >= 10 else 2
        rul_cv = cross_val_score(regressor, X_scaled, y_rul, cv=cv_folds, scoring="neg_mean_absolute_error")
        fail_cv = cross_val_score(classifier, X_scaled, y_fail, cv=cv_folds, scoring="f1")

        self._metrics[asset_id] = {
            "rul_mae_cv": float(-np.mean(rul_cv)),
            "rul_mae_std": float(np.std(rul_cv)),
            "fail_f1_cv": float(np.mean(fail_cv)),
            "fail_f1_std": float(np.std(fail_cv)),
            "n_training_samples": float(len(X)),
        }

        self._is_trained[asset_id] = True
        logger.info(
            "Trained models for %s: RUL MAE=%.1fh, Fail F1=%.3f (n=%d)",
            asset_id,
            self._metrics[asset_id]["rul_mae_cv"],
            self._metrics[asset_id]["fail_f1_cv"],
            len(X),
        )
        return True

    def _physics_based_rul(self, fv: FeatureVector) -> float:
        """Fallback RUL estimate using exponential degradation model.

        When ML models aren't trained yet, estimate RUL from vibration
        trend. Based on ISO 13373 condition monitoring standards:
        - Healthy vibration < 2.8 mm/s
        - Alarm at 4.5 mm/s
        - Danger at 7.1 mm/s
        - Failure predicted at 11.2 mm/s

        Uses the current vibration RMS and its trend (rate of change)
        to project forward to the failure threshold.
        """
        # Current vibration severity
        vib = fv.vib_composite_rms
        vib_trend = fv.rolling_std_10 - fv.rolling_std_50 if fv.rolling_std_50 > 0 else 0.0

        # Thresholds from ISO 10816
        warning_level = 4.5
        danger_level = 7.1
        failure_level = 11.2

        if vib < warning_level:
            # Healthy — long RUL
            margin = warning_level - vib
            rate = max(vib_trend, 0.01)  # Minimum degradation rate
            hours_to_warning = margin / rate
            return min(720.0, hours_to_warning + 48.0)  # +48h buffer after warning
        elif vib < danger_level:
            # Warning — moderate RUL
            margin = danger_level - vib
            rate = max(vib_trend, 0.05)
            return max(0.0, margin / rate)
        elif vib < failure_level:
            # Danger — short RUL
            margin = failure_level - vib
            rate = max(vib_trend, 0.1)
            return max(0.0, margin / rate)
        else:
            # Beyond failure threshold
            return 0.0

    def _determine_failure_mode(self, fv: FeatureVector) -> str:
        """Infer the most likely failure mode from feature patterns.

        Uses heuristics based on vibration/thermal/pressure signatures:
        - High vibration + normal temp → bearing wear
        - High vibration + high temp → lubrication failure
        - Low pressure + normal vibration → seal/gasket leak
        - High current + high temp → electrical fault
        - High spectral entropy → multi-fault (advanced degradation)
        """
        vib = fv.vib_composite_rms
        temp = fv.temperature_current
        pres = fv.pressure_current
        curr = fv.power_estimate
        entropy = fv.spectral_entropy

        # Decision tree heuristics
        if vib > 5.0 and temp > 85.0:
            return "lubrication_failure"
        elif vib > 5.0 and temp <= 85.0:
            return "bearing_wear"
        elif pres < 2.0 and vib < 3.0:
            return "seal_leakage"
        elif curr > 18.0 and temp > 80.0:
            return "electrical_fault"
        elif entropy > 0.7:
            return "multi_fault_degradation"
        elif fv.harmonic_ratio > 0.3:
            return "misalignment"
        elif fv.fft_peak_frequency > 50.0:
            return "imbalance"
        else:
            return "general_wear"

    def _recommend_action(
        self, probability: float, rul_hours: float, failure_mode: str
    ) -> tuple[str, Severity]:
        """Generate maintenance recommendation based on prediction.

        Returns:
            Tuple of (action_text, severity_level).
        """
        if probability > 0.8 or rul_hours < 4:
            return (
                f"IMMEDIATE SHUTDOWN recommended. {failure_mode.replace('_', ' ').title()} "
                f"detected with {probability:.0%} confidence. RUL: {format_duration_hours(rul_hours)}. "
                f"Schedule emergency maintenance within {format_duration_hours(min(rul_hours, 2))}.",
                Severity.EMERGENCY,
            )
        elif probability > 0.5 or rul_hours < 12:
            return (
                f"Plan maintenance within {format_duration_hours(rul_hours)}. "
                f"Probable cause: {failure_mode.replace('_', ' ')}. "
                f"Monitor closely — reduce load if possible.",
                Severity.CRITICAL,
            )
        elif probability > 0.3 or rul_hours < 48:
            return (
                f"Schedule preventive maintenance within {format_duration_hours(rul_hours)}. "
                f"Early signs of {failure_mode.replace('_', ' ')}. "
                f"Order spare parts and coordinate with production schedule.",
                Severity.WARNING,
            )
        else:
            return (
                f"Continue monitoring. Minor signs of {failure_mode.replace('_', ' ')}. "
                f"Next planned maintenance window is sufficient.",
                Severity.INFO,
            )

    def predict(self, fv: FeatureVector) -> PredictionResult:
        """Generate failure prediction for a feature vector.

        If ML models are trained, uses them. Otherwise falls back to
        physics-based estimation with reduced confidence.

        Args:
            fv: Engineered feature vector.

        Returns:
            PredictionResult with RUL, probability, and recommendations.
        """
        asset_id = fv.asset_id

        if self._is_trained.get(asset_id, False):
            # Use ML models
            features = fv.to_array().reshape(1, -1)
            X_scaled = self._scalers[asset_id].transform(features)

            rul = float(self._regressors[asset_id].predict(X_scaled)[0])
            rul = max(0.0, min(720.0, rul))

            proba = self._classifiers[asset_id].predict_proba(X_scaled)[0]
            failure_prob = float(proba[1]) if len(proba) > 1 else 0.0

            # Confidence from classifier probability margin and regressor consistency
            margin = abs(proba[0] - proba[1]) if len(proba) > 1 else 0.5
            confidence = min(1.0, 0.5 + margin * 0.5)

            metrics = self._metrics.get(asset_id, {})
            model_metrics = {
                "method": "ml",
                "rul_mae_cv": metrics.get("rul_mae_cv", 0.0),
                "fail_f1_cv": metrics.get("fail_f1_cv", 0.0),
            }
        else:
            # Physics-based fallback
            rul = self._physics_based_rul(fv)
            # Convert RUL to probability using a sigmoid-like function
            failure_prob = 1.0 / (1.0 + math.exp((rul - 24) / 12))
            confidence = 0.3  # Low confidence for physics-based estimate
            model_metrics = {"method": "physics_fallback"}

        failure_mode = self._determine_failure_mode(fv)
        action, severity = self._recommend_action(failure_prob, rul, failure_mode)

        return PredictionResult(
            asset_id=asset_id,
            timestamp=fv.timestamp,
            failure_probability=round(min(1.0, max(0.0, failure_prob)), 4),
            remaining_useful_life_hours=round(rul, 1),
            confidence=round(confidence, 4),
            failure_mode=failure_mode,
            recommended_action=action,
            severity=severity,
            model_metrics=model_metrics,
        )

    def is_trained(self, asset_id: str) -> bool:
        """Check if ML models are trained for an asset."""
        return self._is_trained.get(asset_id, False)

    def get_metrics(self, asset_id: str) -> dict[str, float]:
        """Return training/evaluation metrics for an asset's models."""
        return self._metrics.get(asset_id, {})
