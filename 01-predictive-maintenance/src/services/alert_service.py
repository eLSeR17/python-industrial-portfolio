"""Multi-tier alert service for predictive maintenance alerts.

Manages the lifecycle of alerts from generation through resolution.
Implements a tiered alerting system aligned with ISA-18.2 alarm
management standards:

Tier 1 — Threshold Alerts (immediate, deterministic):
  Triggered when sensor readings exceed physical limits.
  No ML required; fast and interpretable.

Tier 2 — Trend Alerts (statistical):
  Triggered when sensor trends indicate developing faults.
  Uses rolling statistics and rate-of-change analysis.

Tier 3 — ML Alerts (predictive):
  Triggered by anomaly detection and failure prediction outputs.
  Catches complex multi-sensor patterns invisible to thresholds.

Each tier has configurable severity escalation rules and cooldown
periods to prevent alarm flooding (a major concern in industrial
control rooms per ISA-18.2).
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from src.models.schemas import Alert, AlertStatus, Severity
from src.services.anomaly_detector import AnomalyResult
from src.services.failure_predictor import PredictionResult
from src.utils.helpers import utc_now

logger = logging.getLogger(__name__)


@dataclass
class AlertCooldown:
    """Tracks cooldown state for an asset+metric combination.

    Prevents alarm flooding by suppressing repeated alerts for the
    same condition within the cooldown window.
    """

    asset_id: str
    metric_name: str
    last_alert_time: object  # datetime
    cooldown_seconds: float = 300.0  # 5 minutes default

    def is_cooled_down(self) -> bool:
        """Check if enough time has passed since the last alert."""
        elapsed = (utc_now() - self.last_alert_time).total_seconds()
        return elapsed >= self.cooldown_seconds


class AlertService:
    """Manages alert generation, escalation, and lifecycle.

    The service deduplicates alerts, enforces cooldown periods,
    and maintains the active alert inventory. It processes outputs
    from all three detection tiers (threshold, trend, ML) and
    produces a unified alert feed.

    Usage:
        service = AlertService()
        # From threshold check:
        alerts = service.check_thresholds(reading)
        # From anomaly detection:
        alerts += service.check_anomaly(anomaly_result)
        # From failure prediction:
        alerts += service.check_prediction(prediction_result)
    """

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        """Initialize the alert service.

        Args:
            cooldown_seconds: Minimum seconds between repeated alerts
                             for the same asset+metric combination.
        """
        self._cooldown_seconds = cooldown_seconds
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._cooldowns: dict[str, AlertCooldown] = {}
        self._max_history: int = 10000

    def _cooldown_key(self, asset_id: str, metric: str) -> str:
        """Generate a unique key for cooldown tracking."""
        return f"{asset_id}:{metric}"

    def _check_cooldown(self, asset_id: str, metric: str) -> bool:
        """Check if an alert for this asset+metric is allowed (not in cooldown).

        Returns:
            True if the alert is allowed (not suppressed).
        """
        key = self._cooldown_key(asset_id, metric)
        cooldown = self._cooldowns.get(key)
        if cooldown is None:
            return True
        return cooldown.is_cooled_down()

    def _record_cooldown(self, asset_id: str, metric: str) -> None:
        """Record the time of the latest alert for cooldown tracking."""
        key = self._cooldown_key(asset_id, metric)
        self._cooldowns[key] = AlertCooldown(
            asset_id=asset_id,
            metric_name=metric,
            last_alert_time=utc_now(),
            cooldown_seconds=self._cooldown_seconds,
        )

    def _create_alert(
        self,
        asset_id: str,
        severity: Severity,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        recommended_action: str = "",
    ) -> Alert:
        """Create and register a new alert.

        Deduplication: if an ACTIVE alert already exists for this
        asset+metric at the same severity, it's not duplicated.
        Instead, the existing alert's timestamp is updated.
        """
        # Check for existing active alert
        for existing in self._active_alerts.values():
            if (
                existing.asset_id == asset_id
                and existing.metric_name == metric_name
                and existing.severity == severity
                and existing.status == AlertStatus.ACTIVE
            ):
                # Update timestamp instead of creating duplicate
                self._active_alerts[existing.alert_id] = existing.model_copy(
                    update={"timestamp": utc_now()}
                )
                return self._active_alerts[existing.alert_id]

        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:12].upper()}",
            asset_id=asset_id,
            timestamp=utc_now(),
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            recommended_action=recommended_action,
        )

        self._active_alerts[alert.alert_id] = alert
        self._alert_history.append(alert)
        self._record_cooldown(asset_id, metric_name)

        logger.warning(
            "ALERT [%s] %s %s: %s (value=%.2f, threshold=%.2f)",
            severity.value.upper(), asset_id, metric_name, message,
            metric_value, threshold,
        )
        return alert

    def check_thresholds(self, reading: "SensorReading") -> list[Alert]:
        """Tier 1: Check sensor readings against physical thresholds.

        Evaluates each sensor channel against warning and critical
        thresholds. Returns new alerts for any exceeded thresholds.

        Args:
            reading: Latest sensor reading from the data collector.

        Returns:
            List of new alerts generated (empty if no violations).
        """
        from src.models.schemas import SensorReading  # avoid circular

        alerts: list[Alert] = []
        checks = [
            ("temperature", reading.temperature, "temperature_warning", "temperature_critical",
             "°C", "Check cooling system, verify ambient temperature"),
            ("vibration_x", reading.vibration_x, "vibration_warning", "vibration_critical",
             "mm/s", "Inspect bearings, check alignment and balance"),
            ("vibration_y", reading.vibration_y, "vibration_warning", "vibration_critical",
             "mm/s", "Inspect bearings, check alignment and balance"),
            ("vibration_z", reading.vibration_z, "vibration_warning", "vibration_critical",
             "mm/s", "Inspect bearings, check alignment and balance"),
            ("pressure", reading.pressure, "pressure_warning", "pressure_critical",
             "bar", "Check seals and gaskets, verify pressure relief valves"),
        ]

        for name, value, warn_attr, crit_attr, unit, action in checks:
            # Import here to avoid circular at module level
            from config.settings import get_settings
            settings = get_settings()
            warn_threshold = getattr(settings.alerts, warn_attr)
            crit_threshold = getattr(settings.alerts, crit_attr)

            if not self._check_cooldown(reading.asset_id, name):
                continue

            if value >= crit_threshold:
                alerts.append(self._create_alert(
                    asset_id=reading.asset_id,
                    severity=Severity.CRITICAL,
                    message=f"{name} at critical level: {value:.1f}{unit} (limit: {crit_threshold}{unit})",
                    metric_name=name,
                    metric_value=value,
                    threshold=crit_threshold,
                    recommended_action=action,
                ))
            elif value >= warn_threshold:
                alerts.append(self._create_alert(
                    asset_id=reading.asset_id,
                    severity=Severity.WARNING,
                    message=f"{name} approaching limit: {value:.1f}{unit} (warning: {warn_threshold}{unit})",
                    metric_name=name,
                    metric_value=value,
                    threshold=warn_threshold,
                    recommended_action=f"Monitor {name} closely, schedule inspection",
                ))

        return alerts

    def check_anomaly(self, result: AnomalyResult) -> list[Alert]:
        """Tier 3: Generate alerts from ML anomaly detection results.

        Translates anomaly detection output into maintenance alerts
        with severity based on anomaly score and type.

        Args:
            result: Anomaly detection result from AnomalyDetector.

        Returns:
            List of alerts (empty if no anomaly detected).
        """
        alerts: list[Alert] = []
        if not result.is_anomaly:
            return alerts

        metric = "anomaly_score"
        if not self._check_cooldown(result.asset_id, metric):
            return alerts

        if result.anomaly_type == "both":
            severity = Severity.CRITICAL
            score_threshold = 0.7
        elif result.anomaly_type == "ml":
            severity = Severity.WARNING
            score_threshold = 0.5
        else:
            severity = Severity.WARNING
            score_threshold = 0.3

        if result.anomaly_score >= score_threshold:
            feature_list = ", ".join(result.contributing_features[:5])
            alerts.append(self._create_alert(
                asset_id=result.asset_id,
                severity=severity,
                message=(
                    f"ML anomaly detected (score={result.anomaly_score:.3f}, "
                    f"type={result.anomaly_type}). Contributing features: {feature_list}"
                ),
                metric_name=metric,
                metric_value=result.anomaly_score,
                threshold=score_threshold,
                recommended_action="Investigate sensor readings and recent maintenance history",
            ))

        return alerts

    def check_prediction(self, prediction: PredictionResult) -> list[Alert]:
        """Tier 3: Generate alerts from failure prediction results.

        Maps prediction probability and severity to maintenance alerts.
        Higher probabilities and shorter RUL produce higher-severity alerts.

        Args:
            prediction: Failure prediction from FailurePredictor.

        Returns:
            List of alerts generated from the prediction.
        """
        alerts: list[Alert] = []
        metric = "failure_probability"
        if not self._check_cooldown(prediction.asset_id, metric):
            return alerts

        prob = prediction.failure_probability
        rul = prediction.remaining_useful_life_hours

        if prob >= 0.7 or rul < 4:
            alerts.append(self._create_alert(
                asset_id=prediction.asset_id,
                severity=Severity.EMERGENCY,
                message=(
                    f"Imminent failure predicted: {prob:.0%} probability, "
                    f"RUL={rul:.1f}h. Mode: {prediction.failure_mode}"
                ),
                metric_name=metric,
                metric_value=prob,
                threshold=0.7,
                recommended_action=prediction.recommended_action,
            ))
        elif prob >= 0.3 or rul < 24:
            alerts.append(self._create_alert(
                asset_id=prediction.asset_id,
                severity=Severity.WARNING,
                message=(
                    f"Elevated failure risk: {prob:.0%} probability, "
                    f"RUL={rul:.1f}h. Mode: {prediction.failure_mode}"
                ),
                metric_name=metric,
                metric_value=prob,
                threshold=0.3,
                recommended_action=prediction.recommended_action,
            ))

        return alerts

    def acknowledge_alert(self, alert_id: str) -> Alert | None:
        """Acknowledge an active alert.

        Transitions the alert from ACTIVE to ACKNOWLEDGED state.
        This is typically done by an operator in the control room.

        Returns:
            Updated Alert or None if not found.
        """
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            return None
        updated = alert.model_copy(update={"status": AlertStatus.ACKNOWLEDGED})
        self._active_alerts[alert_id] = updated
        return updated

    def resolve_alert(self, alert_id: str) -> Alert | None:
        """Resolve an alert after maintenance action is taken.

        Returns:
            Updated Alert or None if not found.
        """
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            return None
        updated = alert.model_copy(update={"status": AlertStatus.RESOLVED})
        self._active_alerts[alert_id] = updated
        # Move to history
        self._alert_history.append(updated)
        # Remove from active
        del self._active_alerts[alert_id]
        return updated

    def get_active_alerts(
        self,
        asset_id: str | None = None,
        min_severity: Severity | None = None,
    ) -> list[Alert]:
        """Retrieve currently active alerts with optional filtering.

        Args:
            asset_id: Filter to specific asset (None = all assets).
            min_severity: Only return alerts at or above this severity.

        Returns:
            List of active alerts, newest first.
        """
        severity_order = {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.CRITICAL: 2,
            Severity.EMERGENCY: 3,
        }
        min_sev_val = severity_order.get(min_severity, 0) if min_severity else 0

        alerts = list(self._active_alerts.values())
        if asset_id:
            alerts = [a for a in alerts if a.asset_id == asset_id]
        if min_severity:
            alerts = [a for a in alerts if severity_order.get(a.severity, 0) >= min_sev_val]

        # Sort by severity (highest first), then by time (newest first)
        alerts.sort(
            key=lambda a: (-severity_order.get(a.severity, 0), a.timestamp),
            reverse=False,
        )
        return alerts

    def get_alert_count_by_severity(self) -> dict[str, int]:
        """Return count of active alerts grouped by severity.

        Useful for dashboard summary displays.
        """
        counts: dict[str, int] = defaultdict(int)
        for alert in self._active_alerts.values():
            counts[alert.severity.value] += 1
        return dict(counts)
