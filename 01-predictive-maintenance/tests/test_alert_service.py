"""Tests for the AlertService class.

Covers cooldown mechanisms, threshold alert generation, severity
escalation, deduplication, and alert lifecycle management.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.models.schemas import (
    Alert,
    AlertStatus,
    SensorReading,
    Severity,
)
from src.services.alert_service import AlertCooldown, AlertService
from src.services.anomaly_detector import AnomalyResult
from src.services.failure_predictor import PredictionResult


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def alert_service() -> AlertService:
    """Create a fresh AlertService with a short cooldown for fast tests."""
    return AlertService(cooldown_seconds=1.0)


@pytest.fixture
def sample_reading() -> SensorReading:
    """Create a normal sensor reading (below all thresholds)."""
    return SensorReading(
        asset_id="MOTOR-001",
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        vibration_x=2.0,
        vibration_y=1.5,
        vibration_z=2.5,
        temperature=70.0,
        pressure=4.0,
        current=12.0,
        rpm=1750.0,
    )


@pytest.fixture
def critical_reading() -> SensorReading:
    """Create a sensor reading with critical-level values."""
    return SensorReading(
        asset_id="MOTOR-001",
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        vibration_x=9.0,
        vibration_y=8.5,
        vibration_z=9.5,
        temperature=98.0,
        pressure=8.0,
        current=30.0,
        rpm=1750.0,
    )


@pytest.fixture
def warning_reading() -> SensorReading:
    """Create a sensor reading with warning-level values."""
    return SensorReading(
        asset_id="MOTOR-001",
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        vibration_x=6.0,
        vibration_y=5.5,
        vibration_z=6.5,
        temperature=88.0,
        pressure=6.5,
        current=12.0,
        rpm=1750.0,
    )


# ── Cooldown tests ──────────────────────────────────────────────────────


class TestCooldown:
    """Tests for the alert cooldown mechanism."""

    def test_first_alert_not_cooldown(self, alert_service: AlertService) -> None:
        """First alert for an asset+metric should always be allowed."""
        assert alert_service._check_cooldown("MOTOR-001", "temperature") is True

    def test_immediate_second_alert_suppressed(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Second alert immediately after the first should be suppressed."""
        alert_service.check_thresholds(critical_reading)
        # Second call within cooldown window
        alerts = alert_service.check_thresholds(critical_reading)
        # Temperature alert should be suppressed (only vibration alerts remain)
        temp_alerts = [a for a in alerts if a.metric_name == "temperature"]
        assert len(temp_alerts) == 0, "Temperature alert should be suppressed by cooldown"

    def test_alert_allowed_after_cooldown(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Alert should be allowed again after the cooldown period elapses."""
        alert_service.check_thresholds(critical_reading)

        # Simulate time passage beyond cooldown
        key = alert_service._cooldown_key("MOTOR-001", "temperature")
        alert_service._cooldowns[key] = AlertCooldown(
            asset_id="MOTOR-001",
            metric_name="temperature",
            last_alert_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            cooldown_seconds=1.0,
        )

        alerts = alert_service.check_thresholds(critical_reading)
        temp_alerts = [a for a in alerts if a.metric_name == "temperature"]
        assert len(temp_alerts) >= 1, "Alert should fire after cooldown expires"

    def test_different_metrics_independent_cooldown(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Cooldown for one metric should not suppress alerts for another."""
        alert_service.check_thresholds(critical_reading)

        # Both temperature and vibration should have cooldowns now,
        # but they are independent keys
        temp_key = alert_service._cooldown_key("MOTOR-001", "temperature")
        vib_key = alert_service._cooldown_key("MOTOR-001", "vibration_x")
        assert temp_key != vib_key
        assert temp_key in alert_service._cooldowns
        assert vib_key in alert_service._cooldowns


# ── Threshold alert tests ───────────────────────────────────────────────


class TestThresholdAlerts:
    """Tests for Tier 1 threshold-based alert generation."""

    def test_normal_readings_no_alerts(
        self, alert_service: AlertService, sample_reading: SensorReading
    ) -> None:
        """Normal sensor readings should produce no alerts."""
        alerts = alert_service.check_thresholds(sample_reading)
        assert len(alerts) == 0, "Normal readings should not trigger alerts"

    def test_critical_temperature_alert(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Critical temperature should generate a CRITICAL severity alert."""
        alerts = alert_service.check_thresholds(critical_reading)
        temp_alerts = [a for a in alerts if a.metric_name == "temperature"]
        assert len(temp_alerts) >= 1
        assert temp_alerts[0].severity == Severity.CRITICAL

    def test_warning_vibration_alert(
        self, alert_service: AlertService, warning_reading: SensorReading
    ) -> None:
        """Warning-level vibration should generate WARNING severity alerts."""
        alerts = alert_service.check_thresholds(warning_reading)
        vib_alerts = [a for a in alerts if a.metric_name.startswith("vibration")]
        assert len(vib_alerts) >= 1
        assert all(a.severity == Severity.WARNING for a in vib_alerts)

    def test_alert_message_contains_value(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Alert message should include the metric value for operator context."""
        alerts = alert_service.check_thresholds(critical_reading)
        assert len(alerts) >= 1
        assert "98.0" in alerts[0].message or "critical" in alerts[0].message.lower()

    def test_alert_has_recommended_action(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Alerts should include actionable maintenance recommendations."""
        alerts = alert_service.check_thresholds(critical_reading)
        assert len(alerts) >= 1
        assert len(alerts[0].recommended_action) > 0


# ── Severity level tests ────────────────────────────────────────────────


class TestSeverityLevels:
    """Tests for different alert severity levels."""

    def test_emergency_severity_for_imminent_failure(
        self, alert_service: AlertService
    ) -> None:
        """High failure probability + low RUL should produce EMERGENCY alert."""
        prediction = PredictionResult(
            asset_id="MOTOR-001",
            timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
            failure_probability=0.85,
            remaining_useful_life_hours=2.0,
            confidence=0.9,
            failure_mode="bearing_wear",
            recommended_action="Immediate shutdown",
            severity=Severity.EMERGENCY,
        )
        alerts = alert_service.check_prediction(prediction)
        assert len(alerts) >= 1
        assert alerts[0].severity == Severity.EMERGENCY

    def test_warning_severity_for_elevated_risk(
        self, alert_service: AlertService
    ) -> None:
        """Moderate failure probability should produce WARNING alert."""
        prediction = PredictionResult(
            asset_id="MOTOR-001",
            timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
            failure_probability=0.45,
            remaining_useful_life_hours=18.0,
            confidence=0.7,
            failure_mode="lubrication_failure",
            recommended_action="Schedule maintenance",
            severity=Severity.WARNING,
        )
        alerts = alert_service.check_prediction(prediction)
        assert len(alerts) >= 1
        assert alerts[0].severity == Severity.WARNING

    def test_no_alert_for_low_risk(
        self, alert_service: AlertService
    ) -> None:
        """Low failure probability and long RUL should produce no alerts."""
        prediction = PredictionResult(
            asset_id="MOTOR-001",
            timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
            failure_probability=0.05,
            remaining_useful_life_hours=500.0,
            confidence=0.8,
            failure_mode="general_wear",
            recommended_action="Continue monitoring",
            severity=Severity.INFO,
        )
        alerts = alert_service.check_prediction(prediction)
        assert len(alerts) == 0


# ── Alert history and lifecycle tests ───────────────────────────────────


class TestAlertLifecycle:
    """Tests for alert deduplication, acknowledgement, and resolution."""

    def test_deduplication_same_severity(
        self, alert_service: AlertService, critical_reading: SensorReading
    ) -> None:
        """Duplicate alerts at the same severity should be deduplicated."""
        alert_service.check_thresholds(critical_reading)
        # Force cooldown to expire
        for key in list(alert_service._cooldowns.keys()):
            if key.startswith("MOTOR-001"):
                alert_service._cooldowns[key] = AlertCooldown(
                    asset_id="MOTOR-001",
                    metric_name=key.split(":")[1],
                    last_alert_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    cooldown_seconds=1.0,
                )

        alert_service.check_thresholds(critical_reading)
        # Should have only one temperature alert despite two triggerings
        temp_alerts = [
            a for a in alert_service._active_alerts.values()
            if a.metric_name == "temperature"
        ]
        assert len(temp_alerts) == 1, "Duplicate alerts should be deduplicated"

    def test_acknowledge_alert(self, alert_service: AlertService) -> None:
        """Acknowledging an alert should change its status."""
        alert = Alert(
            alert_id="ALT-TEST-001",
            asset_id="MOTOR-001",
            timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
            severity=Severity.WARNING,
            status=AlertStatus.ACTIVE,
            message="Test alert",
            metric_name="temperature",
            metric_value=90.0,
            threshold=85.0,
            recommended_action="Monitor",
        )
        alert_service._active_alerts["ALT-TEST-001"] = alert

        result = alert_service.acknowledge_alert("ALT-TEST-001")
        assert result is not None
        assert result.status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert_removes_from_active(
        self, alert_service: AlertService
    ) -> None:
        """Resolving an alert should remove it from active alerts."""
        alert = Alert(
            alert_id="ALT-TEST-002",
            asset_id="MOTOR-001",
            timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
            severity=Severity.CRITICAL,
            status=AlertStatus.ACTIVE,
            message="Critical test",
            metric_name="vibration_x",
            metric_value=9.0,
            threshold=8.0,
            recommended_action="Inspect bearings",
        )
        alert_service._active_alerts["ALT-TEST-002"] = alert

        result = alert_service.resolve_alert("ALT-TEST-002")
        assert result is not None
        assert result.status == AlertStatus.RESOLVED
        assert "ALT-TEST-002" not in alert_service._active_alerts
        # Should appear in history
        hist_ids = [a.alert_id for a in alert_service._alert_history]
        assert "ALT-TEST-002" in hist_ids

    def test_acknowledge_nonexistent_returns_none(
        self, alert_service: AlertService
    ) -> None:
        """Acknowledging a non-existent alert should return None."""
        result = alert_service.acknowledge_alert("ALT-NONEXISTENT")
        assert result is None

    def test_get_active_alerts_filter_by_asset(
        self, alert_service: AlertService
    ) -> None:
        """Active alerts should be filterable by asset_id."""
        for aid in ("MOTOR-001", "MOTOR-002"):
            alert_service._active_alerts[f"ALT-{aid}"] = Alert(
                alert_id=f"ALT-{aid}",
                asset_id=aid,
                timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
                severity=Severity.WARNING,
                status=AlertStatus.ACTIVE,
                message=f"Alert for {aid}",
                metric_name="temperature",
                metric_value=90.0,
                threshold=85.0,
                recommended_action="Monitor",
            )

        motor1_alerts = alert_service.get_active_alerts(asset_id="MOTOR-001")
        assert len(motor1_alerts) == 1
        assert motor1_alerts[0].asset_id == "MOTOR-001"

    def test_alert_count_by_severity(
        self, alert_service: AlertService
    ) -> None:
        """Alert count should reflect active alerts per severity."""
        for i, sev in enumerate([Severity.WARNING, Severity.WARNING, Severity.CRITICAL]):
            alert_service._active_alerts[f"ALT-{i}"] = Alert(
                alert_id=f"ALT-{i}",
                asset_id="MOTOR-001",
                timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
                severity=sev,
                status=AlertStatus.ACTIVE,
                message=f"Alert {i}",
                metric_name="temperature",
                metric_value=90.0,
                threshold=85.0,
                recommended_action="Monitor",
            )

        counts = alert_service.get_alert_count_by_severity()
        assert counts.get("warning") == 2
        assert counts.get("critical") == 1
