"""Tests for the anomaly detection engine."""

import math
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.services.anomaly_detector import (
    detect_baseline_shift,
    detect_consumption_spikes,
    detect_equipment_left_on,
    detect_pf_drops,
    run_full_anomaly_detection,
)
from src.services.meter_reader import generate_synthetic_readings


@pytest.fixture
def facility_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def clean_readings(facility_id: uuid.UUID) -> pd.DataFrame:
    """Clean readings with no injected anomalies."""
    end = datetime(2025, 3, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=14)
    return generate_synthetic_readings(
        facility_id=facility_id, start=start, end=end,
        interval_minutes=60, baseload_kw=100.0, peak_kw=300.0,
    )


@pytest.fixture
def spike_readings(facility_id: uuid.UUID) -> pd.DataFrame:
    """Readings with an injected demand spike."""
    df = generate_synthetic_readings(
        facility_id=facility_id,
        start=datetime(2025, 3, 1, tzinfo=timezone.utc) - timedelta(days=14),
        end=datetime(2025, 3, 1, tzinfo=timezone.utc),
        interval_minutes=60,
    )
    # Inject a massive spike at hour 100
    spike_idx = 100
    if spike_idx < len(df):
        df.loc[spike_idx, "demand_kw"] = 2000.0  # way above normal
    return df


@pytest.fixture
def pf_drop_readings(facility_id: uuid.UUID) -> pd.DataFrame:
    """Readings with sustained power factor drop."""
    df = generate_synthetic_readings(
        facility_id=facility_id,
        start=datetime(2025, 3, 1, tzinfo=timezone.utc) - timedelta(days=14),
        end=datetime(2025, 3, 1, tzinfo=timezone.utc),
        interval_minutes=60,
    )
    # Inject 5 consecutive low-PF readings
    for i in range(50, 55):
        if i < len(df):
            df.loc[i, "power_factor"] = 0.70
    return df


@pytest.fixture
def off_hours_readings(facility_id: uuid.UUID) -> pd.DataFrame:
    """Readings with high demand during off-hours (22:00-05:00)."""
    df = generate_synthetic_readings(
        facility_id=facility_id,
        start=datetime(2025, 3, 1, tzinfo=timezone.utc) - timedelta(days=14),
        end=datetime(2025, 3, 1, tzinfo=timezone.utc),
        interval_minutes=60,
    )
    # Inject off-hours waste
    for i in range(0, len(df)):
        ts = df.loc[i, "timestamp"]
        if hasattr(ts, "hour") and ts.hour >= 22:
            df.loc[i, "demand_kw"] = 250.0  # abnormally high off-hours
    return df


class TestSpikeDetection:
    """Tests for Z-score based spike detection."""

    def test_detects_injected_spike(self, spike_readings, facility_id):
        anomalies = detect_consumption_spikes(spike_readings, facility_id, zscore_threshold=3.0)
        types = [a.anomaly_type for a in anomalies]
        assert "spike" in types

    def test_clean_data_few_or_no_spikes(self, clean_readings, facility_id):
        anomalies = detect_consumption_spikes(clean_readings, facility_id, zscore_threshold=4.0)
        # Clean data should have very few or no extreme spikes
        assert len(anomalies) <= 5

    def test_anomaly_has_valid_fields(self, spike_readings, facility_id):
        anomalies = detect_consumption_spikes(spike_readings, facility_id)
        for a in anomalies:
            assert a.facility_id == facility_id
            assert a.anomaly_type == "spike"
            assert a.measured_value > 0
            assert a.expected_value > 0

    def test_empty_input(self, facility_id):
        result = detect_consumption_spikes(pd.DataFrame(), facility_id)
        assert result == []


class TestBaselineShiftDetection:
    """Tests for baseline shift detection."""

    def test_shift_detection_with_synthetic(self, facility_id):
        # Create data with an artificial shift in the last 3 days
        end = datetime(2025, 3, 1, tzinfo=timezone.utc)
        start = end - timedelta(days=30)
        df = generate_synthetic_readings(facility_id, start=start, end=end, interval_minutes=60)

        # Double the demand in the last 3 days
        recent_cutoff = end - timedelta(days=3)
        mask = df["timestamp"] >= recent_cutoff
        df.loc[mask, "demand_kw"] *= 2.0

        anomalies = detect_baseline_shift(df, facility_id, comparison_days=3, shift_threshold_pct=15.0)
        assert len(anomalies) >= 1, "Should detect the baseline shift"

    def test_no_shift_in平稳_data(self, clean_readings, facility_id):
        anomalies = detect_baseline_shift(clean_readings, facility_id, shift_threshold_pct=50.0)
        assert len(anomalies) == 0


class TestEquipmentLeftOn:
    """Tests for equipment left on detection."""

    def test_detects_off_hours_waste(self, off_hours_readings, facility_id):
        anomalies = detect_equipment_left_on(
            off_hours_readings, facility_id,
            off_hours=(22, 5), min_offhour_kw=80.0,
        )
        assert len(anomalies) > 0
        assert all(a.anomaly_type == "equipment_left_on" for a in anomalies)

    def test_no_false_positives_clean(self, clean_readings, facility_id):
        anomalies = detect_equipment_left_on(
            clean_readings, facility_id,
            off_hours=(22, 5), min_offhour_kw=500.0,  # very high threshold
        )
        assert len(anomalies) == 0


class TestPFDropDetection:
    """Tests for sustained power factor drop detection."""

    def test_detects_injected_pf_drop(self, pf_drop_readings, facility_id):
        anomalies = detect_pf_drops(pf_drop_readings, facility_id, pf_threshold=0.85, consecutive_hours=3)
        pf_anomalies = [a for a in anomalies if a.anomaly_type == "pf_drop"]
        assert len(pf_anomalies) >= 1

    def test_no_pf_anomalies_when_pf_good(self, clean_readings, facility_id):
        anomalies = detect_pf_drops(clean_readings, facility_id, pf_threshold=0.75)
        assert len(anomalies) == 0


class TestFullDetection:
    """Integration test for the full anomaly detection pipeline."""

    def test_returns_response_model(self, clean_readings, facility_id):
        result = run_full_anomaly_detection(clean_readings, facility_id)
        assert result.facility_id == facility_id
        assert result.total_anomalies == len(result.anomalies)

    def test_anomalies_sorted_by_severity(self, spike_readings, facility_id):
        result = run_full_anomaly_detection(spike_readings, facility_id)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        severities = [severity_order[a.severity.value] for a in result.anomalies]
        assert severities == sorted(severities)
