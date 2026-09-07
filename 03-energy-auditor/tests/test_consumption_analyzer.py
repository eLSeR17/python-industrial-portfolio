"""Tests for the consumption analyzer service."""

import math
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.services.consumption_analyzer import (
    analyze_demand,
    analyze_power_factor,
    build_load_profile,
    generate_tou_cost_breakdown,
    calculate_load_shift_savings,
)
from src.services.meter_reader import generate_synthetic_readings


@pytest.fixture
def facility_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def synthetic_readings(facility_id: uuid.UUID) -> pd.DataFrame:
    """30 days of hourly synthetic readings."""
    end = datetime(2025, 6, 15, tzinfo=timezone.utc)
    start = end - timedelta(days=30)
    return generate_synthetic_readings(
        facility_id=facility_id,
        meter_id="MTR-TEST",
        start=start,
        end=end,
        interval_minutes=60,
        baseload_kw=100.0,
        peak_kw=400.0,
        pf_nominal=0.91,
    )


@pytest.fixture
def empty_readings() -> pd.DataFrame:
    return pd.DataFrame()


class TestBuildLoadProfile:
    """Tests for hourly load profile generation."""

    def test_returns_24_buckets(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        assert len(profile.buckets) == 24
        assert all(0 <= b.hour <= 23 for b in profile.buckets)

    def test_peak_hour_in_business_hours(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        peak_bucket = max(profile.buckets, key=lambda b: b.avg_kw)
        assert 6 <= peak_bucket.hour <= 20, "Peak should be during business hours"

    def test_baseload_positive(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        assert profile.baseload_kw > 0

    def test_load_factor_between_0_and_1(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        assert 0 < profile.load_factor <= 1

    def test_empty_input_returns_empty(self, facility_id):
        profile = build_load_profile(pd.DataFrame(), facility_id)
        assert len(profile.buckets) == 0
        assert profile.peak_demand_kw == 0.0

    def test_tariff_classification(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        periods = {b.tariff_period.value for b in profile.buckets}
        assert "peak" in periods, "Should have peak periods in profile"
        assert "offpeak" in periods, "Should have off-peak periods in profile"

    def test_bucket_counts_match_data(self, synthetic_readings, facility_id):
        profile = build_load_profile(synthetic_readings, facility_id)
        total_readings = sum(b.readings_count for b in profile.buckets)
        assert total_readings == len(synthetic_readings)


class TestAnalyzeDemand:
    """Tests for demand analysis."""

    def test_returns_peak_and_avg(self, synthetic_readings, facility_id):
        result = analyze_demand(synthetic_readings, facility_id, contract_demand_kva=500.0)
        assert result.peak_demand_kw > 0
        assert result.avg_demand_kw > 0
        assert result.peak_demand_kw >= result.avg_demand_kw

    def test_utilization_pct(self, synthetic_readings, facility_id):
        result = analyze_demand(synthetic_readings, facility_id, contract_demand_kva=500.0)
        assert 0 <= result.demand_utilization_pct <= 200  # can exceed contract

    def test_contract_exceeded_detection(self, synthetic_readings, facility_id):
        # Set low contract to force exceedances
        result = analyze_demand(synthetic_readings, facility_id, contract_demand_kva=200.0)
        assert result.demand_exceeded_hours > 0, "Low contract should show exceedances"

    def test_recommended_demand_percentile(self, synthetic_readings, facility_id):
        result = analyze_demand(synthetic_readings, facility_id, contract_demand_kva=500.0)
        assert result.recommended_contract_kva > 0

    def test_empty_readings(self, facility_id):
        result = analyze_demand(pd.DataFrame(), facility_id, 500.0)
        assert result.peak_demand_kw == 0.0


class TestPowerFactorAnalysis:
    """Tests for power factor analysis."""

    def test_avg_pf_in_valid_range(self, synthetic_readings, facility_id):
        result = analyze_power_factor(synthetic_readings, facility_id)
        assert 0 < result.avg_power_factor <= 1.0

    def test_min_pf_le_avg_pf(self, synthetic_readings, facility_id):
        result = analyze_power_factor(synthetic_readings, facility_id)
        assert result.min_power_factor <= result.avg_power_factor

    def test_capacitor_bank_non_negative(self, synthetic_readings, facility_id):
        result = analyze_power_factor(synthetic_readings, facility_id)
        assert result.capacitor_bank_kvar >= 0

    def test_corrected_pf_after_install(self, synthetic_readings, facility_id):
        result = analyze_power_factor(synthetic_readings, facility_id, target_pf=0.95)
        if result.capacitor_bank_kvar > 0:
            assert result.estimated_pf_after_correction >= result.avg_power_factor


class TestTouCostBreakdown:
    """Tests for TOU cost breakdown."""

    def test_breakdown_has_all_periods(self, synthetic_readings):
        breakdown = generate_tou_cost_breakdown(synthetic_readings)
        assert "peak_kwh" in breakdown
        assert "shoulder_kwh" in breakdown
        assert "offpeak_kwh" in breakdown

    def test_total_equals_sum(self, synthetic_readings):
        breakdown = generate_tou_cost_breakdown(synthetic_readings)
        period_sum = breakdown["peak_cost"] + breakdown["shoulder_cost"] + breakdown["offpeak_cost"]
        assert abs(breakdown["total_cost"] - period_sum) < 0.01

    def test_kwh_sum_matches_total(self, synthetic_readings):
        breakdown = generate_tou_cost_breakdown(synthetic_readings)
        total_kwh = breakdown["peak_kwh"] + breakdown["shoulder_kwh"] + breakdown["offpeak_kwh"]
        assert abs(total_kwh - synthetic_readings["active_energy_kwh"].sum()) < 1.0

    def test_empty_readings(self):
        breakdown = generate_tou_cost_breakdown(pd.DataFrame())
        assert breakdown["total_cost"] == 0


class TestLoadShiftSavings:
    """Tests for load shift savings estimation."""

    def test_savings_non_negative(self, synthetic_readings):
        result = calculate_load_shift_savings(synthetic_readings)
        assert result["savings_usd"] >= 0

    def test_savings_within_range(self, synthetic_readings):
        result = calculate_load_shift_savings(synthetic_readings, shiftable_fraction=0.30)
        if result["savings_usd"] > 0:
            assert result["savings_pct"] > 0
            assert result["savings_pct"] < 50  # sanity check
