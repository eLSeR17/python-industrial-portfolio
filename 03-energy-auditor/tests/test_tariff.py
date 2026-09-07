"""Tests for tariff calculations and unit conversions."""

import math
from datetime import datetime, time, timezone

import pandas as pd
import pytest

from src.utils.tariff import (
    TOU_GENERAL,
    classify_tou_period,
    energy_charge_for_reading,
    calculate_demand_charge,
    calculate_pf_penalty,
    calculate_monthly_bill,
    get_tariff,
    optimal_load_shift_savings,
    IndustrialTariff,
    TOUPeriod,
)
from src.utils.units import (
    kw_to_kva,
    kva_to_kw,
    reactive_power_kw,
    power_factor,
    apparent_power,
    capacitor_bank_kvar,
    kwh_to_joules,
    joules_to_kwh,
    kw_to_btuh,
    btuh_to_kw,
    co2_from_kwh,
)


# ── Tariff Tests ───────────────────────────────────────────────────────────

class TestTouClassification:
    """Tests for TOU period classification."""

    def test_peak_at_1500(self):
        ts = datetime(2025, 6, 15, 15, 0, tzinfo=timezone.utc)  # Sunday
        period = classify_tou_period(ts, TOU_GENERAL)
        # Sunday is weekend → offpeak
        assert period == "offpeak"

    def test_peak_at_1500_weekday(self):
        ts = datetime(2025, 6, 16, 15, 0, tzinfo=timezone.utc)  # Monday
        period = classify_tou_period(ts, TOU_GENERAL)
        assert period == "peak"

    def test_offpeak_at_0300(self):
        ts = datetime(2025, 6, 16, 3, 0, tzinfo=timezone.utc)  # Monday
        period = classify_tou_period(ts, TOU_GENERAL)
        assert period == "offpeak"

    def test_shoulder_at_1000(self):
        ts = datetime(2025, 6, 16, 10, 0, tzinfo=timezone.utc)  # Monday
        period = classify_tou_period(ts, TOU_GENERAL)
        assert period == "shoulder"

    def test_all_periods_covered(self):
        periods_seen = set()
        for h in range(24):
            ts = datetime(2025, 6, 16, h, 0, tzinfo=timezone.utc)  # Monday
            periods_seen.add(classify_tou_period(ts, TOU_GENERAL))
        assert "peak" in periods_seen
        assert "shoulder" in periods_seen
        assert "offpeak" in periods_seen


class TestEnergyCharges:
    """Tests for per-reading energy charge calculation."""

    def test_peak_charge_highest(self):
        ts_peak = datetime(2025, 6, 16, 15, 0, tzinfo=timezone.utc)  # Monday peak
        ts_offpeak = datetime(2025, 6, 16, 3, 0, tzinfo=timezone.utc)

        charge_peak = energy_charge_for_reading(ts_peak, 100.0, TOU_GENERAL)
        charge_offpeak = energy_charge_for_reading(ts_offpeak, 100.0, TOU_GENERAL)

        assert charge_peak > charge_offpeak

    def test_zero_kwh_zero_charge(self):
        ts = datetime(2025, 6, 16, 15, 0, tzinfo=timezone.utc)
        charge = energy_charge_for_reading(ts, 0.0, TOU_GENERAL)
        assert charge == 0.0

    def test_summer_surcharge(self):
        ts_summer = datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc)  # July
        ts_spring = datetime(2025, 4, 15, 15, 0, tzinfo=timezone.utc)  # April

        charge_summer = energy_charge_for_reading(ts_summer, 100.0, TOU_GENERAL)
        charge_spring = energy_charge_for_reading(ts_spring, 100.0, TOU_GENERAL)

        assert charge_summer > charge_spring


class TestDemandCharge:
    """Tests for demand charge calculation."""

    def test_proportional_to_peak(self):
        charge_100 = calculate_demand_charge(100.0, TOU_GENERAL)
        charge_200 = calculate_demand_charge(200.0, TOU_GENERAL)
        assert charge_200 == 2 * charge_100

    def test_zero_demand(self):
        charge = calculate_demand_charge(0.0, TOU_GENERAL)
        assert charge == 0.0


class TestPFPenalty:
    """Tests for power factor penalty calculation."""

    def test_no_penalty_above_threshold(self):
        penalty = calculate_pf_penalty(0.92, TOU_GENERAL)
        assert penalty == 0.0

    def test_penalty_below_threshold(self):
        penalty = calculate_pf_penalty(0.85, TOU_GENERAL)
        assert penalty > 0.0

    def test_worse_pf_higher_penalty(self):
        penalty_88 = calculate_pf_penalty(0.88, TOU_GENERAL)
        penalty_85 = calculate_pf_penalty(0.85, TOU_GENERAL)
        assert penalty_85 > penalty_88


class TestMonthlyBill:
    """Tests for monthly bill calculation."""

    def test_returns_all_components(self):
        import pandas as pd
        dates = pd.date_range("2025-06-01", periods=720, freq="1h", tz=timezone.utc)
        df = pd.DataFrame({
            "timestamp": dates,
            "active_energy_kwh": [100.0] * 720,
            "power_factor": [0.92] * 720,
        })
        bill = calculate_monthly_bill(df, TOU_GENERAL, peak_kva=400.0)
        assert "energy_charge" in bill
        assert "demand_charge" in bill
        assert "pf_penalty" in bill
        assert "total" in bill
        assert bill["total"] > 0

    def test_empty_bill(self):
        bill = calculate_monthly_bill(pd.DataFrame(), TOU_GENERAL, 400.0)
        assert bill["energy_charge"] == 0.0
        assert bill["demand_charge"] == 0.0


class TestLoadShiftSavings:
    """Tests for load shift optimization."""

    def test_positive_savings(self):
        import pandas as pd
        dates = pd.date_range("2025-06-16", periods=168, freq="1h", tz=timezone.utc)
        # Heavy peak, minimal off-peak
        kwh = [200 if 13 <= ts.hour <= 19 else 50 for ts in dates]
        df = pd.DataFrame({
            "timestamp": dates,
            "active_energy_kwh": kwh,
            "power_factor": [0.92] * 168,
        })
        result = optimal_load_shift_savings(df, TOU_GENERAL, shiftable_fraction=0.30)
        assert result["savings_usd"] > 0
        assert result["optimized_cost"] < result["current_cost"]


# ── Unit Conversion Tests ─────────────────────────────────────────────────

class TestPowerConversions:
    """Tests for electrical unit conversions."""

    def test_kw_to_kva_roundtrip(self):
        kw, pf = 100.0, 0.90
        kva = kw_to_kva(kw, pf)
        recovered = kva_to_kw(kva, pf)
        assert abs(recovered - kw) < 0.001

    def test_kw_to_kva_pf_one(self):
        assert kw_to_kva(100.0, 1.0) == 100.0

    def test_kva_always_ge_kw(self):
        kva = kw_to_kva(100.0, 0.80)
        assert kva >= 100.0

    def test_reactive_power_zero_at_pf_one(self):
        q = reactive_power_kw(100.0, 1.0)
        assert abs(q) < 0.001

    def test_reactive_power_positive(self):
        q = reactive_power_kw(100.0, 0.80)
        assert q > 0

    def test_power_factor_calculation(self):
        pf = power_factor(80.0, 100.0)
        assert abs(pf - 0.80) < 0.001

    def test_apparent_power_pythagorean(self):
        s = apparent_power(3.0, 4.0)
        assert abs(s - 5.0) < 0.001

    def test_capacitor_bank_sizing(self):
        kvar = capacitor_bank_kvar(0.80, 0.95, 100.0)
        assert kvar > 0

    def test_capacitor_bank_already_good(self):
        kvar = capacitor_bank_kvar(0.95, 0.95, 100.0)
        assert kvar == 0.0

    def test_kwh_joules_roundtrip(self):
        assert abs(joules_to_kwh(kwh_to_joules(1.0)) - 1.0) < 0.001

    def test_kw_btuh_roundtrip(self):
        assert abs(btuh_to_kw(kw_to_btuh(1.0)) - 1.0) < 0.001

    def test_co2_calculation(self):
        co2 = co2_from_kwh(1000.0, 0.42)
        assert abs(co2 - 420.0) < 0.01

    def test_invalid_pf_raises(self):
        with pytest.raises(ValueError):
            kw_to_kva(100.0, 0.0)
        with pytest.raises(ValueError):
            kw_to_kva(100.0, 1.5)


class TestGetTariff:
    """Tests for tariff registry."""

    def test_known_profile(self):
        tariff = get_tariff("tou_general")
        assert tariff.name == "TOU General Industrial"

    def test_unknown_falls_back(self):
        tariff = get_tariff("nonexistent_profile")
        assert tariff.name == "TOU General Industrial"

    def test_custom_tariff(self):
        custom = IndustrialTariff(
            name="Custom",
            tou_periods=[TOUPeriod("flat", time(0, 0), time(23, 59, 59), 0.10)],
        )
        assert custom.tou_periods[0].rate_kwh == 0.10
