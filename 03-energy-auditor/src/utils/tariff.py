"""Industrial electricity tariff structures and cost calculations.

Supports time-of-use (TOU) tariffs, demand charges, power-factor penalties,
and seasonal variations common in industrial electricity billing.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone

import pandas as pd


@dataclass(frozen=True)
class TOUPeriod:
    """A single time-of-use period within a tariff."""
    name: str  # peak | shoulder | offpeak
    start: time
    end: time
    rate_kwh: float  # $/kWh
    is_weekend_offpeak: bool = False  # weekends counted as offpeak regardless


@dataclass
class IndustrialTariff:
    """Complete industrial electricity tariff structure.

    Typical US industrial tariff has three components:
    1. Energy charge ($/kWh) – varies by TOU period
    2. Demand charge ($/kVA or $/kW) – based on peak demand in billing cycle
    3. Power factor adjustment – penalty for PF below threshold
    """
    name: str
    tou_periods: list[TOUPeriod] = field(default_factory=list)
    demand_charge_per_kva: float = 15.0  # $/kVA per month
    demand_charge_per_kw: float | None = None  # alternative to kVA-based
    pf_threshold: float = 0.90
    pf_penalty_rate: float = 0.02  # surcharge per 0.01 below threshold
    fixed_charge_monthly: float = 0.0
    tax_rate: float = 0.08
    seasonal_multiplier: dict[int, float] = field(default_factory=dict)  # month -> multiplier


# ── Pre-built tariff profiles ─────────────────────────────────────────────

TOU_GENERAL = IndustrialTariff(
    name="TOU General Industrial",
    tou_periods=[
        TOUPeriod("peak", time(13, 0), time(19, 0), 0.18),
        TOUPeriod("shoulder", time(7, 0), time(13, 0), 0.12),
        TOUPeriod("shoulder", time(19, 0), time(23, 0), 0.12),
        TOUPeriod("offpeak", time(0, 0), time(7, 0), 0.07, is_weekend_offpeak=True),
    ],
    demand_charge_per_kva=15.0,
    pf_threshold=0.90,
    pf_penalty_rate=0.02,
    fixed_charge_monthly=45.0,
    seasonal_multiplier={6: 1.15, 7: 1.20, 8: 1.20, 9: 1.15},  # summer surcharge
)

TOU_COMPACT = IndustrialTariff(
    name="TOU Compact Industrial",
    tou_periods=[
        TOUPeriod("peak", time(12, 0), time(18, 0), 0.22),
        TOUPeriod("shoulder", time(6, 0), time(12, 0), 0.14),
        TOUPeriod("shoulder", time(18, 0), time(22, 0), 0.14),
        TOUPeriod("offpeak", time(22, 0), time(6, 0), 0.08, is_weekend_offpeak=True),
    ],
    demand_charge_per_kva=18.0,
    pf_threshold=0.92,
    pf_penalty_rate=0.025,
    fixed_charge_monthly=60.0,
    seasonal_multiplier={6: 1.25, 7: 1.30, 8: 1.30, 9: 1.25},
)

FLAT_RATE = IndustrialTariff(
    name="Flat Rate",
    tou_periods=[
        TOUPeriod("flat", time(0, 0), time(23, 59, 59), 0.11),
    ],
    demand_charge_per_kva=12.0,
    pf_threshold=0.90,
    pf_penalty_rate=0.015,
)

TARIFF_REGISTRY: dict[str, IndustrialTariff] = {
    "tou_general": TOU_GENERAL,
    "tou_compact": TOU_COMPACT,
    "flat_rate": FLAT_RATE,
}


def get_tariff(profile_name: str) -> IndustrialTariff:
    """Retrieve a tariff profile by name. Falls back to TOU_GENERAL."""
    return TARIFF_REGISTRY.get(profile_name, TOU_GENERAL)


# ── Cost calculation functions ─────────────────────────────────────────────

def classify_tou_period(ts: datetime, tariff: IndustrialTariff) -> str:
    """Determine the TOU period for a given timestamp.

    If the tariff defines weekend off-peak override and it's a weekend,
    ALL hours are classified as off-peak regardless of time window.
    """
    local_hour = ts.hour
    local_time = time(local_hour, ts.minute, ts.second)
    is_weekend = ts.weekday() >= 5  # Saturday=5, Sunday=6

    # If any period has weekend off-peak override, weekends are always off-peak
    has_weekend_override = any(p.is_weekend_offpeak for p in tariff.tou_periods)
    if is_weekend and has_weekend_override:
        return "offpeak"

    for period in tariff.tou_periods:
        if period.start <= period.end:
            in_window = period.start <= local_time <= period.end
        else:  # wraps midnight (e.g. 22:00-06:00)
            in_window = local_time >= period.start or local_time <= period.end

        if in_window:
            return period.name

    return "offpeak"  # fallback


def energy_charge_for_reading(
    ts: datetime,
    kwh: float,
    tariff: IndustrialTariff,
) -> float:
    """Calculate the energy charge for a single meter reading."""
    period = classify_tou_period(ts, tariff)
    rate = 0.07  # default offpeak
    for p in tariff.tou_periods:
        if p.name == period:
            rate = p.rate_kwh
            break

    multiplier = tariff.seasonal_multiplier.get(ts.month, 1.0)
    return kwh * rate * multiplier


def calculate_demand_charge(
    peak_kva: float,
    tariff: IndustrialTariff,
) -> float:
    """Monthly demand charge based on peak kVA in billing cycle."""
    return peak_kva * tariff.demand_charge_per_kva


def calculate_pf_penalty(
    avg_pf: float,
    tariff: IndustrialTariff,
) -> float:
    """Power-factor penalty surcharge as fraction of energy + demand charges."""
    if avg_pf >= tariff.pf_threshold:
        return 0.0

    steps_below = (tariff.pf_threshold - avg_pf) / 0.01
    return steps_below * tariff.pf_penalty_rate


def calculate_monthly_bill(
    readings_df: pd.DataFrame,
    tariff: IndustrialTariff,
    peak_kva: float,
) -> dict[str, float]:
    """Full monthly bill breakdown from a DataFrame of readings.

    Parameters
    ----------
    readings_df : DataFrame with columns [timestamp, active_energy_kwh]
    tariff : Industrial tariff structure
    peak_kva : Peak kVA in the billing period

    Returns
    -------
    dict with keys: energy_charge, demand_charge, pf_penalty, fixed_charge,
                    subtotal, tax, total
    """
    if readings_df.empty:
        return {
            "energy_charge": 0.0,
            "demand_charge": 0.0,
            "pf_penalty": 0.0,
            "fixed_charge": tariff.fixed_charge_monthly,
            "subtotal": tariff.fixed_charge_monthly,
            "tax": tariff.fixed_charge_monthly * tariff.tax_rate,
            "total": tariff.fixed_charge_monthly * (1 + tariff.tax_rate),
        }

    energy_total = 0.0
    for _, row in readings_df.iterrows():
        ts = row["timestamp"]
        kwh = row["active_energy_kwh"]
        energy_total += energy_charge_for_reading(ts, kwh, tariff)

    demand = calculate_demand_charge(peak_kva, tariff)

    avg_pf = readings_df["power_factor"].mean() if "power_factor" in readings_df.columns else 1.0
    pf_penalty_amount = calculate_pf_penalty(avg_pf, tariff) * energy_total

    subtotal = energy_total + demand + pf_penalty_amount + tariff.fixed_charge_monthly
    tax = subtotal * tariff.tax_rate
    total = subtotal + tax

    return {
        "energy_charge": round(energy_total, 2),
        "demand_charge": round(demand, 2),
        "pf_penalty": round(pf_penalty_amount, 2),
        "fixed_charge": round(tariff.fixed_charge_monthly, 2),
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
    }


def optimal_load_shift_savings(
    readings_df: pd.DataFrame,
    tariff: IndustrialTariff,
    shiftable_fraction: float = 0.30,
) -> dict[str, float]:
    """Estimate savings from shifting shiftable load from peak to off-peak.

    Assumes that `shiftable_fraction` of peak-hour consumption can be moved
    to off-peak hours without changing total daily consumption.

    Returns
    -------
    dict with current_cost, optimized_cost, savings_usd, savings_pct
    """
    if readings_df.empty or "active_energy_kwh" not in readings_df.columns:
        return {"current_cost": 0.0, "optimized_cost": 0.0, "savings_usd": 0.0, "savings_pct": 0.0}

    readings_df = readings_df.copy()
    readings_df["tou_period"] = readings_df["timestamp"].apply(
        lambda ts: classify_tou_period(ts, tariff)
    )
    readings_df["energy_cost"] = readings_df.apply(
        lambda r: energy_charge_for_reading(r["timestamp"], r["active_energy_kwh"], tariff),
        axis=1,
    )

    current_cost = readings_df["energy_cost"].sum()

    peak_kwh = readings_df.loc[readings_df["tou_period"] == "peak", "active_energy_kwh"].sum()
    shiftable_kwh = peak_kwh * shiftable_fraction

    offpeak_rate = 0.07
    peak_rate = 0.18
    for p in tariff.tou_periods:
        if p.name == "peak":
            peak_rate = p.rate_kwh
        if p.name == "offpeak":
            offpeak_rate = p.rate_kwh

    savings = shiftable_kwh * (peak_rate - offpeak_rate)
    optimized_cost = current_cost - savings

    return {
        "current_cost": round(current_cost, 2),
        "optimized_cost": round(optimized_cost, 2),
        "savings_usd": round(savings, 2),
        "savings_pct": round((savings / current_cost * 100) if current_cost > 0 else 0.0, 2),
    }
