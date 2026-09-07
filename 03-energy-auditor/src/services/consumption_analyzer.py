"""Load profiling, demand analysis, and tariff optimization engine.

Core analysis service that processes time-series meter data to extract:
- Hourly/daily load profiles
- Demand patterns and contract utilization
- Power factor trends and penalty exposure
- Load factor and baseload characterization
- TOU cost breakdown and load-shift optimization
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.schemas import (
    DemandAnalysisResponse,
    LoadProfileBucket,
    LoadProfileResponse,
    PowerFactorAnalysis,
    TariffPeriod,
)
from src.utils.tariff import (
    classify_tou_period,
    get_tariff,
    calculate_monthly_bill,
    optimal_load_shift_savings,
)
from src.utils.units import capacitor_bank_kvar


def build_load_profile(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    tariff_name: str = "tou_general",
) -> LoadProfileResponse:
    """Build an hourly load profile from meter readings.

    Aggregates all readings into 24 hourly buckets, computing average, min,
    max demand and power factor for each hour. Classifies each bucket into
    its TOU tariff period.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw, power_factor].
    facility_id : Facility UUID.
    tariff_name : Tariff profile for TOU classification.

    Returns
    -------
    LoadProfileResponse with 24 hourly buckets.
    """
    if readings_df.empty:
        return LoadProfileResponse(
            facility_id=facility_id,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            buckets=[],
            baseload_kw=0.0,
            peak_demand_kw=0.0,
            peak_demand_timestamp=None,
            load_factor=0.0,
        )

    tariff = get_tariff(tariff_name)
    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour

    buckets: list[LoadProfileBucket] = []
    for h in range(24):
        hour_data = df[df["hour"] == h]
        if hour_data.empty:
            continue

        avg_kw = float(hour_data["demand_kw"].mean())
        max_kw = float(hour_data["demand_kw"].max())
        min_kw = float(hour_data["demand_kw"].min())
        avg_pf = float(hour_data["power_factor"].mean()) if "power_factor" in hour_data.columns else 1.0

        # Classify using a representative timestamp
        representative = hour_data["timestamp"].iloc[0]
        tou = classify_tou_period(representative, tariff)

        buckets.append(LoadProfileBucket(
            hour=h,
            avg_kw=round(avg_kw, 2),
            max_kw=round(max_kw, 2),
            min_kw=round(min_kw, 2),
            avg_pf=round(avg_pf, 4),
            readings_count=len(hour_data),
            tariff_period=TariffPeriod(tou),
        ))

    # Overall metrics
    peak_demand_kw = float(df["demand_kw"].max())
    peak_idx = df["demand_kw"].idxmax()
    peak_ts = df.loc[peak_idx, "timestamp"] if peak_idx is not None else None
    avg_demand = float(df["demand_kw"].mean())
    baseload_kw = float(df["demand_kw"].quantile(0.05))

    return LoadProfileResponse(
        facility_id=facility_id,
        period_start=df["timestamp"].min(),
        period_end=df["timestamp"].max(),
        buckets=buckets,
        baseload_kw=round(baseload_kw, 2),
        peak_demand_kw=round(peak_demand_kw, 2),
        peak_demand_timestamp=peak_ts,
        load_factor=round(avg_demand / peak_demand_kw if peak_demand_kw > 0 else 0.0, 4),
    )


def analyze_demand(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    contract_demand_kva: float,
) -> DemandAnalysisResponse:
    """Analyze demand patterns and contract utilization.

    Identifies peak demand, how often contract demand is exceeded,
    and recommends optimal contract demand level.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw, demand_kva].
    facility_id : Facility UUID.
    contract_demand_kva : Contracted demand level in kVA.

    Returns
    -------
    DemandAnalysisResponse with demand metrics and recommendations.
    """
    if readings_df.empty:
        return DemandAnalysisResponse(
            facility_id=facility_id,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            avg_demand_kw=0.0,
            peak_demand_kw=0.0,
            peak_demand_timestamp=None,
            contract_demand_kva=contract_demand_kva,
            demand_utilization_pct=0.0,
            demand_exceeded_hours=0,
            demand_exceeded_pct=0.0,
            recommended_contract_kva=contract_demand_kva,
        )

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    avg_kw = float(df["demand_kw"].mean())
    peak_kw = float(df["demand_kw"].max())
    peak_idx = df["demand_kw"].idxmax()
    peak_ts = df.loc[peak_idx, "timestamp"] if peak_idx is not None else None

    if "demand_kva" in df.columns:
        avg_kva = float(df["demand_kva"].mean())
        peak_kva = float(df["demand_kva"].max())
        # Convert demand_kw to equivalent kVA for threshold comparison
        demand_kva_series = df["demand_kva"]
    else:
        peak_kva = peak_kw  # assume PF=1 if no kVA data
        avg_kva = avg_kw
        demand_kva_series = df["demand_kw"]

    exceeded_mask = demand_kva_series > contract_demand_kva
    exceeded_hours = int(exceeded_mask.sum())
    total_hours = len(df)
    exceeded_pct = (exceeded_hours / total_hours * 100) if total_hours > 0 else 0.0
    utilization = (peak_kva / contract_demand_kva * 100) if contract_demand_kva > 0 else 0.0

    # Recommendation: contract demand = 95th percentile of kVA demand
    recommended = float(demand_kva_series.quantile(0.95)) if len(demand_kva_series) > 0 else contract_demand_kva

    return DemandAnalysisResponse(
        facility_id=facility_id,
        period_start=df["timestamp"].min(),
        period_end=df["timestamp"].max(),
        avg_demand_kw=round(avg_kw, 2),
        peak_demand_kw=round(peak_kw, 2),
        peak_demand_timestamp=peak_ts,
        contract_demand_kva=contract_demand_kva,
        demand_utilization_pct=round(utilization, 2),
        demand_exceeded_hours=exceeded_hours,
        demand_exceeded_pct=round(exceeded_pct, 2),
        recommended_contract_kva=round(recommended, 2),
    )


def analyze_power_factor(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    tariff_name: str = "tou_general",
    target_pf: float = 0.95,
) -> PowerFactorAnalysis:
    """Power factor analysis with capacitor bank sizing.

    Calculates penalty exposure, recommends capacitor bank size,
    and estimates savings from power factor correction.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw, power_factor].
    facility_id : Facility UUID.
    tariff_name : Tariff profile for penalty calculation.
    target_pf : Desired power factor after correction.

    Returns
    -------
    PowerFactorAnalysis with correction recommendations.
    """
    tariff = get_tariff(tariff_name)

    if readings_df.empty:
        return PowerFactorAnalysis(
            facility_id=facility_id,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            avg_power_factor=1.0,
            min_power_factor=1.0,
            avg_reactive_kvar=0.0,
            penalty_hours=0,
            estimated_penalty_usd=0.0,
            capacitor_bank_kvar=0.0,
            estimated_pf_after_correction=1.0,
            annual_savings_usd=0.0,
        )

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    avg_pf = float(df["power_factor"].mean())
    min_pf = float(df["power_factor"].min())

    # Reactive power per interval
    hours_per_reading = 1.0  # assume hourly data
    if len(df) > 1:
        dt = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        hours_per_reading = max(dt, 0.25)

    reactive_kw = df["demand_kw"] * np.tan(np.arccos(df["power_factor"].clip(0.3, 1.0)))
    avg_reactive_kvar = float(reactive_kw.mean())

    # Penalty exposure
    below_threshold = df["power_factor"] < tariff.pf_threshold
    penalty_hours = int(below_threshold.sum())

    # Penalty = fraction of energy cost
    energy_charge = float(df["demand_kw"].sum() * 0.12)  # rough avg rate
    steps_below = ((tariff.pf_threshold - df.loc[below_threshold, "power_factor"]) / 0.01).mean() if penalty_hours > 0 else 0
    penalty_usd = float(steps_below * tariff.pf_penalty_rate * energy_charge) if penalty_hours > 0 else 0.0

    # Capacitor bank sizing
    avg_kw = float(df["demand_kw"].mean())
    cap_kvar = capacitor_bank_kvar(avg_pf, target_pf, avg_kw) if avg_pf < target_pf else 0.0

    # Estimated PF after correction
    corrected_pf = target_pf if cap_kvar > 0 else avg_pf

    # Annual savings estimate
    monthly_penalty = penalty_usd * (30.0 / max(len(df) / 24, 1))  # extrapolate to monthly
    annual_savings = monthly_penalty * 12 if penalty_usd > 0 else 0.0

    return PowerFactorAnalysis(
        facility_id=facility_id,
        period_start=df["timestamp"].min(),
        period_end=df["timestamp"].max(),
        avg_power_factor=round(avg_pf, 4),
        min_power_factor=round(min_pf, 4),
        avg_reactive_kvar=round(avg_reactive_kvar, 2),
        penalty_hours=penalty_hours,
        estimated_penalty_usd=round(penalty_usd, 2),
        capacitor_bank_kvar=round(cap_kvar, 2),
        estimated_pf_after_correction=round(corrected_pf, 4),
        annual_savings_usd=round(annual_savings, 2),
    )


def generate_tou_cost_breakdown(
    readings_df: pd.DataFrame,
    tariff_name: str = "tou_general",
) -> dict[str, float]:
    """Break down energy cost by TOU period.

    Returns
    -------
    Dict with keys like 'peak_cost', 'shoulder_cost', 'offpeak_cost',
    'peak_kwh', 'shoulder_kwh', 'offpeak_kwh'.
    """
    tariff = get_tariff(tariff_name)
    if readings_df.empty:
        return {"peak_cost": 0, "shoulder_cost": 0, "offpeak_cost": 0,
                "peak_kwh": 0, "shoulder_kwh": 0, "offpeak_kwh": 0,
                "total_cost": 0}

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tou_period"] = df["timestamp"].apply(lambda ts: classify_tou_period(ts, tariff))

    result: dict[str, float] = {}
    for period in ["peak", "shoulder", "offpeak"]:
        mask = df["tou_period"] == period
        kwh = float(df.loc[mask, "active_energy_kwh"].sum())
        rate = next((p.rate_kwh for p in tariff.tou_periods if p.name == period), 0.07)
        result[f"{period}_kwh"] = round(kwh, 2)
        result[f"{period}_cost"] = round(kwh * rate, 2)

    result["total_cost"] = round(sum(v for k, v in result.items() if k.endswith("_cost")), 2)
    return result


def calculate_load_shift_savings(
    readings_df: pd.DataFrame,
    tariff_name: str = "tou_general",
    shiftable_fraction: float = 0.30,
) -> dict[str, float]:
    """Estimate cost savings from shifting load to off-peak hours."""
    tariff = get_tariff(tariff_name)
    return optimal_load_shift_savings(readings_df, tariff, shiftable_fraction)
