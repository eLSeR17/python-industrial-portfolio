"""ISO 50001 Energy Baseline and benchmarking engine.

Implements:
- Energy Performance Indicators (EnPIs) per ISO 50001
- Baseline establishment and tracking
- Cross-facility benchmarking with normalization
- Production-output normalization (kWh/unit, kWh/m²)
- Annual improvement tracking toward EnPI targets
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.schemas import BenchmarkEntry, BenchmarkResponse


def calculate_enpi(
    total_kwh: float,
    production_units: float | None = None,
    floor_area_sqm: float | None = None,
    degree_days: float | None = None,
    operating_hours: float | None = None,
) -> dict[str, float]:
    """Calculate Energy Performance Indicators (EnPIs).

    EnPIs normalize energy consumption against relevant variables to enable
    fair comparison across time periods and between facilities.

    Parameters
    ----------
    total_kwh : Total energy consumption in kWh.
    production_units : Total production output (units, tonnes, etc.).
    floor_area_sqm : Facility floor area in m².
    degree_days : Heating/cooling degree-days for weather normalization.
    operating_hours : Total operating hours in the period.

    Returns
    -------
    Dict of calculated EnPIs (only non-None inputs are computed).
    """
    enpis: dict[str, float] = {}

    if production_units and production_units > 0:
        enpis["kwh_per_unit"] = round(total_kwh / production_units, 4)
    if floor_area_sqm and floor_area_sqm > 0:
        enpis["kwh_per_sqm"] = round(total_kwh / floor_area_sqm, 4)
    if degree_days and degree_days > 0:
        enpis["kwh_per_degree_day"] = round(total_kwh / degree_days, 4)
    if operating_hours and operating_hours > 0:
        enpis["kwh_per_hour"] = round(total_kwh / operating_hours, 4)
        if production_units and production_units > 0:
            enpis["kw_avg"] = round(total_kwh / operating_hours, 2)

    enpis["total_kwh"] = round(total_kwh, 2)
    return enpis


def build_energy_baseline(
    historical_df: pd.DataFrame,
    facility_id: uuid.UUID,
    baseline_year: int = 2024,
    production_units: float | None = None,
    floor_area_sqm: float | None = None,
) -> dict:
    """Establish an energy baseline per ISO 50001 methodology.

    The baseline period is the specified year. It includes:
    - Total consumption
    - EnPIs
    - Monthly profile for normalization
    - Regression data (kWh vs. production output if available)

    Parameters
    ----------
    historical_df : DataFrame with [timestamp, active_energy_kwh] (multi-year).
    facility_id : Facility UUID.
    baseline_year : Year to use as baseline.
    production_units : Annual production for the baseline year.
    floor_area_sqm : Facility floor area.

    Returns
    -------
    Dict with baseline metrics and monthly profile.
    """
    if historical_df.empty:
        return {"facility_id": str(facility_id), "baseline_year": baseline_year, "error": "No data"}

    df = historical_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    baseline_data = df[df["timestamp"].dt.year == baseline_year]

    if baseline_data.empty:
        return {"facility_id": str(facility_id), "baseline_year": baseline_year, "error": "No baseline year data"}

    total_kwh = float(baseline_data["active_energy_kwh"].sum())
    enpis = calculate_enpi(total_kwh, production_units, floor_area_sqm)

    # Monthly profile
    baseline_data = baseline_data.copy()
    baseline_data["month"] = baseline_data["timestamp"].dt.month
    monthly = baseline_data.groupby("month")["active_energy_kwh"].sum()

    # Day-of-week profile
    baseline_data["dow"] = baseline_data["timestamp"].dt.dayofweek
    daily_profile = baseline_data.groupby("dow")["active_energy_kwh"].sum()

    return {
        "facility_id": str(facility_id),
        "baseline_year": baseline_year,
        "total_kwh": round(total_kwh, 2),
        "enpis": enpis,
        "monthly_kwh": {int(m): round(float(v), 2) for m, v in monthly.items()},
        "daily_profile_kwh": {int(d): round(float(v), 2) for d, v in daily_profile.items()},
        "data_points": len(baseline_data),
    }


def compare_to_baseline(
    current_df: pd.DataFrame,
    baseline: dict,
    facility_id: uuid.UUID,
    production_units: float | None = None,
    floor_area_sqm: float | None = None,
    target_reduction_pct: float = 2.0,
) -> dict:
    """Compare current period consumption against the baseline.

    Parameters
    ----------
    current_df : Current period readings.
    baseline : Baseline dict from build_energy_baseline().
    facility_id : Facility UUID.
    production_units : Production output for the current period.
    floor_area_sqm : Floor area.
    target_reduction_pct : ISO 50001 annual improvement target (%).

    Returns
    -------
    Dict with comparison metrics and compliance status.
    """
    if current_df.empty or "error" in baseline:
        return {"facility_id": str(facility_id), "error": "Insufficient data"}

    total_kwh = float(current_df["active_energy_kwh"].sum())
    baseline_kwh = baseline.get("total_kwh", 0)
    baseline_enpis = baseline.get("enpis", {})

    current_enpis = calculate_enpi(total_kwh, production_units, floor_area_sqm)

    # Calculate changes
    kwh_change_pct = ((total_kwh - baseline_kwh) / baseline_kwh * 100) if baseline_kwh > 0 else 0.0

    # EnPI comparison (use kwh_per_unit if available)
    enpi_current = current_enpis.get("kwh_per_unit", total_kwh)
    enpi_baseline = baseline_enpis.get("kwh_per_unit", baseline_kwh)
    enpi_change_pct = ((enpi_current - enpi_baseline) / enpi_baseline * 100) if enpi_baseline > 0 else 0.0

    # ISO 50001 compliance: EnPI must decrease by target_reduction_pct annually
    compliant = enpi_change_pct <= -target_reduction_pct if enpi_change_pct != 0 else True

    # Extrapolate annual savings
    annualized_savings_kwh = -enpi_change_pct / 100 * baseline_kwh if enpi_change_pct < 0 else 0.0

    return {
        "facility_id": str(facility_id),
        "baseline_year": baseline.get("baseline_year"),
        "current_total_kwh": round(total_kwh, 2),
        "baseline_total_kwh": round(baseline_kwh, 2),
        "kwh_change_pct": round(kwh_change_pct, 2),
        "enpi_current": current_enpis,
        "enpi_baseline": baseline_enpis,
        "enpi_change_pct": round(enpi_change_pct, 2),
        "iso50001_compliant": compliant,
        "target_reduction_pct": target_reduction_pct,
        "annualized_savings_kwh": round(annualized_savings_kwh, 2),
    }


def benchmark_facilities(
    facility_data: list[dict],
) -> BenchmarkResponse:
    """Benchmark multiple facilities against each other.

    Parameters
    ----------
    facility_data : List of dicts, each with:
        - facility_id (UUID)
        - facility_name (str)
        - total_kwh (float)
        - production_units (float, optional)
        - floor_area_sqm (float, optional)
        - avg_power_factor (float)
        - baseline_kwh (float, optional)
        - baseline_enpi (float, optional)

    Returns
    -------
    BenchmarkResponse with ranked entries.
    """
    entries: list[BenchmarkEntry] = []

    for fd in facility_data:
        fid = fd["facility_id"]
        total_kwh = fd.get("total_kwh", 0)
        prod = fd.get("production_units")
        area = fd.get("floor_area_sqm")
        pf = fd.get("avg_power_factor", 1.0)
        baseline_kwh = fd.get("baseline_kwh", total_kwh)
        baseline_enpi = fd.get("baseline_enpi")

        kwh_per_unit = total_kwh / prod if prod and prod > 0 else 0.0
        kwh_per_sqm = total_kwh / area if area and area > 0 else 0.0
        demand_intensity = kwh_per_unit * pf  # simplified

        # EnPI score (composite: lower is better)
        enpi_score = kwh_per_unit if kwh_per_unit > 0 else kwh_per_sqm

        # vs baseline
        if baseline_enpi and baseline_enpi > 0:
            vs_baseline = ((enpi_score - baseline_enpi) / baseline_enpi * 100)
        elif baseline_kwh and baseline_kwh > 0:
            vs_baseline = ((total_kwh - baseline_kwh) / baseline_kwh * 100)
        else:
            vs_baseline = 0.0

        compliant = vs_baseline <= -2.0  # 2% annual improvement target

        entries.append(BenchmarkEntry(
            facility_id=fid,
            facility_name=fd.get("facility_name", str(fid)),
            kwh_per_unit=round(kwh_per_unit, 4),
            kwh_per_sqm=round(kwh_per_sqm, 4),
            demand_intensity_kw_per_unit=round(demand_intensity, 4),
            power_factor_avg=round(pf, 4),
            enpi_score=round(enpi_score, 4),
            vs_baseline_pct=round(vs_baseline, 2),
            iso50001_compliant=compliant,
        ))

    # Rank by EnPI (lower is better)
    ranked_ids = [e.facility_id for e in sorted(entries, key=lambda e: e.enpi_score)]

    return BenchmarkResponse(entries=entries, rank_by_enpi=ranked_ids)
