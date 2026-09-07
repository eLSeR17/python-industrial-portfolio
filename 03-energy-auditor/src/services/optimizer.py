"""Energy optimization engine: HVAC scheduling, peak shaving, load shifting.

Provides actionable optimization recommendations by analyzing:
- Current load profile vs. TOU tariff windows
- Peak demand contribution analysis
- HVAC scheduling optimization
- Load shifting potential
- Combined savings estimates
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.schemas import SavingsRecommendation, SavingsReportResponse, Priority
from src.services.consumption_analyzer import (
    build_load_profile,
    calculate_load_shift_savings,
    generate_tou_cost_breakdown,
)
from src.services.power_factor_analysis import compute_correction_savings
from src.utils.tariff import get_tariff, calculate_monthly_bill, optimal_load_shift_savings
from src.utils.units import capacitor_bank_kvar, co2_from_kwh


def analyze_peak_shaving(
    readings_df: pd.DataFrame,
    contract_demand_kva: float,
    peak_reduction_target_pct: float = 10.0,
) -> dict:
    """Analyze potential savings from peak demand reduction.

    Peak shaving reduces demand charges by cutting the highest demand peaks,
    typically via battery storage, generator辅助, or load scheduling.

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw, demand_kva].
    contract_demand_kva : Current contracted demand.
    peak_reduction_target_pct : Target peak reduction (%).

    Returns
    -------
    Dict with peak shaving analysis.
    """
    if readings_df.empty:
        return {"feasible": False, "reason": "No data"}

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    current_peak = float(df["demand_kva"].max()) if "demand_kva" in df.columns else float(df["demand_kw"].max())
    target_peak = current_peak * (1 - peak_reduction_target_pct / 100)
    target_demand_charge = target_peak * 15.0  # $15/kVA
    current_demand_charge = current_peak * 15.0
    monthly_savings = current_demand_charge - target_demand_charge

    # Number of hours exceeding target
    demand_col = "demand_kva" if "demand_kva" in df.columns else "demand_kw"
    hours_exceeding = int((df[demand_col] > target_peak).sum())

    return {
        "feasible": hours_exceeding > 0,
        "current_peak_kva": round(current_peak, 2),
        "target_peak_kva": round(target_peak, 2),
        "reduction_pct": peak_reduction_target_pct,
        "hours_exceeding": hours_exceeding,
        "monthly_demand_savings_usd": round(monthly_savings, 2),
        "annual_demand_savings_usd": round(monthly_savings * 12, 2),
    }


def optimize_hvac_schedule(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    comfort_band: tuple[float, float] = (20.0, 26.0),
) -> dict:
    """Optimize HVAC scheduling to reduce energy waste.

    Analyzes temperature and load data to identify:
    - HVAC running during unoccupied periods
    - Over-cooling/heating beyond comfort band
    - Pre-conditioning opportunities (thermal mass)

    Parameters
    ----------
    readings_df : DataFrame with [timestamp, demand_kw, temperature_c].
    facility_id : Facility UUID.
    comfort_band : Acceptable temperature range (min, max) in °C.

    Returns
    -------
    Dict with HVAC optimization recommendations.
    """
    if readings_df.empty or "temperature_c" not in readings_df.columns:
        return {"recommendations": [], "estimated_savings_kwh": 0.0}

    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour

    # Identify HVAC load (portion of demand correlated with temperature)
    temp = df["temperature_c"].values
    demand = df["demand_kw"].values

    # Simple linear regression: demand = a + b * temp
    if len(temp) > 10:
        coeffs = np.polyfit(temp, demand, 1)
        temp_coefficient = coeffs[0]  # kW per °C
    else:
        temp_coefficient = 0.0

    # Find hours where demand seems excessive relative to temperature
    recommendations = []

    # Off-hours HVAC waste
    off_hours_mask = (df["hour"] < 6) | (df["hour"] >= 22)
    off_hours_demand = df[off_hours_mask]
    if not off_hours_demand.empty:
        off_avg = float(off_hours_demand["demand_kw"].mean())
        on_hours_demand = df[~off_hours_mask]
        on_avg = float(on_hours_demand["demand_kw"].mean()) if not on_hours_demand.empty else off_avg

        if off_avg > on_avg * 0.6:  # off-hours demand too high relative to daytime
            waste_kw = off_avg - on_avg * 0.3
            hours = len(off_hours_mask)
            daily_savings = waste_kw * hours * 0.4  # 40% of excess is shiftable

            recommendations.append({
                "type": "hvac_off_hours",
                "title": "HVAC scheduling during unoccupied hours",
                "description": (
                    f"Off-hours demand ({off_avg:.1f} kW avg) is high relative to "
                    f"daytime ({on_avg:.1f} kW). Consider setback scheduling from "
                    f"22:00-06:00."
                ),
                "estimated_daily_savings_kwh": round(daily_savings, 2),
                "estimated_monthly_savings_kwh": round(daily_savings * 22, 2),
            })

    # Temperature overshoot
    hot_days = df[df["temperature_c"] > comfort_band[1]]
    cold_days = df[df["temperature_c"] < comfort_band[0]]

    if len(hot_days) > len(df) * 0.05:
        avg_excess_temp = float(hot_days["temperature_c"].mean() - comfort_band[1])
        recommendations.append({
            "type": "hvac_overshoot",
            "title": "Temperature overshoot detected",
            "description": (
                f"Average temperature {hot_days['temperature_c'].mean():.1f}°C "
                f"exceeds comfort band max of {comfort_band[1]}°C by {avg_excess_temp:.1f}°C. "
                f"Review cooling setpoints and thermostat calibration."
            ),
            "estimated_daily_savings_kwh": round(avg_excess_temp * temp_coefficient * 8, 2),
            "estimated_monthly_savings_kwh": round(avg_excess_temp * temp_coefficient * 8 * 22, 2),
        })

    total_monthly_kwh = sum(r.get("estimated_monthly_savings_kwh", 0) for r in recommendations)

    return {
        "facility_id": str(facility_id),
        "comfort_band": comfort_band,
        "temperature_coefficient_kw_per_c": round(temp_coefficient, 3),
        "recommendations": recommendations,
        "estimated_savings_kwh": round(total_monthly_kwh, 2),
        "estimated_savings_usd": round(total_monthly_kwh * 0.12, 2),
    }


def generate_savings_report(
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    tariff_name: str = "tou_general",
    production_units: float | None = None,
) -> SavingsReportResponse:
    """Generate savings report with optimization opportunities.

    Combines power factor correction, load shifting, peak shaving, and HVAC
    optimization into a single actionable report.

    Parameters
    ----------
    readings_df : Meter readings DataFrame.
    facility_id : Facility UUID.
    tariff_name : Tariff profile.
    production_units : Production output for normalization.

    Returns
    -------
    SavingsReportResponse with ranked recommendations.
    """
    tariff = get_tariff(tariff_name)
    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Current cost analysis
    demand_col = "demand_kva" if "demand_kva" in df.columns else "demand_kw"
    peak_kva = float(df[demand_col].max())
    avg_pf = float(df["power_factor"].mean()) if "power_factor" in df.columns else 0.92
    current_bill = calculate_monthly_bill(df, tariff, peak_kva)

    recommendations: list[SavingsRecommendation] = []

    # 1. Power Factor Correction
    if avg_pf < tariff.pf_threshold:
        avg_kw = float(df["demand_kw"].mean())
        target_pf = 0.95
        cap_kvar = capacitor_bank_kvar(avg_pf, target_pf, avg_kw)
        # Estimate savings: penalty elimination + reduced losses
        pf_savings_monthly = current_bill["pf_penalty"] * 12 if current_bill["pf_penalty"] > 0 else avg_kw * 0.02 * 12
        capex = cap_kvar * 45  # ~$45/kVAR installed

        recommendations.append(SavingsRecommendation(
            category="power_factor",
            title=f"Install capacitor bank ({cap_kvar:.0f} kVAR)",
            description=(
                f"Current average PF: {avg_pf:.3f} (threshold: {tariff.pf_threshold}). "
                f"Installing a {cap_kvar:.0f} kVAR capacitor bank would raise PF to ~{target_pf}, "
                f"eliminating penalty charges and reducing I²R losses."
            ),
            estimated_savings_usd=round(pf_savings_monthly, 2),
            estimated_savings_kwh=round(avg_kw * 0.03 * 22, 2),  # ~3% loss reduction
            implementation_cost_usd=round(capex, 2),
            payback_months=round(capex / pf_savings_monthly, 1) if pf_savings_monthly > 0 else None,
            priority=Priority.HIGH,
        ))

    # 2. Load Shifting
    shift_result = optimal_load_shift_savings(df, tariff, shiftable_fraction=0.25)
    if shift_result["savings_usd"] > 0:
        recommendations.append(SavingsRecommendation(
            category="load_shifting",
            title="Shift 25% of peak load to off-peak hours",
            description=(
                f"Current TOU cost: ${shift_result['current_cost']:.2f}. "
                f"By shifting 25% of peak-hour consumption to off-peak periods "
                f"(e.g., pre-heating, batch scheduling), potential savings: "
                f"${shift_result['savings_usd']:.2f}."
            ),
            estimated_savings_usd=round(shift_result["savings_usd"], 2),
            estimated_savings_kwh=0,
            implementation_cost_usd=5000.0,  # scheduling software + process changes
            payback_months=round(5000 / shift_result["savings_usd"], 1) if shift_result["savings_usd"] > 0 else None,
            priority=Priority.HIGH,
        ))

    # 3. Peak Shaving
    peak_analysis = analyze_peak_shaving(df, peak_kva, peak_reduction_target_pct=10.0)
    if peak_analysis["feasible"] and peak_analysis["annual_demand_savings_usd"] > 0:
        recommendations.append(SavingsRecommendation(
            category="peak_shaving",
            title="Reduce peak demand by 10%",
            description=(
                f"Current peak demand: {peak_analysis['current_peak_kva']:.0f} kVA. "
                f"Target: {peak_analysis['target_peak_kva']:.0f} kVA. "
                f"This can be achieved through load scheduling, battery storage, "
                f"or demand response programs."
            ),
            estimated_savings_usd=round(peak_analysis["monthly_demand_savings_usd"], 2),
            estimated_savings_kwh=0,
            implementation_cost_usd=25000.0,  # battery or controls
            payback_months=round(25000 / peak_analysis["monthly_demand_savings_usd"], 1) if peak_analysis["monthly_demand_savings_usd"] > 0 else None,
            priority=Priority.MEDIUM,
        ))

    # 4. HVAC Optimization
    hvac = optimize_hvac_schedule(df, facility_id)
    if hvac["estimated_savings_usd"] > 0:
        recommendations.append(SavingsRecommendation(
            category="hvac",
            title="Optimize HVAC scheduling",
            description="\n".join(r["description"] for r in hvac["recommendations"]),
            estimated_savings_usd=round(hvac["estimated_savings_usd"], 2),
            estimated_savings_kwh=round(hvac["estimated_savings_kwh"], 2),
            implementation_cost_usd=2000.0,  # BMS programming
            payback_months=round(2000 / hvac["estimated_savings_usd"], 1) if hvac["estimated_savings_usd"] > 0 else None,
            priority=Priority.MEDIUM,
        ))

    # 5. Equipment audit
    avg_demand = float(df["demand_kw"].mean())
    baseload = float(df["demand_kw"].quantile(0.05))
    if baseload > avg_demand * 0.4:
        excess_baseload = baseload - avg_demand * 0.25
        recommendations.append(SavingsRecommendation(
            category="equipment",
            title="Investigate high baseload",
            description=(
                f"Baseload (5th percentile): {baseload:.1f} kW, which is "
                f"{baseload / avg_demand * 100:.0f}% of average demand ({avg_demand:.1f} kW). "
                f"A healthy baseload should be <25% of average. "
                f"Check for equipment left running, compressed air leaks, "
                f"or standby power waste."
            ),
            estimated_savings_usd=round(excess_baseload * 0.12 * 24 * 30, 2),
            estimated_savings_kwh=round(excess_baseload * 24 * 30, 2),
            implementation_cost_usd=1000.0,  # audit + fixes
            payback_months=round(1000 / (excess_baseload * 0.12 * 24 * 30), 1),
            priority=Priority.MEDIUM,
        ))

    # Aggregate
    total_savings_usd = sum(r.estimated_savings_usd for r in recommendations)
    total_savings_kwh = sum(r.estimated_savings_kwh for r in recommendations)
    total_capex = sum(r.implementation_cost_usd for r in recommendations)

    # Weighted payback (by savings amount)
    if total_savings_usd > 0:
        weighted_payback = sum(
            (r.implementation_cost_usd / r.estimated_savings_usd) * r.estimated_savings_usd
            for r in recommendations if r.estimated_savings_usd > 0
        ) / total_savings_usd
    else:
        weighted_payback = None

    potential_cost = current_bill["total"] - total_savings_usd
    annual_savings = total_savings_usd * 12

    # Sort by savings (highest first)
    recommendations.sort(key=lambda r: r.estimated_savings_usd, reverse=True)

    return SavingsReportResponse(
        facility_id=facility_id,
        report_date=datetime.now(timezone.utc),
        current_monthly_cost_usd=round(current_bill["total"], 2),
        potential_monthly_cost_usd=round(max(0, potential_cost), 2),
        monthly_savings_usd=round(total_savings_usd, 2),
        annual_savings_usd=round(annual_savings, 2),
        savings_pct=round(total_savings_usd / current_bill["total"] * 100 if current_bill["total"] > 0 else 0, 2),
        recommendations=recommendations,
        total_capex_usd=round(total_capex, 2),
        weighted_payback_months=round(weighted_payback, 1) if weighted_payback else None,
    )
