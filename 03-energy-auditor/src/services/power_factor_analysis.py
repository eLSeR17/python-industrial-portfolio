"""Power factor correction analysis helper for the optimizer.

Separated to avoid circular imports while keeping the optimizer clean.
"""

import pandas as pd

from src.utils.units import capacitor_bank_kvar


def compute_correction_savings(
    readings_df: pd.DataFrame,
    target_pf: float = 0.95,
    penalty_rate: float = 0.02,
    threshold: float = 0.90,
) -> dict:
    """Estimate savings from power factor correction.

    Parameters
    ----------
    readings_df : Meter readings with power_factor column.
    target_pf : Desired power factor.
    penalty_rate : Penalty per 0.01 below threshold (fraction of energy cost).
    threshold : PF threshold below which penalties apply.

    Returns
    -------
    Dict with correction sizing and savings estimates.
    """
    if readings_df.empty or "power_factor" not in readings_df.columns:
        return {"savings_usd": 0.0, "capacitor_kvar": 0.0, "corrected_pf": 1.0}

    avg_pf = float(readings_df["power_factor"].mean())
    avg_kw = float(readings_df["demand_kw"].mean())

    if avg_pf >= target_pf:
        return {"savings_usd": 0.0, "capacitor_kvar": 0.0, "corrected_pf": avg_pf}

    cap_kvar = capacitor_bank_kvar(avg_pf, target_pf, avg_kw)

    # Estimate penalty savings
    energy_charge = avg_kw * 24 * 30 * 0.12  # monthly rough
    if avg_pf < threshold:
        steps = (threshold - avg_pf) / 0.01
        current_penalty = steps * penalty_rate * energy_charge
    else:
        current_penalty = 0.0

    # Loss reduction: ~ (1 - PF_old²/PF_new²) of I²R losses
    loss_reduction_pct = 1 - (avg_pf ** 2 / target_pf ** 2)
    loss_savings = energy_charge * loss_reduction_pct * 0.5  # half of losses are avoidable

    total_savings = current_penalty + loss_savings

    return {
        "savings_usd": round(total_savings, 2),
        "capacitor_kvar": round(cap_kvar, 2),
        "corrected_pf": target_pf,
        "current_pf": round(avg_pf, 4),
        "monthly_penalty_eliminated": round(current_penalty, 2),
        "monthly_loss_reduction_savings": round(loss_savings, 2),
    }
