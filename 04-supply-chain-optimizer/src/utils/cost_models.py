"""Cost models for landed cost, transportation, holding, and sensitivity analysis."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def total_landed_cost(
    material_cost: float,
    transport_cost: float,
    duty_cost: float,
    handling_cost: float,
    storage_cost: float,
    insurance_cost: float,
    quantity: float,
) -> dict[str, Any]:
    """Compute total landed cost and per-unit cost.

    Landed cost includes all costs to deliver goods to the point of use:
    purchase price, freight, customs duties, insurance, handling, and storage.

    Args:
        material_cost: Total material/purchase cost.
        transport_cost: Total transportation cost.
        duty_cost: Customs duties and tariffs.
        handling_cost: Loading, unloading, and processing.
        storage_cost: Warehouse storage during transit.
        insurance_cost: Cargo insurance.
        quantity: Number of units.

    Returns:
        Landed cost breakdown with totals and per-unit figures.
    """
    components = {
        "material": material_cost,
        "transport": transport_cost,
        "duty": duty_cost,
        "handling": handling_cost,
        "storage": storage_cost,
        "insurance": insurance_cost,
    }
    total = sum(components.values())
    per_unit = total / quantity if quantity > 0 else 0.0

    pct_breakdown = {
        k: (v / total * 100.0) if total > 0 else 0.0
        for k, v in components.items()
    }

    return {
        "total_landed_cost": round(total, 2),
        "per_unit_cost": round(per_unit, 2),
        "quantity": quantity,
        "components": {k: round(v, 2) for k, v in components.items()},
        "component_pct": {k: round(v, 1) for k, v in pct_breakdown.items()},
    }


def transportation_cost_model(
    distance_km: float,
    weight_tons: float,
    cost_per_km_per_ton: float = 0.35,
    fuel_surcharge_pct: float = 0.0,
    minimum_charge: float = 50.0,
) -> dict[str, Any]:
    """Model transportation cost based on distance and weight.

    Uses a linear cost model with optional fuel surcharge and minimum
    charge floor. This captures the core economics of road freight.

    Args:
        distance_km: Distance of the shipment route.
        weight_tons: Weight of goods in metric tons.
        cost_per_km_per_ton: Base rate (cost per km per ton).
        fuel_surcharge_pct: Additional fuel surcharge as percentage.
        minimum_charge: Minimum shipment cost.

    Returns:
        Cost breakdown with base, surcharge, and total.
    """
    base_cost = distance_km * weight_tons * cost_per_km_per_ton
    fuel_surcharge = base_cost * fuel_surcharge_pct
    total = base_cost + fuel_surcharge
    total = max(total, minimum_charge)

    return {
        "distance_km": distance_km,
        "weight_tons": weight_tons,
        "base_cost": round(base_cost, 2),
        "fuel_surcharge": round(fuel_surcharge, 2),
        "minimum_charge": minimum_charge,
        "total_transport_cost": round(total, 2),
        "cost_per_unit_weight": round(total / weight_tons, 2) if weight_tons > 0 else 0.0,
    }


def holding_cost_model(
    avg_inventory_units: float,
    unit_cost: float,
    annual_holding_cost_pct: float = 0.25,
    capital_cost_pct: float = 0.08,
    insurance_pct: float = 0.02,
    storage_pct: float = 0.05,
    obsolescence_pct: float = 0.05,
    shrinkage_pct: float = 0.03,
) -> dict[str, Any]:
    """Decompose annual inventory holding cost into its components.

    Holding cost typically runs 20-30% of inventory value annually.
    This model breaks it into standard components:
    - Capital cost: opportunity cost of tied-up capital
    - Insurance: inventory insurance premiums
    - Storage: warehouse rent, utilities, labor
    - Obsolescence: risk of inventory becoming unsellable
    - Shrinkage: theft, damage, counting errors

    Args:
        avg_inventory_units: Average inventory on hand.
        unit_cost: Cost per unit.
        annual_holding_cost_pct: Total holding cost rate.
        capital_cost_pct: Cost of capital rate.
        insurance_pct: Insurance rate on inventory value.
        storage_pct: Storage cost rate.
        obsolescence_pct: Obsolescence risk rate.
        shrinkage_pct: Shrinkage rate.

    Returns:
        Detailed holding cost decomposition.
    """
    inventory_value = avg_inventory_units * unit_cost

    components = {
        "capital": inventory_value * capital_cost_pct,
        "insurance": inventory_value * insurance_pct,
        "storage": inventory_value * storage_pct,
        "obsolescence": inventory_value * obsolescence_pct,
        "shrinkage": inventory_value * shrinkage_pct,
    }
    total_holding = inventory_value * annual_holding_cost_pct
    computed_total = sum(components.values())

    return {
        "inventory_units": avg_inventory_units,
        "unit_cost": unit_cost,
        "inventory_value": round(inventory_value, 2),
        "total_annual_holding_cost": round(total_holding, 2),
        "components": {k: round(v, 2) for k, v in components.items()},
        "component_pct": {
            k: round(v / computed_total * 100, 1) if computed_total > 0 else 0.0
            for k, v in components.items()
        },
        "verified_total": round(computed_total, 2),
    }


def sensitivity_analysis(
    base_value: float,
    parameters: dict[str, tuple[float, float]],
    cost_function: Callable[[str, float], float] | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Perform one-at-a-time sensitivity analysis on cost parameters.

    Sweeps each parameter across its range while holding others constant,
    recording the output cost for each value. This identifies which
    parameters have the greatest influence on total cost.

    Args:
        base_value: Baseline cost output.
        parameters: Dict mapping parameter names to (min_factor, max_factor).
        cost_function: Optional callable(param_name, factor) -> cost. If None,
            uses linear approximation.

    Returns:
        Dict mapping each parameter to its sweep results.
    """
    results: dict[str, list[dict[str, float]]] = {}

    for param_name, (min_factor, max_factor) in parameters.items():
        factors = np.linspace(min_factor, max_factor, 21)
        sweep = []

        for f in factors:
            if cost_function:
                cost = cost_function(param_name, f)
            else:
                # Linear approximation
                cost = base_value * f

            sweep.append({
                "factor": round(float(f), 4),
                "cost": round(float(cost), 2),
                "delta": round(float(cost - base_value), 2),
                "delta_pct": round(float((f - 1.0) * 100), 2),
            })

        # Compute elasticity at the base point
        costs = [s["cost"] for s in sweep]
        factors_arr = [s["factor"] for s in sweep]
        if len(costs) > 2 and costs[0] != costs[-1]:
            # Central difference at the middle (factor=1.0 if in range)
            mid_idx = len(factors_arr) // 2
            if mid_idx > 0 and mid_idx < len(costs) - 1:
                elasticity = (
                    (costs[mid_idx + 1] - costs[mid_idx - 1])
                    / (factors_arr[mid_idx + 1] - factors_arr[mid_idx - 1])
                    * (factors_arr[mid_idx] / costs[mid_idx])
                    if costs[mid_idx] != 0
                    else 0.0
                )
            else:
                elasticity = 0.0
        else:
            elasticity = 0.0

        results[param_name] = {
            "sweep": sweep,
            "elasticity_at_base": round(float(elasticity), 4),
        }

    return results


def cost_optimization_report(
    current_cost: float,
    optimized_cost: float,
    implementation_cost: float = 0.0,
    implementation_time_months: int = 0,
) -> dict[str, Any]:
    """Generate an executive cost optimization report.

    Summarizes the savings from optimization and computes ROI, payback
    period, and annual savings for executive decision-making.

    Args:
        current_cost: Baseline (unoptimized) annual cost.
        optimized_cost: Optimized annual cost.
        implementation_cost: One-time cost to implement changes.
        implementation_time_months: Time to implement in months.

    Returns:
        Executive summary with financial metrics.
    """
    annual_savings = current_cost - optimized_cost
    savings_pct = (annual_savings / current_cost * 100.0) if current_cost > 0 else 0.0

    # ROI and payback
    if implementation_cost > 0:
        roi = (annual_savings / implementation_cost * 100.0) if implementation_cost > 0 else 0.0
        payback_months = (
            (implementation_cost / annual_savings * 12.0)
            if annual_savings > 0
            else float("inf")
        )
    else:
        roi = float("inf")
        payback_months = 0.0

    # 3-year NPV (10% discount rate)
    discount_rate = 0.10
    npv_3year = sum(
        annual_savings / (1 + discount_rate) ** y for y in range(1, 4)
    ) - implementation_cost

    # Classification
    if savings_pct >= 15:
        impact = "transformative"
    elif savings_pct >= 10:
        impact = "significant"
    elif savings_pct >= 5:
        impact = "moderate"
    else:
        impact = "marginal"

    return {
        "current_annual_cost": round(current_cost, 2),
        "optimized_annual_cost": round(optimized_cost, 2),
        "annual_savings": round(annual_savings, 2),
        "savings_pct": round(savings_pct, 1),
        "implementation_cost": round(implementation_cost, 2),
        "implementation_time_months": implementation_time_months,
        "roi_pct": round(roi, 1) if roi != float("inf") else "infinite",
        "payback_months": round(payback_months, 1) if payback_months != float("inf") else "immediate",
        "npv_3year": round(npv_3year, 2),
        "impact_classification": impact,
        "recommendation": (
            "Proceed with implementation" if savings_pct >= 5
            else "Savings may not justify implementation cost"
        ),
    }
