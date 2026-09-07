"""Electrical unit conversions and power calculations.

Covers active/reactive/apparent power relationships, power factor,
kVA/kW conversions, and common industrial electrical units.
"""

import math


def kw_to_kva(kw: float, pf: float) -> float:
    """Convert active power (kW) to apparent power (kVA).

    Parameters
    ----------
    kw : Active power in kilowatts.
    pf : Power factor (0 < pf <= 1).

    Returns
    -------
    Apparent power in kVA.
    """
    if pf <= 0 or pf > 1:
        raise ValueError(f"Power factor must be in (0, 1], got {pf}")
    return kw / pf


def kva_to_kw(kva: float, pf: float) -> float:
    """Convert apparent power (kVA) to active power (kW)."""
    if pf <= 0 or pf > 1:
        raise ValueError(f"Power factor must be in (0, 1], got {pf}")
    return kva * pf


def reactive_power_kw(kw: float, pf: float) -> float:
    """Calculate reactive power (kVAR) from active power and power factor.

    Q = P * tan(arccos(PF))
    """
    if pf <= 0 or pf > 1:
        raise ValueError(f"Power factor must be in (0, 1], got {pf}")
    angle = math.acos(pf)
    return kw * math.tan(angle)


def power_factor(kw: float, kva: float) -> float:
    """Calculate power factor from active and apparent power.

    PF = P / S
    """
    if kva <= 0:
        raise ValueError(f"Apparent power must be > 0, got {kva}")
    pf = kw / kva
    return min(max(pf, 0.0), 1.0)


def apparent_power(active_kw: float, reactive_kvar: float) -> float:
    """Calculate apparent power from active and reactive components.

    S = sqrt(P^2 + Q^2)
    """
    return math.sqrt(active_kw**2 + reactive_kvar**2)


def capacitor_bank_kvar(
    current_pf: float,
    target_pf: float,
    active_power_kw: float,
) -> float:
    """Calculate the capacitor bank size needed to improve power factor.

    Qc = P * (tan(arccos(PF_old)) - tan(arccos(PF_new)))

    Parameters
    ----------
    current_pf : Current (measured) power factor.
    target_pf : Desired power factor after correction.
    active_power_kw : Average active power in kW.

    Returns
    -------
    Required capacitor bank size in kVAR.
    """
    if not (0 < current_pf <= 1):
        raise ValueError(f"current_pf must be in (0, 1], got {current_pf}")
    if not (0 < target_pf <= 1):
        raise ValueError(f"target_pf must be in (0, 1], got {target_pf}")
    if target_pf < current_pf:
        raise ValueError("target_pf must be >= current_pf")

    q_before = reactive_power_kw(active_power_kw, current_pf)
    q_after = reactive_power_kw(active_power_kw, target_pf)
    return q_before - q_after


def kwh_to_joules(kwh: float) -> float:
    """Convert kilowatt-hours to joules. 1 kWh = 3.6 MJ."""
    return kwh * 3_600_000.0


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kilowatt-hours."""
    return joules / 3_600_000.0


def kw_to_btuh(kw: float) -> float:
    """Convert kW to BTU/hour. 1 kW ≈ 3412.14 BTU/h."""
    return kw * 3412.14


def btuh_to_kw(btuh: float) -> float:
    """Convert BTU/hour to kW."""
    return btuh / 3412.14


def co2_from_kwh(kwh: float, emission_factor_kg_per_kwh: float = 0.42) -> float:
    """Estimate CO₂ emissions from electricity consumption.

    Default factor 0.42 kg CO₂/kWh represents a typical US grid mix.
    Regional factors range from 0.05 (hydro/nuclear-heavy) to 0.9 (coal-heavy).
    """
    return kwh * emission_factor_kg_per_kwh
