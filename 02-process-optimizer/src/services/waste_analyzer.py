"""Statistical Process Control (SPC) and waste analysis engine.

SPC charts (X-bar, R), CUSUM change detection, Western Electric zone
rules, and OEE calculation with bottleneck identification. Consumes
historical process data and produces control charts with violation
annotations so operators can separate real process disturbances from
normal variation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config.settings import settings
from src.models.schemas import (
    AlarmSeverity,
    OEEComponents,
    SPCAlarm,
    SPCChart,
    SPCResult,
)
from src.utils.time_series import CUSUMDetector, linear_trend, SlidingWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control Limit Computation
# ---------------------------------------------------------------------------

# Standard SPC constants for sample size n=2..10.
# A2: multiplier for X-bar chart UCL/LCL from R-bar.
# D3, D4: multipliers for R-chart LCL/UCL.
# d2: expected value of R / σ (for sigma estimation from R-bar).
_SPC_CONSTANTS: dict[int, dict[str, float]] = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267, "d2": 1.128},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574, "d2": 1.693},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282, "d2": 2.059},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114, "d2": 2.326},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004, "d2": 2.534},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078},
}


def _get_constants(n: int) -> dict[str, float]:
    """Retrieve SPC constants for subgroup size *n*.

    For n > 10 we extrapolate using the closest available constants.
    """
    n_capped = max(2, min(n, 10))
    return _SPC_CONSTANTS[n_capped]


def compute_xbar_r_chart(
    data: NDArray[np.float64],
    subgroup_size: int = 5,
) -> tuple[SPCChart, SPCChart]:
    """Compute X-bar and R (range) control charts.

    The data is divided into subgroups of *subgroup_size* consecutive
    observations. For each subgroup:
        - X-bar = mean of the subgroup
        - R = max − min of the subgroup

    Control limits for the X-bar chart:
        UCL = X̿ + A2 · R̄
        LCL = X̿ − A2 · R̄

    Control limits for the R chart:
        UCL = D4 · R̄
        LCL = D3 · R̄

    Args:
        data: 1-D array of process measurements.
        subgroup_size: Number of observations per subgroup (2–10 recommended).

    Returns:
        Tuple of (xbar_chart, r_chart).
    """
    if data.ndim != 1 or len(data) < subgroup_size * 2:
        raise ValueError(
            f"Need at least {subgroup_size * 2} data points, got {len(data)}"
        )

    # Reshape into subgroups.
    n_complete = (len(data) // subgroup_size) * subgroup_size
    trimmed = data[:n_complete].reshape(-1, subgroup_size)

    xbars = np.mean(trimmed, axis=1)
    ranges = np.ptp(trimmed, axis=1)  # max - min per row.

    xbar_bar = float(np.mean(xbars))
    r_bar = float(np.mean(ranges))

    constants = _get_constants(subgroup_size)
    a2 = constants["A2"]
    d3 = constants["D3"]
    d4 = constants["D4"]

    xbar_ucl = xbar_bar + a2 * r_bar
    xbar_lcl = xbar_bar - a2 * r_bar

    r_ucl = d4 * r_bar
    r_lcl = d3 * r_bar

    # Identify violations (points outside control limits).
    xbar_violations = [int(i) for i in np.where((xbars > xbar_ucl) | (xbars < xbar_lcl))[0]]
    r_violations = [int(i) for i in np.where((ranges > r_ucl) | (ranges < r_lcl))[0]]

    xbar_chart = SPCChart(
        variable="xbar",
        chart_type="xbar",
        center_line=xbar_bar,
        upper_control_limit=xbar_ucl,
        lower_control_limit=xbar_lcl,
        data_points=xbars.tolist(),
        violations=xbar_violations,
    )

    r_chart = SPCChart(
        variable="r",
        chart_type="r",
        center_line=r_bar,
        upper_control_limit=r_ucl,
        lower_control_limit=r_lcl,
        data_points=ranges.tolist(),
        violations=r_violations,
    )

    return xbar_chart, r_chart


# ---------------------------------------------------------------------------
# Western Electric Rules
# ---------------------------------------------------------------------------

def western_electric_violations(
    data: NDArray[np.float64],
    center_line: float,
    ucl: float,
    lcl: float,
) -> list[dict[str, Any]]:
    """Apply the Western Electric (Nelson) rules to a time series.

    Zones are defined relative to the control limits:
        Zone C: within ±1σ of center line  (±1/3 of distance from CL to limit)
        Zone B: between 1σ and 2σ          (1/3 to 2/3)
        Zone A: between 2σ and 3σ          (2/3 to the limit)

    Rules:
        1. Any point beyond Zone A (outside control limits).
        2. Two out of three consecutive points in Zone A or beyond (same side).
        3. Four out of five consecutive points in Zone B or beyond (same side).
        4. Eight consecutive points on the same side of the center line.

    Args:
        data: Time series of X-bar values.
        center_line: Center line (X̿).
        ucl: Upper control limit.
        lcl: Lower control limit.

    Returns:
        List of violation dicts with ``rule``, ``index``, and ``message``.
    """
    sigma = (ucl - center_line) / 3.0 if ucl != center_line else 1.0
    violations: list[dict[str, Any]] = []

    for i in range(len(data)):
        val = float(data[i])
        deviation = val - center_line

        # Rule 1: point beyond 3σ (outside control limits).
        if val > ucl or val < lcl:
            violations.append({
                "rule": "Rule 1 — Point beyond 3σ",
                "index": i,
                "message": f"Point {val:.4f} exceeds control limit [{lcl:.4f}, {ucl:.4f}]",
            })
            continue

        # Determine zone (A, B, C) and side (+, -).
        abs_dev = abs(deviation)
        side = "above" if deviation >= 0 else "below"

        if abs_dev > 2.0 * sigma:
            zone = "A"
        elif abs_dev > 1.0 * sigma:
            zone = "B"
        else:
            zone = "C"

        # Rule 2: 2 of 3 in Zone A or beyond (same side).
        if i >= 2 and zone == "A":
            window = data[i - 2 : i + 1]
            same_side_count = sum(
                1 for v in window
                if (v - center_line) * deviation > 0 and abs(v - center_line) >= 2.0 * sigma
            )
            if same_side_count >= 2:
                violations.append({
                    "rule": f"Rule 2 — 2 of 3 in Zone A ({side})",
                    "index": i,
                    "message": f"2 of last 3 points in Zone A or beyond on the {side} side",
                })

        # Rule 3: 4 of 5 in Zone B or beyond (same side).
        if i >= 4:
            window = data[i - 4 : i + 1]
            same_side_beyond_b = sum(
                1 for v in window
                if (v - center_line) * deviation > 0 and abs(v - center_line) >= 1.0 * sigma
            )
            if same_side_beyond_b >= 4:
                violations.append({
                    "rule": f"Rule 3 — 4 of 5 in Zone B ({side})",
                    "index": i,
                    "message": f"4 of last 5 points in Zone B or beyond on the {side} side",
                })

        # Rule 4: 8 consecutive on same side.
        if i >= 7:
            window = data[i - 7 : i + 1]
            all_same = all((v - center_line) * deviation > 0 for v in window)
            if all_same:
                violations.append({
                    "rule": f"Rule 4 — 8 consecutive {side}",
                    "index": i,
                    "message": f"8 consecutive points on the {side} side of center line",
                })

    return violations


# ---------------------------------------------------------------------------
# Process Capability Indices
# ---------------------------------------------------------------------------

def compute_capability(
    data: NDArray[np.float64],
    usl: float,
    lsl: float,
) -> dict[str, float]:
    """Compute process capability indices Cp and Cpk.

    Cp = (USL − LSL) / (6σ)          — potential capability (ignores centering)
    Cpk = min(Cpu, Cpl)              — actual capability (accounts for centering)
        Cpu = (USL − μ) / (3σ)
        Cpl = (μ − LSL) / (3σ)

    Args:
        data: Process measurements.
        usl: Upper specification limit.
        lsl: Lower specification limit.

    Returns:
        Dict with Cp, Cpk, Cpu, Cpl, and sigma estimate.
    """
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    if sigma == 0:
        return {"Cp": float("inf"), "Cpk": float("inf"), "Cpu": float("inf"), "Cpl": float("inf"), "sigma": 0.0}

    cp = (usl - lsl) / (6.0 * sigma)
    cpu = (usl - mu) / (3.0 * sigma)
    cpl = (mu - lsl) / (3.0 * sigma)
    cpk = min(cpu, cpl)

    return {
        "Cp": round(cp, 4),
        "Cpk": round(cpk, 4),
        "Cpu": round(cpu, 4),
        "Cpl": round(cpl, 4),
        "sigma": round(sigma, 6),
    }


# ---------------------------------------------------------------------------
# OEE (Overall Equipment Effectiveness)
# ---------------------------------------------------------------------------

@dataclass
class StageData:
    """Data for a single stage in a multi-stage process line."""

    name: str
    planned_downtime_hours: float = 0.0
    actual_runtime_hours: float = 24.0
    ideal_cycle_time_seconds: float = 1.0
    actual_parts_produced: int = 0
    theoretical_max_parts: int = 0
    good_parts: int = 0
    total_parts: int = 0


def compute_oee(stages: list[StageData]) -> dict[str, OEEComponents]:
    """Calculate OEE for each stage and the overall line.

    OEE = Availability × Performance × Quality

    Args:
        stages: List of StageData for each process stage.

    Returns:
        Dict with per-stage OEE and a key ``"_overall"`` for the line.
    """
    results: dict[str, OEEComponents] = {}
    availabilities = []
    performances = []
    qualities = []

    for stage in stages:
        planned_hours = settings.oee_planned_production_hours
        available_hours = planned_hours - stage.planned_downtime_hours
        availability = available_hours / planned_hours if planned_hours > 0 else 0.0
        availability = min(max(availability, 0.0), 1.0)

        if stage.ideal_cycle_time_seconds > 0 and available_hours > 0:
            theoretical_max = int(available_hours * 3600.0 / stage.ideal_cycle_time_seconds)
            performance = stage.actual_parts_produced / theoretical_max if theoretical_max > 0 else 0.0
        else:
            performance = 0.0
        performance = min(max(performance, 0.0), 1.0)

        quality = stage.good_parts / stage.total_parts if stage.total_parts > 0 else 0.0
        quality = min(max(quality, 0.0), 1.0)

        oee_comp = OEEComponents(
            availability=round(availability, 4),
            performance=round(performance, 4),
            quality=round(quality, 4),
        )
        results[stage.name] = oee_comp
        availabilities.append(availability)
        performances.append(performance)
        qualities.append(quality)

    # Overall: geometric mean is more accurate than arithmetic for multiplicative factors.
    if results:
        results["_overall"] = OEEComponents(
            availability=round(float(np.mean(availabilities)), 4),
            performance=round(float(np.mean(performances)), 4),
            quality=round(float(np.mean(qualities)), 4),
        )

    return results


def identify_bottleneck(stages: list[StageData]) -> dict[str, Any]:
    """Identify the bottleneck stage (highest utilization / slowest cycle time).

    Args:
        stages: List of StageData.

    Returns:
        Dict with ``bottleneck_stage``, ``cycle_times``, and ``utilizations``.
    """
    cycle_times: dict[str, float] = {}
    utilizations: dict[str, float] = {}

    for stage in stages:
        ct = stage.ideal_cycle_time_seconds
        cycle_times[stage.name] = ct

        if stage.actual_runtime_hours > 0:
            util = stage.actual_parts_produced / (stage.actual_runtime_hours * 3600.0 / max(ct, 0.01))
        else:
            util = 0.0
        utilizations[stage.name] = min(round(util, 4), 1.0)

    bottleneck = max(cycle_times, key=cycle_times.get)  # type: ignore[arg-type]

    return {
        "bottleneck_stage": bottleneck,
        "cycle_times": cycle_times,
        "utilizations": utilizations,
    }


# ---------------------------------------------------------------------------
# High-Level SPC Analyzer
# ---------------------------------------------------------------------------

class WasteAnalyzer:
    """Orchestrates SPC analysis, OEE, and bottleneck detection.

    This is the service called by the API layer. It combines the low-level
    chart computation with alarm generation and trend detection.
    """

    def __init__(self) -> None:
        self._cusum_detectors: dict[str, CUSUMDetector] = {}

    def analyze(
        self,
        process_id: str,
        variable_data: dict[str, NDArray[np.float64]],
        subgroup_size: int = 5,
        specification_limits: dict[str, tuple[float, float]] | None = None,
    ) -> SPCResult:
        """Run a full SPC analysis on all tracked variables.

        Args:
            process_id: Process identifier.
            variable_data: Dict of {variable_name: 1-D array of recent measurements}.
            subgroup_size: Subgroup size for X-bar/R charts.
            specification_limits: Optional {variable: (LSL, USL)} for capability.

        Returns:
            SPCResult with charts, alarms, CUSUM state, and capability indices.
        """
        charts: list[SPCChart] = []
        alarms: list[SPCAlarm] = []
        cusum_state: dict[str, float] = {}
        capability: dict[str, float] = {}

        for var_name, data in variable_data.items():
            if len(data) < subgroup_size * 2:
                logger.debug("Skipping variable '%s': insufficient data (%d points)", var_name, len(data))
                continue

            # --- X-bar and R charts ---
            try:
                xbar_chart, r_chart = compute_xbar_r_chart(data, subgroup_size)
                xbar_chart.variable = var_name
                r_chart.variable = f"{var_name}_range"
                charts.extend([xbar_chart, r_chart])
            except ValueError as e:
                logger.warning("Chart computation failed for '%s': %s", var_name, e)
                continue

            # --- Western Electric violations on X-bar chart ---
            xbar_data = np.array(xbar_chart.data_points, dtype=np.float64)
            we_violations = western_electric_violations(
                xbar_data,
                xbar_chart.center_line,
                xbar_chart.upper_control_limit,
                xbar_chart.lower_control_limit,
            )
            for v in we_violations:
                alarms.append(SPCAlarm(
                    rule=v["rule"],
                    severity=AlarmSeverity.WARNING,
                    variable=var_name,
                    message=v["message"],
                    value=float(xbar_data[min(v["index"], len(xbar_data) - 1)]),
                    limit=xbar_chart.upper_control_limit,
                ))

            # --- CUSUM change detection ---
            key = f"{process_id}::{var_name}"
            if key not in self._cusum_detectors:
                self._cusum_detectors[key] = CUSUMDetector(
                    target=xbar_chart.center_line,
                    threshold=settings.spc_cusum_threshold,
                    drift=settings.spc_cusum_drift * float(np.std(xbar_data, ddof=1) or 1.0),
                )
            detector = self._cusum_detectors[key]
            for val in xbar_data:
                result = detector.update(float(val))
                if result["alarm"]:
                    alarms.append(SPCAlarm(
                        rule=f"CUSUM {result['direction']} shift",
                        severity=AlarmSeverity.CRITICAL,
                        variable=var_name,
                        message=(
                            f"CUSUM detected a sustained {result['direction']} shift "
                            f"in {var_name} (S+={result['s_pos']:.2f}, S-={result['s_neg']:.2f})"
                        ),
                    ))
            cusum_state[var_name] = detector.state["s_pos"] + detector.state["s_neg"]

            # --- Process capability ---
            if specification_limits and var_name in specification_limits:
                lsl, usl = specification_limits[var_name]
                cap = compute_capability(data, usl, lsl)
                capability[var_name] = cap.get("Cpk", 0.0)

        # --- R-chart violations as alarms ---
        for chart in charts:
            if chart.chart_type == "r" and chart.violations:
                alarms.append(SPCAlarm(
                    rule="R-chart out of control",
                    severity=AlarmSeverity.WARNING,
                    variable=chart.variable,
                    message=f"Range chart for {chart.variable} has {len(chart.violations)} out-of-control point(s)",
                ))

        logger.info(
            "SPC analysis for '%s': %d charts, %d alarms",
            process_id,
            len(charts),
            len(alarms),
        )

        return SPCResult(
            process_id=process_id,
            charts=charts,
            alarms=alarms,
            cusum_state=cusum_state,
            process_capability=capability,
        )

    def reset_cusum(self, process_id: str, variable_name: str) -> None:
        """Reset the CUSUM detector for a variable (e.g., after alarm acknowledgment)."""
        key = f"{process_id}::{variable_name}"
        if key in self._cusum_detectors:
            self._cusum_detectors[key].reset()


# Module-level singleton.
waste_analyzer = WasteAnalyzer()
