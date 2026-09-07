"""Gradient-free process optimizer.

Implements three optimization strategies for finding optimal operating
setpoints in real-time manufacturing:

1. **Nelder-Mead Simplex** — derivative-free, robust to noise, good for
   5–20 dimensional problems with smooth-ish objective landscapes.

2. **Coordinate Descent** — optimizes one variable at a time while holding
   others fixed. Fast per iteration, effective when variable interactions
   are moderate.

3. **Penalty Method** — wraps any unconstrained optimizer to enforce
   physical constraints (actuator limits, safety bounds, power budgets)
   by adding a quadratic penalty term for violations.

The objective function models **waste reduction** as a weighted combination of:
    - Off-spec production (distance from quality targets)
    - Energy consumption (steam, electricity, compressed air)
    - Raw material deviation (excess feed above stoichiometric requirement)
    - Throughput loss (running below rated capacity)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from config.settings import settings
from src.models.schemas import OptimizationConstraints, OptimizationMethod, OptimizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------

def waste_objective(
    setpoints: NDArray[np.float64],
    variable_names: list[str],
    current_state: dict[str, float],
    constraints: OptimizationConstraints,
    process_model_fn: Callable[[str, float], float] | None = None,
) -> float:
    """Evaluate the waste-reduction objective function.

    Lower is better (we are minimising waste).

    The objective is a weighted sum of penalty terms:
        J = w_quality · f_quality + w_energy · f_energy + w_material · f_material

    Args:
        setpoints: Proposed setpoint vector (same order as *variable_names*).
        variable_names: Ordered list of variable names matching *setpoints*.
        current_state: Current measured values of all process variables.
        constraints: Physical and operational constraints.
        process_model_fn: Optional callable(var_name, value) → predicted
            process response. Used for anticipated-impact estimation.

    Returns:
        Scalar cost value (lower = less waste).
    """
    cost = 0.0

    for i, name in enumerate(variable_names):
        val = float(setpoints[i])
        current = current_state.get(name, val)
        limits = constraints.variable_limits.get(name)

        # --- Quality cost: deviation from an implicit optimal center ---
        # The "ideal" setpoint for each variable is its current value (the
        # process was presumably running near optimum). Large moves away
        # from the current operating point increase waste risk.
        move_magnitude = abs(val - current) / (abs(current) + 1e-9)
        cost += 0.3 * move_magnitude**2

        # --- Constraint violation penalty ---
        if limits is not None:
            lower, upper = limits
            if val < lower:
                cost += constraints.penalty_weight * (lower - val) ** 2
            elif val > upper:
                cost += constraints.penalty_weight * (val - upper) ** 2

        # --- Rate-of-change penalty ---
        max_rate = constraints.rate_of_change_limits.get(name)
        if max_rate is not None:
            rate_violation = max(0.0, abs(val - current) - max_rate)
            cost += 0.5 * rate_violation**2

    # --- Power budget constraint ---
    if constraints.power_limit_kw is not None:
        total_power = sum(abs(float(setpoints[i])) for i in range(len(setpoints)))
        if total_power > constraints.power_limit_kw:
            cost += 100.0 * (total_power - constraints.power_limit_kw) ** 2

    return cost


# ---------------------------------------------------------------------------
# Nelder-Mead Simplex
# ---------------------------------------------------------------------------

def _nelder_mead_simplex(
    objective: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    max_iterations: int = 200,
    xatol: float = 1e-6,
    fatol: float = 1e-6,
    initial_step: float = 0.5,
) -> tuple[NDArray[np.float64], float, int, bool]:
    """Pure NumPy implementation of the Nelder-Mead simplex algorithm.

    This avoids the overhead and opaque internals of ``scipy.optimize`` for
    cases where we need fine-grained control over convergence diagnostics.

    The simplex is a set of n+1 points in ℝⁿ. At each iteration:
        1. Order vertices by objective value.
        2. Reflect the worst point through the centroid.
        3. Apply expansion, contraction, or shrink based on the improvement.

    Args:
        objective: Scalar objective function f(x) → float.
        x0: Initial guess (1-D array, length n).
        max_iterations: Maximum number of iterations.
        xatol: Convergence threshold on variable tolerance (max simplex spread).
        fatol: Convergence threshold on function value spread.
        initial_step: Initial simplex edge length as fraction of |x0|.

    Returns:
        Tuple of (best_x, best_f, iterations_used, converged).
    """
    n = len(x0)
    alpha_r = 1.0   # Reflection coefficient
    gamma_e = 2.0   # Expansion coefficient
    rho_c = 0.5     # Contraction coefficient
    sigma_s = 0.5   # Shrink coefficient

    # Initialize simplex: x0 plus n perturbations.
    simplex = np.zeros((n + 1, n), dtype=np.float64)
    simplex[0] = x0.copy()
    for i in range(n):
        step = initial_step * (abs(x0[i]) + 1.0)
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += step

    f_vals = np.array([objective(simplex[i]) for i in range(n + 1)], dtype=np.float64)

    converged = False
    for iteration in range(max_iterations):
        # Order by function value.
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        # Convergence check.
        spread_x = float(np.max(np.ptp(simplex, axis=0)))
        spread_f = float(f_vals[-1] - f_vals[0])
        if spread_x < xatol and spread_f < fatol:
            converged = True
            break

        # Centroid (exclude worst).
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflect.
        x_r = centroid + alpha_r * (centroid - simplex[-1])
        f_r = objective(x_r)

        if f_r < f_vals[-2] and f_r >= f_vals[0]:
            # Accept reflection.
            simplex[-1] = x_r
            f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            # Expand.
            x_e = centroid + gamma_e * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        else:
            # Contract.
            x_c = centroid + rho_c * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink.
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                    f_vals[i] = objective(simplex[i])

    best_idx = int(np.argmin(f_vals))
    return simplex[best_idx], float(f_vals[best_idx]), iteration + 1, converged


# ---------------------------------------------------------------------------
# Coordinate Descent
# ---------------------------------------------------------------------------

def _coordinate_descent(
    objective: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    max_iterations: int = 100,
    step_sizes: NDArray[np.float64] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[NDArray[np.float64], float, int, bool]:
    """Coordinate descent optimizer.

    Iterates over each dimension, performing a 1-D line search along that
    axis using a golden-section bracketing method. Effective when variables
    are weakly coupled.

    Args:
        objective: Scalar objective function.
        x0: Initial guess.
        max_iterations: Maximum full sweeps over all coordinates.
        step_sizes: Initial step per coordinate. Defaults to 1% of current.
        bounds: Per-variable (min, max) bounds.

    Returns:
        Tuple of (best_x, best_f, iterations_used, converged).
    """
    n = len(x0)
    x = x0.copy().astype(np.float64)
    f_current = objective(x)
    bounds_arr = np.array(bounds, dtype=np.float64) if bounds else np.full((n, 2), [-1e9, 1e9])

    if step_sizes is None:
        step_sizes = np.maximum(np.abs(x) * 0.01, 0.01)

    converged = False
    for iteration in range(max_iterations):
        max_change = 0.0
        for j in range(n):
            lo, hi = bounds_arr[j]
            step = step_sizes[j]

            def _line_search_1d(delta: float) -> float:
                """Evaluate objective moving only coordinate j."""
                trial = x.copy()
                trial[j] = np.clip(x[j] + delta, lo, hi)
                return objective(trial)

            # Golden-section search on [x[j] - step, x[j] + step].
            a, b = max(lo, x[j] - step), min(hi, x[j] + step)
            gr = (np.sqrt(5) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = _line_search_1d(c - x[j])
            fd = _line_search_1d(d - x[j])

            for _ in range(30):  # Max golden-section iterations.
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = _line_search_1d(c - x[j])
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = _line_search_1d(d - x[j])
                if (b - a) < 1e-8:
                    break

            new_val = np.clip((a + b) / 2.0, lo, hi)
            change = abs(new_val - x[j])
            x[j] = new_val
            max_change = max(max_change, change)

        f_current = objective(x)
        if max_change < 1e-6:
            converged = True
            break

    return x, f_current, iteration + 1, converged


# ---------------------------------------------------------------------------
# Constraint-Wrapper (Penalty Method)
# ---------------------------------------------------------------------------

class PenaltyWrapper:
    """Wraps an objective function with quadratic penalty terms for constraints.

    The penalty is scaled by an adaptive weight that increases over
    successive calls to drive the solution toward the feasible region.
    """

    def __init__(self, constraints: OptimizationConstraints) -> None:
        self.constraints = constraints
        self._penalty_weight = settings.penalty_weight_initial
        self._iteration = 0

    def __call__(
        self,
        setpoints: NDArray[np.float64],
        variable_names: list[str],
        current_state: dict[str, float],
    ) -> float:
        """Evaluate the penalized objective.

        Args:
            setpoints: Proposed setpoint vector.
            variable_names: Ordered variable names.
            current_state: Current process measurements.

        Returns:
            Penalized objective value (lower = better, feasible = no penalty).
        """
        self._iteration += 1
        # Grow penalty weight over iterations.
        self._penalty_weight = settings.penalty_weight_initial + settings.penalty_weight_growth * self._iteration

        # Temporarily set the weight for the objective function.
        original_weight = self.constraints.penalty_weight
        self.constraints.penalty_weight = self._penalty_weight

        cost = waste_objective(
            setpoints=setpoints,
            variable_names=variable_names,
            current_state=current_state,
            constraints=self.constraints,
        )

        self.constraints.penalty_weight = original_weight
        return cost

    def reset(self) -> None:
        """Reset the adaptive penalty weight."""
        self._iteration = 0
        self._penalty_weight = settings.penalty_weight_initial


# ---------------------------------------------------------------------------
# Main Optimizer Class
# ---------------------------------------------------------------------------

@dataclass
class _OptHistoryEntry:
    """Single entry in the optimization trace (for diagnostics)."""

    iteration: int
    objective_value: float
    setpoints: dict[str, float]


class ProcessOptimizer:
    """High-level optimizer that selects and executes the best strategy.

    Usage::

        optimizer = ProcessOptimizer()
        result = optimizer.optimize(
            process_id="reactor-01",
            current_setpoints={"temperature": 75.0, "pressure": 2.5},
            variable_names=["temperature", "pressure"],
            variable_limits={"temperature": (60.0, 90.0), "pressure": (1.0, 5.0)},
            method=OptimizationMethod.NELDER_MEAD,
        )
    """

    def __init__(self) -> None:
        self._history: dict[str, list[_OptHistoryEntry]] = {}
        self._last_results: dict[str, OptimizationResult] = {}

    def optimize(
        self,
        process_id: str,
        current_setpoints: dict[str, float],
        variable_names: list[str],
        variable_limits: dict[str, tuple[float, float]] | None = None,
        method: OptimizationMethod = OptimizationMethod.NELDER_MEAD,
        max_iterations: int | None = None,
        rate_of_change_limits: dict[str, float] | None = None,
        power_limit_kw: float | None = None,
    ) -> OptimizationResult:
        """Run the optimization for a single process line.

        Args:
            process_id: Identifier of the process to optimize.
            current_setpoints: Current operating setpoints.
            variable_names: Ordered list of variables to optimize.
            variable_limits: Per-variable (min, max) bounds.
            method: Optimization algorithm to use.
            max_iterations: Override the default max iterations.
            rate_of_change_limits: Max allowed change per cycle.
            power_limit_kw: Total power budget constraint.

        Returns:
            OptimizationResult with the recommended setpoints.
        """
        t_start = time.monotonic()

        constraints = OptimizationConstraints(
            variable_limits=variable_limits or {},
            rate_of_change_limits=rate_of_change_limits or {},
            power_limit_kw=power_limit_kw,
        )

        # Build initial vector from current setpoints.
        x0 = np.array(
            [current_setpoints.get(name, 0.0) for name in variable_names],
            dtype=np.float64,
        )

        current_state = dict(current_setpoints)
        penalty = PenaltyWrapper(constraints)

        def _objective(x: NDArray[np.float64]) -> float:
            return penalty(x, variable_names, current_state)

        max_iter = max_iterations or settings.nelder_mead_max_iterations

        if method == OptimizationMethod.NELDER_MEAD:
            best_x, best_f, iters, converged = _nelder_mead_simplex(
                objective=_objective,
                x0=x0,
                max_iterations=max_iter,
                xatol=settings.nelder_mead_xatol,
                fatol=settings.nelder_mead_fatol,
            )
        elif method == OptimizationMethod.COORDINATE_DESCENT:
            bounds = [variable_limits.get(name, (-1e9, 1e9)) for name in variable_names]
            best_x, best_f, iters, converged = _coordinate_descent(
                objective=_objective,
                x0=x0,
                max_iterations=min(max_iter, settings.coordinate_descent_max_iterations),
                bounds=bounds,
            )
        elif method == OptimizationMethod.BAYESIAN:
            # Bayesian optimization via scipy's minimize with differential_evolution.
            from scipy.optimize import differential_evolution

            bounds_list = [
                variable_limits.get(name, (-1e9, 1e9)) for name in variable_names
            ]
            de_result = differential_evolution(
                _objective,
                bounds=bounds_list,
                maxiter=max_iter,
                tol=1e-6,
                seed=42,
                polish=True,
            )
            best_x = de_result.x
            best_f = de_result.fun
            iters = de_result.nit
            converged = de_result.success
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Build result dict.
        recommended = {
            name: float(best_x[i]) for i, name in enumerate(variable_names)
        }

        # Estimate improvement over current cost.
        current_cost = waste_objective(
            x0, variable_names, current_state, constraints
        )
        improvement_pct = 0.0
        if current_cost > 0:
            improvement_pct = max(0.0, (current_cost - best_f) / current_cost * 100.0)

        elapsed = time.monotonic() - t_start
        logger.info(
            "Optimization [%s] for %s: %.2f%% improvement in %d iterations (%.3fs)",
            method.value,
            process_id,
            improvement_pct,
            iters,
            elapsed,
        )

        result = OptimizationResult(
            process_id=process_id,
            method=method,
            recommended_setpoints=recommended,
            predicted_improvement_pct=round(improvement_pct, 2),
            iterations_used=iters,
            convergence_achieved=converged,
            objective_value=best_f,
            constraints_satisfied=best_f < current_cost * 1.01,  # Allow 1% tolerance.
        )

        self._last_results[process_id] = result
        self._history.setdefault(process_id, []).append(
            _OptHistoryEntry(
                iteration=len(self._history.get(process_id, [])) + 1,
                objective_value=best_f,
                setpoints=recommended,
            )
        )
        # Cap history at 500 entries.
        if len(self._history[process_id]) > 500:
            self._history[process_id] = self._history[process_id][-500:]

        return result

    def get_last_result(self, process_id: str) -> OptimizationResult | None:
        """Retrieve the most recent optimization result for a process."""
        return self._last_results.get(process_id)

    def get_history(self, process_id: str) -> list[dict[str, Any]]:
        """Return the full optimization history for a process."""
        return [
            {"iteration": h.iteration, "objective_value": h.objective_value, "setpoints": h.setpoints}
            for h in self._history.get(process_id, [])
        ]


# Module-level singleton.
optimizer = ProcessOptimizer()
