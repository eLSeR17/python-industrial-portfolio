"""Tests for the Nelder-Mead and coordinate descent optimizers.

All tests use synthetic objective functions — no external services or
Kafka/Redis connections required. The synthetic functions are standard
optimization benchmarks (Rosenbrock, Sphere, Ackley) that verify the
optimizer finds the known global minimum.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.models.schemas import OptimizationConstraints, OptimizationMethod
from src.services.optimizer import (
    ProcessOptimizer,
    _coordinate_descent,
    _nelder_mead_simplex,
    waste_objective,
)


# ---------------------------------------------------------------------------
# Synthetic Objective Functions
# ---------------------------------------------------------------------------

def _sphere(x: NDArray[np.float64]) -> float:
    """Sphere function: f(x) = Σ(x_i²). Minimum = 0 at x = [0, 0, ...]."""
    return float(np.sum(x**2))


def _rosenbrock(x: NDArray[np.float64]) -> float:
    """Rosenbrock function: f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²].

    Minimum = 0 at x = [1, 1, ...].
    """
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return total


def _ackley(x: NDArray[np.float64]) -> float:
    """Ackley function — multimodal with many local minima.

    Global minimum ≈ 0 at x = [0, 0, ...].
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20.0
        + np.e
    )


# ---------------------------------------------------------------------------
# Nelder-Mead Tests
# ---------------------------------------------------------------------------

class TestNelderMead:
    """Test the pure NumPy Nelder-Mead simplex implementation."""

    def test_minimizes_sphere_2d(self) -> None:
        """Should find the minimum of a 2-D sphere function near origin."""
        x0 = np.array([5.0, -3.0], dtype=np.float64)
        best_x, best_f, iters, converged = _nelder_mead_simplex(
            objective=_sphere,
            x0=x0,
            max_iterations=500,
            xatol=1e-8,
            fatol=1e-10,
        )
        assert converged, f"Did not converge after {iters} iterations"
        assert best_f < 1e-6, f"Objective value {best_f} is not close to 0"
        assert np.allclose(best_x, 0.0, atol=1e-4), f"Best x {best_x} not near origin"

    def test_minimizes_sphere_10d(self) -> None:
        """Should handle 10-dimensional problems (within budget)."""
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-10, 10, size=10)
        best_x, best_f, _, converged = _nelder_mead_simplex(
            objective=_sphere,
            x0=x0,
            max_iterations=2000,
            xatol=1e-6,
            fatol=1e-8,
        )
        assert best_f < 1.0, f"10-D sphere: objective {best_f} too high"
        # Nelder-Mead struggles in high dimensions — just check it moved toward 0.
        assert np.mean(np.abs(best_x)) < np.mean(np.abs(x0))

    def test_minimizes_rosenbrock_2d(self) -> None:
        """Should find the Rosenbrock minimum near (1, 1)."""
        x0 = np.array([0.0, 0.0], dtype=np.float64)
        best_x, best_f, _, _ = _nelder_mead_simplex(
            objective=_rosenbrock,
            x0=x0,
            max_iterations=1000,
            xatol=1e-8,
            fatol=1e-10,
        )
        assert best_f < 1.0, f"Rosenbrock: objective {best_f}"
        assert np.allclose(best_x, 1.0, atol=0.1), f"Best x {best_x} not near (1, 1)"

    def test_convergence_budget(self) -> None:
        """With very few iterations, should return even if not converged."""
        x0 = np.array([100.0, 100.0])
        _, _, iters, converged = _nelder_mead_simplex(
            objective=_sphere,
            x0=x0,
            max_iterations=5,
        )
        assert iters <= 5
        # With 5 iterations it almost certainly won't converge.
        assert not converged

    def test_one_dimensional(self) -> None:
        """Should work on 1-D problems (degenerate simplex)."""
        def f(x: NDArray[np.float64]) -> float:
            return float((x[0] - 7.0) ** 2)

        best_x, best_f, _, converged = _nelder_mead_simplex(f, np.array([0.0]), max_iterations=100)
        assert converged
        assert abs(best_x[0] - 7.0) < 0.01
        assert best_f < 1e-6


# ---------------------------------------------------------------------------
# Coordinate Descent Tests
# ---------------------------------------------------------------------------

class TestCoordinateDescent:
    """Test the coordinate descent optimizer."""

    def test_minimizes_sphere(self) -> None:
        """Coordinate descent should find the sphere minimum."""
        x0 = np.array([3.0, -4.0, 2.0], dtype=np.float64)
        best_x, best_f, _, converged = _coordinate_descent(
            objective=_sphere,
            x0=x0,
            max_iterations=200,
        )
        assert best_f < 0.01, f"Sphere via CD: objective {best_f}"
        assert np.allclose(best_x, 0.0, atol=0.05)

    def test_respects_bounds(self) -> None:
        """Variables should stay within specified bounds."""
        def f(x: NDArray[np.float64]) -> float:
            return float(x[0] + x[1])  # Minimized at lower bound.

        x0 = np.array([5.0, 5.0])
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        best_x, _, _, _ = _coordinate_descent(f, x0, bounds=bounds)
        assert 0.0 <= best_x[0] <= 10.0
        assert 0.0 <= best_x[1] <= 10.0

    def test_minimizes_rosenbrock(self) -> None:
        """Should find a reasonable solution to Rosenbrock in 2-D."""
        x0 = np.array([-1.0, 1.0], dtype=np.float64)
        best_x, best_f, _, _ = _coordinate_descent(
            objective=_rosenbrock,
            x0=x0,
            max_iterations=500,
        )
        assert best_f < 10.0  # CD may not reach exact minimum but should be close.


# ---------------------------------------------------------------------------
# Waste Objective Tests
# ---------------------------------------------------------------------------

class TestWasteObjective:
    """Test the waste-reduction objective function."""

    def test_no_change_zero_cost(self) -> None:
        """If setpoints equal current state, cost should be near zero."""
        x = np.array([50.0, 2.5, 100.0])
        names = ["temperature", "pressure", "flow"]
        current = {"temperature": 50.0, "pressure": 2.5, "flow": 100.0}
        constraints = OptimizationConstraints()
        cost = waste_objective(x, names, current, constraints)
        assert cost < 0.01, f"Expected near-zero cost, got {cost}"

    def test_violation_penalty(self) -> None:
        """Violating bounds should produce a large penalty."""
        x = np.array([150.0])  # Way above limit.
        names = ["temperature"]
        current = {"temperature": 50.0}
        constraints = OptimizationConstraints(variable_limits={"temperature": (0.0, 100.0)})
        cost = waste_objective(x, names, current, constraints)
        assert cost > 10.0, f"Expected large penalty for bound violation, got {cost}"

    def test_large_move_penalty(self) -> None:
        """Moving far from the current operating point should cost more."""
        x_small = np.array([55.0])
        x_large = np.array([100.0])
        names = ["temperature"]
        current = {"temperature": 50.0}
        constraints = OptimizationConstraints()
        cost_small = waste_objective(x_small, names, current, constraints)
        cost_large = waste_objective(x_large, names, current, constraints)
        assert cost_large > cost_small


# ---------------------------------------------------------------------------
# Integration: ProcessOptimizer
# ---------------------------------------------------------------------------

class TestProcessOptimizer:
    """Integration tests for the ProcessOptimizer class."""

    def test_nelder_mead_optimization(self) -> None:
        """Run a full optimization cycle with Nelder-Mead."""
        proc = ProcessOptimizer()
        current = {"temperature": 75.0, "pressure": 2.5}
        result = proc.optimize(
            process_id="test-01",
            current_setpoints=current,
            variable_names=["temperature", "pressure"],
            variable_limits={"temperature": (60.0, 90.0), "pressure": (1.0, 5.0)},
            method=OptimizationMethod.NELDER_MEAD,
            max_iterations=100,
        )
        assert result.process_id == "test-01"
        assert result.method == OptimizationMethod.NELDER_MEAD
        assert "temperature" in result.recommended_setpoints
        assert "pressure" in result.recommended_setpoints
        # The optimizer should respect bounds.
        assert 60.0 <= result.recommended_setpoints["temperature"] <= 90.0
        assert 1.0 <= result.recommended_setpoints["pressure"] <= 5.0

    def test_coordinate_descent_optimization(self) -> None:
        """Run a full optimization cycle with coordinate descent."""
        proc = ProcessOptimizer()
        current = {"temp": 50.0}
        result = proc.optimize(
            process_id="test-02",
            current_setpoints=current,
            variable_names=["temp"],
            variable_limits={"temp": (0.0, 100.0)},
            method=OptimizationMethod.COORDINATE_DESCENT,
            max_iterations=50,
        )
        assert result.process_id == "test-02"
        assert result.iterations_used > 0

    def test_history_tracking(self) -> None:
        """Optimization history should accumulate across runs."""
        proc = ProcessOptimizer()
        for i in range(3):
            proc.optimize(
                process_id="test-hist",
                current_setpoints={"x": float(i)},
                variable_names=["x"],
                method=OptimizationMethod.NELDER_MEAD,
                max_iterations=10,
            )
        history = proc.get_history("test-hist")
        assert len(history) == 3

    def test_get_last_result(self) -> None:
        """get_last_result should return the most recent OptimizationResult."""
        proc = ProcessOptimizer()
        result = proc.optimize(
            process_id="test-last",
            current_setpoints={"x": 1.0},
            variable_names=["x"],
            max_iterations=10,
        )
        last = proc.get_last_result("test-last")
        assert last is not None
        assert last.result_id == result.result_id
