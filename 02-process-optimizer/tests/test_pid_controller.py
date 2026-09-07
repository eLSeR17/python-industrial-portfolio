"""Tests for the PID controller.

Validates the discrete PID update equation, anti-windup behavior,
derivative filtering, Ziegler-Nichols auto-tuning (step-response method),
and the output clamping logic. All tests use synthetic process responses
— no hardware in the loop.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.services.pid_controller import PIDController, PIDMode, PIDTuning


# ---------------------------------------------------------------------------
# Synthetic Process for Closed-Loop Tests
# ---------------------------------------------------------------------------

class _FakeProcess:
    """Simulates a simple first-order process for testing the PID in a loop.

    The process is: dy/dt = (K · u - y) / τ
    Discrete: y[k+1] = y[k] + dt/τ · (K · u[k] - y[k])
    """

    def __init__(self, gain: float = 2.0, tau: float = 10.0, dt: float = 1.0) -> None:
        self.gain = gain
        self.tau = tau
        self.dt = dt
        self.y = 0.0

    def step(self, u: float) -> float:
        """Advance one time step with input *u*."""
        dy = (self.gain * u - self.y) / self.tau * self.dt
        self.y += dy
        return self.y

    def reset(self, y0: float = 0.0) -> None:
        self.y = y0


# ---------------------------------------------------------------------------
# PID Update Tests
# ---------------------------------------------------------------------------

class TestPIDUpdate:
    """Verify the core PID update equation and its components."""

    def test_proportional_only(self) -> None:
        """With Ki=Kd=0, output should be Kp * error."""
        pid = PIDController("p_only", kp=2.0, ki=0.0, kd=0.0)
        output = pid.update(setpoint=10.0, measurement=5.0, dt=1.0)
        assert abs(output - 10.0) < 1e-6, f"Expected ~10.0 (2.0 * 5.0), got {output}"

    def test_integral_accumulates(self) -> None:
        """Integral term should accumulate error over time."""
        pid = PIDController("i_test", kp=0.0, ki=1.0, kd=0.0)
        for _ in range(5):
            output = pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
        # After 5 steps with error=10 and Ki=1, I = 5 * 1.0 * 10 = 50.
        assert abs(output - 50.0) < 1e-6, f"Expected ~50.0, got {output}"

    def test_derivative_responds_to_change(self) -> None:
        """Derivative term should react to the rate of change of error."""
        pid = PIDController("d_test", kp=0.0, ki=0.0, kd=2.0)
        # First step: error goes from 0 to 10.
        output_step1 = pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
        # Second step: error stays at 10 (no change).
        output_step2 = pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
        # The filtered derivative should have decayed significantly (EMA with alpha=0.1).
        assert abs(output_step2) < abs(output_step1) * 0.5, (
            f"D term should decrease when error is constant: step1={output_step1}, step2={output_step2}"
        )

    def test_output_clamped(self) -> None:
        """Output should be clamped to [output_min, output_max]."""
        pid = PIDController("clamp_test", kp=100.0, ki=0.0, kd=0.0)
        pid.set_limits(output_min=0.0, output_max=50.0)
        output = pid.update(setpoint=100.0, measurement=0.0, dt=1.0)
        assert output == 50.0, f"Expected clamped output 50.0, got {output}"

    def test_manual_mode_returns_constant(self) -> None:
        """In manual mode, the controller should not update."""
        pid = PIDController("manual", kp=5.0)
        pid.mode = PIDMode.MANUAL
        output1 = pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
        output2 = pid.update(setpoint=20.0, measurement=0.0, dt=1.0)
        assert output1 == output2, "Manual mode output should be constant"

    def test_zero_dt_produces_finite_output(self) -> None:
        """A very small dt should not cause division by zero."""
        pid = PIDController("small_dt", kp=1.0, ki=1.0, kd=1.0)
        output = pid.update(setpoint=5.0, measurement=3.0, dt=1e-8)
        assert np.isfinite(output), f"Output is not finite: {output}"


# ---------------------------------------------------------------------------
# Anti-Windup Tests
# ---------------------------------------------------------------------------

class TestAntiWindup:
    """Verify that integral anti-windup prevents excessive overshoot recovery."""

    def test_integral_bounded_by_clamp(self) -> None:
        """Integral term should not exceed the anti-windup clamp bounds."""
        pid = PIDController("aw", kp=0.0, ki=10.0, kd=0.0)
        # Large sustained error should saturate the integrator.
        for _ in range(100):
            pid.update(setpoint=1000.0, measurement=0.0, dt=1.0)
        # The integral should be clamped, not 1000 * 10.
        assert pid._integral <= 100.0 + 1.0, (
            f"Integral {pid._integral} exceeds clamp bound"
        )

    def test_back_calculation_reduces_integral(self) -> None:
        """When output saturates, back-calculation should pull the integral down."""
        pid = PIDController("bc", kp=1.0, ki=1.0, kd=0.0)
        # Use wide output limits so output never saturates — integral accumulates freely.
        pid.set_limits(output_min=-1000.0, output_max=1000.0)
        for _ in range(50):
            pid.update(setpoint=100.0, measurement=0.0, dt=1.0)
        integral_before = pid._integral  # Integral accumulated to the anti-windup clamp bound.

        # Now tighten limits so output saturates — back-calculation activates.
        pid.set_limits(output_min=0.0, output_max=100.0)
        for _ in range(10):
            pid.update(setpoint=100.0, measurement=0.0, dt=1.0)

        # The integral should have decreased due to back-calculation.
        assert pid._integral < integral_before, (
            f"Integral did not decrease: {pid._integral} >= {integral_before}"
        )


# ---------------------------------------------------------------------------
# Derivative Filter Tests
# ---------------------------------------------------------------------------

class TestDerivativeFilter:
    """Verify the first-order EMA derivative filter suppresses noise."""

    def test_filter_smooths_noise(self) -> None:
        """The filtered derivative should be less noisy than the raw derivative."""
        pid = PIDController("df", kp=0.0, ki=0.0, kd=1.0)
        rng = np.random.default_rng(99)
        # Generate noisy constant error.
        errors = [10.0 + rng.normal(0, 2.0) for _ in range(50)]
        raw_derivs: list[float] = []
        filtered_derivs: list[float] = []

        prev_err = 0.0
        for e in errors:
            pid.update(setpoint=10.0, measurement=10.0 - e, dt=1.0)
            raw = e - prev_err
            raw_derivs.append(raw)
            filtered_derivs.append(pid._derivative_filtered)
            prev_err = e

        raw_std = float(np.std(raw_derivs[5:]))
        filt_std = float(np.std(filtered_derivs[5:]))
        # Filtered should be significantly less variable.
        assert filt_std < raw_std, (
            f"Filter did not reduce noise: filtered_std={filt_std:.4f} >= raw_std={raw_std:.4f}"
        )


# ---------------------------------------------------------------------------
# Auto-Tuning Tests
# ---------------------------------------------------------------------------

class TestAutoTuning:
    """Test Ziegler-Nichols auto-tuning via step response."""

    def test_step_response_tuning(self) -> None:
        """Should compute reasonable Kp, Ki, Kd from a step response."""
        # Generate a synthetic step response.
        t = np.linspace(0, 100, 500)
        K_true, tau_true, L_true = 3.0, 20.0, 5.0
        y = np.zeros_like(t)
        mask = t > L_true
        y[mask] = K_true * (1.0 - np.exp(-(t[mask] - L_true) / tau_true))

        pid = PIDController("tune_test")
        tuning = pid.auto_tune_step_response(t, y, dead_time=L_true, time_constant=tau_true)

        assert tuning.kp > 0, "Kp should be positive"
        assert tuning.ki >= 0, "Ki should be non-negative"
        assert tuning.kd >= 0, "Kd should be non-negative"
        assert tuning.method == "ziegler_nichols_step"

    def test_tuning_applies_to_controller(self) -> None:
        """set_tuning should update Kp, Ki, Kd."""
        pid = PIDController("apply_tune")
        tuning = PIDTuning(kp=5.0, ki=2.0, kd=0.5)
        pid.set_tuning(tuning)
        assert pid.kp == 5.0
        assert pid.ki == 2.0
        assert pid.kd == 0.5


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------

class TestPIDReset:
    """Verify that reset() clears all internal state."""

    def test_reset_clears_integral(self) -> None:
        pid = PIDController("reset")
        for _ in range(20):
            pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0
        assert pid._derivative_filtered == 0.0
