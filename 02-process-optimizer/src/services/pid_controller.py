"""Advanced PID controller with auto-tuning, anti-windup, and derivative filter.

This module implements a discrete PID controller suitable for real-time
industrial control loops. Key features:

1. **Ziegler-Nichols auto-tuning** from a relay feedback test (the
   controller oscillates the actuator and measures the ultimate gain and
   period).

2. **Integral anti-windup** via back-calculation: when the actuator
   saturates, the integral term is clamped and a feedback gain *Kt*
   unwinds the integrator to prevent overshoot recovery lag.

3. **Derivative filter** (first-order EMA) to suppress high-frequency
   noise amplification that would otherwise make the derivative term
   unusable with real sensor signals.

Discrete-time update equation::

    e[k] = setpoint - measurement
    P = Kp · e[k]

    I[k] = I[k-1] + Ki · dt · e[k]      (with anti-windup clamping)

    D_filtered = α · D_filtered[k-1] + (1-α) · (e[k] - e[k-1]) / dt
    D = Kd · D_filtered

    output = P + I + D
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from config.settings import settings

logger = logging.getLogger(__name__)


class PIDMode(str, Enum):
    """Controller operating mode."""

    MANUAL = "manual"
    AUTO = "auto"
    CASCADED = "cascaded"


@dataclass
class PIDTuning:
    """Ziegler-Nichols tuning parameters derived from a relay test."""

    kp: float
    ki: float
    kd: float
    ultimate_gain: float = 0.0
    ultimate_period: float = 0.0
    method: str = "ziegler_nichols_step"
    tuned_at: float = field(default_factory=time.time)


class PIDController:
    """Discrete PID controller with anti-windup and derivative filtering.

    Usage::

        pid = PIDController(name="reactor_temp")
        pid.set_tuning(PIDTuning(kp=2.0, ki=0.5, kd=0.1))
        pid.set_limits(output_min=0.0, output_max=100.0)

        while True:
            measurement = read_sensor()
            output = pid.update(setpoint=75.0, measurement=measurement, dt=1.0)
            write_actuator(output)
    """

    def __init__(
        self,
        name: str = "pid",
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
    ) -> None:
        self.name = name
        self.mode: PIDMode = PIDMode.AUTO

        # Tuning parameters.
        self.kp = kp if kp is not None else settings.pid_default_kp
        self.ki = ki if ki is not None else settings.pid_default_ki
        self.kd = kd if kd is not None else settings.pid_default_kd

        # Anti-windup back-calculation gain (typically Kt = 1 / Kp).
        self._kt: float = 1.0 / max(self.kp, 1e-9)

        # Output limits.
        self.output_min = settings.pid_anti_windup_clamp_min
        self.output_max = settings.pid_anti_windup_clamp_max

        # Internal state.
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._derivative_filtered = 0.0
        self._derivative_alpha = settings.pid_derivative_filter_alpha
        self._last_update = 0.0

        # History for auto-tuning analysis.
        self._output_history: list[float] = []
        self._error_history: list[float] = []

    # ------------------------------------------------------------------
    # Core Update
    # ------------------------------------------------------------------

    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: float | None = None,
    ) -> float:
        """Execute one PID control step.

        Args:
            setpoint: Desired target value.
            measurement: Current process variable reading.
            dt: Time since last update in seconds. If None, computed from
                wall clock.

        Returns:
            Controller output (actuator command), clamped to output limits.
        """
        if self.mode != PIDMode.AUTO:
            return self._prev_measurement  # Return last output in manual mode.

        now = time.monotonic()
        if dt is None:
            dt = now - self._last_update if self._last_update > 0 else 1.0
        self._last_update = now

        dt = max(dt, 1e-6)  # Guard against zero/negative dt.

        error = setpoint - measurement

        # --- Proportional term ---
        p_term = self.kp * error

        # --- Integral term with anti-windup (clamping) ---
        self._integral += self.ki * dt * error
        self._integral = np.clip(
            self._integral,
            settings.pid_anti_windup_clamp_min,
            settings.pid_anti_windup_clamp_max,
        )

        # --- Derivative term with first-order EMA filter ---
        raw_derivative = (error - self._prev_error) / dt
        self._derivative_filtered = (
            self._derivative_alpha * self._derivative_filtered
            + (1.0 - self._derivative_alpha) * raw_derivative
        )
        d_term = self.kd * self._derivative_filtered

        # --- Combine ---
        output = p_term + self._integral + d_term

        # --- Anti-windup back-calculation ---
        # When the output is saturated, pull the integrator back proportionally.
        clamped_output = np.clip(output, self.output_min, self.output_max)
        if output != clamped_output:
            # Back-calculate: remove the excess from the integrator.
            self._integral += self._kt * (clamped_output - output)

        self._prev_error = error
        self._prev_measurement = measurement

        # Record for tuning / analysis.
        self._output_history.append(float(clamped_output))
        self._error_history.append(error)
        # Keep bounded.
        if len(self._output_history) > 10000:
            self._output_history = self._output_history[-5000:]
            self._error_history = self._error_history[-5000:]

        return float(clamped_output)

    # ------------------------------------------------------------------
    # Auto-Tuning (Ziegler-Nichols)
    # ------------------------------------------------------------------

    def auto_tune_relay(
        self,
        sensor_fn: callable,
        actuator_fn: callable,
        relay_amplitude: float = 10.0,
        num_oscillations: int = 5,
        dt: float = 0.5,
    ) -> PIDTuning:
        """Perform Ziegler-Nichols relay feedback auto-tuning.

        The relay method works by applying a relay (bang-bang) feedback to
        the process. The process oscillates at the ultimate frequency. From
        the oscillation period (*Tu*) and relay amplitude (*a*), the
        ultimate gain is estimated as *Ku = 4a / (π · d)*, where *d* is
        the amplitude of the oscillation around the setpoint. Ziegler-Nichols
        rules then compute Kp, Ki, Kd.

        Args:
            sensor_fn: Callable that returns the current process measurement.
            actuator_fn: Callable that accepts a float and commands the actuator.
            relay_amplitude: Magnitude of the relay output (±relay_amplitude).
            num_oscillations: Number of full oscillations to observe.
            dt: Sampling interval during the relay test (seconds).

        Returns:
            PIDTuning with the identified parameters.

        Raises:
            RuntimeError: If the relay test does not produce measurable
                oscillations within the allowed time.
        """
        logger.info("Starting relay auto-tuning for PID '%s'", self.name)

        # Save current state to restore later.
        saved_kp, saved_ki, saved_kd = self.kp, self.ki, self.kd
        saved_mode = self.mode
        self.mode = PIDMode.AUTO

        setpoint = float(sensor_fn())
        output_center = 0.0  # Assume centered around 0 for the relay.
        sign = 1.0

        crossings: list[float] = []
        peaks: list[tuple[float, float]] = []  # (time, value)
        t_start = time.monotonic()
        max_duration = 300.0  # Safety timeout: 5 minutes.

        prev_value = float(sensor_fn())
        prev_sign = 1.0

        while len(crossings) < num_oscillations * 2 + 1:
            elapsed = time.monotonic() - t_start
            if elapsed > max_duration:
                raise RuntimeError(
                    f"Relay test did not converge after {max_duration}s — "
                    "check process connectivity or increase relay_amplitude."
                )

            measurement = float(sensor_fn())

            # Relay logic: if measurement > setpoint, push down; else push up.
            if measurement > setpoint:
                actuator_fn(output_center - relay_amplitude)
                sign = -1.0
            else:
                actuator_fn(output_center + relay_amplitude)
                sign = 1.0

            # Detect zero crossings (sign change).
            current_sign = 1.0 if measurement > setpoint else -1.0
            if prev_sign != current_sign and len(crossings) > 0:
                crossings.append(elapsed)

            # Detect peaks.
            if len(peaks) >= 2:
                prev_val = peaks[-1][1]
                if (prev_val > setpoint and measurement < prev_val) or (
                    prev_val < setpoint and measurement > prev_val
                ):
                    peaks.append((elapsed, measurement))

            if not peaks or (measurement > peaks[-1][1] and measurement > setpoint):
                peaks.append((elapsed, measurement))
            elif measurement < peaks[-1][1] and measurement < setpoint:
                peaks.append((elapsed, measurement))

            prev_value = measurement
            prev_sign = current_sign
            time.sleep(dt)

        # Calculate ultimate period from zero crossings.
        if len(crossings) < 2:
            raise RuntimeError("Insufficient zero crossings for tuning")

        periods = np.diff(crossings[1:])  # Skip first partial cycle.
        tu = float(np.mean(periods)) * 2.0  # Full period = 2 × half-period.

        # Estimate oscillation amplitude from peaks above and below setpoint.
        above = [v for _, v in peaks if v > setpoint]
        below = [v for _, v in peaks if v < setpoint]
        a = (float(np.mean(above)) - float(np.mean(below))) / 2.0 if above and below else relay_amplitude

        # Ultimate gain: Ku = 4a / (π · d), where d = relay amplitude.
        ku = (4.0 * a) / (np.pi * relay_amplitude) if relay_amplitude > 0 else 1.0
        ku = max(ku, 0.01)

        # Ziegler-Nichols tuning rules (classic).
        kp_new = 0.6 * ku
        ki_new = 2.0 * kp_new / tu if tu > 0 else 0.0
        kd_new = kp_new * tu / 8.0 if tu > 0 else 0.0

        # Restore normal operation.
        self.kp, self.ki, self.kd = kp_new, ki_new, kd_new
        self._kt = 1.0 / max(kp_new, 1e-9)
        self.mode = saved_mode
        self._integral = 0.0
        self._derivative_filtered = 0.0
        actuator_fn(0.0)

        tuning = PIDTuning(
            kp=kp_new,
            ki=ki_new,
            kd=kd_new,
            ultimate_gain=ku,
            ultimate_period=tu,
        )
        logger.info(
            "Auto-tuning complete for '%s': Ku=%.4f, Tu=%.2fs → Kp=%.4f, Ki=%.4f, Kd=%.6f",
            self.name,
            ku,
            tu,
            kp_new,
            ki_new,
            kd_new,
        )
        return tuning

    def auto_tune_step_response(
        self,
        step_data_time: NDArray[np.float64],
        step_data_output: NDArray[np.float64],
        dead_time: float | None = None,
        time_constant: float | None = None,
    ) -> PIDTuning:
        """Compute PID tuning from step-response characteristics (no relay needed).

        Uses the Ziegler-Nichols open-loop method:
            - L = dead time
            - τ = time constant
            - Kp = 1.2 · τ / (K · L)
            - Ki = Kp / (2 · L)
            - Kd = Kp · 0.5 · L

        Args:
            step_data_time: Time array from a step response test.
            step_data_output: Output array from the same test.
            dead_time: Pre-computed dead time. If None, estimated from data.
            time_constant: Pre-computed time constant. If None, estimated.

        Returns:
            PIDTuning with identified parameters.
        """
        t = np.asarray(step_data_time, dtype=np.float64)
        y = np.asarray(step_data_output, dtype=np.float64)

        y_initial = float(y[0])
        y_final = float(y[-1])
        K = y_final - y_initial  # Process gain for unit step.

        if K == 0:
            raise ValueError("Step response shows no change — cannot identify tuning")

        if dead_time is None:
            # Estimate L as the time when the output first deviates > 2% from initial.
            threshold = y_initial + 0.02 * K
            mask = y > threshold if K > 0 else y < threshold
            idx = int(np.argmax(mask))
            dead_time = float(t[idx]) if idx > 0 else 0.1

        if time_constant is None:
            # τ = time to reach 63.2% of final value from dead_time end.
            y_63 = y_initial + 0.632 * K
            idx_63 = int(np.argmin(np.abs(y - y_63)))
            time_constant = max(0.1, float(t[idx_63]) - dead_time)

        dead_time = max(dead_time, 0.1)
        L = dead_time

        # Ziegler-Nichols open-loop rules.
        kp = 1.2 * time_constant / (K * L) if K != 0 and L != 0 else 1.0
        ki = kp / (2.0 * L) if L > 0 else 0.0
        kd = kp * 0.5 * L

        tuning = PIDTuning(
            kp=kp,
            ki=ki,
            kd=kd,
            method="ziegler_nichols_step",
        )
        logger.info(
            "Step-response tuning for '%s': L=%.2fs, τ=%.2fs → Kp=%.4f, Ki=%.4f, Kd=%.6f",
            self.name,
            dead_time,
            time_constant,
            kp,
            ki,
            kd,
        )
        return tuning

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_tuning(self, tuning: PIDTuning) -> None:
        """Apply a set of tuning parameters."""
        self.kp = tuning.kp
        self.ki = tuning.ki
        self.kd = tuning.kd
        self._kt = 1.0 / max(self.kp, 1e-9)
        logger.info("Applied tuning to '%s': Kp=%.4f Ki=%.4f Kd=%.6f", self.name, self.kp, self.ki, self.kd)

    def set_limits(self, output_min: float, output_max: float) -> None:
        """Set actuator output bounds."""
        if output_min >= output_max:
            raise ValueError(f"output_min ({output_min}) must be < output_max ({output_max})")
        self.output_min = output_min
        self.output_max = output_max

    def reset(self) -> None:
        """Reset all internal state (integral, derivative, history)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._derivative_filtered = 0.0
        self._last_update = 0.0

    @property
    def state(self) -> dict[str, float]:
        """Return a snapshot of the controller's internal state."""
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "integral": self._integral,
            "derivative_filtered": self._derivative_filtered,
            "mode": self.mode.value,
        }
