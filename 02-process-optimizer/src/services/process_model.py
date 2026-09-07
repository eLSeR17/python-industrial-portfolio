"""Digital process model based on First-Order Plus Dead Time (FOPDT).

A FOPDT model describes the response of a process to a step input change:

    G(s) = K · e^(-L·s) / (τ·s + 1)

Where:
    K  = process gain (% output / % input)
    τ  = time constant (seconds) — how fast the process responds
    L  = dead time / delay (seconds) — transport lag

Step-response simulation, parameter identification from experimental
data via least-squares fitting, and a ProcessModelStore for caching
fitted models per process line.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FOPDTParameters:
    """Identified parameters of a FOPDT model."""

    gain: float
    time_constant: float
    dead_time: float
    identified_at: datetime = field(default_factory=datetime.utcnow)
    fit_quality: float = 0.0  # R² of the fit (0–1, higher = better)

    def __post_init__(self) -> None:
        """Validate physical plausibility of the identified parameters."""
        if self.gain < settings.fopdt_min_gain or self.gain > settings.fopdt_max_gain:
            logger.warning(
                "Gain %.4f outside expected range [%.2f, %.2f]",
                self.gain,
                settings.fopdt_min_gain,
                settings.fopdt_max_gain,
            )
        if self.time_constant < settings.fopdt_min_time_constant:
            logger.warning("Time constant %.2f is unrealistically small", self.time_constant)

    def to_dict(self) -> dict[str, float]:
        """Serialize to a flat dict for JSON / Redis."""
        return {
            "gain": self.gain,
            "time_constant": self.time_constant,
            "dead_time": self.dead_time,
            "fit_quality": self.fit_quality,
        }


def simulate_step_response(
    K: float,
    tau: float,
    L: float,
    dt: float = 1.0,
    duration: float | None = None,
    input_amplitude: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate the time-domain step response of a FOPDT system.

    The output at time *t* for a unit step at *t = 0* is:
        y(t) = K · input_amplitude · (1 - exp(-(t - L) / τ))   if t > L
        y(t) = 0                                               if t ≤ L

    Args:
        K: Process gain.
        tau: Time constant (seconds).
        L: Dead time (seconds).
        dt: Time step (seconds). Smaller = more accurate.
        duration: Total simulation time. Defaults to 5 × tau + L.
        input_amplitude: Magnitude of the step input.

    Returns:
        Tuple of (time_array, output_array), both 1-D NumPy arrays.
    """
    if duration is None:
        duration = 5.0 * tau + L + dt
    t = np.arange(0.0, duration, dt, dtype=np.float64)
    y = np.zeros_like(t)

    mask = t > L
    y[mask] = K * input_amplitude * (1.0 - np.exp(-(t[mask] - L) / tau))
    return t, y


def fit_fopdt(
    time_data: NDArray[np.float64],
    output_data: NDArray[np.float64],
    input_amplitude: float = 1.0,
) -> FOPDTParameters:
    """Identify FOPDT parameters from step-response data.

    The fitting procedure:
    1. Estimate the steady-state gain K from the final value.
    2. Find the 63.2% response point to estimate τ + L.
    3. Use least-squares optimization over (K, τ, L) to minimize RMSE.

    Args:
        time_data: Time stamps of the measured response (seconds).
        output_data: Process variable measurements.
        input_amplitude: Magnitude of the step input that produced the response.

    Returns:
        Identified FOPDTParameters with fit_quality (R²).
    """
    if len(time_data) < 5:
        raise ValueError("Need at least 5 data points for FOPDT fitting")
    if time_data.ndim != 1 or output_data.ndim != 1:
        raise ValueError("time_data and output_data must be 1-D arrays")

    t = np.asarray(time_data, dtype=np.float64)
    y = np.asarray(output_data, dtype=np.float64)

    y_final = float(y[-1])
    y_initial = float(y[0])
    K_init = (y_final - y_initial) / input_amplitude if input_amplitude != 0 else 1.0

    # 63.2% point estimate for τ + L.
    y_63 = y_initial + 0.632 * (y_final - y_initial)
    idx_63 = int(np.argmin(np.abs(y - y_63)))
    tau_plus_l = float(t[idx_63]) if idx_63 > 0 else 10.0

    L_init = max(0.0, tau_plus_l * 0.2)
    tau_init = max(0.1, tau_plus_l - L_init)

    def _model(params: NDArray[np.float64], tt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate FOPDT model at given parameter values."""
        K, tau, dead = params
        tau = max(tau, 0.01)
        dead = max(dead, 0.0)
        out = np.zeros_like(tt)
        mask = tt > dead
        out[mask] = y_initial + K * input_amplitude * (1.0 - np.exp(-(tt[mask] - dead) / tau))
        return out

    def _cost(params: NDArray[np.float64]) -> float:
        """Sum of squared residuals."""
        predicted = _model(params, t)
        return float(np.sum((y - predicted) ** 2))

    from scipy.optimize import minimize as sp_minimize

    x0 = np.array([K_init, tau_init, L_init], dtype=np.float64)
    bounds = [
        (settings.fopdt_min_gain, settings.fopdt_max_gain),
        (settings.fopdt_min_time_constant, settings.fopdt_max_time_constant),
        (settings.fopdt_min_dead_time, settings.fopdt_max_dead_time),
    ]

    result = sp_minimize(_cost, x0, method="L-BFGS-B", bounds=bounds)
    K_opt, tau_opt, L_opt = result.x

    # Compute R² for fit quality.
    y_pred = _model(result.x, t)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    params = FOPDTParameters(
        gain=float(K_opt),
        time_constant=float(tau_opt),
        dead_time=float(L_opt),
        fit_quality=r_squared,
    )
    logger.info(
        "FOPDT fit: K=%.4f, τ=%.2fs, L=%.2fs, R²=%.4f",
        params.gain,
        params.time_constant,
        params.dead_time,
        params.fit_quality,
    )
    return params


def predict_output(
    params: FOPDTParameters,
    current_output: float,
    new_input: float,
    dt: float = 1.0,
) -> float:
    """Predict the next-step output using the discrete FOPDT approximation.

    Uses a first-order forward Euler discretization:
        y[k+1] = α · y[k] + K · (1 - α) · u[k - d]

    Where α = exp(-dt / τ) and d = round(L / dt).

    Args:
        params: Identified FOPDT model parameters.
        current_output: Current measured output.
        new_input: Proposed new input value.
        dt: Discrete time step.

    Returns:
        Predicted next-step output value.
    """
    alpha = np.exp(-dt / max(params.time_constant, 0.01))
    return alpha * current_output + params.gain * (1.0 - alpha) * new_input


# ---------------------------------------------------------------------------
# Model Store
# ---------------------------------------------------------------------------

class ProcessModelStore:
    """In-memory cache of FOPDT models, keyed by (process_id, variable_name).

    In production this would be backed by PostgreSQL. For the portfolio
    project the in-memory dict is sufficient.
    """

    def __init__(self) -> None:
        self._models: dict[str, FOPDTParameters] = {}

    def _key(self, process_id: str, variable_name: str) -> str:
        return f"{process_id}::{variable_name}"

    def store(
        self,
        process_id: str,
        variable_name: str,
        params: FOPDTParameters,
    ) -> None:
        """Cache a fitted model."""
        key = self._key(process_id, variable_name)
        self._models[key] = params
        logger.info(
            "Stored FOPDT model for %s/%s (R²=%.4f)",
            process_id,
            variable_name,
            params.fit_quality,
        )

    def get(self, process_id: str, variable_name: str) -> FOPDTParameters | None:
        """Retrieve a cached model, or None if not yet identified."""
        return self._models.get(self._key(process_id, variable_name))

    def has_model(self, process_id: str, variable_name: str) -> bool:
        """Check whether a model exists."""
        return self._key(process_id, variable_name) in self._models

    def remove(self, process_id: str, variable_name: str) -> bool:
        """Delete a cached model. Returns True if it existed."""
        key = self._key(process_id, variable_name)
        return self._models.pop(key, None) is not None

    @property
    def model_count(self) -> int:
        return len(self._models)

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all cached models (for the dashboard)."""
        result = []
        for key, params in self._models.items():
            pid, var = key.split("::", 1)
            result.append(
                {
                    "process_id": pid,
                    "variable": var,
                    "gain": params.gain,
                    "time_constant": params.time_constant,
                    "dead_time": params.dead_time,
                    "fit_quality": params.fit_quality,
                    "identified_at": params.identified_at.isoformat(),
                }
            )
        return result


# Module-level singleton.
model_store = ProcessModelStore()
