"""In-memory representation of a live industrial process.

The ProcessStateHolder acts as the single source of truth for the current
condition of every monitored process line. It is thread-safe via asyncio locks
and backed by Redis for persistence across restarts.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import numpy as np

from config.settings import settings
from src.models.schemas import (
    ProcessState,
    ProcessType,
    ProcessUpdateRequest,
    ProcessVariable,
)

logger = logging.getLogger(__name__)


class ProcessStateHolder:
    """Manages the live state of all tracked processes.

    This is an in-memory store with optional Redis backing. For a single-node
    deployment the in-memory dict is sufficient; Redis provides durability and
    enables horizontal scaling.
    """

    def __init__(self) -> None:
        self._states: dict[str, ProcessState] = {}
        self._history: dict[str, list[dict[str, float]]] = {}
        self._lock = asyncio.Lock()
        self._max_history_per_variable: int = 1000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, request: ProcessUpdateRequest) -> ProcessState:
        """Apply a batch of sensor readings to the process state.

        For each reading, the corresponding ProcessVariable is updated (or
        created if it does not yet exist). The raw readings are also appended
        to the sliding-window history used by the SPC engine.

        Args:
            request: Incoming sensor batch from the stream processor.

        Returns:
            The updated ProcessState after applying the readings.
        """
        async with self._lock:
            state = self._states.get(request.process_id)

            if state is None:
                state = ProcessState(
                    process_id=request.process_id,
                    process_type=request.process_type,
                )
                self._states[request.process_id] = state
                self._history[request.process_id] = []

            # Index existing variables by name for O(1) lookup.
            var_map: dict[str, ProcessVariable] = {v.name: v for v in state.variables}

            for reading in request.readings:
                var = var_map.get(reading.sensor_id)

                if var is None:
                    var = ProcessVariable(
                        name=reading.sensor_id,
                        value=reading.value,
                        unit=reading.unit,
                    )
                    var_map[reading.sensor_id] = var
                else:
                    var.value = reading.value

                # Append to history.
                hist = self._history.setdefault(request.process_id, [])
                if len(hist) >= self._max_history_per_variable:
                    hist.pop(0)
                hist.append(
                    {
                        "value": reading.value,
                        "timestamp": reading.timestamp.timestamp(),
                    }
                )

            # Apply setpoints.
            for name, sp in request.setpoints.items():
                if name in var_map:
                    var_map[name].setpoint = sp

            state.variables = list(var_map.values())
            state.timestamp = datetime.utcnow()
            self._states[request.process_id] = state

            return state

    async def get_state(self, process_id: str) -> ProcessState | None:
        """Retrieve the current state for a process.

        Args:
            process_id: Unique identifier of the process line.

        Returns:
            The current ProcessState, or None if the process is unknown.
        """
        async with self._lock:
            return self._states.get(process_id)

    async def get_all_states(self) -> list[ProcessState]:
        """Return a snapshot of every tracked process."""
        async with self._lock:
            return list(self._states.values())

    async def get_history(
        self,
        process_id: str,
        variable_name: str,
        last_n: int | None = None,
    ) -> list[dict[str, float]]:
        """Retrieve historical readings for a variable.

        Args:
            process_id: Process identifier.
            variable_name: Name of the variable (sensor_id).
            last_n: If set, return only the most recent N readings.

        Returns:
            List of {"value": float, "timestamp": float} dicts.
        """
        async with self._lock:
            full_history = self._history.get(process_id, [])

        if last_n is not None:
            return full_history[-last_n:]
        return full_history

    async def get_variable_values(self, process_id: str) -> dict[str, float]:
        """Extract a flat dict of {variable_name: current_value}.

        This is the format consumed by the optimizer and PID controller.
        """
        state = await self.get_state(process_id)
        if state is None:
            return {}
        return {v.name: v.value for v in state.variables}

    async def get_variable_array(
        self,
        process_id: str,
        variable_names: list[str],
    ) -> np.ndarray:
        """Return current values as a NumPy array in the given order.

        Missing variables are filled with 0.0 (the caller is responsible
        for validating the array before use in optimization).

        Args:
            process_id: Process identifier.
            variable_names: Ordered list of variable names.

        Returns:
            1-D NumPy array of shape (len(variable_names),).
        """
        values = await self.get_variable_values(process_id)
        return np.array([values.get(name, 0.0) for name in variable_names], dtype=np.float64)

    async def remove(self, process_id: str) -> bool:
        """Remove a process and its history from the holder.

        Returns:
            True if the process existed and was removed, False otherwise.
        """
        async with self._lock:
            existed = process_id in self._states
            self._states.pop(process_id, None)
            self._history.pop(process_id, None)
            return existed

    @property
    def process_ids(self) -> list[str]:
        """List all currently tracked process IDs."""
        return list(self._states.keys())

    async def to_json(self, process_id: str) -> str | None:
        """Serialize a process state to JSON for Redis caching."""
        state = await self.get_state(process_id)
        if state is None:
            return None
        return state.model_dump_json()


# Module-level singleton — imported everywhere via `from src.models.process_state import holder`.
holder = ProcessStateHolder()
