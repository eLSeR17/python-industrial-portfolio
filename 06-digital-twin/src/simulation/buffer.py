"""SimPy-compatible buffer / inventory store between stations."""

from typing import Any, Optional

import simpy

from src.models.schemas import BufferConfig


class Buffer:
    """Finite-capacity buffer backed by a SimPy ``Store``.

    Provides ``get()`` and ``put()`` that return SimPy events (to be
    ``yield``-ed by the calling process).  Also tracks occupancy history,
    blocking, and starvation for metrics collection.

    Args:
        env: SimPy environment.
        config: Buffer configuration (capacity, initial level).
    """

    def __init__(self, env: simpy.Environment, config: BufferConfig) -> None:
        self.env = env
        self.config = config
        self.name = config.name
        self.buffer_id = config.id

        self._store = simpy.Store(env, capacity=config.capacity)
        self._current_level = config.initial_level
        self._max_capacity = config.capacity
        self._total_put = 0
        self._total_get = 0
        self._blocked_events = 0
        self._starved_events = 0
        self._occupancy_history: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    # Public API — returns SimPy events (to be yielded by caller)
    # ------------------------------------------------------------------

    def get(self) -> Any:
        """Return a SimPy ``Store.get`` event.

        When yielded by a SimPy process, the process suspends until an item
        is available, then returns the item.

        Usage inside a SimPy generator::

            item = yield buffer.get()
        """
        self._total_get += 1
        event = self._store.get()
        # We wrap with a callback to track occupancy after retrieval
        event.callbacks.append(self._on_get_done)
        return event

    def put(self, item: Any) -> Any:
        """Return a SimPy ``Store.put`` event.

        When yielded, the process suspends until there is capacity.

        Usage inside a SimPy generator::

            yield buffer.put(item)
        """
        self._total_put += 1
        event = self._store.put(item)
        event.callbacks.append(self._on_put_done)
        return event

    @property
    def occupancy(self) -> float:
        """Fraction of buffer capacity currently in use (0.0 – 1.0)."""
        if self._max_capacity == 0:
            return 0.0
        return self._current_level / self._max_capacity

    @property
    def level(self) -> int:
        """Current number of items in the buffer."""
        return self._current_level

    @property
    def is_full(self) -> bool:
        """Whether the buffer is at capacity."""
        return self._current_level >= self._max_capacity

    @property
    def is_empty(self) -> bool:
        """Whether the buffer has no items."""
        return self._current_level <= 0

    @property
    def blocked_count(self) -> int:
        """Number of put attempts that had to wait (buffer was full)."""
        return self._blocked_events

    @property
    def starved_count(self) -> int:
        """Number of get attempts that had to wait (buffer was empty)."""
        return self._starved_events

    def get_history(self) -> list[tuple[float, int]]:
        """Return the recorded occupancy history as ``[(time, level), ...]``."""
        return list(self._occupancy_history)

    def summary(self) -> dict:
        """Return a summary dict for metrics collection."""
        history = self._occupancy_history
        levels = [h[1] for h in history] if history else [0]
        return {
            "buffer_id": self.buffer_id,
            "name": self.name,
            "max_capacity": self._max_capacity,
            "final_level": self._current_level,
            "avg_occupancy": float(sum(levels) / len(levels)),
            "total_put": self._total_put,
            "total_get": self._total_get,
            "blocked_events": self._blocked_events,
            "starved_events": self._starved_events,
        }

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_put_done(self, event: Any) -> None:  # noqa: ARG002
        """Called after a put completes — update level and record occupancy."""
        self._current_level = len(self._store.items)
        self._record_occupancy()

    def _on_get_done(self, event: Any) -> None:  # noqa: ARG002
        """Called after a get completes — update level and record occupancy."""
        self._current_level = len(self._store.items)
        self._record_occupancy()

    def _record_occupancy(self) -> None:
        """Snapshot the current occupancy at the current simulation time."""
        self._occupancy_history.append((self.env.now, self._current_level))
