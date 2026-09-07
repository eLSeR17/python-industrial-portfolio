"""Conveyor transport process between two buffers in the simulation."""

from typing import Generator

import simpy

from src.models.schemas import ConveyorConfig
from src.simulation.buffer import Buffer


class Conveyor:
    """Continuously transports items from a source buffer to a destination buffer.

    The conveyor is modelled as a SimPy ``Process`` that pulls items from
    *source_buffer*, waits for a configurable travel time, and then pushes
    them into *dest_buffer*.

    Args:
        env: SimPy environment.
        config: Conveyor configuration.
        source_buffer: The upstream :class:`Buffer` to pull from.
        dest_buffer: The downstream :class:`Buffer` to push to.
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: ConveyorConfig,
        source_buffer: Buffer,
        dest_buffer: Buffer,
    ) -> None:
        self.env = env
        self.config = config
        self.source = source_buffer
        self.dest = dest_buffer
        self.conveyor_id = config.id
        self.name = config.name

        self._travel_time = 1.0 / config.speed if config.speed > 0 else 1.0
        self._items_transported = 0
        self._items_rejected = 0
        self._process: simpy.Process | None = None

    def start(self) -> None:
        """Kick off the conveyor process."""
        self._process = self.env.process(self.run())

    def run(self) -> Generator:
        """SimPy process: keep moving items from source to destination."""
        while True:
            # Wait for an item in the source buffer
            item = yield self.source.get()

            # Simulate travel time
            yield self.env.timeout(self._travel_time)

            # Place in destination (blocks if dest is full)
            yield self.dest.put(item)
            self._items_transported += 1

    @property
    def items_transported(self) -> int:
        """Total items successfully moved by this conveyor."""
        return self._items_transported

    @property
    def items_rejected(self) -> int:
        """Items that could not be placed in the destination (full buffer)."""
        return self._items_rejected

    def summary(self) -> dict:
        """Return a summary dict for metrics collection."""
        return {
            "conveyor_id": self.conveyor_id,
            "name": self.name,
            "from_station": self.config.from_station,
            "to_station": self.config.to_station,
            "travel_time": self._travel_time,
            "items_transported": self._items_transported,
            "items_rejected": self._items_rejected,
        }
