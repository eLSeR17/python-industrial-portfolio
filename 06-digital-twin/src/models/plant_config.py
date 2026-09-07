"""Dataclass-based plant topology representation."""

from dataclasses import dataclass, field
from typing import Optional

from src.models.schemas import BufferConfig, ConveyorConfig, MachineConfig


@dataclass
class Station:
    """A production station combining a machine and its input buffer."""

    id: str
    name: str
    machine_config: MachineConfig
    buffer_config: BufferConfig


@dataclass
class PlantTopology:
    """Full topology of a manufacturing plant: stations linked by conveyors."""

    stations: list[Station] = field(default_factory=list)
    conveyors: list[ConveyorConfig] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------------

    def get_station(self, station_id: str) -> Optional[Station]:
        """Return the station with *station_id* or ``None``."""
        for s in self.stations:
            if s.id == station_id:
                return s
        return None

    def get_downstream(self, station_id: str) -> Optional[str]:
        """Return the id of the next station downstream, or ``None``."""
        for c in self.conveyors:
            if c.from_station == station_id:
                return c.to_station
        return None

    def get_upstream(self, station_id: str) -> Optional[str]:
        """Return the id of the previous station upstream, or ``None``."""
        for c in self.conveyors:
            if c.to_station == station_id:
                return c.from_station
        return None

    # -- validation ------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check topology connectivity.  Returns a list of error messages (empty = OK)."""
        errors: list[str] = []
        station_ids = {s.id for s in self.stations}

        if not self.stations:
            errors.append("Topology has no stations.")
            return errors

        for conv in self.conveyors:
            if conv.from_station not in station_ids:
                errors.append(
                    f"Conveyor '{conv.id}' references unknown from_station '{conv.from_station}'."
                )
            if conv.to_station not in station_ids:
                errors.append(
                    f"Conveyor '{conv.id}' references unknown to_station '{conv.to_station}'."
                )

        # Check that at least one station has no downstream (an end station).
        downstream_ids = {c.from_station for c in self.conveyors}
        upstream_ids = {c.to_station for c in self.conveyors}
        has_end = any(s.id not in downstream_ids for s in self.stations)
        if not has_end:
            errors.append("No end station found (every station has a downstream link).")

        return errors
