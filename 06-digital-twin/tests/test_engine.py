"""Tests for the SimulationEngine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schemas import (
    BufferConfig,
    ConveyorConfig,
    MachineConfig,
    SimulationConfig,
)
from src.simulation.engine import SimulationEngine


def _two_machine_config(seed: int = 42, duration: float = 500.0) -> SimulationConfig:
    """Build a simple two-machine-one-buffer line configuration."""
    return SimulationConfig(
        id="test-line",
        name="2-machine line",
        duration=duration,
        warmup_period=50.0,
        machines=[
            MachineConfig(
                id="m1",
                name="CNC-1",
                processing_time_mean=10.0,
                processing_time_std=1.0,
                failure_rate=0.005,
                repair_time_mean=20.0,
                repair_time_std=3.0,
                capacity=1,
            ),
            MachineConfig(
                id="m2",
                name="CNC-2",
                processing_time_mean=12.0,
                processing_time_std=1.5,
                failure_rate=0.003,
                repair_time_mean=25.0,
                repair_time_std=5.0,
                capacity=1,
            ),
        ],
        buffers=[
            BufferConfig(id="b1", name="WIP", capacity=20, initial_level=0),
        ],
        conveyors=[
            ConveyorConfig(
                id="c1",
                name="Belt-1",
                from_station="m1",
                to_station="m2",
                speed=1.0,
                capacity=5,
            ),
        ],
        schedule_type="FIFO",
        random_seed=seed,
    )


def test_single_run_produces_output() -> None:
    """A 500-minute run with two machines should produce > 0 units."""
    config = _two_machine_config(duration=500.0)
    engine = SimulationEngine(config)
    result = engine.run()

    assert result.throughput > 0, f"Expected positive throughput, got {result.throughput}"
    assert result.total_produced > 0, f"Expected produced > 0, got {result.total_produced}"
    assert result.avg_cycle_time > 0, f"Expected positive cycle time, got {result.avg_cycle_time}"


def test_machines_have_utilization() -> None:
    """Each machine should show some utilization after a run."""
    config = _two_machine_config(duration=500.0)
    engine = SimulationEngine(config)
    result = engine.run()

    assert "m1" in result.utilization, "Machine m1 not in utilization dict"
    assert "m2" in result.utilization, "Machine m2 not in utilization dict"
    assert result.utilization["m1"] > 0, "m1 utilization should be > 0"
    assert result.utilization["m2"] > 0, "m2 utilization should be > 0"


def test_replications_are_consistent() -> None:
    """Three replications of the same config should produce throughput
    within a factor of 2 of each other (same mean processing time)."""
    config = _two_machine_config(duration=500.0)
    engine = SimulationEngine(config)
    results = engine.run_replications(3)

    assert len(results) == 3
    throughputs = [r.throughput for r in results]
    assert max(throughputs) < 2 * max(min(throughputs), 0.01), (
        f"Replications inconsistent: {throughputs}"
    )
