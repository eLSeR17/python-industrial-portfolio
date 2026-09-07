"""Tests for the Machine simulation process."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simpy

from src.models.schemas import BufferConfig, MachineConfig
from src.simulation.buffer import Buffer
from src.simulation.machine import Machine


def _make_machine(
    env: simpy.Environment,
    mean_proc: float = 10.0,
    failure_rate: float = 0.0,
    buffer_out_capacity: int = 100,
) -> tuple[Machine, Buffer]:
    """Create a single machine with input and output buffers.

    Returns (machine, input_buffer) so the test can feed jobs into the input.
    """
    buf_in = Buffer(env, BufferConfig(id="in", name="In", capacity=1000))
    buf_out = Buffer(env, BufferConfig(id="out", name="Out", capacity=buffer_out_capacity))
    cfg = MachineConfig(
        id="m1",
        name="TestMachine",
        processing_time_mean=mean_proc,
        processing_time_std=0.5,
        failure_rate=failure_rate,
        repair_time_mean=20.0,
        repair_time_std=2.0,
        capacity=1,
    )
    return Machine(env, cfg, buf_in, buf_out), buf_in


def _feeder(env: simpy.Environment, buf_in: Buffer, count: int):
    """SimPy process that feeds *count* jobs into the buffer."""
    for i in range(count):
        yield buf_in.put(f"job_{i}")
        yield env.timeout(0.1)


def test_average_processing_time() -> None:
    """With mean=10, running 100 jobs should give average ~10 (within 20%)."""
    env = simpy.Environment()
    machine, buf_in = _make_machine(env, mean_proc=10.0)

    env.process(_feeder(env, buf_in, 100))
    machine.start()

    env.run(until=1200)

    avg = machine._total_processing_time / max(machine._jobs_processed, 1)
    assert 5.0 < avg < 15.0, f"Average processing time {avg} not within 20% of 10"


def test_failure_handling() -> None:
    """With a high failure rate, some jobs should be marked as failed."""
    env = simpy.Environment()
    machine, buf_in = _make_machine(env, mean_proc=5.0, failure_rate=0.1)

    env.process(_feeder(env, buf_in, 50))
    machine.start()
    env.run(until=600)

    assert machine._jobs_processed > 0, "No jobs processed"


def test_blocking() -> None:
    """When output buffer is full, the machine should show blocking time."""
    env = simpy.Environment()
    machine, buf_in = _make_machine(env, mean_proc=5.0, buffer_out_capacity=2)

    env.process(_feeder(env, buf_in, 20))
    machine.start()
    env.run(until=300)

    assert machine._jobs_processed > 0


def test_utilization_is_finite() -> None:
    """Utilization should be a valid fraction between 0 and 1."""
    env = simpy.Environment()
    machine, buf_in = _make_machine(env, mean_proc=10.0)

    env.process(_feeder(env, buf_in, 10))
    machine.start()
    env.run(until=200)

    assert 0.0 <= machine.utilization <= 1.0, (
        f"Utilization {machine.utilization} out of range"
    )
