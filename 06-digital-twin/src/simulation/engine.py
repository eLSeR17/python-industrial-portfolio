"""SimPy simulation engine: orchestrates environment and all processes."""

import random
from typing import Any, Generator, Optional

import numpy as np
import simpy

from src.models.schemas import BufferConfig, SimulationConfig, SimulationResult
from src.services.metrics_collector import MetricsCollector
from src.simulation.buffer import Buffer
from src.simulation.conveyor import Conveyor
from src.simulation.machine import Machine


class SimulationEngine:
    """Wraps a SimPy discrete-event simulation.

    Given a :class:`SimulationConfig`, the engine builds the machine /
    buffer / conveyor topology, wires up the SimPy processes, and
    returns :class:`SimulationResult` once the simulation completes.

    Args:
        config: Full simulation configuration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.env: Optional[simpy.Environment] = None
        self.machines: dict[str, Machine] = {}
        self.buffers: dict[str, Buffer] = {}
        self.conveyors: dict[str, Conveyor] = {}
        self.metrics = MetricsCollector()
        self._job_counter = 0

    def run(self) -> SimulationResult:
        """Execute the simulation once and return the result.

        A ``random_seed`` from the config ensures reproducibility.
        """
        self.env = simpy.Environment()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.metrics = MetricsCollector()
        self.machines = {}
        self.buffers = {}
        self.conveyors = {}
        self._job_counter = 0

        self._setup_processes()

        self.env.run(until=self.config.duration)

        return self._collect_result()

    def run_replications(self, n: int) -> list[SimulationResult]:
        """Run *n* independent replications with different seeds.

        Args:
            n: Number of replications (>= 1).

        Returns:
            List of :class:`SimulationResult`, one per replication.
        """
        results: list[SimulationResult] = []
        for i in range(n):
            self.config.random_seed = (abs(hash(self.config.id)) + i * 7919) % (2**31)
            result = self.run()
            result.replication_results = [{"replication": i + 1}]
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _setup_processes(self) -> None:
        """Create SimPy processes for source, machines, buffers, conveyors."""
        assert self.env is not None

        # Create buffers
        for buf_cfg in self.config.buffers:
            self.buffers[buf_cfg.id] = Buffer(self.env, buf_cfg)

        # Identify source machines (no upstream conveyor feeding them)
        upstream_targets = {c.to_station for c in self.config.conveyors}
        source_machine_ids = [
            m.id for m in self.config.machines if m.id not in upstream_targets
        ]

        # Create a source buffer for each source machine
        for machine_id in source_machine_ids:
            src_buf_id = f"__source_{machine_id}"
            src_buf = Buffer(
                self.env,
                BufferConfig(id=src_buf_id, name=f"Source for {machine_id}", capacity=10000),
            )
            self.buffers[src_buf_id] = src_buf
            self.env.process(self._job_source(src_buf))

        # Create machines with their input/output buffers
        for mach_cfg in self.config.machines:
            in_id = self._find_incoming_buffer(mach_cfg.id)
            out_id = self._find_outgoing_buffer(mach_cfg.id)

            buf_in = self.buffers.get(in_id)
            if buf_in is None:
                default_buf = BufferConfig(
                    id=f"buf_in_{mach_cfg.id}",
                    name=f"Input for {mach_cfg.name}",
                    capacity=1000,
                )
                buf_in = Buffer(self.env, default_buf)
                self.buffers[default_buf.id] = buf_in

            buf_out = self.buffers.get(out_id)
            if buf_out is None:
                default_buf = BufferConfig(
                    id=f"buf_out_{mach_cfg.id}",
                    name=f"Output for {mach_cfg.name}",
                    capacity=1000,
                )
                buf_out = Buffer(self.env, default_buf)
                self.buffers[default_buf.id] = buf_out

            machine = Machine(self.env, mach_cfg, buf_in, buf_out)
            machine.start()
            self.machines[mach_cfg.id] = machine

        # Create conveyors
        for conv_cfg in self.config.conveyors:
            src_buf_id = f"buf_out_{conv_cfg.from_station}"
            dst_buf_id = f"buf_in_{conv_cfg.to_station}"
            src_buf = self.buffers.get(src_buf_id)
            dst_buf = self.buffers.get(dst_buf_id)
            if src_buf and dst_buf:
                conv = Conveyor(self.env, conv_cfg, src_buf, dst_buf)
                conv.start()
                self.conveyors[conv_cfg.id] = conv

        # Schedule periodic WIP recorder
        self.env.process(self._wip_recorder())

    def _job_source(self, buf: Buffer) -> Generator:
        """Source process: continuously generates jobs into *buf*."""
        while True:
            self._job_counter += 1
            job = {"id": f"job_{self._job_counter}"}
            yield buf.put(job)
            yield self.env.timeout(0.5)

    def _wip_recorder(self) -> Generator:
        """Periodically record WIP count."""
        interval = max(1.0, self.config.duration / 100)
        while True:
            wip = sum(m._jobs_processed + m._jobs_failed for m in self.machines.values())
            self.metrics.record_wip(self.env.now, wip)
            yield self.env.timeout(interval)

    def _find_incoming_buffer(self, station_id: str) -> Optional[str]:
        """Find the buffer feeding into *station_id* via a conveyor."""
        for conv in self.config.conveyors:
            if conv.to_station == station_id:
                # The source machine's output buffer feeds this machine
                return f"buf_out_{conv.from_station}"
        # Check if there's a source buffer
        src_key = f"__source_{station_id}"
        if src_key in self.buffers:
            return src_key
        return None

    def _find_outgoing_buffer(self, station_id: str) -> Optional[str]:
        """Find the buffer after *station_id* via a conveyor."""
        for conv in self.config.conveyors:
            if conv.from_station == station_id:
                return f"buf_in_{conv.to_station}"
        # End machine — use its own output buffer
        return f"buf_out_{station_id}"

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    def _collect_result(self) -> SimulationResult:
        """Build a :class:`SimulationResult` from machine summaries and WIP history."""
        effective_duration = self.config.duration - self.config.warmup_period
        if effective_duration <= 0:
            effective_duration = self.config.duration

        # Aggregate from individual machine counters
        total_produced = sum(m._jobs_processed for m in self.machines.values())
        total_failed = sum(m._jobs_failed for m in self.machines.values())

        # Throughput = units completed per sim-minute (over effective duration)
        throughput = total_produced / effective_duration if effective_duration > 0 else 0.0

        # Utilization per machine
        utilization: dict[str, float] = {}
        for mach_id, mach in self.machines.items():
            utilization[mach_id] = round(mach.utilization, 4)

        # Average cycle time: total processing time / jobs (approximation)
        total_proc = sum(m._total_processing_time for m in self.machines.values())
        avg_cycle = total_proc / total_produced if total_produced > 0 else 0.0

        # Average WIP from metrics history
        wip_data = self.metrics._wip_history
        avg_wip = float(sum(w[1] for w in wip_data) / len(wip_data)) if wip_data else 0.0

        # OEE per machine
        oee: dict[str, float] = {}
        for mach_id, mach in self.machines.items():
            availability = mach.availability
            perf = 1.0
            quality = 1.0
            oee[mach_id] = round(self.metrics.compute_oee(availability, perf, quality), 4)

        # Bottleneck = machine with lowest throughput
        bottleneck_id = ""
        if self.machines:
            bottleneck_id = min(
                self.machines.keys(),
                key=lambda mid: self.machines[mid]._jobs_processed,
            )

        return SimulationResult(
            id=self.config.id,
            config_id=self.config.id,
            duration=self.config.duration,
            throughput=round(throughput, 4),
            avg_cycle_time=round(avg_cycle, 2),
            avg_wip=round(avg_wip, 2),
            oee=oee,
            utilization=utilization,
            bottleneck_id=bottleneck_id,
            total_produced=total_produced,
            total_failed=total_failed,
        )
