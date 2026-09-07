"""Machine process model for discrete-event simulation."""

import random as _rng
from typing import Any, Generator, Optional

import simpy

from src.models.schemas import MachineConfig, MachineStatus
from src.simulation.buffer import Buffer
from src.simulation.failure_model import RepairModel, WeibullFailureModel


class Machine:
    """SimPy process representing a single machine / workstation.

    The machine repeatedly:
      1. Retrieves a job from its input buffer.
      2. Processes the job (duration drawn from a normal distribution).
      3. Attempts to place the result in its output buffer.

    While processing, a Weibull-based failure model determines whether a
    breakdown occurs; if so, the machine enters a repair phase before resuming.

    Args:
        env: SimPy environment.
        config: Machine configuration.
        buffer_in: Upstream buffer (source of jobs).
        buffer_out: Downstream buffer (destination for processed jobs).
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: MachineConfig,
        buffer_in: Buffer,
        buffer_out: Buffer,
    ) -> None:
        self.env = env
        self.config = config
        self.buffer_in = buffer_in
        self.buffer_out = buffer_out
        self.machine_id = config.id
        self.name = config.name

        # Status tracking
        self.status: MachineStatus = MachineStatus.IDLE
        self._process: Optional[simpy.Process] = None

        # Failure / repair models
        if config.failure_rate > 0:
            mtbf = 1.0 / config.failure_rate if config.failure_rate > 0 else float("inf")
            self._failure_model = WeibullFailureModel.from_mtbf_mttr(
                mtbf=mtbf,
                mttr=config.repair_time_mean or 1.0,
                shape=2.0,
            )
            self._repair_model = RepairModel(
                mean=config.repair_time_mean or 1.0,
                std=config.repair_time_std,
            )
        else:
            self._failure_model = None
            self._repair_model = None

        # Counters
        self._jobs_processed = 0
        self._jobs_failed = 0
        self._total_processing_time = 0.0
        self._total_idle_time = 0.0
        self._total_blocked_time = 0.0
        self._total_failed_time = 0.0
        self._status_log: list[tuple[float, str]] = []

        # Track time in current state for failure checks
        self._last_state_change = 0.0

    def start(self) -> None:
        """Start the machine's SimPy process."""
        self._process = self.env.process(self.run())

    def run(self) -> Generator:
        """Main simulation loop for this machine."""
        while True:
            # --- IDLE: wait for a job from the input buffer ---
            self._set_status(MachineStatus.IDLE)
            item = yield self.buffer_in.get()

            # --- PROCESSING ---
            self._set_status(MachineStatus.PROCESSING)
            proc_time = max(
                0.1,
                self.config.processing_time_mean
                + self.config.processing_time_std * _rng.gauss(0, 1),
            )
            yield self.env.timeout(proc_time)
            self._total_processing_time += proc_time
            self._jobs_processed += 1

            # --- Check for failure during processing ---
            if self._should_fail():
                yield from self._handle_failure()

            # --- BLOCKED: try to place in output buffer ---
            yield self.buffer_out.put(item)

    def _handle_failure(self) -> Generator:
        """Simulate a breakdown and repair cycle."""
        if self._repair_model is None:
            return  # type: ignore[misc]
        repair_time = self._repair_model.sample()
        self._set_status(MachineStatus.FAILED)
        yield self.env.timeout(repair_time * 0.1)
        self._set_status(MachineStatus.REPAIRING)
        yield self.env.timeout(repair_time * 0.9)
        self._total_failed_time += repair_time
        self._jobs_failed += 1

    def _should_fail(self) -> bool:
        """Check if the machine fails right now (Weibull CDF-based)."""
        if self._failure_model is None:
            return False
        elapsed = self.env.now - self._last_state_change
        return self._failure_model.should_fail(elapsed)

    def _set_status(self, new_status: MachineStatus) -> None:
        """Update status and record the transition."""
        self._status_log.append((self.env.now, new_status.value))
        self._last_state_change = self.env.now
        self.status = new_status

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def utilization(self) -> float:
        """Fraction of time the machine spent PROCESSING."""
        total = self.env.now if self.env.now > 0 else 1.0
        return self._total_processing_time / total

    @property
    def availability(self) -> float:
        """Fraction of time the machine was NOT in FAILED/REPAIRING state."""
        total = self.env.now if self.env.now > 0 else 1.0
        downtime = self._total_failed_time
        return (total - downtime) / total

    def get_status_log(self) -> list[dict[str, Any]]:
        """Return the full status-change log."""
        return [{"time": t, "status": s} for t, s in self._status_log]

    def summary(self) -> dict:
        """Return a summary dict for metrics collection."""
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "jobs_processed": self._jobs_processed,
            "jobs_failed": self._jobs_failed,
            "utilization": round(self.utilization, 4),
            "availability": round(self.availability, 4),
            "total_processing_time": round(self._total_processing_time, 2),
            "total_blocked_time": round(self._total_blocked_time, 2),
            "total_failed_time": round(self._total_failed_time, 2),
            "final_status": self.status.value,
        }
