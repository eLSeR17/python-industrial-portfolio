"""Metrics collection for OEE, throughput, and utilization."""

from collections import defaultdict
from typing import Any, Optional


class MetricsCollector:
    """Collects and aggregates simulation metrics during a run.

    Tracks per-machine status changes, job completions, and WIP over time.
    Provides OEE, utilization, and bottleneck identification.
    """

    def __init__(self) -> None:
        self._machine_status: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self._job_completions: list[dict[str, Any]] = []
        self._wip_history: list[tuple[float, int]] = []
        self._machine_totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "processing": 0.0,
                "idle": 0.0,
                "blocked": 0.0,
                "failed": 0.0,
                "repairing": 0.0,
            }
        )
        self._last_status_change: dict[str, tuple[float, str]] = {}

    def record_machine_status(
        self, machine_id: str, status: str, timestamp: float
    ) -> None:
        """Record a status transition for a machine.

        Args:
            machine_id: Identifier of the machine.
            status: New status string (IDLE, PROCESSING, FAILED, etc.).
            timestamp: Current simulation time.
        """
        # Close previous interval
        if machine_id in self._last_status_change:
            prev_time, prev_status = self._last_status_change[machine_id]
            duration = timestamp - prev_time
            key = prev_status.lower()
            if key in self._machine_totals[machine_id]:
                self._machine_totals[machine_id][key] += duration

        self._machine_status[machine_id].append((timestamp, status))
        self._last_status_change[machine_id] = (timestamp, status)

    def record_job_completion(
        self,
        machine_id: str,
        job_id: str,
        start: float,
        end: float,
        quality: str = "good",
    ) -> None:
        """Record that a job completed processing.

        Args:
            machine_id: Which machine processed the job.
            job_id: Unique job identifier.
            start: Time processing started.
            end: Time processing ended.
            quality: ``"good"`` or ``"defective"``.
        """
        self._job_completions.append(
            {
                "machine_id": machine_id,
                "job_id": job_id,
                "start": start,
                "end": end,
                "cycle_time": end - start,
                "quality": quality,
            }
        )

    def record_wip(self, timestamp: float, wip_count: int) -> None:
        """Record work-in-progress count at a point in time.

        Args:
            timestamp: Simulation time.
            wip_count: Number of jobs currently in the system.
        """
        self._wip_history.append((timestamp, wip_count))

    def compute_oee(
        self, availability: float, performance: float, quality: float
    ) -> float:
        """Compute Overall Equipment Effectiveness.

        OEE = Availability × Performance × Quality.

        Args:
            availability: Fraction of scheduled time the machine is available.
            performance: Speed ratio (actual / ideal cycle rate).
            quality: Fraction of good units.

        Returns:
            OEE as a value between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, availability * performance * quality))

    def compute_summary(self, sim_duration: float = 1.0) -> dict[str, Any]:
        """Compute aggregate metrics from all collected data.

        Args:
            sim_duration: Total simulation duration (used for throughput calc).

        Returns:
            Dictionary with throughput, avg_cycle_time, avg_wip, per-machine
            utilization, and bottleneck_id.
        """
        total_produced = len(self._job_completions)
        total_good = sum(
            1 for j in self._job_completions if j["quality"] == "good"
        )
        throughput = total_produced / sim_duration if sim_duration > 0 else 0.0

        cycle_times = [j["cycle_time"] for j in self._job_completions]
        avg_cycle_time = float(sum(cycle_times) / len(cycle_times)) if cycle_times else 0.0

        wip_values = [w[1] for w in self._wip_history]
        avg_wip = float(sum(wip_values) / len(wip_values)) if wip_values else 0.0

        # Per-machine utilization
        utilization: dict[str, float] = {}
        for machine_id, totals in self._machine_totals.items():
            total_time = sum(totals.values())
            if total_time > 0:
                utilization[machine_id] = round(totals["processing"] / total_time, 4)
            else:
                utilization[machine_id] = 0.0

        # Bottleneck = machine with lowest throughput contribution
        machine_counts: dict[str, int] = defaultdict(int)
        for j in self._job_completions:
            machine_counts[j["machine_id"]] += 1
        bottleneck_id = ""
        if machine_counts:
            bottleneck_id = min(machine_counts, key=machine_counts.get)  # type: ignore[arg-type]

        return {
            "throughput": round(throughput, 4),
            "avg_cycle_time": round(avg_cycle_time, 2),
            "avg_wip": round(avg_wip, 2),
            "total_produced": total_produced,
            "total_good": total_good,
            "total_failed": total_produced - total_good,
            "utilization": utilization,
            "bottleneck_id": bottleneck_id,
        }

    def get_machine_timeline(self, machine_id: str) -> list[dict[str, Any]]:
        """Return status changes for a specific machine.

        Args:
            machine_id: The machine to query.

        Returns:
            List of ``{"time": float, "status": str}`` dicts.
        """
        return [
            {"time": t, "status": s}
            for t, s in self._machine_status.get(machine_id, [])
        ]
