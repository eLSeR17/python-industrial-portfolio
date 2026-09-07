"""Job scheduling strategies for sequencing work through machines."""

from typing import Any, Optional


class JobScheduler:
    """Dispatches the next job from a queue according to a chosen strategy.

    Supported strategies:
        * **FIFO** – first in, first out.
        * **SPT** – shortest processing time first.
        * **EDD** – earliest due date first.
        * **CRITICAL_RATIO** – lowest critical ratio first.

    Args:
        strategy: One of ``FIFO``, ``SPT``, ``EDD``, ``CRITICAL_RATIO``.
    """

    VALID_STRATEGIES = {"FIFO", "SPT", "EDD", "CRITICAL_RATIO"}

    def __init__(self, strategy: str = "FIFO") -> None:
        strategy_upper = strategy.upper()
        if strategy_upper not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}."
            )
        self.strategy = strategy_upper

    def next_job(self, jobs: list[dict[str, Any]], now: float = 0.0) -> Optional[dict[str, Any]]:
        """Return the next job to process according to the scheduling strategy.

        Each job is a dict with at least ``"id"``.  Strategy-specific keys:
            * **SPT**: ``"processing_time"``
            * **EDD**: ``"due_date"``
            * **CRITICAL_RATIO**: ``"due_date"`` and ``"processing_time"``

        Args:
            jobs: Non-empty list of candidate jobs.
            now: Current simulation time (used by CRITICAL_RATIO).

        Returns:
            The selected job dict, or ``None`` if *jobs* is empty.
        """
        if not jobs:
            return None

        if self.strategy == "FIFO":
            return self._fifo(jobs)
        if self.strategy == "SPT":
            return self._spt(jobs)
        if self.strategy == "EDD":
            return self._edd(jobs)
        if self.strategy == "CRITICAL_RATIO":
            return self._critical_ratio(jobs, now)
        # Defensive – should never reach here after __init__ validation.
        return jobs[0]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _fifo(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the first job in the list."""
        return jobs[0]

    @staticmethod
    def _spt(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the job with the shortest processing time."""
        return min(jobs, key=lambda j: j.get("processing_time", float("inf")))

    @staticmethod
    def _edd(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the job with the earliest due date."""
        return min(jobs, key=lambda j: j.get("due_date", float("inf")))

    @staticmethod
    def _critical_ratio(
        jobs: list[dict[str, Any]], now: float
    ) -> dict[str, Any]:
        """Return the job with the lowest critical ratio.

        CR = (due_date - now) / processing_time.

        A CR < 1 means the job is already late.
        """
        def _cr(job: dict[str, Any]) -> float:
            due = job.get("due_date", float("inf"))
            proc = job.get("processing_time", 1.0)
            if proc <= 0:
                proc = 1.0
            return (due - now) / proc

        return min(jobs, key=_cr)
