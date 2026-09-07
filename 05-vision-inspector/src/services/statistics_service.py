"""In-memory statistics service for quality metrics and SPC.

In production this would back onto Redis; the current implementation
uses a plain dictionary so the service runs with zero external
dependencies.
"""

import math
from datetime import datetime, timezone

from src.models.schemas import (
    AlertInfo,
    DefectTypeInfo,
    DefectType,
    InspectionResult,
    StatisticsResponse,
    SeverityLevel,
)

_SEVERITY_WEIGHTS: dict[SeverityLevel, float] = {
    SeverityLevel.COSMETIC: 0.25,
    SeverityLevel.MINOR: 0.5,
    SeverityLevel.MAJOR: 0.75,
    SeverityLevel.CRITICAL: 1.0,
}


class StatisticsService:
    """Collect inspection results and compute quality metrics."""

    def __init__(self, alert_threshold_pct: float = 5.0) -> None:
        self._results: list[InspectionResult] = []
        self._alert_threshold_pct = alert_threshold_pct

    def record_inspection(self, result: InspectionResult) -> None:
        """Store an inspection result and update running counters.

        Parameters
        ----------
        result:
            Completed inspection output.
        """
        self._results.append(result)

    def get_statistics(
        self, line_id: str, period_minutes: int = 60
    ) -> StatisticsResponse:
        """Compute aggregated quality statistics for a line.

        Parameters
        ----------
        line_id:
            Production line identifier.
        period_minutes:
            Look-back window in minutes.

        Returns
        -------
        StatisticsResponse
            Aggregated metrics.
        """
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - period_minutes * 60

        relevant = [
            r
            for r in self._results
            if r.line_id == line_id and r.timestamp.timestamp() >= cutoff
        ]

        total = len(relevant)
        passed = sum(1 for r in relevant if r.passed)

        type_counts: dict[str, int] = {}
        for r in relevant:
            for d in r.defects:
                key = d.type.value
                type_counts[key] = type_counts.get(key, 0) + 1

        top_defects = self._build_top_defects(type_counts, total)

        return StatisticsResponse(
            line_id=line_id,
            period_minutes=period_minutes,
            total_inspected=total,
            pass_rate=round(passed / total * 100, 2) if total else 100.0,
            defect_rate=round((total - passed) / total * 100, 2) if total else 0.0,
            defect_type_counts=type_counts,
            top_defects=top_defects,
        )

    def get_pareto_analysis(self, line_id: str) -> list[DefectTypeInfo]:
        """Return defect types sorted by frequency (80/20 rule).

        Parameters
        ----------
        line_id:
            Production line identifier.

        Returns
        -------
        list[DefectTypeInfo]
            Sorted descending by count.
        """
        type_counts: dict[str, int] = {}
        severity_sums: dict[str, float] = {}
        severity_counts: dict[str, int] = {}

        for r in self._results:
            if r.line_id != line_id:
                continue
            for d in r.defects:
                key = d.type.value
                type_counts[key] = type_counts.get(key, 0) + 1
                severity_sums[key] = severity_sums.get(key, 0.0) + _SEVERITY_WEIGHTS.get(
                    d.severity, 0.5
                )
                severity_counts[key] = severity_counts.get(key, 0) + 1

        total_defects = sum(type_counts.values()) or 1
        infos: list[DefectTypeInfo] = []
        for dtype, count in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            avg_w = severity_sums[dtype] / severity_counts[dtype] if severity_counts[dtype] else 0.5
            severity_label = self._weight_to_label(avg_w)
            infos.append(
                DefectTypeInfo(
                    type=DefectType(dtype),
                    count=count,
                    percentage=round(count / total_defects * 100, 2),
                    avg_severity=severity_label,
                )
            )
        return infos

    def get_trend(self, line_id: str, window: int = 20) -> dict:
        """Compute rolling defect rate and C-chart data.

        Parameters
        ----------
        line_id:
            Production line identifier.
        window:
            Rolling window size for trend calculation.

        Returns
        -------
        dict
            ``rolling_defect_rates``, ``c_chart``, ``trend_direction``.
        """
        relevant = [r for r in self._results if r.line_id == line_id]
        defect_counts = [r.defect_count for r in relevant]

        rolling_rates: list[float] = []
        for i in range(len(defect_counts)):
            start = max(0, i - window + 1)
            window_data = defect_counts[start : i + 1]
            avg = sum(window_data) / len(window_data)
            rolling_rates.append(round(avg, 3))

        c_chart = self._c_chart_data(defect_counts)

        trend_direction = "stable"
        if len(rolling_rates) >= 4:
            recent = rolling_rates[-4:]
            if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                trend_direction = "increasing"
            elif all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                trend_direction = "decreasing"

        return {
            "rolling_defect_rates": rolling_rates,
            "c_chart": c_chart,
            "trend_direction": trend_direction,
        }

    def check_alerts(self, line_id: str) -> list[AlertInfo]:
        """Return active alerts when defect rate exceeds the threshold.

        Parameters
        ----------
        line_id:
            Production line identifier.

        Returns
        -------
        list[AlertInfo]
            Alerts for the last 60 minutes.
        """
        stats = self.get_statistics(line_id, period_minutes=60)
        alerts: list[AlertInfo] = []
        if stats.total_inspected > 0 and stats.defect_rate > self._alert_threshold_pct:
            alerts.append(
                AlertInfo(
                    line_id=line_id,
                    alert_type="HIGH_DEFECT_RATE",
                    message=(
                        f"Defect rate {stats.defect_rate:.1f}% exceeds "
                        f"threshold {self._alert_threshold_pct:.1f}%"
                    ),
                    defect_rate=stats.defect_rate,
                    threshold=self._alert_threshold_pct,
                )
            )
        critical_count = sum(
            1
            for r in self._results
            if r.line_id == line_id
            and (datetime.now(timezone.utc) - r.timestamp).total_seconds() < 3600
            and any(d.severity == SeverityLevel.CRITICAL for d in r.defects)
        )
        if critical_count > 0:
            alerts.append(
                AlertInfo(
                    line_id=line_id,
                    alert_type="CRITICAL_DEFECT_DETECTED",
                    message=f"{critical_count} critical defect(s) in the last hour",
                    defect_rate=stats.defect_rate,
                    threshold=self._alert_threshold_pct,
                )
            )
        return alerts

    def _c_chart_data(self, defect_counts: list[int]) -> dict:
        """Compute C-chart centre line, UCL, and LCL.

        Parameters
        ----------
        defect_counts:
            Sequence of per-inspection defect counts.

        Returns
        -------
        dict
            ``center_line``, ``ucl``, ``lcl``, ``data``.
        """
        if not defect_counts:
            return {"center_line": 0, "ucl": 0, "lcl": 0, "data": []}

        c_bar = sum(defect_counts) / len(defect_counts)
        if c_bar == 0:
            return {"center_line": 0, "ucl": 0, "lcl": 0, "data": defect_counts}
        ucl = c_bar + 3 * math.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * math.sqrt(c_bar))

        return {
            "center_line": round(c_bar, 3),
            "ucl": round(ucl, 3),
            "lcl": round(lcl, 3),
            "data": defect_counts,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_top_defects(
        type_counts: dict[str, int], total_inspected: int
    ) -> list[DefectTypeInfo]:
        total_defects = sum(type_counts.values()) or 1
        infos: list[DefectTypeInfo] = []
        for dtype, count in sorted(type_counts.items(), key=lambda kv: -kv[1])[:5]:
            infos.append(
                DefectTypeInfo(
                    type=DefectType(dtype),
                    count=count,
                    percentage=round(count / total_defects * 100, 2),
                    avg_severity="MINOR",
                )
            )
        return infos

    @staticmethod
    def _weight_to_label(w: float) -> str:
        if w < 0.35:
            return SeverityLevel.COSMETIC.value
        if w < 0.6:
            return SeverityLevel.MINOR.value
        if w < 0.85:
            return SeverityLevel.MAJOR.value
        return SeverityLevel.CRITICAL.value
