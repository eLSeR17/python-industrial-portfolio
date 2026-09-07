"""Tests for the StatisticsService.

Uses synthetic InspectionResult objects — no network or file I/O.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.models.schemas import (
    BoundingBox,
    DefectDetection,
    DefectType,
    InspectionResult,
    SeverityLevel,
)
from src.services.statistics_service import StatisticsService


def _make_result(
    line_id: str = "LINE-A",
    defect_count: int = 0,
    passed: bool = True,
    minutes_ago: int = 0,
    defect_type: DefectType = DefectType.SCRATCH,
    severity: SeverityLevel = SeverityLevel.MINOR,
) -> InspectionResult:
    defects: list[DefectDetection] = []
    for _ in range(defect_count):
        defects.append(
            DefectDetection(
                type=defect_type,
                severity=severity,
                confidence=0.8,
                bbox=BoundingBox(x=10, y=10, w=50, h=50),
                area=2500.0,
            )
        )
    return InspectionResult(
        id=uuid.uuid4().hex[:12],
        image_id=uuid.uuid4().hex[:12],
        line_id=line_id,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        defects=defects,
        passed=passed,
        total_defect_area=sum(d.area for d in defects),
        defect_count=defect_count,
        processing_time_ms=12.5,
    )


@pytest.fixture()
def service() -> StatisticsService:
    return StatisticsService(alert_threshold_pct=5.0)


class TestRecordInspection:
    def test_stores_result(self, service: StatisticsService) -> None:
        r = _make_result()
        service.record_inspection(r)
        assert len(service._results) == 1

    def test_stores_multiple(self, service: StatisticsService) -> None:
        for _ in range(5):
            service.record_inspection(_make_result())
        assert len(service._results) == 5


class TestGetStatistics:
    def test_no_data_returns_zero(self, service: StatisticsService) -> None:
        stats = service.get_statistics("LINE-A")
        assert stats.total_inspected == 0
        assert stats.pass_rate == 100.0
        assert stats.defect_rate == 0.0

    def test_all_pass(self, service: StatisticsService) -> None:
        for _ in range(10):
            service.record_inspection(_make_result(passed=True, defect_count=0))
        stats = service.get_statistics("LINE-A")
        assert stats.total_inspected == 10
        assert stats.pass_rate == 100.0

    def test_all_fail(self, service: StatisticsService) -> None:
        for _ in range(4):
            service.record_inspection(_make_result(passed=False, defect_count=3))
        stats = service.get_statistics("LINE-A")
        assert stats.defect_rate == 100.0

    def test_mixed(self, service: StatisticsService) -> None:
        for i in range(10):
            service.record_inspection(_make_result(passed=(i < 7), defect_count=0 if i < 7 else 1))
        stats = service.get_statistics("LINE-A")
        assert stats.pass_rate == 70.0
        assert stats.defect_rate == 30.0

    def test_different_lines(self, service: StatisticsService) -> None:
        service.record_inspection(_make_result(line_id="LINE-A", passed=True))
        service.record_inspection(_make_result(line_id="LINE-B", passed=False))
        stats_a = service.get_statistics("LINE-A")
        stats_b = service.get_statistics("LINE-B")
        assert stats_a.total_inspected == 1
        assert stats_b.total_inspected == 1
        assert stats_a.pass_rate == 100.0
        assert stats_b.pass_rate == 0.0

    def test_defect_type_counts(self, service: StatisticsService) -> None:
        service.record_inspection(
            _make_result(defect_count=2, defect_type=DefectType.SCRATCH)
        )
        service.record_inspection(
            _make_result(defect_count=1, defect_type=DefectType.DENT)
        )
        stats = service.get_statistics("LINE-A")
        assert stats.defect_type_counts.get("SCRATCH", 0) == 2
        assert stats.defect_type_counts.get("DENT", 0) == 1


class TestParetoAnalysis:
    def test_sorted_by_count(self, service: StatisticsService) -> None:
        for _ in range(5):
            service.record_inspection(
                _make_result(defect_count=1, defect_type=DefectType.SCRATCH)
            )
        for _ in range(2):
            service.record_inspection(
                _make_result(defect_count=1, defect_type=DefectType.DENT)
            )
        pareto = service.get_pareto_analysis("LINE-A")
        assert len(pareto) == 2
        assert pareto[0].type == DefectType.SCRATCH
        assert pareto[0].count == 5

    def test_empty_line(self, service: StatisticsService) -> None:
        pareto = service.get_pareto_analysis("NON-EXISTENT")
        assert pareto == []


class TestGetTrend:
    def test_stable_trend(self, service: StatisticsService) -> None:
        for _ in range(10):
            service.record_inspection(_make_result(defect_count=1))
        trend = service.get_trend("LINE-A", window=5)
        assert "rolling_defect_rates" in trend
        assert "c_chart" in trend
        assert trend["c_chart"]["center_line"] == 1.0

    def test_increasing_trend(self, service: StatisticsService) -> None:
        for i in range(10):
            service.record_inspection(_make_result(defect_count=i))
        trend = service.get_trend("LINE-A", window=4)
        assert trend["trend_direction"] == "increasing"

    def test_empty_data(self, service: StatisticsService) -> None:
        trend = service.get_trend("EMPTY")
        assert trend["c_chart"]["center_line"] == 0
        assert trend["rolling_defect_rates"] == []


class TestCheckAlerts:
    def test_no_alert_when_below_threshold(self, service: StatisticsService) -> None:
        for _ in range(20):
            service.record_inspection(_make_result(passed=True, defect_count=0))
        alerts = service.check_alerts("LINE-A")
        assert len(alerts) == 0

    def test_alert_when_above_threshold(self, service: StatisticsService) -> None:
        for _ in range(10):
            service.record_inspection(_make_result(passed=False, defect_count=2))
        alerts = service.check_alerts("LINE-A")
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "HIGH_DEFECT_RATE"

    def test_critical_defect_alert(self, service: StatisticsService) -> None:
        r = _make_result(
            passed=False,
            defect_count=1,
            severity=SeverityLevel.CRITICAL,
        )
        service.record_inspection(r)
        alerts = service.check_alerts("LINE-A")
        critical_alerts = [a for a in alerts if a.alert_type == "CRITICAL_DEFECT_DETECTED"]
        assert len(critical_alerts) == 1


class TestCChartData:
    def test_basic_c_chart(self, service: StatisticsService) -> None:
        counts = [2, 3, 1, 4, 2, 1, 3, 2]
        chart = service._c_chart_data(counts)
        assert chart["center_line"] == pytest.approx(2.25, abs=0.01)
        assert chart["ucl"] > chart["center_line"]
        assert chart["lcl"] >= 0
        assert chart["data"] == counts

    def test_empty_c_chart(self, service: StatisticsService) -> None:
        chart = service._c_chart_data([])
        assert chart["center_line"] == 0
        assert chart["ucl"] == 0
        assert chart["lcl"] == 0

    def test_zero_defects(self, service: StatisticsService) -> None:
        chart = service._c_chart_data([0, 0, 0])
        assert chart["center_line"] == 0
        assert chart["lcl"] == 0
