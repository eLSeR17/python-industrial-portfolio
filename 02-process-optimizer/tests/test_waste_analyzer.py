"""Tests for the WasteAnalyzer (SPC, CUSUM, OEE, Western Electric rules).

Uses NumPy-generated synthetic process data with known characteristics:
    - In-control data: normally distributed around a target.
    - Out-of-control data: shifted mean or increased variance.
    - Step-change data: a clear shift midway through.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.models.schemas import AlarmSeverity, OEEComponents
from src.services.waste_analyzer import (
    compute_capability,
    compute_oee,
    compute_xbar_r_chart,
    identify_bottleneck,
    StageData,
    western_electric_violations,
    WasteAnalyzer,
)
from src.utils.time_series import CUSUMDetector, SlidingWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_control_data(
    n: int = 200,
    mean: float = 100.0,
    sigma: float = 2.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate in-control normally distributed process data."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sigma, size=n)


def _shifted_data(
    n: int = 200,
    mean_before: float = 100.0,
    mean_after: float = 103.0,
    sigma: float = 2.0,
    shift_at: int = 100,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate data with a mean shift at a known point."""
    rng = np.random.default_rng(seed)
    before = rng.normal(mean_before, sigma, size=shift_at)
    after = rng.normal(mean_after, sigma, size=n - shift_at)
    return np.concatenate([before, after])


# ---------------------------------------------------------------------------
# Sliding Window Tests
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_push_and_retrieve(self) -> None:
        w = SlidingWindow(size=5)
        for i in range(5):
            w.push(float(i))
        assert w.is_full
        np.testing.assert_array_almost_equal(w.data, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_overflow_overwrites_oldest(self) -> None:
        w = SlidingWindow(size=3)
        w.push(1.0)
        w.push(2.0)
        w.push(3.0)
        w.push(4.0)  # Should overwrite 1.0.
        np.testing.assert_array_almost_equal(w.data, [2.0, 3.0, 4.0])

    def test_mean_and_std(self) -> None:
        w = SlidingWindow(size=100)
        rng = np.random.default_rng(0)
        values = rng.normal(50.0, 5.0, size=100)
        for v in values:
            w.push(float(v))
        assert abs(w.mean() - 50.0) < 2.0
        assert abs(w.std() - 5.0) < 2.0

    def test_reset(self) -> None:
        w = SlidingWindow(size=5)
        w.push(1.0)
        w.reset()
        assert w.count == 0
        assert not w.is_full


# ---------------------------------------------------------------------------
# X-bar / R Chart Tests
# ---------------------------------------------------------------------------

class TestXbarRChart:
    def test_in_control_chart(self) -> None:
        """In-control data should produce no violations."""
        data = _in_control_data(n=150, mean=100.0, sigma=2.0)
        xbar, r_chart = compute_xbar_r_chart(data, subgroup_size=5)

        # Control limits should bracket the data.
        assert xbar.upper_control_limit > xbar.center_line
        assert xbar.lower_control_limit < xbar.center_line
        # With in-control data, very few or no violations expected.
        assert len(xbar.violations) < 5, f"Too many violations: {len(xbar.violations)}"

    def test_shifted_data_produces_violations(self) -> None:
        """A mean shift should trigger violations on the X-bar chart."""
        data = _shifted_data(n=200, mean_before=100.0, mean_after=105.0, sigma=2.0)
        xbar, _ = compute_xbar_r_chart(data, subgroup_size=5)
        # After the shift, points should exceed the upper control limit.
        assert len(xbar.violations) > 0, "Expected violations after mean shift"

    def test_control_limits_sanity(self) -> None:
        """UCL should be > LCL and center line should be between them."""
        data = _in_control_data(n=100)
        xbar, r_chart = compute_xbar_r_chart(data, subgroup_size=5)
        assert xbar.upper_control_limit > xbar.center_line > xbar.lower_control_limit
        assert r_chart.upper_control_limit > r_chart.center_line >= r_chart.lower_control_limit

    def test_insufficient_data_raises(self) -> None:
        """Should raise ValueError with too few data points."""
        with pytest.raises(ValueError, match="Need at least"):
            compute_xbar_r_chart(np.array([1.0, 2.0]), subgroup_size=5)


# ---------------------------------------------------------------------------
# CUSUM Tests
# ---------------------------------------------------------------------------

class TestCUSUM:
    def test_no_alarm_on_in_control(self) -> None:
        """CUSUM should not alarm on stable in-control data."""
        detector = CUSUMDetector(target=100.0, threshold=5.0, drift=0.5)
        data = _in_control_data(n=100, mean=100.0, sigma=1.0)
        alarms = [detector.update(float(v)) for v in data]
        assert not any(a["alarm"] for a in alarms), "CUSUM should not alarm on stable data"

    def test_alarm_on_shift(self) -> None:
        """CUSUM should detect a persistent upward shift."""
        detector = CUSUMDetector(target=100.0, threshold=3.0, drift=0.5)
        # 50 points at 100, then 50 points at 103.
        data = np.concatenate([
            np.full(50, 100.0),
            np.full(50, 103.0),
        ])
        results = [detector.update(float(v)) for v in data]
        # After the shift, CUSUM should eventually alarm.
        assert any(r["alarm"] for r in results[40:]), "CUSUM should detect the shift"

    def test_reset(self) -> None:
        """After reset, accumulators should return to zero."""
        detector = CUSUMDetector(target=0.0, threshold=2.0, drift=0.1)
        detector.update(10.0)
        detector.update(10.0)
        detector.reset()
        state = detector.state
        assert state["s_pos"] == 0.0
        assert state["s_neg"] == 0.0


# ---------------------------------------------------------------------------
# Western Electric Rules Tests
# ---------------------------------------------------------------------------

class TestWesternElectric:
    def test_rule_1_beyond_limits(self) -> None:
        """A point beyond UCL should trigger Rule 1."""
        data = np.array([50.0, 50.0, 50.0, 50.0, 100.0])  # Last point way above UCL.
        violations = western_electric_violations(data, center_line=50.0, ucl=55.0, lcl=45.0)
        rule1 = [v for v in violations if "Rule 1" in v["rule"]]
        assert len(rule1) == 1
        assert rule1[0]["index"] == 4

    def test_rule_4_eight_consecutive(self) -> None:
        """8 consecutive points on one side should trigger Rule 4."""
        data = np.array([52.0] * 8)  # All above center_line=50.
        violations = western_electric_violations(data, center_line=50.0, ucl=55.0, lcl=45.0)
        rule4 = [v for v in violations if "Rule 4" in v["rule"]]
        assert len(rule4) >= 1

    def test_no_violations_on_clean_data(self) -> None:
        """Alternating data around center should produce no violations."""
        data = np.array([50.0, 51.0, 50.0, 49.0] * 5)
        violations = western_electric_violations(data, center_line=50.0, ucl=53.0, lcl=47.0)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Process Capability Tests
# ---------------------------------------------------------------------------

class TestCapability:
    def test_capable_process(self) -> None:
        """A well-centered, low-variance process should have high Cpk."""
        rng = np.random.default_rng(42)
        data = rng.normal(50.0, 1.0, size=200)  # σ=1, limits at ±10.
        cap = compute_capability(data, usl=60.0, lsl=40.0)
        assert cap["Cpk"] > 2.0, f"Cpk {cap['Cpk']} should be > 2 for σ=1, limits ±10"

    def test_off_center_process(self) -> None:
        """An off-center process should have lower Cpk than Cp."""
        rng = np.random.default_rng(42)
        data = rng.normal(55.0, 1.0, size=200)  # Centered at 55, limits 40-60.
        cap = compute_capability(data, usl=60.0, lsl=40.0)
        assert cap["Cpk"] < cap["Cp"], "Cpk should be less than Cp for off-center process"


# ---------------------------------------------------------------------------
# OEE Tests
# ---------------------------------------------------------------------------

class TestOEE:
    def test_perfect_oee(self) -> None:
        """A perfect stage should have OEE = 1.0."""
        stage = StageData(
            name="perfect",
            planned_downtime_hours=0.0,
            actual_runtime_hours=24.0,
            ideal_cycle_time_seconds=1.0,
            actual_parts_produced=86400,
            good_parts=86400,
            total_parts=86400,
        )
        result = compute_oee([stage])
        assert result["perfect"].oee == pytest.approx(1.0, abs=0.01)

    def test_downtime_reduces_availability(self) -> None:
        """Planned downtime should reduce availability below 1.0."""
        stage = StageData(
            name="downtime",
            planned_downtime_hours=6.0,
            actual_runtime_hours=18.0,
            ideal_cycle_time_seconds=1.0,
            actual_parts_produced=64800,
            good_parts=64800,
            total_parts=64800,
        )
        result = compute_oee([stage])
        assert result["downtime"].availability < 1.0
        assert result["downtime"].availability == pytest.approx(0.75, abs=0.01)

    def test_defects_reduce_quality(self) -> None:
        """Defective parts should reduce quality below 1.0."""
        stage = StageData(
            name="defects",
            planned_downtime_hours=0.0,
            actual_runtime_hours=24.0,
            ideal_cycle_time_seconds=1.0,
            actual_parts_produced=86400,
            good_parts=77760,
            total_parts=86400,
        )
        result = compute_oee([stage])
        assert result["defects"].quality == pytest.approx(0.9, abs=0.01)

    def test_overall_oee(self) -> None:
        """Overall OEE should be computed for multiple stages."""
        stages = [
            StageData(name="stage1", good_parts=90, total_parts=100,
                      actual_parts_produced=100, ideal_cycle_time_seconds=1.0),
            StageData(name="stage2", good_parts=95, total_parts=100,
                      actual_parts_produced=100, ideal_cycle_time_seconds=1.0),
        ]
        result = compute_oee(stages)
        assert "_overall" in result


# ---------------------------------------------------------------------------
# Bottleneck Identification
# ---------------------------------------------------------------------------

class TestBottleneck:
    def test_identifies_slowest_stage(self) -> None:
        stages = [
            StageData(name="fast", ideal_cycle_time_seconds=1.0, actual_parts_produced=1000, actual_runtime_hours=1.0),
            StageData(name="slow", ideal_cycle_time_seconds=5.0, actual_parts_produced=200, actual_runtime_hours=1.0),
        ]
        result = identify_bottleneck(stages)
        assert result["bottleneck_stage"] == "slow"


# ---------------------------------------------------------------------------
# WasteAnalyzer Integration
# ---------------------------------------------------------------------------

class TestWasteAnalyzer:
    def test_analyze_produces_charts_and_alarms(self) -> None:
        """Full analysis should return charts and potentially alarms."""
        analyzer = WasteAnalyzer()
        rng = np.random.default_rng(42)
        variable_data = {
            "temperature": rng.normal(75.0, 0.5, size=200),
            "pressure": rng.normal(2.5, 0.1, size=200),
        }
        result = analyzer.analyze(
            process_id="test-analyzer",
            variable_data=variable_data,
            subgroup_size=5,
        )
        assert result.process_id == "test-analyzer"
        assert len(result.charts) > 0
        # CUSUM state should exist for each variable.
        assert "temperature" in result.cusum_state
        assert "pressure" in result.cusum_state

    def test_analyze_with_spec_limits(self) -> None:
        """Spec limits should produce capability indices."""
        analyzer = WasteAnalyzer()
        rng = np.random.default_rng(42)
        data = rng.normal(50.0, 1.0, size=200)
        result = analyzer.analyze(
            process_id="cap-test",
            variable_data={"x": data},
            subgroup_size=5,
            specification_limits={"x": (40.0, 60.0)},
        )
        assert "x" in result.process_capability
        assert result.process_capability["x"] > 1.0  # Capable process.

    def test_insufficient_data_skips_variable(self) -> None:
        """Variables with too few points should be skipped, not crash."""
        analyzer = WasteAnalyzer()
        data = {"sparse": np.array([1.0, 2.0, 3.0])}
        result = analyzer.analyze("skip-test", data, subgroup_size=5)
        assert len(result.charts) == 0  # Not enough data for charts.
