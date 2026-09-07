"""Scenario runner: execute what-if comparisons across configurations."""

import math
from typing import Any, Optional, Type

from src.models.schemas import (
    ScenarioComparison,
    SimulationConfig,
    SimulationResult,
)


class ScenarioRunner:
    """Runs multiple simulation scenarios and compares their results.

    Args:
        engine_class: The engine class to instantiate (default:
            :class:`SimulationEngine`).  Useful for dependency injection in
            tests.
    """

    def __init__(self, engine_class: Type[SimulationEngine] | None = None) -> None:
        self.engine_class = engine_class or SimulationEngine

    def run_scenario(
        self, config: SimulationConfig, replications: int = 10
    ) -> SimulationResult:
        """Run *replications* independent replications of a single scenario.

        The returned :class:`SimulationResult` aggregates results across
        replications and includes a 95 % confidence interval for throughput.

        Args:
            config: Scenario configuration.
            replications: Number of independent runs (>= 1).

        Returns:
            Aggregated :class:`SimulationResult`.
        """
        if replications < 1:
            raise ValueError(f"replications must be >= 1, got {replications}")

        # Lazy import to avoid circular dependency
        from src.simulation.engine import SimulationEngine
        engine_cls = self.engine_class or SimulationEngine

        base_seed = config.random_seed
        throughputs: list[float] = []
        cycle_times: list[float] = []
        wips: list[float] = []
        all_results: list[dict[str, Any]] = []

        for i in range(replications):
            config.random_seed = (base_seed + i * 7919) % (2**31)
            engine = self.engine_class(config)
            result = engine.run()
            throughputs.append(result.throughput)
            cycle_times.append(result.avg_cycle_time)
            wips.append(result.avg_wip)
            all_results.append(
                {
                    "replication": i + 1,
                    "throughput": result.throughput,
                    "avg_cycle_time": result.avg_cycle_time,
                    "avg_wip": result.avg_wip,
                    "seed": config.random_seed,
                }
            )

        ci_low, ci_high = self._compute_confidence_interval(throughputs)

        avg_throughput = sum(throughputs) / len(throughputs)
        avg_cycle = sum(cycle_times) / len(cycle_times)
        avg_wip_val = sum(wips) / len(wips)

        # Re-run once more for per-machine metrics with the median seed
        config.random_seed = base_seed
        engine = engine_cls(config)
        final = engine.run()

        return SimulationResult(
            id=config.id,
            config_id=config.id,
            duration=config.duration,
            throughput=round(avg_throughput, 4),
            avg_cycle_time=round(avg_cycle, 2),
            avg_wip=round(avg_wip_val, 2),
            oee=final.oee,
            utilization=final.utilization,
            bottleneck_id=final.bottleneck_id,
            total_produced=final.total_produced,
            total_failed=final.total_failed,
            replication_results=all_results,
        )

    def compare_scenarios(
        self, configs: list[SimulationConfig], replications: int = 5
    ) -> ScenarioComparison:
        """Run and compare multiple scenarios.

        Args:
            configs: List of scenario configurations.
            replications: Replications per scenario.

        Returns:
            :class:`ScenarioComparison` with best performers and recommendations.
        """
        if not configs:
            return ScenarioComparison()

        results: list[SimulationResult] = []
        for cfg in configs:
            results.append(self.run_scenario(cfg, replications))

        best_tp = max(results, key=lambda r: r.throughput)
        best_oee = max(results, key=lambda r: sum(r.oee.values()) / max(len(r.oee), 1))

        recommendations: list[str] = []
        throughputs = {r.config_id: r.throughput for r in results}
        if len(results) > 1:
            worst = min(results, key=lambda r: r.throughput)
            recommendations.append(
                f"Scenario '{worst.config_id}' has the lowest throughput "
                f"({worst.throughput:.2f}/min). Consider increasing buffer capacity "
                f"or reducing processing time at bottleneck '{worst.bottleneck_id}'."
            )
            best = max(results, key=lambda r: r.throughput)
            recommendations.append(
                f"Scenario '{best.config_id}' achieves best throughput "
                f"({best.throughput:.2f}/min)."
            )

        return ScenarioComparison(
            scenarios=results,
            best_throughput=best_tp.config_id,
            best_oee=best_oee.config_id,
            recommendations=recommendations,
        )

    @staticmethod
    def _compute_confidence_interval(
        results: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute the confidence interval for a list of observations.

        Uses the t-distribution for small samples.

        Args:
            results: Observed values.
            confidence: Confidence level (default 0.95 → 95 %).

        Returns:
            (lower, upper) bounds.
        """
        n = len(results)
        if n < 2:
            val = results[0] if results else 0.0
            return (val, val)

        mean = sum(results) / n
        variance = sum((x - mean) ** 2 for x in results) / (n - 1)
        std_err = math.sqrt(variance / n)

        # Approximate t-value for 95 % CI (two-tailed) with finite df
        # Using a simple approximation for common df values
        if n <= 30:
            # Rough t-values by df (simplified)
            t_table = {
                1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
            }
            df = n - 1
            t_val = t_table.get(df, 2.0)
        else:
            t_val = 1.96  # z-approximation for large n

        margin = t_val * std_err
        return (mean - margin, mean + margin)
