"""Landed cost decomposition and network-wide cost analysis."""

from typing import Any

import numpy as np

from src.models.network import SupplyChainNetwork
from src.models.schemas import (
    CostBreakdownRequest,
    CostBreakdownResponse,
    NodeType,
)


class CostAnalyzer:
    """Decompose and analyze supply chain costs.

    Computes total landed cost, cost breakdowns by component, per-unit
    costs, and supports what-if analysis by adjusting input parameters.
    The landed cost model captures the full chain from material procurement
    through final delivery.
    """

    def __init__(self, network: SupplyChainNetwork) -> None:
        """Initialize with a supply chain network.

        Args:
            network: The supply chain graph to analyze.
        """
        self.network = network
        self.graph = network.graph

    # ------------------------------------------------------------------
    # Landed cost from network
    # ------------------------------------------------------------------

    def compute_landed_costs(
        self,
        from_node: str,
        to_node: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Compute total landed cost for shipping from one node to another.

        Captures material cost, transport cost, handling, and estimated
        storage during transit. Uses the shortest-cost path through the
        network.

        Args:
            from_node: Origin node ID (supplier or warehouse).
            to_node: Destination node ID (warehouse or customer).
            quantity: Number of units to ship.

        Returns:
            Dict with cost components and totals.

        Raises:
            ValueError: If no path exists between the nodes.
        """
        path, transport_cost_per_unit = self.network.shortest_cost_path(
            from_node, to_node
        )
        if not path:
            raise ValueError(f"No path from {from_node} to {to_node}")

        material_cost = self.graph.nodes[from_node].get("variable_cost", 0.0)
        handling_cost = 0.50  # per unit
        storage_cost = self._compute_storage_cost(path, quantity)
        insurance_cost = material_cost * 0.02  # 2% of material value

        total_per_unit = (
            material_cost
            + transport_cost_per_unit
            + handling_cost
            + storage_cost
            + insurance_cost
        )

        return {
            "path": path,
            "quantity": quantity,
            "material_cost_per_unit": material_cost,
            "transport_cost_per_unit": transport_cost_per_unit,
            "handling_cost_per_unit": handling_cost,
            "storage_cost_per_unit": storage_cost,
            "insurance_cost_per_unit": insurance_cost,
            "total_per_unit": total_per_unit,
            "total_cost": total_per_unit * quantity,
        }

    def _compute_storage_cost(
        self, path: list[str], quantity: float
    ) -> float:
        """Estimate per-unit storage cost based on transit time."""
        total_hours = sum(
            self.graph[path[i]][path[i + 1]].get("transit_time_hours", 0)
            for i in range(len(path) - 1)
        )
        storage_rate_per_unit_day = 0.15
        days = total_hours / 24.0
        return storage_rate_per_unit_day * days

    # ------------------------------------------------------------------
    # Full cost breakdown
    # ------------------------------------------------------------------

    def cost_breakdown(self, request: CostBreakdownRequest) -> CostBreakdownResponse:
        """Decompose a total cost into its constituent parts.

        Provides an itemized breakdown with absolute amounts and
        percentages, useful for identifying cost drivers.

        Args:
            request: Cost components and quantity.

        Returns:
            Full breakdown with per-unit and total costs.
        """
        components = {
            "material": request.material_cost,
            "transport": request.transport_cost,
            "duty": request.duty_cost,
            "handling": request.handling_cost,
            "storage": request.storage_cost,
            "insurance": request.insurance_cost,
        }

        # Add overhead allocations
        for name, amount in request.overhead_allocations.items():
            components[name] = amount

        base_total = sum(components.values())
        component_pct = {
            k: (v / base_total * 100.0) if base_total > 0 else 0.0
            for k, v in components.items()
        }

        total_with_markup = base_total * (1 + request.markup_pct)
        per_unit = total_with_markup / request.quantity

        return CostBreakdownResponse(
            total_landed_cost=base_total,
            per_unit_cost=per_unit,
            components=components,
            component_pct=component_pct,
            total_with_markup=total_with_markup,
        )

    # ------------------------------------------------------------------
    # Network-wide cost analysis
    # ------------------------------------------------------------------

    def total_network_cost(self) -> dict[str, Any]:
        """Compute total cost across the entire supply chain network.

        Aggregates fixed facility costs, variable handling costs, and
        transport costs for all active edges. Provides a baseline for
        optimization comparisons.

        Returns:
            Cost summary with breakdown by category.
        """
        # Fixed costs for operating facilities
        fixed_costs = {
            n: self.graph.nodes[n].get("fixed_cost", 0.0)
            for n in self.graph.nodes()
        }
        total_fixed = sum(fixed_costs.values())

        # Variable costs based on capacity utilization
        variable_costs = {}
        for n, d in self.graph.nodes(data=True):
            capacity = d.get("capacity", 0.0)
            var_cost_per_unit = d.get("variable_cost", 0.0)
            # Assume 70% average utilization
            variable_costs[n] = capacity * 0.70 * var_cost_per_unit
        total_variable = sum(variable_costs.values())

        # Transport costs (capacity-weighted)
        transport_costs = {}
        for u, v, d in self.graph.edges(data=True):
            capacity = d.get("capacity", 0.0)
            cost_per_unit = d.get("cost_per_unit", 0.0)
            transport_costs[f"{u}->{v}"] = capacity * 0.60 * cost_per_unit
        total_transport = sum(transport_costs.values())

        total = total_fixed + total_variable + total_transport

        return {
            "total_cost": total,
            "fixed_costs": {"total": total_fixed, "detail": fixed_costs},
            "variable_costs": {"total": total_variable, "detail": variable_costs},
            "transport_costs": {"total": total_transport, "detail": transport_costs},
            "cost_by_node_type": self._cost_by_node_type(
                fixed_costs, variable_costs
            ),
        }

    def _cost_by_node_type(
        self,
        fixed_costs: dict[str, float],
        variable_costs: dict[str, float],
    ) -> dict[str, float]:
        """Group costs by node type for reporting."""
        result: dict[str, float] = {}
        for n in self.graph.nodes():
            node_type = self.graph.nodes[n].get("node_type", "unknown")
            combined = fixed_costs.get(n, 0) + variable_costs.get(n, 0)
            result[node_type] = result.get(node_type, 0.0) + combined
        return result

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        base_cost: float,
        parameters: dict[str, tuple[float, float]],
    ) -> dict[str, list[dict[str, float]]]:
        """Run one-at-a-time sensitivity analysis on cost parameters.

        For each parameter, sweeps across a range of values and reports
        the resulting total cost, identifying which parameters have the
        greatest impact on the bottom line.

        Args:
            base_cost: Baseline total cost.
            parameters: Dict mapping parameter names to (min_factor, max_factor)
                ranges. Factor 1.0 = no change.

        Returns:
            Dict mapping parameter names to lists of (factor, cost) pairs.
        """
        results: dict[str, list[dict[str, float]]] = {}

        for param_name, (min_factor, max_factor) in parameters.items():
            factors = np.linspace(min_factor, max_factor, 11)
            sweep = []
            for f in factors:
                perturbed_cost = base_cost * f
                sweep.append({
                    "factor": round(float(f), 3),
                    "cost": round(perturbed_cost, 2),
                    "delta_pct": round((f - 1.0) * 100, 1),
                })
            results[param_name] = sweep

        return results
