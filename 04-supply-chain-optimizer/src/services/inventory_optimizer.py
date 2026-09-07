"""EOQ-based inventory optimization and multi-echelon bullwhip analysis."""

import math
from typing import Any

import numpy as np

from src.models.schemas import (
    BullwhipRequest,
    EchelonInventory,
    InventoryOptimizeRequest,
    InventoryOptimizeResponse,
)


class InventoryOptimizer:
    """Optimize inventory levels using classical and multi-echelon methods.

    Implements:
    - Economic Order Quantity (EOQ) for single-item, single-location
    - Safety stock calculation from demand uncertainty and lead time
    - Multi-echelon bullwhip effect quantification
    - Cost comparison against naive ordering strategies

    The EOQ balances ordering costs against holding costs to find the
    quantity that minimizes total annual inventory cost.
    """

    def __init__(
        self,
        holding_cost_pct: float = 0.25,
        ordering_cost: float = 50.0,
        lead_time_days: float = 7.0,
        service_level: float = 0.95,
        days_per_year: int = 365,
    ) -> None:
        """Initialize with default inventory parameters.

        Args:
            holding_cost_pct: Annual holding cost as fraction of unit cost.
            ordering_cost: Fixed cost per order placed.
            lead_time_days: Supplier lead time in days.
            service_level: Target cycle service level (0-1).
            days_per_year: Operating days per year.
        """
        self.holding_cost_pct = holding_cost_pct
        self.ordering_cost = ordering_cost
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.days_per_year = days_per_year

    # ------------------------------------------------------------------
    # Core EOQ
    # ------------------------------------------------------------------

    def optimize(self, request: InventoryOptimizeRequest) -> InventoryOptimizeResponse:
        """Compute optimal inventory parameters using EOQ model.

        The Economic Order Quantity minimizes the sum of annual ordering
        and holding costs. Safety stock protects against demand variability
        during the lead time.

        Args:
            request: Demand, cost, and uncertainty parameters.

        Returns:
            Optimal order quantity, reorder point, safety stock, and cost analysis.
        """
        D = request.annual_demand
        S = request.ordering_cost
        H = request.holding_cost_pct * request.unit_cost
        L = request.lead_time_days
        sigma_d = request.demand_std_dev
        cs = request.service_level
        N = request.days_per_year

        # Daily demand
        daily_demand = D / N

        # --- EOQ ---
        if H <= 0 or D <= 0:
            eoq = 0.0
        else:
            eoq = math.sqrt((2.0 * D * S) / H)

        # --- Safety stock ---
        z = self._z_score(cs)
        lead_time_demand_std = sigma_d * math.sqrt(L) if sigma_d > 0 else 0.0
        safety_stock = z * lead_time_demand_std

        # --- Reorder point ---
        lead_time_demand = daily_demand * L
        reorder_point = lead_time_demand + safety_stock

        # --- Inventory levels ---
        max_inventory = eoq + safety_stock
        average_inventory = (eoq / 2.0) + safety_stock

        # --- Costs ---
        orders_per_year = D / eoq if eoq > 0 else 0.0
        annual_ordering_cost = orders_per_year * S
        annual_holding_cost = average_inventory * H
        annual_total = annual_ordering_cost + annual_holding_cost

        # --- Baseline comparison (ordering once per lead time period) ---
        baseline_orders = D / max(1.0, daily_demand * L)
        baseline_ordering = baseline_orders * S
        baseline_avg_inv = (daily_demand * L / 2.0) + safety_stock
        baseline_holding = baseline_avg_inv * H
        baseline_total = baseline_ordering + baseline_holding

        savings_pct = (
            ((baseline_total - annual_total) / baseline_total * 100.0)
            if baseline_total > 0
            else 0.0
        )

        return InventoryOptimizeResponse(
            eoq=round(eoq, 2),
            reorder_point=round(reorder_point, 2),
            safety_stock=round(safety_stock, 2),
            max_inventory=round(max_inventory, 2),
            average_inventory=round(average_inventory, 2),
            orders_per_year=round(orders_per_year, 2),
            annual_ordering_cost=round(annual_ordering_cost, 2),
            annual_holding_cost=round(annual_holding_cost, 2),
            annual_total_cost=round(annual_total, 2),
            total_cost_savings_pct=round(savings_pct, 2),
        )

    # ------------------------------------------------------------------
    # Multi-echelon bullwhip analysis
    # ------------------------------------------------------------------

    def analyze_bullwhip(self, request: BullwhipRequest) -> dict[str, Any]:
        """Quantify the bullwhip effect across supply chain echelons.

        The bullwhip effect measures demand variability amplification
        as orders propagate upstream. Each echelon's order variance is
        computed from its demand variance, lead time, and ordering
        policy (EOQ-based).

        Args:
            request: List of echelons from downstream (customer) to
                upstream (supplier).

        Returns:
            Bullwhip ratios per echelon and aggregate metrics.
        """
        echelons = request.echelons
        if not echelons:
            return {"echelons": [], "max_bullwhip": 0.0, "recommendation": ""}

        results = []
        prev_order_variance = None

        for i, echelon in enumerate(echelons):
            daily_demand = echelon.demand_mean / 365.0
            demand_var = echelon.demand_std_dev ** 2
            L = echelon.lead_time_days
            H = echelon.holding_cost_pct * echelon.unit_cost
            S = echelon.ordering_cost

            # EOQ for this echelon
            if H > 0 and echelon.demand_mean > 0:
                eoq = math.sqrt(2 * echelon.demand_mean * S / H)
            else:
                eoq = 1.0

            # Order variance under EOQ policy (standard formula)
            # Var(orders) = Var(demand) * [1 + (2*L/ROP) + (2*EOQ^2)/(ROP*demand)]
            if daily_demand > 0:
                rop = daily_demand * L + math.sqrt(max(0, demand_var * L))
                order_variance = demand_var * (
                    1.0
                    + (2.0 * L / max(1.0, rop))
                    + (2.0 * eoq ** 2) / (max(1.0, rop) * daily_demand)
                )
            else:
                order_variance = demand_var

            # Bullwhip ratio = order variance / demand variance
            bullwhip_ratio = (
                order_variance / demand_var if demand_var > 0 else 1.0
            )

            # Downstream demand variance (what this echelon sees)
            input_demand_var = prev_order_variance if prev_order_variance is not None else demand_var

            results.append({
                "echelon_index": i,
                "name": echelon.name,
                "demand_mean": round(echelon.demand_mean, 2),
                "demand_std_dev": round(echelon.demand_std_dev, 2),
                "lead_time_days": L,
                "eoq": round(eoq, 2),
                "input_demand_variance": round(input_demand_var, 2),
                "output_order_variance": round(order_variance, 2),
                "bullwhip_ratio": round(bullwhip_ratio, 4),
                "amplification": round(bullwhip_ratio - 1.0, 4),
            })

            prev_order_variance = order_variance

        ratios = [e["bullwhip_ratio"] for e in results]
        max_bullwhip = max(ratios) if ratios else 1.0

        # Generate recommendation
        if max_bullwhip > 3.0:
            rec = (
                "Critical bullwhip effect detected (ratio > 3.0). "
                "Implement vendor-managed inventory (VMI) and information "
                "sharing to reduce demand amplification."
            )
        elif max_bullwhip > 2.0:
            rec = (
                "Significant bullwhip effect. Consider reducing lead times, "
                "sharing POS data with upstream suppliers, and smoothing "
                "order patterns."
            )
        else:
            rec = (
                "Bullwhip effect is within acceptable range. "
                "Continue monitoring and maintain collaborative planning."
            )

        return {
            "echelons": results,
            "max_bullwhip_ratio": max_bullwhip,
            "total_amplification": round(
                (ratios[-1] - 1.0) * 100.0 if ratios else 0.0, 2
            ),
            "recommendation": rec,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _z_score(service_level: float) -> float:
        """Inverse normal CDF approximation for common service levels.

        Uses the rational approximation by Abramowitz and Stegun for
        the normal quantile function. Accurate to ~4.5e-4 for
        0.5 < p < 0.999.
        """
        if service_level <= 0.0:
            return -3.0
        if service_level >= 1.0:
            return 3.09
        if service_level < 0.5:
            return -InventoryOptimizer._z_score(1.0 - service_level)

        # Rational approximation (Abramowitz & Stegun 26.2.23)
        p = service_level
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        )
        return round(z, 4)
