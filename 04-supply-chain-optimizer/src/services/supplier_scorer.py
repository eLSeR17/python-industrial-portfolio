"""Multi-criteria supplier evaluation, ranking, and sensitivity analysis."""

from typing import Any

import numpy as np

from src.models.schemas import (
    SupplierAnalyzeRequest,
    SupplierAnalyzeResponse,
    SupplierMetrics,
    SupplierScore,
    SupplierWeights,
)


class SupplierScorer:
    """Multi-criteria supplier evaluation and ranking.

    Implements a weighted scoring model that normalizes supplier attributes
    onto a common 0-1 scale and applies configurable weights. The scorer
    handles both "lower is better" metrics (price, lead time, defect rate)
    and "higher is better" metrics (on-time delivery, financial stability).

    Sensitivity analysis perturbs each weight by +/- the specified fraction
    to identify which criteria most affect the ranking.
    """

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def analyze(self, request: SupplierAnalyzeRequest) -> SupplierAnalyzeResponse:
        """Score, rank, and compare suppliers.

        Args:
            request: Supplier metrics, weights, and sensitivity settings.

        Returns:
            Ranked scores, sensitivity analysis, and recommendation.
        """
        suppliers = request.suppliers
        weights = request.weights

        # Normalize each criterion to 0-1 (1 = best)
        normalized = self._normalize_suppliers(suppliers)

        # Weighted scoring
        scored = self._weighted_scores(suppliers, normalized, weights)

        # Rank
        scored.sort(key=lambda s: s.weighted_score, reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1

        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            suppliers, weights, request.sensitivity_perturbation
        )

        recommended = scored[0].supplier_id if scored else ""

        return SupplierAnalyzeResponse(
            scores=scored,
            sensitivity=sensitivity,
            recommended_supplier=recommended,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_suppliers(
        self, suppliers: list[SupplierMetrics]
    ) -> dict[str, dict[str, float]]:
        """Normalize each criterion to [0, 1] where 1 = best.

        Returns:
            Dict mapping supplier_id -> criterion -> normalized_score.
        """
        # Collect raw values
        raw: dict[str, dict[str, float]] = {}
        criteria = ["price", "quality", "lead_time", "reliability", "esg_risk"]

        for s in suppliers:
            raw[s.supplier_id] = {
                "price": s.unit_price,
                "quality": s.defect_rate_ppm,
                "lead_time": s.lead_time_days,
                "reliability": s.on_time_delivery_pct / 100.0,
                "esg_risk": s.esg_risk_score,
            }

        # Min-max normalization per criterion
        # "lower is better": price, quality (defects), lead_time, esg_risk
        # "higher is better": reliability
        lower_better = {"price", "quality", "lead_time", "esg_risk"}
        higher_better = {"reliability"}

        normalized: dict[str, dict[str, float]] = {}
        for s in suppliers:
            normalized[s.supplier_id] = {}

        for criterion in criteria:
            values = [raw[s.supplier_id][criterion] for s in suppliers]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            for s in suppliers:
                if range_val == 0:
                    normalized[s.supplier_id][criterion] = 0.5
                elif criterion in lower_better:
                    # Invert: lower raw = higher score
                    normalized[s.supplier_id][criterion] = (
                        (max_val - raw[s.supplier_id][criterion]) / range_val
                    )
                else:
                    # Direct: higher raw = higher score
                    normalized[s.supplier_id][criterion] = (
                        (raw[s.supplier_id][criterion] - min_val) / range_val
                    )

        return normalized

    # ------------------------------------------------------------------
    # Weighted scoring
    # ------------------------------------------------------------------

    def _weighted_scores(
        self,
        suppliers: list[SupplierMetrics],
        normalized: dict[str, dict[str, float]],
        weights: SupplierWeights,
    ) -> list[SupplierScore]:
        """Apply weights to normalized scores and identify strengths/weaknesses."""
        weight_map = {
            "price": weights.price,
            "quality": weights.quality,
            "lead_time": weights.lead_time,
            "reliability": weights.reliability,
            "esg_risk": weights.esg_risk,
        }

        scores: list[SupplierScore] = []
        for s in suppliers:
            criterion_scores = {}
            weighted_sum = 0.0

            for criterion, w in weight_map.items():
                ns = normalized[s.supplier_id][criterion]
                criterion_scores[criterion] = round(ns, 4)
                weighted_sum += ns * w

            # Identify strengths (score > 0.7) and weaknesses (score < 0.3)
            strengths = [
                c for c, v in criterion_scores.items() if v > 0.7
            ]
            weaknesses = [
                c for c, v in criterion_scores.items() if v < 0.3
            ]

            scores.append(SupplierScore(
                supplier_id=s.supplier_id,
                name=s.name,
                weighted_score=round(weighted_sum, 4),
                rank=0,  # set after sorting
                criterion_scores=criterion_scores,
                strengths=strengths,
                weaknesses=weaknesses,
            ))

        return scores

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def _sensitivity_analysis(
        self,
        suppliers: list[SupplierMetrics],
        weights: SupplierWeights,
        perturbation: float,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perturb each weight and observe ranking changes.

        For each criterion weight, sweeps from (weight - perturbation)
        to (weight + perturbation) while redistributing the remainder
        proportionally among other weights.

        Args:
            suppliers: Supplier metrics.
            weights: Base weight vector.
            perturbation: Maximum fractional perturbation per weight.

        Returns:
            Dict mapping criterion name to list of sweep results.
        """
        normalized = self._normalize_suppliers(suppliers)
        criteria = ["price", "quality", "lead_time", "reliability", "esg_risk"]
        weight_list = [weights.price, weights.quality, weights.lead_time,
                       weights.reliability, weights.esg_risk]

        results: dict[str, list[dict[str, Any]]] = {}

        for idx, criterion in enumerate(criteria):
            sweep = []
            base_w = weight_list[idx]

            # Sweep from base - perturbation to base + perturbation
            steps = 5
            deltas = np.linspace(-perturbation, perturbation, 2 * steps + 1)

            for delta in deltas:
                # Perturb this weight
                new_weight_list = weight_list.copy()
                new_weight_list[idx] = max(0.001, base_w + delta * base_w)

                # Redistribute remainder proportionally
                remainder = 1.0 - new_weight_list[idx]
                other_sum = sum(
                    w for j, w in enumerate(weight_list) if j != idx
                )
                for j in range(len(new_weight_list)):
                    if j != idx and other_sum > 0:
                        new_weight_list[j] = (
                            weight_list[j] / other_sum * remainder
                        )

                # Create perturbed weights
                perturbed = SupplierWeights(
                    price=new_weight_list[0],
                    quality=new_weight_list[1],
                    lead_time=new_weight_list[2],
                    reliability=new_weight_list[3],
                    esg_risk=new_weight_list[4],
                )

                # Recompute scores
                scored = self._weighted_scores(suppliers, normalized, perturbed)
                scored.sort(key=lambda s: s.weighted_score, reverse=True)

                sweep.append({
                    "delta_pct": round(delta * 100, 1),
                    "weight": round(new_weight_list[idx], 4),
                    "winner": scored[0].supplier_id,
                    "scores": {
                        s.supplier_id: round(s.weighted_score, 4) for s in scored
                    },
                })

            results[criterion] = sweep

        return results
