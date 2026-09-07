"""Risk scoring engine for compliance findings.

Computes a weighted 0-100 risk score from compliance findings and
extracted entities, then maps it to a :class:`RiskLevel`.
"""

from config.settings import get_settings
from src.models.schemas import ComplianceFinding, ComplianceStatus, ExtractedEntity, RiskLevel


class RiskScorer:
    """Compute a composite risk score from compliance findings.

    The scoring weights are:
    - Missing critical hazard information: 40 points
    - Exceeding concentration limits: 30 points
    - Missing PPE: 20 points
    - Incomplete SDS sections: 10 points
    """

    def score(
        self,
        findings: list[ComplianceFinding],
        entities: list[ExtractedEntity],
    ) -> tuple[float, RiskLevel]:
        """Compute the aggregated risk score and level.

        Args:
            findings: Compliance findings from all checks.
            entities: Extracted entities (used for context).

        Returns:
            ``(risk_score, risk_level)`` tuple.
        """
        category_scores = self._compute_category_scores(findings)
        raw_score = sum(category_scores.values())
        max_possible = 40.0 + 30.0 + 20.0 + 10.0
        normalized = self._normalize_score(raw_score, max_possible)
        level = self._score_to_level(normalized)
        return normalized, level

    def _compute_category_scores(self, findings: list[ComplianceFinding]) -> dict[str, float]:
        """Break down the risk score by compliance category.

        Args:
            findings: All compliance findings.

        Returns:
            Dict mapping category name to its contribution (0-max).
        """
        thresholds = get_settings().risk_thresholds
        scores: dict[str, float] = {
            "hazard_info": 0.0,
            "concentration": 0.0,
            "ppe": 0.0,
            "sections": 0.0,
        }

        for finding in findings:
            if finding.status != ComplianceStatus.FAIL:
                continue

            rule_prefix = finding.rule_id.split("-")[0]
            severity_weight = self._severity_weight(finding.severity)

            if rule_prefix == "GHS":
                scores["hazard_info"] += 40.0 * severity_weight
            elif rule_prefix == "OSHA":
                scores["concentration"] += 30.0 * severity_weight
            elif rule_prefix == "PPE":
                scores["ppe"] += 20.0 * severity_weight
            elif rule_prefix == "SDS":
                scores["sections"] += 10.0 * severity_weight

        for key in scores:
            max_val = {"hazard_info": 40.0, "concentration": 30.0, "ppe": 20.0, "sections": 10.0}[key]
            scores[key] = min(scores[key], max_val)

        return scores

    @staticmethod
    def _severity_weight(level: RiskLevel) -> float:
        """Map a risk level to a numeric weight.

        Args:
            level: Risk level enum value.

        Returns:
            Weight between 0.0 and 1.0.
        """
        return {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.LOW: 0.2,
        }.get(level, 0.1)

    @staticmethod
    def _normalize_score(raw_score: float, max_possible: float) -> float:
        """Normalize a raw score to 0-100.

        Args:
            raw_score: Sum of category scores.
            max_possible: Theoretical maximum.

        Returns:
            Clamped 0-100 score.
        """
        if max_possible <= 0:
            return 0.0
        normalized = (raw_score / max_possible) * 100.0
        return max(0.0, min(100.0, normalized))

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        """Map a numeric score to a :class:`RiskLevel`.

        Args:
            score: Normalized 0-100 risk score.

        Returns:
            Corresponding risk level.
        """
        thresholds = get_settings().risk_thresholds
        if score >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        if score >= thresholds["high"]:
            return RiskLevel.HIGH
        if score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
