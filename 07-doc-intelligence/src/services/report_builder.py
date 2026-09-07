"""Report builder generating HTML and JSON compliance reports.

Uses Jinja2 for HTML rendering and produces structured dicts for JSON.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models.schemas import ComplianceFinding, ComplianceStatus, ExtractedEntity, RiskLevel


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


class ReportBuilder:
    """Build compliance reports in HTML or JSON format.

    Args:
        template_dir: Optional override for the Jinja2 template directory.
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        tdir = Path(template_dir) if template_dir else _TEMPLATE_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(tdir)),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build_html_report(
        self,
        document_id: str,
        metadata: dict[str, Any],
        entities: list[ExtractedEntity],
        findings: list[ComplianceFinding],
        risk_score: float,
        risk_level: RiskLevel,
    ) -> str:
        """Render an HTML compliance report.

        Args:
            document_id: UUID of the processed document.
            metadata: Document metadata dict.
            entities: Extracted entities.
            findings: Compliance findings.
            risk_score: Computed 0-100 risk score.
            risk_level: Mapped risk level.

        Returns:
            Rendered HTML string.
        """
        recommendations = self._generate_recommendations(findings)
        template = self._env.get_template("compliance_report.html")
        return template.render(
            document_id=document_id,
            metadata=metadata,
            entities=entities,
            findings=findings,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendations=recommendations,
        )

    def build_json_report(
        self,
        document_id: str,
        metadata: dict[str, Any],
        entities: list[ExtractedEntity],
        findings: list[ComplianceFinding],
        risk_score: float,
        risk_level: RiskLevel,
    ) -> dict[str, Any]:
        """Build a structured JSON compliance report.

        Args:
            document_id: UUID of the processed document.
            metadata: Document metadata dict.
            entities: Extracted entities.
            findings: Compliance findings.
            risk_score: Computed 0-100 risk score.
            risk_level: Mapped risk level.

        Returns:
            Report as a JSON-serialisable dict.
        """
        recommendations = self._generate_recommendations(findings)
        return {
            "document_id": document_id,
            "metadata": metadata,
            "risk_assessment": {
                "score": round(risk_score, 2),
                "level": risk_level.value,
            },
            "entities": [e.model_dump() for e in entities],
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "status": f.status.value,
                    "message": f.message,
                    "evidence": f.evidence,
                    "severity": f.severity.value,
                    "entity_refs": f.entity_refs,
                }
                for f in findings
            ],
            "summary": {
                "total_findings": len(findings),
                "compliant": sum(1 for f in findings if f.status == ComplianceStatus.PASS),
                "non_compliant": sum(1 for f in findings if f.status == ComplianceStatus.FAIL),
                "warnings": sum(1 for f in findings if f.status == ComplianceStatus.WARNING),
                "unknown": sum(1 for f in findings if f.status == ComplianceStatus.UNKNOWN),
            },
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_recommendations(self, findings: list[ComplianceFinding]) -> list[str]:
        """Generate actionable recommendations from findings.

        Args:
            findings: Compliance findings list.

        Returns:
            Unique recommendation strings.
        """
        recommendations: list[str] = []
        seen: set[str] = set()

        for finding in findings:
            if finding.status != ComplianceStatus.FAIL:
                continue
            rule_prefix = finding.rule_id.split("-")[0]

            if rule_prefix == "GHS" and "GHS" not in seen:
                seen.add("GHS")
                recommendations.append(
                    "Add all required GHS hazard statements for every chemical "
                    "identified in the document. Each substance must have at least "
                    "one hazard code and corresponding precautionary statement."
                )
            elif rule_prefix == "OSHA" and "OSHA" not in seen:
                seen.add("OSHA")
                recommendations.append(
                    "Ensure all chemical concentrations are below OSHA "
                    "Permissible Exposure Limits (PELs). Where concentrations "
                    "exceed PELs, implement engineering controls or provide "
                    "respiratory protection."
                )
            elif rule_prefix == "PPE" and "PPE" not in seen:
                seen.add("PPE")
                recommendations.append(
                    "Document all required Personal Protective Equipment (PPE) "
                    "in the safety data sheet or operational manual. Each hazard "
                    "must have corresponding PPE specified."
                )
            elif rule_prefix == "SDS" and "SDS" not in seen:
                seen.add("SDS")
                recommendations.append(
                    "Complete all 16 mandatory SDS sections. Missing sections "
                    "may result in non-compliance with GHS regulations and "
                    "can endanger workers who rely on this information."
                )

        return recommendations
