"""Compliance checker against GHS, OSHA, and SDS standards.

Cross-references extracted entities and raw text against a regulatory
database to produce a list of :class:`ComplianceFinding` instances.
"""

import re
from typing import Any

from src.models.schemas import ComplianceFinding, ComplianceStatus, DocumentType, ExtractedEntity, RiskLevel
from src.utils.regulation_db import RegulationDB


class ComplianceChecker:
    """Validate documents against industrial safety regulations.

    Args:
        regulation_db: Pre-loaded regulatory reference database.
    """

    def __init__(self, regulation_db: RegulationDB) -> None:
        self._regdb = regulation_db

    def check_document(
        self,
        entities: list[ExtractedEntity],
        raw_text: str,
        doc_type: DocumentType,
    ) -> list[ComplianceFinding]:
        """Run all applicable compliance checks and return findings.

        Args:
            entities: Entities extracted from the document.
            raw_text: Full cleaned text of the document.
            doc_type: Detected or declared document type.

        Returns:
            List of compliance findings.
        """
        findings: list[ComplianceFinding] = []
        findings.extend(self._check_ghs_compliance(entities))
        findings.extend(self._check_osha_limits(entities))
        findings.extend(self._check_ppe_requirements(entities))
        if doc_type == DocumentType.SDS:
            findings.extend(self._check_sds_completeness(raw_text))
        return findings

    # ------------------------------------------------------------------
    # GHS compliance
    # ------------------------------------------------------------------

    def _check_ghs_compliance(self, entities: list[ExtractedEntity]) -> list[ComplianceFinding]:
        """Verify that required GHS hazard statements are present.

        For each chemical found in the document, checks whether at
        least one hazard statement is referenced.

        Args:
            entities: Extracted entities.

        Returns:
            Findings for missing or present GHS statements.
        """
        findings: list[ComplianceFinding] = []
        chemicals = [e for e in entities if e.label == "CHEMICAL"]
        hazard_codes = {e.text.upper() for e in entities if e.label == "HAZARD_CODE"}
        ghs_hazards = self._regdb.get_ghs_hazards()

        if chemicals and not hazard_codes:
            findings.append(ComplianceFinding(
                rule_id="GHS-001",
                status=ComplianceStatus.FAIL,
                message="Chemicals detected but no GHS hazard codes found in the document.",
                evidence=f"Chemicals found: {', '.join(e.text for e in chemicals[:5])}",
                severity=RiskLevel.HIGH,
                entity_refs=[e.text for e in chemicals[:5]],
            ))

        for code in hazard_codes:
            if code in ghs_hazards:
                findings.append(ComplianceFinding(
                    rule_id=f"GHS-{code}",
                    status=ComplianceStatus.PASS,
                    message=f"Hazard statement {code} is present: {ghs_hazards[code]}",
                    evidence=code,
                    severity=RiskLevel.LOW,
                    entity_refs=[code],
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=f"GHS-{code}",
                    status=ComplianceStatus.WARNING,
                    message=f"Hazard code {code} detected but not found in GHS reference database.",
                    evidence=code,
                    severity=RiskLevel.MEDIUM,
                    entity_refs=[code],
                ))

        if not chemicals and not hazard_codes:
            findings.append(ComplianceFinding(
                rule_id="GHS-002",
                status=ComplianceStatus.UNKNOWN,
                message="No chemicals or hazard codes detected; manual review recommended.",
                evidence="",
                severity=RiskLevel.LOW,
            ))

        return findings

    # ------------------------------------------------------------------
    # OSHA PELs
    # ------------------------------------------------------------------

    def _check_osha_limits(self, entities: list[ExtractedEntity]) -> list[ComplianceFinding]:
        """Check concentration entities against OSHA PELs.

        Args:
            entities: Extracted entities.

        Returns:
            Findings for each concentration vs PEL comparison.
        """
        findings: list[ComplianceFinding] = []
        pels = self._regdb.get_osha_pels()
        concentrations = [e for e in entities if e.label == "CONCENTRATION"]
        chemicals = {e.text.lower(): e for e in entities if e.label == "CHEMICAL"}

        for conc_entity in concentrations:
            match = re.match(r"([\d.]+)\s*(ppm|mg/m3|mg/l|%)", conc_entity.text, re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2).lower()

            for chem_name, pel_info in pels.items():
                if any(chem_name in ent.text.lower() or ent.text.lower() in chem_name for ent in chemicals.values()):
                    pel_value = float(pel_info["pel"])
                    pel_unit = pel_info["unit"].lower()

                    if unit == pel_unit and value > pel_value:
                        findings.append(ComplianceFinding(
                            rule_id=f"OSHA-{chem_name.upper().replace(' ', '-')}",
                            status=ComplianceStatus.FAIL,
                            message=(
                                f"Concentration of {conc_entity.text} for {chem_name} "
                                f"exceeds OSHA PEL of {pel_value} {pel_unit}."
                            ),
                            evidence=conc_entity.text,
                            severity=RiskLevel.HIGH,
                            entity_refs=[conc_entity.text, chem_name],
                        ))
                    elif unit == pel_unit:
                        findings.append(ComplianceFinding(
                            rule_id=f"OSHA-{chem_name.upper().replace(' ', '-')}",
                            status=ComplianceStatus.PASS,
                            message=(
                                f"Concentration {conc_entity.text} for {chem_name} "
                                f"is within OSHA PEL of {pel_value} {pel_unit}."
                            ),
                            evidence=conc_entity.text,
                            severity=RiskLevel.LOW,
                            entity_refs=[conc_entity.text, chem_name],
                        ))

        return findings

    # ------------------------------------------------------------------
    # PPE requirements
    # ------------------------------------------------------------------

    def _check_ppe_requirements(self, entities: list[ExtractedEntity]) -> list[ComplianceFinding]:
        """Verify PPE coverage for detected hazards.

        Args:
            entities: Extracted entities.

        Returns:
            Findings for missing PPE.
        """
        findings: list[ComplianceFinding] = []
        hazard_codes = [e.text.upper() for e in entities if e.label == "HAZARD_CODE"]
        ppe_found = {e.text.lower() for e in entities if e.label == "PPE"}

        required_ppe: set[str] = set()
        for code in hazard_codes:
            for item in self._regdb.get_ppe_for_hazard(code):
                required_ppe.add(item)

        if not required_ppe:
            return findings

        for item in sorted(required_ppe):
            if item not in ppe_found:
                findings.append(ComplianceFinding(
                    rule_id=f"PPE-{item.upper().replace(' ', '-')}",
                    status=ComplianceStatus.FAIL,
                    message=f"Required PPE '{item}' not mentioned in the document for detected hazards.",
                    evidence=f"Required for: {', '.join(hazard_codes)}",
                    severity=RiskLevel.HIGH,
                    entity_refs=hazard_codes[:3],
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=f"PPE-{item.upper().replace(' ', '-')}",
                    status=ComplianceStatus.PASS,
                    message=f"PPE '{item}' is properly documented.",
                    evidence=item,
                    severity=RiskLevel.LOW,
                ))

        return findings

    # ------------------------------------------------------------------
    # SDS completeness
    # ------------------------------------------------------------------

    def _check_sds_completeness(self, raw_text: str) -> list[ComplianceFinding]:
        """Check that an SDS document contains all 16 required sections.

        Args:
            raw_text: Full cleaned text of the document.

        Returns:
            Findings for each missing or present SDS section.
        """
        findings: list[ComplianceFinding] = []
        required_sections = self._regdb.get_required_sds_sections()
        lower_text = raw_text.lower()

        for i, section_title in enumerate(required_sections, start=1):
            keywords = [w.lower() for w in section_title.split() if len(w) > 3]
            found = any(kw in lower_text for kw in keywords)

            if found:
                findings.append(ComplianceFinding(
                    rule_id=f"SDS-{i:02d}",
                    status=ComplianceStatus.PASS,
                    message=f"SDS section {i} present: {section_title}",
                    evidence=section_title,
                    severity=RiskLevel.LOW,
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=f"SDS-{i:02d}",
                    status=ComplianceStatus.FAIL,
                    message=f"SDS section {i} missing or not detectable: {section_title}",
                    evidence="",
                    severity=RiskLevel.HIGH if i <= 8 else RiskLevel.MEDIUM,
                ))

        return findings
