"""Tests for the compliance checker module."""

import pytest

from src.models.schemas import (
    ComplianceFinding,
    ComplianceStatus,
    DocumentType,
    ExtractedEntity,
    RiskLevel,
)
from src.services.compliance_checker import ComplianceChecker
from src.utils.regulation_db import RegulationDB


@pytest.fixture
def regdb() -> RegulationDB:
    """Return a RegulationDB with built-in defaults."""
    return RegulationDB(data_path=None)


@pytest.fixture
def checker(regdb: RegulationDB) -> ComplianceChecker:
    """Return a ComplianceChecker wired to the default regulation DB."""
    return ComplianceChecker(regdb)


def _make_entity(text: str, label: str, start: int = 0) -> ExtractedEntity:
    """Helper to create an ExtractedEntity with sensible defaults."""
    return ExtractedEntity(
        text=text,
        label=label,
        start_char=start,
        end_char=start + len(text),
        confidence=0.9,
    )


class TestCheckSDSCompleteness:
    """Tests for SDS 16-section completeness check."""

    def test_missing_all_sections(self, checker: ComplianceChecker) -> None:
        """Empty text fails all 16 sections."""
        findings = checker._check_sds_completeness("")
        failures = [f for f in findings if f.status == ComplianceStatus.FAIL]
        assert len(failures) == 16

    def test_present_sections(self, checker: ComplianceChecker) -> None:
        """Text containing section keywords passes those checks."""
        text = (
            "1. Identification of the substance and company\n"
            "2. Hazard identification\n"
            "Composition information on ingredients\n"
            "First aid measures\n"
            "Fire-fighting measures\n"
            "Accidental release measures\n"
            "Handling and storage\n"
            "Exposure controls personal protection\n"
            "Physical and chemical properties\n"
            "Stability and reactivity\n"
            "Toxicological information\n"
            "Ecological information\n"
            "Disposal considerations\n"
            "Transport information\n"
            "Regulatory information\n"
            "Other information"
        )
        findings = checker._check_sds_completeness(text)
        passes = [f for f in findings if f.status == ComplianceStatus.PASS]
        assert len(passes) == 16

    def test_partial_sections(self, checker: ComplianceChecker) -> None:
        """Text with some sections produces mix of PASS and FAIL."""
        text = "Identification of the substance and company. Hazard identification."
        findings = checker._check_sds_completeness(text)
        statuses = {f.status for f in findings}
        assert ComplianceStatus.PASS in statuses
        assert ComplianceStatus.FAIL in statuses


class TestCheckGHSCompliance:
    """Tests for GHS hazard statement verification."""

    def test_chemicals_no_codes_fails(self, checker: ComplianceChecker) -> None:
        """Chemicals present but no hazard codes triggers a FAIL."""
        entities = [_make_entity("sodium hydroxide", "CHEMICAL")]
        findings = checker._check_ghs_compliance(entities)
        failures = [f for f in findings if f.status == ComplianceStatus.FAIL and f.rule_id == "GHS-001"]
        assert len(failures) == 1

    def test_valid_hazard_code_passes(self, checker: ComplianceChecker) -> None:
        """A recognised H-code produces a PASS finding."""
        entities = [_make_entity("H314", "HAZARD_CODE")]
        findings = checker._check_ghs_compliance(entities)
        passes = [f for f in findings if f.status == ComplianceStatus.PASS and "H314" in f.rule_id]
        assert len(passes) == 1

    def test_no_entities_unknown(self, checker: ComplianceChecker) -> None:
        """Empty entity list yields an UNKNOWN finding."""
        findings = checker._check_ghs_compliance([])
        unknowns = [f for f in findings if f.status == ComplianceStatus.UNKNOWN]
        assert len(unknowns) == 1


class TestCheckOSHALimits:
    """Tests for OSHA PEL checking."""

    def test_within_limit_passes(self, checker: ComplianceChecker) -> None:
        """Concentration below PEL produces a PASS finding."""
        entities = [
            _make_entity("benzene", "CHEMICAL"),
            _make_entity("0.5 ppm", "CONCENTRATION"),
        ]
        findings = checker._check_osha_limits(entities)
        passes = [f for f in findings if f.status == ComplianceStatus.PASS and "benzene" in f.rule_id.lower()]
        assert len(passes) == 1

    def test_exceeding_limit_fails(self, checker: ComplianceChecker) -> None:
        """Concentration above PEL produces a FAIL finding."""
        entities = [
            _make_entity("benzene", "CHEMICAL"),
            _make_entity("5 ppm", "CONCENTRATION"),
        ]
        findings = checker._check_osha_limits(entities)
        failures = [f for f in findings if f.status == ComplianceStatus.FAIL and "benzene" in f.rule_id.lower()]
        assert len(failures) == 1
        assert "exceeds" in failures[0].message.lower()

    def test_no_chemical_no_findings(self, checker: ComplianceChecker) -> None:
        """No chemical entities means no OSHA findings."""
        entities = [_make_entity("100 ppm", "CONCENTRATION")]
        findings = checker._check_osha_limits(entities)
        assert findings == []


class TestCheckPPERequirements:
    """Tests for PPE coverage verification."""

    def test_missing_ppe_fails(self, checker: ComplianceChecker) -> None:
        """Hazard code present but PPE not mentioned triggers FAIL."""
        entities = [_make_entity("H314", "HAZARD_CODE")]
        findings = checker._check_ppe_requirements(entities)
        failures = [f for f in findings if f.status == ComplianceStatus.FAIL]
        assert len(failures) > 0

    def test_ppe_present_passes(self, checker: ComplianceChecker) -> None:
        """Hazard code with matching PPE produces PASS findings."""
        entities = [
            _make_entity("H314", "HAZARD_CODE"),
            _make_entity("gloves", "PPE"),
            _make_entity("goggles", "PPE"),
            _make_entity("face shield", "PPE"),
            _make_entity("lab coat", "PPE"),
        ]
        findings = checker._check_ppe_requirements(entities)
        passes = [f for f in findings if f.status == ComplianceStatus.PASS]
        assert len(passes) == 4

    def test_no_hazards_no_findings(self, checker: ComplianceChecker) -> None:
        """No hazard codes means no PPE requirements to check."""
        entities = [_make_entity("gloves", "PPE")]
        findings = checker._check_ppe_requirements(entities)
        assert findings == []


class TestCheckDocument:
    """Integration test for the main check_document method."""

    def test_sds_document(self, checker: ComplianceChecker) -> None:
        """SDS document type includes SDS section checks."""
        entities = [
            _make_entity("sodium hydroxide", "CHEMICAL"),
            _make_entity("H314", "HAZARD_CODE"),
        ]
        text = "Identification of the substance. Hazard identification. Sodium hydroxide H314."
        findings = checker.check_document(entities, text, DocumentType.SDS)
        prefixes = {f.rule_id.split("-")[0] for f in findings}
        assert "SDS" in prefixes
        assert "GHS" in prefixes

    def test_non_sds_skips_section_check(self, checker: ComplianceChecker) -> None:
        """Non-SDS documents skip the 16-section check."""
        entities = [_make_entity("H301", "HAZARD_CODE")]
        text = "Some manual content"
        findings = checker.check_document(entities, text, DocumentType.MANUAL)
        sds_findings = [f for f in findings if f.rule_id.startswith("SDS-")]
        assert len(sds_findings) == 0
