"""Tests for the entity extractor module."""

import pytest

from src.services.entity_extractor import EntityExtractor


@pytest.fixture
def extractor() -> EntityExtractor:
    """Return an EntityExtractor instance."""
    return EntityExtractor(model_name="en_core_web_sm")


class TestExtractEntities:
    """Tests for entity extraction on synthetic text."""

    def test_extracts_hazard_codes(self, extractor: EntityExtractor) -> None:
        """Detects GHS hazard codes in text."""
        text = "This product is classified as H301 Toxic if swallowed and H314 Causes severe skin burns."
        entities = extractor.extract_entities(text)
        labels = {e.label for e in entities}
        codes = [e.text for e in entities if e.label == "HAZARD_CODE"]
        assert "HAZARD_CODE" in labels
        assert "H301" in codes
        assert "H314" in codes

    def test_extracts_concentrations(self, extractor: EntityExtractor) -> None:
        """Detects concentration values."""
        text = "Contains 50% sodium hydroxide and 100 ppm of formaldehyde."
        entities = extractor.extract_entities(text)
        concs = [e for e in entities if e.label == "CONCENTRATION"]
        texts = [e.text for e in concs]
        assert any("50%" in t for t in texts)
        assert any("100 ppm" in t for t in texts)

    def test_extracts_ppe(self, extractor: EntityExtractor) -> None:
        """Detects PPE equipment mentions."""
        text = "Workers must wear gloves and goggles when handling this substance. A face shield is also recommended."
        entities = extractor.extract_entities(text)
        ppe = [e for e in entities if e.label == "PPE"]
        ppe_texts = [e.text.lower() for e in ppe]
        assert "gloves" in ppe_texts or "goggles" in ppe_texts

    def test_extracts_temperature(self, extractor: EntityExtractor) -> None:
        """Detects temperature measurements."""
        text = "Store below 25°C. Do not heat above 100°F."
        entities = extractor.extract_entities(text)
        temps = [e for e in entities if e.label == "TEMPERATURE"]
        texts = [e.text for e in temps]
        assert any("25°C" in t or "25" in t for t in texts)

    def test_extracts_pressure(self, extractor: EntityExtractor) -> None:
        """Detects pressure measurements."""
        text = "Operating pressure: 150 psi maximum."
        entities = extractor.extract_entities(text)
        pressures = [e for e in entities if e.label == "PRESSURE"]
        texts = [e.text for e in pressures]
        assert any("150 psi" in t or "150" in t for t in texts)

    def test_extracts_chemicals(self, extractor: EntityExtractor) -> None:
        """Detects known chemical names via EntityRuler."""
        text = "Sodium hydroxide is used as a cleaning agent. Methanol is a solvent."
        entities = extractor.extract_entities(text)
        chemicals = [e for e in entities if e.label == "CHEMICAL"]
        chem_texts = [e.text.lower() for e in chemicals]
        assert "sodium hydroxide" in chem_texts or "methanol" in chem_texts

    def test_no_entities_clean_text(self, extractor: EntityExtractor) -> None:
        """Returns empty list for text with no identifiable entities."""
        text = "The weather is nice today and the sky is blue."
        entities = extractor.extract_entities(text)
        assert entities == []

    def test_empty_text(self, extractor: EntityExtractor) -> None:
        """Handles empty input gracefully."""
        entities = extractor.extract_entities("")
        assert entities == []

    def test_positions_are_correct(self, extractor: EntityExtractor) -> None:
        """Entity positions align with the source text."""
        text = "H301 Toxic if swallowed"
        entities = extractor.extract_entities(text)
        hazard = [e for e in entities if e.text == "H301"]
        assert len(hazard) == 1
        assert text[hazard[0].start_char:hazard[0].end_char] == "H301"
