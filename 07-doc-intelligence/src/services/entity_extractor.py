"""Named-entity recognition using spaCy and custom pattern rules.

Combines a spaCy NER pipeline with an EntityRuler and regex-based
extractors for domain-specific industrial entities: chemicals,
concentrations, hazard codes, PPE, and physical measurements.
"""

import re
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from src.models.schemas import ExtractedEntity


class EntityExtractor:
    """Extract industrial-domain entities from document text.

    Uses a spaCy model augmented with an EntityRuler for chemical
    names, and regex-based extractors for structured patterns like
    concentrations, hazard codes, PPE terms, and measurements.

    Args:
        model_name: Name of the spaCy model to load.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            self._nlp: Language = spacy.load(model_name)
        except OSError:
            self._nlp = spacy.load("en_core_web_sm")
        self._setup_patterns()
        self._ppe_terms: list[str] = [
            "gloves", "goggles", "safety goggles", "respirator",
            "face shield", "lab coat", "protective clothing",
            "safety glasses", "hard hat", "steel-toe boots",
            "chemical apron", "fume hood",
        ]

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Run NER and custom pattern matching on *text*.

        Args:
            text: Cleaned document text.

        Returns:
            Deduplicated list of :class:`ExtractedEntity` instances.
        """
        if not text or not text.strip():
            return []

        doc: Doc = self._nlp(text)
        entities: list[ExtractedEntity] = []
        seen_spans: set[tuple[int, int]] = set()

        for ent in doc.ents:
            label = self._map_label(ent.label_)
            if label and (ent.start_char, ent.end_char) not in seen_spans:
                seen_spans.add((ent.start_char, ent.end_char))
                entities.append(ExtractedEntity(
                    text=ent.text,
                    label=label,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.85,
                ))

        for extractor in (
            self._extract_concentrations,
            self._extract_hazard_codes,
            self._extract_ppe,
            self._extract_measurements,
        ):
            for entity in extractor(text):
                span_key = (entity.start_char, entity.end_char)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(entity)

        entities.sort(key=lambda e: e.start_char)
        return entities

    # ------------------------------------------------------------------
    # Pattern setup
    # ------------------------------------------------------------------

    def _setup_patterns(self) -> None:
        """Configure EntityRuler patterns for chemical names."""
        if "entity_ruler" not in self._nlp.pipe_names:
            ruler = self._nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self._nlp.get_pipe("entity_ruler")

        chemicals = [
            "sodium hydroxide", "sodium chloride", "sodium hypochlorite",
            "methanol", "ethanol", "isopropanol", "acetone",
            "sulfuric acid", "hydrochloric acid", "nitric acid",
            "acetic acid", "phosphoric acid", "hydrofluoric acid",
            "benzene", "toluene", "xylene", "chloroform",
            "carbon tetrachloride", "dichloromethane",
            "ammonia", "hydrogen peroxide", "hydrogen sulfide",
            "carbon monoxide", "carbon dioxide", "nitrogen dioxide",
            "sulfur dioxide", "chlorine", "ozone",
            "asbestos", "lead", "mercury", "cadmium", "arsenic",
            "formaldehyde", "acrolein", "phenol",
            "ethylene glycol", "propylene glycol",
            "sodium carbonate", "calcium chloride",
            "potassium hydroxide", "sodium sulfate",
            "sodium nitrate", "ammonium nitrate",
        ]

        patterns = [
            {"label": "CHEMICAL", "pattern": [{"LOWER": {"IN": [c.split()[0]]}, "LOWER": {"IN": c.split()[1:] if len(c.split()) > 1 else []}}]}
            for c in chemicals if len(c.split()) > 1
        ]
        single_words = [c for c in chemicals if len(c.split()) == 1]
        patterns += [{"label": "CHEMICAL", "pattern": [{"LOWER": c}]} for c in single_words]
        patterns += [{"label": "CHEMICAL", "pattern": c} for c in chemicals if " " in c]

        ruler.add_patterns(patterns)

    # ------------------------------------------------------------------
    # NER label mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_label(spacy_label: str) -> str | None:
        """Map spaCy entity labels to domain-specific labels.

        Args:
            spacy_label: Original spaCy NER label.

        Returns:
            Mapped label or ``None`` if unmapped.
        """
        mapping: dict[str, str] = {
            "CHEMICAL": "CHEMICAL",
            "ORG": "CHEMICAL",
            "PRODUCT": "CHEMICAL",
            "QUANTITY": "CONCENTRATION",
            "CARDINAL": "CONCENTRATION",
        }
        return mapping.get(spacy_label)

    # ------------------------------------------------------------------
    # Regex extractors
    # ------------------------------------------------------------------

    def _extract_concentrations(self, text: str) -> list[ExtractedEntity]:
        """Extract concentration values (%, ppm, mg/L, etc.).

        Args:
            text: Document text.

        Returns:
            List of concentration entities.
        """
        pattern = r"\b(\d+(?:\.\d+)?)\s*(%|ppm|ppb|mg/L|mg/m3|g/L|μg/m3|vol%|w/w|v/v)(?=\s|[^a-zA-Z0-9]|$)"
        results: list[ExtractedEntity] = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append(ExtractedEntity(
                text=match.group(0),
                label="CONCENTRATION",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95,
            ))
        return results

    def _extract_hazard_codes(self, text: str) -> list[ExtractedEntity]:
        """Extract GHS hazard codes (H### pattern).

        Args:
            text: Document text.

        Returns:
            List of hazard code entities.
        """
        pattern = r"\b(H\d{3}[a-z]?)\b"
        results: list[ExtractedEntity] = []
        for match in re.finditer(pattern, text):
            code = match.group(1)
            if 200 <= int(re.search(r"\d+", code).group()) <= 999:
                results.append(ExtractedEntity(
                    text=code,
                    label="HAZARD_CODE",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.90,
                ))
        return results

    def _extract_ppe(self, text: str) -> list[ExtractedEntity]:
        """Extract PPE equipment mentions from the text.

        Args:
            text: Document text.

        Returns:
            List of PPE entities.
        """
        results: list[ExtractedEntity] = []
        lower_text = text.lower()
        for term in self._ppe_terms:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(lower_text):
                results.append(ExtractedEntity(
                    text=text[match.start():match.end()],
                    label="PPE",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.90,
                ))
        return results

    def _extract_measurements(self, text: str) -> list[ExtractedEntity]:
        """Extract temperature and pressure measurements.

        Args:
            text: Document text.

        Returns:
            List of temperature and pressure entities.
        """
        results: list[ExtractedEntity] = []
        temp_pattern = r"\b(-?\d+(?:\.\d+)?)\s*°[CF]\b"
        for match in re.finditer(temp_pattern, text):
            results.append(ExtractedEntity(
                text=match.group(0),
                label="TEMPERATURE",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95,
            ))

        pressure_pattern = r"\b(\d+(?:\.\d+)?)\s*(psi|bar|kPa|atm|mmHg|torr)\b"
        for match in re.finditer(pressure_pattern, text, re.IGNORECASE):
            results.append(ExtractedEntity(
                text=match.group(0),
                label="PRESSURE",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95,
            ))

        return results
