"""Regulatory reference database for compliance checks.

Loads GHS hazard statements, OSHA PELs, signal words, required SDS
sections, and PPE mappings from JSON files or built-in defaults.
"""

import json
import os
from pathlib import Path
from typing import Any


class RegulationDB:
    """In-memory store of regulatory reference data.

    Args:
        data_path: Optional filesystem path to a directory containing
            ``ghs_sentences.json``. Falls back to built-in defaults when
            *None* or the file does not exist.
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._data_path = data_path
        self._ghs: dict[str, str] = {}
        self._osha_pels: dict[str, dict[str, Any]] = {}
        self._signal_words: dict[str, str] = {}
        self._ppe_map: dict[str, list[str]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ghs_hazards(self) -> dict[str, str]:
        """Return GHS hazard statements keyed by H-code.

        Returns:
            Mapping like ``{"H301": "Toxic if swallowed", ...}``.
        """
        return dict(self._ghs)

    def get_osha_pels(self) -> dict[str, dict[str, Any]]:
        """Return OSHA Permissible Exposure Limits.

        Returns:
            Mapping of chemical name to a dict with ``pel``,
            ``unit``, and ``cas`` keys.
        """
        return dict(self._osha_pels)

    def get_signal_words(self) -> dict[str, str]:
        """Return signal-word mapping (Danger / Warning).

        Returns:
            Mapping like ``{"H301": "Danger", ...}``.
        """
        return dict(self._signal_words)

    def get_required_sds_sections(self) -> list[str]:
        """Return the 16 mandatory GHS SDS section identifiers.

        Returns:
            Ordered list of section title keywords.
        """
        return [
            "Identification of the substance/mixture and of the company/undertaking",
            "Hazard(s) identification",
            "Composition/information on ingredients",
            "First aid measures",
            "Fire-fighting measures",
            "Accidental release measures",
            "Handling and storage",
            "Exposure controls/personal protection",
            "Physical and chemical properties",
            "Stability and reactivity",
            "Toxicological information",
            "Ecological information",
            "Disposal considerations",
            "Transport information",
            "Regulatory information",
            "Other information",
        ]

    def get_ppe_for_hazard(self, hazard_code: str) -> list[str]:
        """Return required PPE items for a given H-code.

        Args:
            hazard_code: GHS H-code such as ``H314``.

        Returns:
            List of PPE item names (e.g. ``["gloves", "goggles"]``).
        """
        return list(self._ppe_map.get(hazard_code, self._ppe_map.get("default", [])))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load data from JSON file or fall back to built-in defaults."""
        loaded = False
        if self._data_path:
            ghs_path = Path(self._data_path) / "ghs_sentences.json"
            if ghs_path.exists():
                try:
                    with open(ghs_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    self._ghs = {k: str(v) for k, v in data.items()}
                    loaded = True
                except (json.JSONDecodeError, OSError):
                    loaded = False

        if not loaded:
            self._ghs = self._default_ghs()

        self._osha_pels = self._default_osha_pels()
        self._signal_words = self._default_signal_words()
        self._ppe_map = self._default_ppe_map()

    # ------------------------------------------------------------------
    # Built-in defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_ghs() -> dict[str, str]:
        """Return a built-in subset of GHS hazard statements."""
        return {
            "H200": "Explosive; mass explosion hazard",
            "H201": "Explosive; mass explosion hazard, no desensitizing agent",
            "H202": "Explosive; severe projection hazard",
            "H203": "Explosive; fire, blast or projection hazard",
            "H204": "Fire or projection hazard",
            "H205": "May mass explode in fire",
            "H220": "Extremely flammable gas",
            "H221": "Flammable gas",
            "H222": "Extremely flammable aerosol",
            "H223": "Flammable aerosol",
            "H224": "Extremely flammable liquid and vapour",
            "H225": "Highly flammable liquid and vapour",
            "H226": "Flammable liquid and vapour",
            "H228": "Flammable solid",
            "H229": "Pressurised container: may burst if heated",
            "H230": "May react explosively even in the absence of air",
            "H231": "May react explosively under elevated temperature/pressure",
            "H232": "Spontaneous ignition in air",
            "H240": "Heating may cause an explosion",
            "H241": "Heating may cause a fire or explosion",
            "H242": "Heating may cause a fire",
            "H250": "Catches fire spontaneously if exposed to air",
            "H251": "Self-heating; may catch fire",
            "H252": "Self-heating in large quantities; may catch fire",
            "H260": "In contact with water releases flammable gases which may ignite spontaneously",
            "H261": "In contact with water releases flammable gas",
            "H270": "May intensify fire; oxidiser",
            "H271": "May cause fire or explosion; strong oxidiser",
            "H272": "May intensify fire; oxidiser",
            "H280": "Contains gas under pressure; may explode if heated",
            "H281": "Contains refrigerated gas; may cause cryogenic burns or injury",
            "H290": "May be corrosive to metals",
            "H300": "Fatal if swallowed",
            "H301": "Toxic if swallowed",
            "H302": "Harmful if swallowed",
            "H303": "May be harmful if swallowed",
            "H304": "May be fatal if swallowed and enters airways",
            "H305": "May be harmful if swallowed and enters airways",
            "H310": "Fatal in contact with skin",
            "H311": "Toxic in contact with skin",
            "H312": "Harmful in contact with skin",
            "H313": "May be harmful in contact with skin",
            "H314": "Causes severe skin burns and eye damage",
            "H315": "Causes skin irritation",
            "H316": "Causes mild skin irritation",
            "H317": "May cause an allergic skin reaction",
            "H318": "Causes serious eye damage",
            "H319": "Causes serious eye irritation",
            "H320": "Causes eye irritation",
            "H330": "Fatal if inhaled",
            "H331": "Toxic if inhaled",
            "H332": "Harmful if inhaled",
            "H333": "May be harmful if inhaled",
        }

    @staticmethod
    def _default_osha_pels() -> dict[str, dict[str, Any]]:
        """Return OSHA PELs for commonly regulated chemicals."""
        return {
            "asbestos": {"pel": "0.1", "unit": "f/cc", "cas": "1332-21-4"},
            "benzene": {"pel": "1", "unit": "ppm", "cas": "71-43-2"},
            "cadmium": {"pel": "0.005", "unit": "mg/m3", "cas": "7440-43-9"},
            "carbon monoxide": {"pel": "50", "unit": "ppm", "cas": "630-08-0"},
            "chlorine": {"pel": "1", "unit": "ppm", "cas": "7782-50-5"},
            "formaldehyde": {"pel": "0.75", "unit": "ppm", "cas": "50-00-0"},
            "hydrogen sulfide": {"pel": "20", "unit": "ppm", "cas": "7783-06-4"},
            "lead": {"pel": "0.05", "unit": "mg/m3", "cas": "7439-92-1"},
            "mercury": {"pel": "0.05", "unit": "mg/m3", "cas": "7439-97-6"},
            "methanol": {"pel": "200", "unit": "ppm", "cas": "67-56-1"},
            "methylene chloride": {"pel": "25", "unit": "ppm", "cas": "75-09-2"},
            "nitric oxide": {"pel": "25", "unit": "ppm", "cas": "10102-43-9"},
            "ozone": {"pel": "0.1", "unit": "ppm", "cas": "10028-15-6"},
            "sulfur dioxide": {"pel": "5", "unit": "ppm", "cas": "7446-09-5"},
            "sulfuric acid": {"pel": "1", "unit": "mg/m3", "cas": "7664-93-9"},
            "toluene": {"pel": "200", "unit": "ppm", "cas": "108-88-3"},
            "xylene": {"pel": "100", "unit": "ppm", "cas": "1330-20-7"},
        }

    @staticmethod
    def _default_signal_words() -> dict[str, str]:
        """Return H-code to signal-word mapping."""
        danger_codes = {
            "H200", "H201", "H202", "H203", "H204", "H205",
            "H220", "H222", "H224", "H240", "H241", "H250",
            "H251", "H260", "H270", "H271",
            "H280", "H300", "H301", "H310", "H311", "H314",
            "H318", "H330", "H331",
        }
        return {code: "Danger" for code in danger_codes}

    @staticmethod
    def _default_ppe_map() -> dict[str, list[str]]:
        """Return required PPE items for each hazard category."""
        return {
            "H301": ["gloves", "goggles"],
            "H302": ["gloves", "goggles"],
            "H311": ["gloves", "goggles", "lab coat"],
            "H312": ["gloves", "goggles"],
            "H314": ["gloves", "goggles", "face shield", "lab coat"],
            "H318": ["gloves", "goggles", "face shield"],
            "H331": ["gloves", "goggles", "respirator"],
            "H332": ["gloves", "goggles", "respirator"],
            "H220": ["gloves", "goggles"],
            "H224": ["gloves", "goggles"],
            "H225": ["gloves", "goggles"],
            "H250": ["gloves", "goggles", "face shield"],
            "H260": ["gloves", "goggles", "respirator"],
            "default": ["gloves", "goggles"],
        }
