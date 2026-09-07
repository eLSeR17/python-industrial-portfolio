"""Rule-based defect classification and measurement computation.

The classifier assigns a ``SeverityLevel`` to every detection using
threshold tables derived from the ``Settings.defect_severity_thresholds``
configuration and computes metric measurements in millimetres.
"""

import math

import cv2
import numpy as np

from src.models.schemas import DefectType, SeverityLevel


class DefectClassifier:
    """Classify defect type from contour geometry and assign severity.

    Severity rules encode the manufacturing quality standard:

    * **Scratch** — width-based thresholds (cosmetic < 0.1 mm,
      critical > 0.5 mm).
    * **Dent** — area relative to part surface (cosmetic < 0.5 %,
      critical > 2 %).
    * **Crack** — length + branching heuristic.
    * **Stain** — area percentage of surface.
    """

    def classify_severity(
        self, defect_type: DefectType, measurements: dict[str, float]
    ) -> SeverityLevel:
        """Determine severity based on defect type and measurements.

        Parameters
        ----------
        defect_type:
            The category of defect.
        measurements:
            Metric dict with keys like ``width_mm``, ``area_mm2``,
            ``length_mm``, ``area_pct``, ``branch_count``.

        Returns
        -------
        SeverityLevel
            Classified severity.
        """
        if defect_type == DefectType.SCRATCH:
            return self._scratch_severity(measurements)
        if defect_type == DefectType.DENT:
            return self._dent_severity(measurements)
        if defect_type == DefectType.CRACK:
            return self._crack_severity(measurements)
        if defect_type == DefectType.STAIN:
            return self._stain_severity(measurements)
        return SeverityLevel.MINOR

    def classify_defect(self, contour: np.ndarray, image_area: int) -> DefectType:
        """Classify defect type from contour shape analysis.

        Parameters
        ----------
        contour:
            OpenCV contour array.
        image_area:
            Total image area in pixels².

        Returns
        -------
        DefectType
            Best-matching defect category.
        """
        if contour is None or cv2.contourArea(contour) == 0:
            return DefectType.DEFORMATION

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return DefectType.DEFORMATION

        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0.0
        extent = area / (w * h) if (w * h) > 0 else 0.0

        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        if circularity > 0.6 and extent > 0.5:
            if area < image_area * 0.005:
                return DefectType.STAIN
            return DefectType.DENT

        if aspect_ratio > 4.0 or (aspect_ratio < 0.25 and w > 0):
            return DefectType.CRACK

        if solidity < 0.7 or extent < 0.3:
            return DefectType.SCRATCH

        if area > image_area * 0.01:
            return DefectType.DEFORMATION

        return DefectType.STAIN

    def compute_measurements(
        self, contour: np.ndarray, pixels_per_mm: float
    ) -> dict[str, float]:
        """Compute physical measurements from a contour.

        Parameters
        ----------
        contour:
            OpenCV contour.
        pixels_per_mm:
            Conversion factor from pixels to millimetres.

        Returns
        -------
        dict
            Keys: ``width_mm``, ``height_mm``, ``area_mm2``,
            ``length_mm``, ``perimeter_mm``, ``area_pct`` (of 640x480).
        """
        if contour is None or cv2.contourArea(contour) == 0:
            return {
                "width_mm": 0.0,
                "height_mm": 0.0,
                "area_mm2": 0.0,
                "length_mm": 0.0,
                "perimeter_mm": 0.0,
                "area_pct": 0.0,
            }

        x, y, w, h = cv2.boundingRect(contour)
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)

        length_px = max(w, h)

        ref_area = 640.0 * 480.0
        return {
            "width_mm": round(w / pixels_per_mm, 4),
            "height_mm": round(h / pixels_per_mm, 4),
            "area_mm2": round(area_px / (pixels_per_mm ** 2), 4),
            "length_mm": round(length_px / pixels_per_mm, 4),
            "perimeter_mm": round(perimeter_px / pixels_per_mm, 4),
            "area_pct": round(area_px / ref_area * 100, 4),
        }

    # ------------------------------------------------------------------
    # Private severity helpers
    # ------------------------------------------------------------------

    def _scratch_severity(self, m: dict[str, float]) -> SeverityLevel:
        w = m.get("width_mm", 0.0)
        if w < 0.1:
            return SeverityLevel.COSMETIC
        if w <= 0.5:
            return SeverityLevel.MINOR
        return SeverityLevel.CRITICAL

    def _dent_severity(self, m: dict[str, float]) -> SeverityLevel:
        pct = m.get("area_pct", 0.0)
        if pct < 0.5:
            return SeverityLevel.COSMETIC
        if pct < 2.0:
            return SeverityLevel.MINOR
        if pct < 5.0:
            return SeverityLevel.MAJOR
        return SeverityLevel.CRITICAL

    def _crack_severity(self, m: dict[str, float]) -> SeverityLevel:
        length = m.get("length_mm", 0.0)
        branches = int(m.get("branch_count", 0))
        if length < 1.0 and branches == 0:
            return SeverityLevel.MINOR
        if length < 5.0 and branches <= 1:
            return SeverityLevel.MAJOR
        return SeverityLevel.CRITICAL

    def _stain_severity(self, m: dict[str, float]) -> SeverityLevel:
        pct = m.get("area_pct", 0.0)
        if pct < 0.3:
            return SeverityLevel.COSMETIC
        if pct < 1.0:
            return SeverityLevel.MINOR
        if pct < 3.0:
            return SeverityLevel.MAJOR
        return SeverityLevel.CRITICAL
