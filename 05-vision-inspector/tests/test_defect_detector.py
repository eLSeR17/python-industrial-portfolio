"""Tests for the DefectDetector and DefectClassifier.

All tests use synthetic images drawn with OpenCV — no trained model
required.
"""

import numpy as np
import pytest

from src.models.schemas import DefectType, SeverityLevel
from src.services.defect_classifier import DefectClassifier
from src.services.defect_detector import DefectDetector
from src.utils.image_utils import create_synthetic_defect_image


@pytest.fixture()
def detector() -> DefectDetector:
    return DefectDetector(model_path=None, confidence_threshold=0.3)


@pytest.fixture()
def classifier() -> DefectClassifier:
    return DefectClassifier()


class TestClassicalDetection:
    def test_finds_dent(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "dent", (320, 240))
        detections = detector.detect_defects(img)
        assert len(detections) >= 1
        assert any(d.type == DefectType.DENT for d in detections)

    def test_finds_scratch(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "scratch", (320, 240))
        detections = detector.detect_defects(img)
        assert len(detections) >= 1
        types = {d.type for d in detections}
        assert DefectType.SCRATCH in types or DefectType.CRACK in types

    def test_finds_crack(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "crack", (320, 200))
        detections = detector.detect_defects(img)
        assert len(detections) >= 1
        types = {d.type for d in detections}
        assert DefectType.CRACK in types or DefectType.SCRATCH in types

    def test_finds_stain(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "stain", (320, 240))
        detections = detector.detect_defects(img)
        assert len(detections) >= 1

    def test_multiple_defects(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "dent", (200, 200))
        img2 = create_synthetic_defect_image(640, 480, "scratch", (450, 300))
        combined = cv2.add(img, img2)
        detections = detector.detect_defects(combined)
        assert len(detections) >= 1

    def test_empty_image(self, detector: DefectDetector) -> None:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        detections = detector.detect_defects(empty)
        assert detections == []

    def test_clean_image(self, detector: DefectDetector) -> None:
        clean = np.full((480, 640, 3), 180, dtype=np.uint8)
        detections = detector.detect_defects(clean)
        assert len(detections) == 0

    def test_sorted_by_severity(self, detector: DefectDetector) -> None:
        img = create_synthetic_defect_image(640, 480, "crack", (320, 200))
        detections = detector.detect_defects(img)
        if len(detections) > 1:
            order = {
                SeverityLevel.CRITICAL: 0,
                SeverityLevel.MAJOR: 1,
                SeverityLevel.MINOR: 2,
                SeverityLevel.COSMETIC: 3,
            }
            for i in range(len(detections) - 1):
                assert order[detections[i].severity] <= order[detections[i + 1].severity]


# Need cv2 for combining images
import cv2  # noqa: E402


class TestClassifyDefect:
    def test_circular_is_dent(self, classifier: DefectClassifier) -> None:
        pts: list[list[int]] = []
        for angle in range(0, 360, 5):
            rad = np.radians(angle)
            pts.append([int(100 + 40 * np.cos(rad)), int(100 + 40 * np.sin(rad))])
        contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        result = classifier.classify_defect(contour, 640 * 480)
        assert result in (DefectType.DENT, DefectType.STAIN)

    def test_long_thin_is_crack_or_scratch(self, classifier: DefectClassifier) -> None:
        pts = np.array([[100, 100], [120, 200], [110, 300], [115, 400]], dtype=np.int32)
        contour = pts.reshape(-1, 1, 2)
        result = classifier.classify_defect(contour, 640 * 480)
        assert result in (DefectType.CRACK, DefectType.SCRATCH)

    def test_empty_contour(self, classifier: DefectClassifier) -> None:
        empty = np.array([], dtype=np.int32).reshape(0, 1, 2)
        result = classifier.classify_defect(empty, 640 * 480)
        assert result == DefectType.DEFORMATION


class TestComputeMeasurements:
    def test_basic_measurements(self, classifier: DefectClassifier) -> None:
        contour = np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.int32)
        contour = contour.reshape(-1, 1, 2)
        m = classifier.compute_measurements(contour, pixels_per_mm=10.0)
        assert m["width_mm"] > 0
        assert m["height_mm"] > 0
        assert m["area_mm2"] > 0
        assert m["perimeter_mm"] > 0
        assert m["area_pct"] > 0

    def test_zero_contour(self, classifier: DefectClassifier) -> None:
        empty = np.array([], dtype=np.int32).reshape(0, 1, 2)
        m = classifier.compute_measurements(empty, pixels_per_mm=10.0)
        assert m["width_mm"] == 0.0
        assert m["area_mm2"] == 0.0


class TestClassifySeverity:
    def test_scratch_cosmetic(self, classifier: DefectClassifier) -> None:
        assert classifier.classify_severity(DefectType.SCRATCH, {"width_mm": 0.05}) == SeverityLevel.COSMETIC

    def test_scratch_critical(self, classifier: DefectClassifier) -> None:
        assert classifier.classify_severity(DefectType.SCRATCH, {"width_mm": 0.8}) == SeverityLevel.CRITICAL

    def test_dent_minor(self, classifier: DefectClassifier) -> None:
        assert classifier.classify_severity(DefectType.DENT, {"area_pct": 1.0}) == SeverityLevel.MINOR

    def test_crack_critical(self, classifier: DefectClassifier) -> None:
        assert classifier.classify_severity(
            DefectType.CRACK, {"length_mm": 6.0, "branch_count": 2}
        ) == SeverityLevel.CRITICAL

    def test_stain_cosmetic(self, classifier: DefectClassifier) -> None:
        assert classifier.classify_severity(DefectType.STAIN, {"area_pct": 0.1}) == SeverityLevel.COSMETIC
