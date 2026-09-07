"""Defect detection — YOLOv8 inference with classical CV fallback.

When a trained YOLOv8 model is available the pipeline uses it for
high-accuracy detection.  Otherwise it falls back to Canny edge
detection, contour analysis, and simple blob detection that works
out-of-the-box on synthetic or generic imagery.
"""

import math
import uuid

import cv2
import numpy as np

from src.models.schemas import BoundingBox, DefectDetection, DefectType, SeverityLevel
from src.services.defect_classifier import DefectClassifier


class DefectDetector:
    """Detect surface defects in manufacturing images.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` YOLOv8 weights file.  ``None`` disables
        deep-learning detection and forces classical fallback.
    confidence_threshold:
        Minimum confidence to keep a detection.
    """

    def __init__(self, model_path: str | None, confidence_threshold: float = 0.5) -> None:
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._classifier = DefectClassifier()
        self._class_name_map: dict[str, DefectType] = {
            "scratch": DefectType.SCRATCH,
            "dent": DefectType.DENT,
            "crack": DefectType.CRACK,
            "stain": DefectType.STAIN,
            "discoloration": DefectType.DISCOLORATION,
            "deformation": DefectType.DEFORMATION,
        }

        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_defects(self, image: np.ndarray) -> list[DefectDetection]:
        """Run defect detection on an image.

        Parameters
        ----------
        image:
            BGR input image.

        Returns
        -------
        list[DefectDetection]
            Detected defects sorted by severity (most severe first).
        """
        if image.size == 0:
            return []

        if self._model is not None:
            detections = self._yolo_detection(image)
        else:
            detections = self._classical_detection(image)

        severity_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.MAJOR: 1,
            SeverityLevel.MINOR: 2,
            SeverityLevel.COSMETIC: 3,
        }
        detections.sort(key=lambda d: (severity_order.get(d.severity, 9), -d.confidence))
        return detections

    # ------------------------------------------------------------------
    # Classical computer-vision detection
    # ------------------------------------------------------------------

    def _classical_detection(self, image: np.ndarray) -> list[DefectDetection]:
        """Fallback detector using Canny edges, contours, and blobs."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = image.shape[0] * image.shape[1]
        detections: list[DefectDetection] = []

        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 50:
                continue

            measurements = self._classifier.compute_measurements(contour, pixels_per_mm=10.0)
            defect_type = self._classifier.classify_defect(contour, image_area)
            severity = self._classifier.classify_severity(defect_type, measurements)

            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(0.95, max(0.4, area_px / (image_area * 0.01)))

            detections.append(
                DefectDetection(
                    type=defect_type,
                    severity=severity,
                    confidence=round(confidence, 3),
                    bbox=BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                    area=round(area_px, 2),
                    description=f"Classical CV detection — {defect_type.value} ({severity.value})",
                )
            )

        # Simple blob detection for stain-like regions
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = 80
        detector_params.filterByCircularity = True
        detector_params.minCircularity = 0.4
        detector = cv2.SimpleBlobDetector_create(detector_params)
        keypoints = detector.detect(gray)

        for kp in keypoints:
            cx, cy = kp.pt
            r = kp.size / 2
            area_px = math.pi * r * r
            if area_px < 50:
                continue
            x = max(0, int(cx - r))
            y = max(0, int(cy - r))
            w = int(2 * r)
            h = int(2 * r)
            contour_approx = self._circular_contour(cx, cy, r)
            measurements = self._classifier.compute_measurements(contour_approx, pixels_per_mm=10.0)
            severity = self._classifier.classify_severity(DefectType.STAIN, measurements)
            confidence = min(0.9, max(0.35, area_px / (image_area * 0.005)))

            detections.append(
                DefectDetection(
                    type=DefectType.STAIN,
                    severity=severity,
                    confidence=round(confidence, 3),
                    bbox=BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                    area=round(area_px, 2),
                    description="Blob detection — stain candidate",
                )
            )

        return detections

    # ------------------------------------------------------------------
    # YOLOv8 detection
    # ------------------------------------------------------------------

    def _yolo_detection(self, image: np.ndarray) -> list[DefectDetection]:
        """Run YOLOv8 model inference and map to ``DefectDetection``."""
        assert self._model is not None
        results = self._model(image, verbose=False)
        detections: list[DefectDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                raw_name = self._model.names.get(cls_id, "unknown")
                defect_type = self._class_name_map.get(raw_name.lower(), DefectType.DEFORMATION)
                w, h = x2 - x1, y2 - y1
                area = w * h
                measurements = {
                    "width_mm": w / 10.0,
                    "height_mm": h / 10.0,
                    "area_mm2": area / 100.0,
                }
                severity = self._classifier.classify_severity(defect_type, measurements)
                detections.append(
                    DefectDetection(
                        type=defect_type,
                        severity=severity,
                        confidence=round(conf, 3),
                        bbox=BoundingBox(x=float(x1), y=float(y1), w=float(w), h=float(h)),
                        area=round(area, 2),
                        description=f"YOLOv8 detection — {raw_name} ({severity.value})",
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, path: str) -> None:
        """Lazily import and load ultralytics YOLO."""
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(path)
        except Exception:
            self._model = None

    @staticmethod
    def _circular_contour(cx: float, cy: float, r: float) -> np.ndarray:
        """Build an approximate circular contour for measurement."""
        pts: list[list[int]] = []
        for angle_deg in range(0, 360, 10):
            rad = math.radians(angle_deg)
            px = int(cx + r * math.cos(rad))
            py = int(cy + r * math.sin(rad))
            pts.append([px, py])
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
