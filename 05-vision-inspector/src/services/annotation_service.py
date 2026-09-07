"""Visual annotation of inspection results on images.

Draws bounding boxes coloured by severity, text labels, and a
statistics overlay on top of the original image.
"""

import cv2
import numpy as np

from src.models.schemas import (
    DefectDetection,
    InspectionResult,
    SeverityLevel,
)

_SEVERITY_COLOURS: dict[SeverityLevel, tuple[int, int, int]] = {
    SeverityLevel.COSMETIC: (144, 238, 144),   # light green  (BGR)
    SeverityLevel.MINOR: (0, 200, 255),         # yellow
    SeverityLevel.MAJOR: (0, 128, 255),         # orange
    SeverityLevel.CRITICAL: (0, 0, 255),        # red
}


class AnnotationService:
    """Draw detection boxes, labels, and statistics on images."""

    def draw_detections(
        self, image: np.ndarray, detections: list[DefectDetection]
    ) -> np.ndarray:
        """Draw coloured bounding boxes and labels for every detection.

        Parameters
        ----------
        image:
            Input BGR image (will be copied).
        detections:
            Defects to visualise.

        Returns
        -------
        np.ndarray
            Annotated image (BGR).
        """
        if image.size == 0:
            return image.copy()

        canvas = image.copy()

        for det in detections:
            colour = _SEVERITY_COLOURS.get(det.severity, (200, 200, 200))
            bx, by = int(det.bbox.x), int(det.bbox.y)
            bw, bh = int(det.bbox.w), int(det.bbox.h)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), colour, 2)

            label = f"{det.type.value} {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(canvas, (bx, by - th - 8), (bx + tw + 4, by), colour, -1)
            cv2.putText(
                canvas,
                label,
                (bx + 2, by - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return canvas

    def draw_statistics_overlay(
        self, image: np.ndarray, stats: dict
    ) -> np.ndarray:
        """Render a semi-transparent statistics panel.

        Parameters
        ----------
        image:
            Input BGR image.
        stats:
            Keys: ``defect_count`` (int), ``passed`` (bool),
            ``processing_time_ms`` (float).

        Returns
        -------
        np.ndarray
            Image with overlay.
        """
        if image.size == 0:
            return image.copy()

        canvas = image.copy()
        h, w = canvas.shape[:2]

        overlay = canvas.copy()
        panel_h = 90
        cv2.rectangle(overlay, (10, h - panel_h - 10), (280, h - 10), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y0 = h - panel_h
        defect_count = stats.get("defect_count", 0)
        passed = stats.get("passed", True)
        proc_ms = stats.get("processing_time_ms", 0.0)

        cv2.putText(canvas, f"Defects: {defect_count}", (20, y0 + 20), font, 0.5, (255, 255, 255), 1)
        status = "PASS" if passed else "FAIL"
        status_color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.putText(canvas, f"Status: {status}", (20, y0 + 45), font, 0.5, status_color, 1)
        cv2.putText(
            canvas,
            f"Time: {proc_ms:.1f} ms",
            (20, y0 + 70),
            font,
            0.5,
            (200, 200, 200),
            1,
        )

        return canvas

    def create_annotated_copy(
        self,
        image: np.ndarray,
        detections: list[DefectDetection],
        result: InspectionResult,
    ) -> np.ndarray:
        """Produce a fully annotated image (detections + stats panel).

        Parameters
        ----------
        image:
            Original BGR image.
        detections:
            Defect detections.
        result:
            Inspection result for statistics overlay.

        Returns
        -------
        np.ndarray
            Fully annotated image.
        """
        annotated = self.draw_detections(image, detections)
        stats = {
            "defect_count": result.defect_count,
            "passed": result.passed,
            "processing_time_ms": result.processing_time_ms,
        }
        return self.draw_statistics_overlay(annotated, stats)
