"""FastAPI router definitions for the Vision Inspector API.

Endpoints expose the full inspection pipeline (preprocess, detect,
classify, annotate, record) plus statistical queries and health checks.
"""

import time

from fastapi import APIRouter, HTTPException

from src.models.schemas import (
    DefectType,
    InspectionRequest,
    InspectionResult,
    SeverityLevel,
)
from src.services.annotation_service import AnnotationService
from src.services.defect_detector import DefectDetector
from src.services.image_processor import ImageProcessor
from src.services.statistics_service import StatisticsService
from src.utils.image_utils import decode_image_base64, encode_image_base64

router = APIRouter(prefix="/api/v1")

_processor = ImageProcessor()
_detector = DefectDetector(model_path=None, confidence_threshold=0.5)
_annotator = AnnotationService()
_statistics = StatisticsService(alert_threshold_pct=5.0)

_DEFECT_TYPE_DESCRIPTIONS: dict[str, str] = {
    DefectType.SCRATCH.value: "Surface scratch caused by handling or tooling contact",
    DefectType.DENT.value: "Indentation from impact or compression",
    DefectType.CRACK.value: "Fracture or fissure in the material",
    DefectType.STAIN.value: "Chemical residue, oil, or discolouration spot",
    DefectType.DISCOLORATION.value: "Uneven colour indicating thermal or chemical damage",
    DefectType.DEFORMATION.value: "Shape deviation beyond tolerance limits",
}


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(request: InspectionRequest) -> InspectionResult:
    """Run the full inspection pipeline on a single image.

    The pipeline executes in order:

    1. Decode and validate the Base64 image.
    2. Preprocess (resize + normalise).
    3. Detect defects (YOLOv8 or classical fallback).
    4. Classify severity.
    5. Annotate the image.
    6. Record the result for statistics.
    """
    start = time.perf_counter()

    try:
        image_bytes = __import__("base64").b64decode(request.image)
        image = _processor.validate_image(image_bytes)
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    preprocessed = _processor.preprocess(image, target_size=(640, 480))
    enhanced = _processor.enhance((preprocessed * 255).astype("uint8"))

    detections = _detector.detect_defects(enhanced)

    passed = not any(d.severity in (SeverityLevel.MAJOR, SeverityLevel.CRITICAL) for d in detections)
    total_area = sum(d.area for d in detections)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = InspectionResult(
        line_id=request.line_id,
        defects=detections,
        passed=passed,
        total_defect_area=round(total_area, 2),
        defect_count=len(detections),
        processing_time_ms=round(elapsed_ms, 2),
    )

    _statistics.record_inspection(result)

    annotated = _annotator.create_annotated_copy(image, detections, result)
    _annotated_b64 = encode_image_base64(annotated)

    return result


@router.get("/statistics/{line_id}", response_model=dict)
async def get_statistics(line_id: str, period_minutes: int = 60) -> dict:
    """Return aggregated quality statistics for a production line."""
    stats = _statistics.get_statistics(line_id, period_minutes)
    return stats.model_dump()


@router.get("/statistics/{line_id}/pareto", response_model=list)
async def get_pareto(line_id: str) -> list[dict]:
    """Return Pareto analysis (80/20 rule) of defect types."""
    analysis = _statistics.get_pareto_analysis(line_id)
    return [item.model_dump() for item in analysis]


@router.get("/defect-types", response_model=list[dict])
async def list_defect_types() -> list[dict]:
    """List all supported defect types with descriptions."""
    return [
        {"type": dt.value, "description": desc}
        for dt, desc in zip(DefectType, _DEFECT_TYPE_DESCRIPTIONS.values(), strict=True)
    ]


@router.get("/alerts/{line_id}", response_model=list)
async def get_alerts(line_id: str) -> list[dict]:
    """Return active quality alerts for a production line."""
    alerts = _statistics.check_alerts(line_id)
    return [a.model_dump() for a in alerts]


@router.get("/health")
async def health_check() -> dict:
    """Simple health-check endpoint."""
    return {"status": "healthy", "service": "vision-inspector"}
