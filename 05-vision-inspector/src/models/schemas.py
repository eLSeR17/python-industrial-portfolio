"""Pydantic v2 models for the Vision Inspector API.

Defines request/response schemas, enums for defect taxonomy, and data
classes used throughout the inspection pipeline.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class DefectType(str, Enum):
    """Known defect categories detected by the inspection pipeline."""

    SCRATCH = "SCRATCH"
    DENT = "DENT"
    CRACK = "CRACK"
    STAIN = "STAIN"
    DISCOLORATION = "DISCOLORATION"
    DEFORMATION = "DEFORMATION"


class SeverityLevel(str, Enum):
    """Severity classification for a detected defect."""

    COSMETIC = "COSMETIC"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x: float
    y: float
    w: float
    h: float


class DefectDetection(BaseModel):
    """A single defect found during inspection."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: DefectType
    severity: SeverityLevel
    confidence: float
    bbox: BoundingBox
    area: float
    description: str = ""


class InspectionRequest(BaseModel):
    """Incoming inspection request carrying an image and metadata."""

    image: str = Field(description="Base64-encoded image bytes")
    line_id: str
    product_type: str = "default"
    metadata: dict = Field(default_factory=dict)


class InspectionResult(BaseModel):
    """Full result of an inspection run."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    image_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    line_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    defects: list[DefectDetection] = Field(default_factory=list)
    passed: bool = True
    total_defect_area: float = 0.0
    defect_count: int = 0
    processing_time_ms: float = 0.0


class DefectTypeInfo(BaseModel):
    """Summary for a single defect type in statistics."""

    type: DefectType
    count: int
    percentage: float
    avg_severity: str


class StatisticsResponse(BaseModel):
    """Aggregated quality statistics for a production line."""

    line_id: str
    period_minutes: int
    total_inspected: int
    pass_rate: float
    defect_rate: float
    defect_type_counts: dict[str, int]
    top_defects: list[DefectTypeInfo]


class AlertInfo(BaseModel):
    """Quality alert triggered when thresholds are breached."""

    line_id: str
    alert_type: str
    message: str
    defect_rate: float
    threshold: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
