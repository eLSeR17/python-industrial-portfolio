"""Pydantic v2 schemas and enums for the Document Intelligence API."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    """Supported industrial document categories."""

    SDS = "SDS"
    MANUAL = "MANUAL"
    CERTIFICATE = "CERTIFICATE"
    AUDIT_REPORT = "AUDIT_REPORT"
    REGULATORY_FILING = "REGULATORY_FILING"


class ComplianceStatus(str, Enum):
    """Outcome of a single compliance rule check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    """Risk classification derived from the aggregated score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ProcessingStatus(str, Enum):
    """Lifecycle status of an uploaded document."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Entity / Finding models
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """A named entity extracted from the document text."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = Field(ge=0.0, le=1.0)


class ComplianceFinding(BaseModel):
    """A single compliance check result."""

    model_config = ConfigDict(from_attributes=True)

    rule_id: str
    status: ComplianceStatus
    message: str
    evidence: str = ""
    severity: RiskLevel = RiskLevel.LOW
    entity_refs: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    """Aggregated compliance analysis for a document."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    findings: list[ComplianceFinding] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    entities_count: int = 0
    findings_count: int = 0
    compliant_findings: int = 0
    non_compliant_findings: int = 0


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata about an uploaded document."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: DocumentType = DocumentType.SDS
    page_count: int = 0
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    status: ProcessingStatus = ProcessingStatus.PENDING


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    """Request parameters for report generation."""

    document_id: str
    format: str = "html"
    include_recommendations: bool = True


class BatchUploadRequest(BaseModel):
    """Payload for batch document upload."""

    files: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response returned after a document is uploaded."""

    document_id: str
    task_id: Optional[str] = None
    status: ProcessingStatus
    message: str
