"""FastAPI route definitions for the Document Intelligence API.

Provides endpoints for document upload, status checking, compliance
retrieval, report generation, batch processing, and regulation queries.
"""

import base64
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from src.models.db import Base, ComplianceCheck, Document, Extraction, RiskAssessment
from src.models.schemas import (
    ComplianceResult,
    ComplianceStatus,
    DocumentType,
    ProcessingStatus,
    ReportRequest,
    RiskLevel,
    UploadResponse,
)
from src.services.compliance_checker import ComplianceChecker
from src.services.document_parser import DocumentParser
from src.services.entity_extractor import EntityExtractor
from src.services.report_builder import ReportBuilder
from src.services.risk_scorer import RiskScorer
from src.utils.regulation_db import RegulationDB

router = APIRouter(prefix="/api/v1", tags=["Document Intelligence"])

settings = get_settings()

engine = create_engine(settings.database_url.replace("+asyncpg", "+psycopg2"))
session_factory = sessionmaker(bind=engine)


def _get_session() -> Session:
    """Create a new synchronous database session."""
    return session_factory()


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a document for compliance analysis.

    Accepts PDF, DOCX, or TXT files up to the configured size limit.
    Returns a document ID and task ID for tracking.
    """
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_size_mb} MB",
        )

    doc_id = str(uuid.uuid4())
    session: Session = _get_session()
    try:
        doc = Document(
            id=uuid.UUID(doc_id),
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            file_size=len(file_bytes),
            doc_type="SDS",
            status="PENDING",
            uploaded_at=datetime.utcnow(),
        )
        session.add(doc)
        session.commit()

        from src.workers.celery_app import process_document
        encoded = base64.b64encode(file_bytes).decode("ascii")
        task = process_document.delay(doc_id, encoded, file.filename or "unknown", file.content_type or "")

        return UploadResponse(
            document_id=doc_id,
            task_id=task.id,
            status=ProcessingStatus.PENDING,
            message="Document uploaded and queued for processing.",
        )
    finally:
        session.close()


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@router.get("/status/{task_id}")
async def get_status(task_id: str) -> dict[str, Any]:
    """Check the processing status of an async task.

    Returns the Celery task state and result (if completed).
    """
    from src.workers.celery_app import celery_app as app
    result = app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
    }


# ------------------------------------------------------------------
# Compliance results
# ------------------------------------------------------------------

@router.get("/compliance/{doc_id}", response_model=ComplianceResult)
async def get_compliance(doc_id: str) -> ComplianceResult:
    """Retrieve compliance analysis results for a document.

    Raises 404 if the document has not been processed yet.
    """
    session: Session = _get_session()
    try:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        checks = session.query(ComplianceCheck).filter(ComplianceCheck.document_id == doc.id).all()
        extractions = session.query(Extraction).filter(Extraction.document_id == doc.id).all()
        risk = session.query(RiskAssessment).filter(RiskAssessment.document_id == doc.id).first()

        compliant = sum(1 for c in checks if c.status == "PASS")
        non_compliant = sum(1 for c in checks if c.status == "FAIL")

        return ComplianceResult(
            document_id=doc_id,
            findings=[
                {
                    "rule_id": c.rule_id,
                    "status": c.status,
                    "message": c.message,
                    "evidence": c.evidence or "",
                    "severity": c.severity,
                    "entity_refs": [],
                }
                for c in checks
            ],
            risk_score=risk.score if risk else 0.0,
            risk_level=RiskLevel(risk.level) if risk else RiskLevel.LOW,
            entities_count=len(extractions),
            findings_count=len(checks),
            compliant_findings=compliant,
            non_compliant_findings=non_compliant,
        )
    finally:
        session.close()


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

@router.get("/report/{doc_id}")
async def get_report(
    doc_id: str,
    format: str = Query("html", regex="^(html|json)$"),
) -> Any:
    """Generate and return a compliance report.

    Args:
        doc_id: Document UUID.
        format: ``html`` or ``json``.

    Returns:
        HTML string or JSON dict depending on format.
    """
    session: Session = _get_session()
    try:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        checks = session.query(ComplianceCheck).filter(ComplianceCheck.document_id == doc.id).all()
        extractions = session.query(Extraction).filter(Extraction.document_id == doc.id).all()
        risk = session.query(RiskAssessment).filter(RiskAssessment.document_id == doc.id).first()

        from src.models.schemas import ComplianceFinding, ExtractedEntity as EntModel
        findings = [
            ComplianceFinding(
                rule_id=c.rule_id,
                status=ComplianceStatus(c.status),
                message=c.message,
                evidence=c.evidence or "",
                severity=RiskLevel(c.severity),
                entity_refs=[],
            )
            for c in checks
        ]
        entities = [
            EntModel(
                text=e.text,
                label=e.label,
                start_char=e.start_char,
                end_char=e.end_char,
                confidence=e.confidence,
            )
            for e in extractions
        ]

        metadata = {
            "filename": doc.filename,
            "content_type": doc.content_type,
            "file_size": doc.file_size,
            "doc_type": doc.doc_type,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else "",
            "processed_at": doc.processed_at.isoformat() if doc.processed_at else "",
        }

        builder = ReportBuilder()
        if format == "json":
            return builder.build_json_report(
                doc_id, metadata, entities, findings,
                risk.score if risk else 0.0,
                RiskLevel(risk.level) if risk else RiskLevel.LOW,
            )
        return builder.build_html_report(
            doc_id, metadata, entities, findings,
            risk.score if risk else 0.0,
            RiskLevel(risk.level) if risk else RiskLevel.LOW,
        )
    finally:
        session.close()


# ------------------------------------------------------------------
# Batch check
# ------------------------------------------------------------------

@router.post("/batch-check", status_code=202)
async def batch_check(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload multiple documents for parallel compliance analysis.

    Returns a list of document IDs and task IDs.
    """
    results: list[dict[str, str]] = []
    for file in files:
        file_bytes = await file.read()
        doc_id = str(uuid.uuid4())

        session: Session = _get_session()
        try:
            doc = Document(
                id=uuid.UUID(doc_id),
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                file_size=len(file_bytes),
                doc_type="SDS",
                status="PENDING",
                uploaded_at=datetime.utcnow(),
            )
            session.add(doc)
            session.commit()

            from src.workers.celery_app import process_document
            encoded = base64.b64encode(file_bytes).decode("ascii")
            task = process_document.delay(doc_id, encoded, file.filename or "unknown", file.content_type or "")

            results.append({"document_id": doc_id, "task_id": task.id})
        finally:
            session.close()

    return {"documents": results, "count": len(results)}


# ------------------------------------------------------------------
# Regulation queries
# ------------------------------------------------------------------

@router.get("/regulations/ghs")
async def get_ghs_regulations() -> dict[str, str]:
    """Return all GHS hazard statements from the reference database."""
    regdb = RegulationDB(data_path=None)
    return regdb.get_ghs_hazards()


@router.get("/regulations/osha-pels")
async def get_osha_pels() -> dict[str, dict[str, Any]]:
    """Return OSHA Permissible Exposure Limits for common chemicals."""
    regdb = RegulationDB(data_path=None)
    return regdb.get_osha_pels()


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"status": "healthy", "service": "doc-intelligence"}
