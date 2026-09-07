"""Celery worker configuration for async document processing.

Defines the Celery app instance and the document-processing task
that orchestrates parsing, entity extraction, compliance checking,
risk scoring, and report storage.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any

from celery import Celery

from config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "doc_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="process_document")
def process_document(
    self,
    document_id: str,
    file_bytes_b64: str,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    """Run the full document processing pipeline asynchronously.

    Steps: parse → extract entities → check compliance → score risk.

    Args:
        self: Celery task instance (for status updates).
        document_id: UUID string of the uploaded document.
        file_bytes_b64: Base64-encoded file content.
        filename: Original filename.
        content_type: MIME type of the upload.

    Returns:
        Summary dict with status, timing, and result counts.
    """
    import base64

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    from src.models.db import Base, ComplianceCheck, Document, Extraction, RiskAssessment
    from src.services.compliance_checker import ComplianceChecker
    from src.services.document_parser import DocumentParser
    from src.services.entity_extractor import EntityExtractor
    from src.services.risk_scorer import RiskScorer
    from src.utils.regulation_db import RegulationDB

    start_time = time.time()

    engine = create_engine(settings.database_url.replace("+asyncpg", "+psycopg2"))
    session_factory = sessionmaker(bind=engine)
    session: Session = session_factory()

    try:
        doc = session.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            return {"status": "error", "message": f"Document {document_id} not found"}

        doc.status = "PROCESSING"
        session.commit()

        file_bytes = base64.b64decode(file_bytes_b64)

        parser = DocumentParser()
        full_text, metadata = parser.parse(file_bytes, filename)

        doc.raw_text = full_text
        doc.doc_type = metadata.get("doc_type", "SDS")
        session.commit()

        extractor = EntityExtractor(model_name=settings.spacy_model)
        entities = extractor.extract_entities(full_text)

        for ent in entities:
            session.add(Extraction(
                document_id=doc.id,
                text=ent.text,
                label=ent.label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=ent.confidence,
            ))

        regulation_db = RegulationDB(data_path=None)
        checker = ComplianceChecker(regulation_db)
        from src.models.schemas import DocumentType
        try:
            dtype = DocumentType(doc.doc_type)
        except ValueError:
            dtype = DocumentType.SDS
        findings = checker.check_document(entities, full_text, dtype)

        for finding in findings:
            session.add(ComplianceCheck(
                document_id=doc.id,
                rule_id=finding.rule_id,
                status=finding.status.value,
                message=finding.message,
                evidence=finding.evidence,
                severity=finding.severity.value,
            ))

        scorer = RiskScorer()
        risk_score, risk_level = scorer.score(findings, entities)

        session.add(RiskAssessment(
            document_id=doc.id,
            score=risk_score,
            level=risk_level.value,
            details_json=json.dumps({
                "entities_count": len(entities),
                "findings_count": len(findings),
            }),
        ))

        doc.status = "COMPLETED"
        doc.processed_at = datetime.utcnow()
        session.commit()

        elapsed = (time.time() - start_time) * 1000
        return {
            "status": "completed",
            "document_id": document_id,
            "entities_count": len(entities),
            "findings_count": len(findings),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level.value,
            "processing_time_ms": round(elapsed, 2),
        }

    except Exception as exc:
        session.rollback()
        if doc is not None:
            doc.status = "FAILED"
            session.commit()
        return {"status": "error", "message": str(exc)}
    finally:
        session.close()
        engine.dispose()
