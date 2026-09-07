"""SQLAlchemy 2.0 ORM models for persistent storage."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class Document(Base):
    """Stores metadata and raw text for each uploaded document."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=False, default="application/octet-stream")
    file_size = Column(Integer, nullable=False, default=0)
    doc_type = Column(String(64), nullable=False, default="SDS")
    raw_text = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="PENDING")
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    extractions = relationship("Extraction", back_populates="document", cascade="all, delete-orphan")
    checks = relationship("ComplianceCheck", back_populates="document", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="document", cascade="all, delete-orphan")


class Extraction(Base):
    """Individual entity extracted from a document."""

    __tablename__ = "extractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    text = Column(String(1024), nullable=False)
    label = Column(String(64), nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    document = relationship("Document", back_populates="extractions")


class ComplianceCheck(Base):
    """Result of a single compliance rule check against a document."""

    __tablename__ = "compliance_checks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    rule_id = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)
    message = Column(Text, nullable=False)
    evidence = Column(Text, nullable=True, default="")
    severity = Column(String(32), nullable=False, default="LOW")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    document = relationship("Document", back_populates="checks")


class RiskAssessment(Base):
    """Aggregated risk score for a document."""

    __tablename__ = "risk_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    score = Column(Float, nullable=False)
    level = Column(String(32), nullable=False)
    details_json = Column(Text, nullable=True)
    assessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    document = relationship("Document", back_populates="risk_assessments")
