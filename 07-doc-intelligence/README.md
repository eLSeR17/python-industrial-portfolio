# Document Intelligence for Compliance

Industrial safety documents — SDS sheets, manuals, audit reports — are
long, tedious to review, and the consequences of missing a non-compliance
are severe. This project extracts entities from those documents using
spaCy NER, cross-references them against GHS/OSHA/SDS regulatory data,
and produces a risk-scored compliance report.

The pipeline runs asynchronously via Celery so uploads return quickly and
processing happens in the background. Reports come out as HTML (for
human review) or JSON (for downstream systems).

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI (port 8000)                       │
│  POST /upload  GET /compliance/{id}  GET /report/{id}  /health  │
└──────────────┬──────────────────────────────────────┬────────────┘
               │                                      │
               ▼                                      ▼
    ┌──────────────────┐                ┌──────────────────────────┐
    │   PostgreSQL      │                │   Redis (Celery broker)  │
    │   documents,      │                │   task queue              │
    │   extractions,    │                │                           │
    │   checks, risks   │                └────────────┬─────────────┘
    └──────────────────┘                             │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Celery Worker        │
                                          │  ┌────────────────┐  │
                                          │  │ DocumentParser  │  │
                                          │  │ (PyMuPDF/DOCX)  │  │
                                          │  └───────┬────────┘  │
                                          │          ▼           │
                                          │  ┌────────────────┐  │
                                          │  │EntityExtractor  │  │
                                          │  │ (spaCy + ruler) │  │
                                          │  └───────┬────────┘  │
                                          │          ▼           │
                                          │  ┌────────────────┐  │
                                          │  │ComplianceChecker│  │
                                          │  │ (GHS/OSHA/SDS)  │  │
                                          │  └───────┬────────┘  │
                                          │          ▼           │
                                          │  ┌────────────────┐  │
                                          │  │  RiskScorer     │  │
                                          │  │  (0–100 score)  │  │
                                          │  └───────┬────────┘  │
                                          │          ▼           │
                                          │  ┌────────────────┐  │
                                          │  │ ReportBuilder   │  │
                                          │  │ (HTML / JSON)   │  │
                                          │  └────────────────┘  │
                                          └──────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| NLP engine | spaCy (en_core_web_sm) |
| PDF parsing | PyMuPDF (fitz) |
| DOCX parsing | python-docx |
| Task queue | Celery + Redis |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 |
| Reports | Jinja2 templates |
| Config | pydantic-settings |
| Container | Docker + Docker Compose |

## Setup

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export DOC_INT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/doc_intelligence
export DOC_INT_REDIS_URL=redis://localhost:6379/0
export DOC_INT_CELERY_BROKER_URL=redis://localhost:6379/0

docker compose up -d postgres redis

# Create tables
python -c "from src.models.db import Base; from sqlalchemy import create_engine; Base.metadata.create_all(create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/doc_intelligence'))"

uvicorn src.main:app --reload --port 8000

# In another terminal:
celery -A src.workers.celery_app worker --loglevel=info
```

### Docker

```bash
docker compose up --build
```

## API

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@safety_data_sheet.pdf"
```

### Check processing status

```bash
curl http://localhost:8000/api/v1/status/<task_id>
```

### Get compliance results

```bash
curl http://localhost:8000/api/v1/compliance/<document_id>
```

### Generate reports

```bash
# HTML
curl http://localhost:8000/api/v1/report/<document_id>?format=html > report.html

# JSON
curl http://localhost:8000/api/v1/report/<document_id>?format=json
```

### Batch upload

```bash
curl -X POST http://localhost:8000/api/v1/batch-check \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "files=@doc3.txt"
```

### Regulation lookups

```bash
# GHS hazard statements
curl http://localhost:8000/api/v1/regulations/ghs

# OSHA PELs
curl http://localhost:8000/api/v1/regulations/osha-pels
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

## Project Structure

```
07-doc-intelligence/
├── config/                 # Application settings (pydantic-settings)
├── src/
│   ├── main.py             # FastAPI application entry-point
│   ├── models/             # Pydantic schemas + SQLAlchemy ORM
│   ├── services/           # Core business logic
│   │   ├── document_parser.py    # PDF/DOCX/TXT parsing
│   │   ├── entity_extractor.py   # spaCy NER + custom patterns
│   │   ├── compliance_checker.py # GHS/OSHA/SDS validation
│   │   ├── risk_scorer.py        # Weighted 0–100 scoring
│   │   └── report_builder.py     # HTML/JSON report generation
│   ├── api/routes.py       # REST endpoint definitions
│   ├── workers/celery_app.py     # Async task configuration
│   └── utils/              # Text cleaning + regulation database
├── templates/              # Jinja2 HTML report template
├── tests/                  # Unit test suite
├── data/regulations/       # GHS reference data (JSON)
├── docker-compose.yml      # Full stack orchestration
├── Dockerfile              # Production container image
└── requirements.txt        # Python dependencies
```

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

## Notes

- The spaCy model `en_core_web_sm` is downloaded automatically in the Docker build. For local dev you need `python -m spacy download en_core_web_sm`.
- Risk scoring weights: hazard info 40%, concentration limits 30%, PPE 20%, section completeness 10%.
- Supported document types: SDS, MANUAL, CERTIFICATE, AUDIT_REPORT, REGULATORY_FILING.
- The regulation database includes 50+ GHS hazard statements and 17 OSHA PEL entries for common industrial chemicals.
