# Data Directory

This directory stores generated reports and any persistent data.

## Reports
HTML audit reports are generated to `data/reports/` by the report generator service.

## Meter Data
In development mode, synthetic meter data is generated in-memory. For production,
meter readings are stored in PostgreSQL via the SQLAlchemy ORM models.
