"""Tests for the REST API routes.

Uses httpx AsyncClient with FastAPI's TestApp transport to exercise
the health, ingest, predictions, and alerts endpoints with synthetic
sensor data.
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import create_app
from src.api.routes import _reset_stores


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Create a test client against the FastAPI app (no lifespan startup).

    The lifespan is skipped so tests run without a database or Redis.
    Stores are reset between tests via _reset_stores().
    """
    _reset_stores()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_stores()


@pytest.fixture
def sensor_payload() -> dict:
    """Build a realistic ingest batch payload with synthetic data."""
    return {
        "readings": [
            {
                "asset_id": "MOTOR-001",
                "asset_type": "motor",
                "timestamp": "2026-09-04T10:30:00Z",
                "vibration_x": 2.34,
                "vibration_y": 1.87,
                "vibration_z": 3.12,
                "temperature": 78.5,
                "pressure": 4.2,
                "current": 15.3,
                "rpm": 1750,
            }
        ]
    }


# ── Health endpoint tests ───────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health system health check."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Health endpoint should return 200 with status healthy."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_includes_service_name(self, client: AsyncClient) -> None:
        """Health response should include the service name."""
        resp = await client.get("/health")
        body = resp.json()
        assert "service" in body
        assert len(body["service"]) > 0


# ── Ingest endpoint tests ──────────────────────────────────────────────


class TestIngestEndpoint:
    """Tests for the POST /api/v1/ingest endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_returns_201(
        self, client: AsyncClient, sensor_payload: dict
    ) -> None:
        """Successful ingestion should return 201 Created."""
        resp = await client.post("/api/v1/ingest", json=sensor_payload)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_ingest_accepts_one_reading(
        self, client: AsyncClient, sensor_payload: dict
    ) -> None:
        """Ingest response should report accepted count matching input."""
        resp = await client.post("/api/v1/ingest", json=sensor_payload)
        body = resp.json()
        assert body["accepted"] == 1
        assert body["rejected"] == 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_readings(
        self, client: AsyncClient
    ) -> None:
        """Ingest should handle batch with multiple readings."""
        payload = {
            "readings": [
                {
                    "asset_id": f"MOTOR-{i:03d}",
                    "asset_type": "motor",
                    "timestamp": f"2026-09-04T10:30:{i:02d}Z",
                    "vibration_x": 2.0 + i * 0.1,
                    "vibration_y": 1.5,
                    "vibration_z": 2.5,
                    "temperature": 70.0,
                    "pressure": 4.0,
                    "current": 12.0,
                    "rpm": 1750,
                }
                for i in range(5)
            ]
        }
        resp = await client.post("/api/v1/ingest", json=payload)
        assert resp.status_code == 201
        assert resp.json()["accepted"] == 5

    @pytest.mark.asyncio
    async def test_ingest_includes_processing_time(
        self, client: AsyncClient, sensor_payload: dict
    ) -> None:
        """Ingest response should include processing time in milliseconds."""
        resp = await client.post("/api/v1/ingest", json=sensor_payload)
        body = resp.json()
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] >= 0.0


# ── Health-by-asset endpoint tests ─────────────────────────────────────


class TestAssetHealthEndpoint:
    """Tests for the GET /api/v1/health/{asset_id} endpoint."""

    @pytest.mark.asyncio
    async def test_health_unknown_asset_returns_404(
        self, client: AsyncClient
    ) -> None:
        """Querying health for an unknown asset should return 404."""
        resp = await client.get("/api/v1/health/NONEXISTENT")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_health_after_ingest(
        self, client: AsyncClient, sensor_payload: dict
    ) -> None:
        """After ingesting data, health should be computable."""
        await client.post("/api/v1/ingest", json=sensor_payload)
        resp = await client.get("/api/v1/health/MOTOR-001")
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["overall_score"] <= 1.0
        assert body["asset_id"] == "MOTOR-001"


# ── Predictions endpoint tests ──────────────────────────────────────────


class TestPredictionsEndpoint:
    """Tests for the GET /api/v1/predictions/{asset_id} endpoint."""

    @pytest.mark.asyncio
    async def test_prediction_unknown_asset_returns_404(
        self, client: AsyncClient
    ) -> None:
        """Querying predictions for an unknown asset should return 404."""
        resp = await client.get("/api/v1/predictions/NONEXISTENT")
        assert resp.status_code == 404


# ── Alerts endpoint tests ──────────────────────────────────────────────


class TestAlertsEndpoint:
    """Tests for the GET /api/v1/alerts endpoint."""

    @pytest.mark.asyncio
    async def test_alerts_empty_when_no_data(
        self, client: AsyncClient
    ) -> None:
        """Alerts endpoint should return empty list when no data exists."""
        resp = await client.get("/api/v1/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_alerts_supports_limit_query(
        self, client: AsyncClient
    ) -> None:
        """Alerts endpoint should respect the limit parameter."""
        resp = await client.get("/api/v1/alerts?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) <= 5


# ── Maintenance schedule endpoint tests ─────────────────────────────────


class TestMaintenanceScheduleEndpoint:
    """Tests for the GET /api/v1/maintenance-schedule endpoint."""

    @pytest.mark.asyncio
    async def test_schedule_returns_empty_when_no_predictions(
        self, client: AsyncClient
    ) -> None:
        """Schedule should be empty when no predictions exist."""
        resp = await client.get("/api/v1/maintenance-schedule")
        assert resp.status_code == 200
        body = resp.json()
        assert body["windows"] == []
        assert body["total_estimated_cost"] == 0.0
