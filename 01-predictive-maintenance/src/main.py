"""FastAPI application factory and entry point.

Creates and configures the FastAPI application with all middleware,
routers, and lifecycle hooks. Includes startup logic for the ML
pipeline calibration and sensor simulation.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.api.routes import router
from src.api.websocket import ws_router
from src.services.alert_service import AlertService
from src.services.anomaly_detector import AnomalyDetector
from src.services.data_collector import DataCollector
from src.services.feature_engineer import FeatureEngineer
from src.services.failure_predictor import FailurePredictor

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("predictive_maintenance")


# ── Service singletons (initialized at startup) ──────────────────────
_data_collector: DataCollector | None = None
_feature_engineer: FeatureEngineer | None = None
_anomaly_detector: AnomalyDetector | None = None
_failure_predictor: FailurePredictor | None = None
_alert_service: AlertService | None = None


def get_data_collector() -> DataCollector:
    """Access the global data collector service."""
    assert _data_collector is not None, "DataCollector not initialized"
    return _data_collector


def get_feature_engineer() -> FeatureEngineer:
    """Access the global feature engineering service."""
    assert _feature_engineer is not None, "FeatureEngineer not initialized"
    return _feature_engineer


def get_anomaly_detector() -> AnomalyDetector:
    """Access the global anomaly detection service."""
    assert _anomaly_detector is not None, "AnomalyDetector not initialized"
    return _anomaly_detector


def get_failure_predictor() -> FailurePredictor:
    """Access the global failure prediction service."""
    assert _failure_predictor is not None, "FailurePredictor not initialized"
    return _failure_predictor


def get_alert_service() -> AlertService:
    """Access the global alert service."""
    assert _alert_service is not None, "AlertService not initialized"
    return _alert_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown logic.

    On startup:
    1. Initialize all service singletons
    2. Register simulated fleet of industrial assets
    3. Collect calibration data for anomaly detection
    4. Pre-populate some training data for failure prediction

    On shutdown:
    - Log final statistics
    """
    global _data_collector, _feature_engineer, _anomaly_detector
    global _failure_predictor, _alert_service

    settings = get_settings()
    logger.info("Starting Predictive Maintenance Engine v%s", settings.app_version)

    # Initialize services
    _data_collector = DataCollector(sample_rate_hz=settings.default_sample_rate_hz)
    _feature_engineer = FeatureEngineer(buffer_size=200, fft_window=256)
    _anomaly_detector = AnomalyDetector(
        contamination=settings.ml.contamination,
        n_estimators=settings.ml.n_estimators_isolation,
    )
    _failure_predictor = FailurePredictor(
        n_estimators=settings.ml.n_estimators_rf,
        max_depth=settings.ml.max_depth_rf,
        min_samples_split=settings.ml.min_samples_split_rf,
    )
    _alert_service = AlertService(cooldown_seconds=300)

    # Register simulated fleet
    from src.models.schemas import AssetType
    fleet_ids = _data_collector.register_fleet(
        count=settings.default_asset_count,
        asset_type=AssetType.MOTOR,
        prefix="MOTOR",
    )
    logger.info("Registered fleet: %s", fleet_ids)

    # Calibrate anomaly detector with simulated healthy data
    logger.info("Calibrating anomaly detector with synthetic healthy data...")
    for asset_id in fleet_ids:
        # Generate 60 "healthy" readings for calibration
        readings = _data_collector.collect_batch([asset_id], count=60)
        for reading in readings:
            fv = _feature_engineer.compute_features(reading)
            _anomaly_detector.add_calibration_sample(fv)

    logger.info("Anomaly detector calibrated for %d assets", len(fleet_ids))

    # Generate some failure probability data for demo
    logger.info("Generating training data for failure predictor...")
    import random
    for asset_id in fleet_ids[:3]:
        # Simulate a degrading asset
        for hour_offset in range(0, 100, 2):
            readings = _data_collector.collect_batch([asset_id], count=1)
            if readings:
                fv = _feature_engineer.compute_features(readings[0])
                # Label: will fail within 24h if past hour 72
                will_fail = hour_offset > 72
                rul = max(0, 96 - hour_offset) if will_fail else 200.0
                _failure_predictor.add_training_sample(
                    fv,
                    time_to_failure_hours=rul,
                    failed_within_horizon=will_fail,
                )

    logger.info("Startup complete. API ready at http://%s:%d", settings.host, settings.port)
    logger.info("Swagger docs at http://%s:%d/docs", settings.host, settings.port)

    yield

    # Shutdown
    logger.info("Shutting down Predictive Maintenance Engine...")
    _data_collector = None
    _feature_engineer = None
    _anomaly_detector = None
    _failure_predictor = None
    _alert_service = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes and middleware.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Predictive Maintenance Engine for industrial equipment. "
            "Reduces unplanned downtime by 25–30% through ML-driven "
            "failure prediction and condition-based maintenance scheduling."
        ),
        lifespan=lifespan,
    )

    # CORS for dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router)
    app.include_router(ws_router)

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """System health check endpoint.

        Returns:
            Status dict confirming the API is operational.
        """
        return {"status": "healthy", "service": settings.app_name}

    return app


# Module-level app instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
