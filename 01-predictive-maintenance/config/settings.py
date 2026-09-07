"""Application configuration management.

Centralizes all settings with environment variable support and sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    url: str = "postgresql+asyncpg://pm_user:pm_secret_2026@localhost:5432/predictive_maintenance"
    url_sync: str = "postgresql+psycopg2://pm_user:pm_secret_2026@localhost:5432/predictive_maintenance"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800

    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisSettings(BaseSettings):
    """Redis connection configuration for caching and pub/sub."""

    url: str = "redis://localhost:6379/0"
    max_connections: int = 20
    socket_timeout: int = 5
    decode_responses: bool = True

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class MLSettings(BaseSettings):
    """Machine learning pipeline configuration."""

    # Isolation Forest parameters
    contamination: float = 0.05
    n_estimators_isolation: int = 200
    max_samples_isolation: str | int = "auto"
    random_state: int = 42

    # Random Forest failure prediction
    n_estimators_rf: int = 300
    max_depth_rf: int = 15
    min_samples_split_rf: int = 5
    test_size: float = 0.2

    # Feature engineering windows (number of samples)
    rolling_windows: list[int] = [10, 50, 200]
    fft_window_size: int = 256
    spectral_entropy_bins: int = 64

    # Model persistence
    model_dir: str = "/app/models" if __name__ == "__main__" else "models"
    retrain_interval_hours: int = 24


class AlertSettings(BaseSettings):
    """Alert threshold configuration."""

    temperature_warning: float = 85.0
    temperature_critical: float = 95.0
    vibration_warning: float = 5.0
    vibration_critical: float = 8.0
    pressure_warning: float = 6.0
    pressure_critical: float = 7.5
    failure_probability_warning: float = 0.3
    failure_probability_critical: float = 0.7
    health_score_warning: float = 0.6
    health_score_critical: float = 0.3


class Settings(BaseSettings):
    """Root application settings aggregating all sub-configurations."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"
    app_name: str = "Predictive Maintenance Engine"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # Sensor simulation defaults
    default_sample_rate_hz: float = 100.0
    default_asset_count: int = 10
    degradation_start_hour: float = 72.0
    failure_threshold_hour: float = 96.0

    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    ml: MLSettings = MLSettings()
    alerts: AlertSettings = AlertSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache  # Reuse single instance — env vars don't change at runtime
def get_settings() -> Settings:
    """Return cached application settings singleton.

    Environment variables override defaults. The cache ensures the same
    Settings instance is reused across the application lifecycle, avoiding
    repeated file reads and validation.

    Returns:
        Validated Settings instance with all sub-configurations.
    """
    return Settings()
