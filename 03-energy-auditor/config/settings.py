"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the energy auditor application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "Industrial Energy Auditor"
    app_version: str = "1.0.0"
    debug: bool = False

    # --- Database ---
    database_url: str = "postgresql+asyncpg://energy:energy@localhost:5432/energy_auditor"
    database_url_sync: str = "postgresql+psycopg2://energy:energy@localhost:5432/energy_auditor"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2

    # --- Tariff defaults (TOU) ---
    peak_rate_kwh: float = 0.18  # USD/kWh during peak hours
    shoulder_rate_kwh: float = 0.12
    offpeak_rate_kwh: float = 0.07
    demand_charge_kva: float = 15.0  # USD/kVA monthly demand charge
    pf_penalty_threshold: float = 0.90  # power factor below this triggers penalty
    pf_penalty_rate: float = 0.02  # surcharge per 0.01 below threshold

    # --- Anomaly detection ---
    zscore_threshold: float = 3.0
    rolling_window_hours: int = 168  # 7 days

    # --- Benchmarking ---
    baseline_year: int = 2024
    enpi_target_reduction_pct: float = 2.0  # ISO 50001 annual improvement target

    # --- Report ---
    report_output_dir: Path = Path("data/reports")
    logo_path: Path | None = None

    # --- CORS ---
    allowed_origins: list[str] = ["*"]


@lru_cache  # Cached singleton — pydantic-settings validates on first access
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()
