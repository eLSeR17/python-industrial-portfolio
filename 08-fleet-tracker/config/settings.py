"""Application configuration using pydantic-settings.

All settings are loaded from environment variables with the ``FLEET_`` prefix.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings.

    Environment variables prefixed with ``FLEET_`` override the defaults.
    """

    model_config = {"env_prefix": "FLEET_", "env_file": ".env", "extra": "ignore"}

    app_name: str = "Fleet & Asset Tracker"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./fleet_tracker.db"
    redis_url: str = "redis://localhost:6379/0"
    geofence_check_interval_sec: int = 10
    maintenance_warning_days: int = 30
    speed_limit_kmh: float = 20.0
    idle_threshold_minutes: int = 5


_settings: Settings | None = None


# Manual singleton — avoids functools import overhead
def get_settings() -> Settings:
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
