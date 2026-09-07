"""Application settings loaded from environment variables.

Uses pydantic-settings with the ``DOC_INT_`` prefix so every setting
can be overridden via an environment variable of the form
``DOC_INT_<SETTING_NAME>``.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Document Intelligence service."""

    model_config = SettingsConfigDict(
        env_prefix="DOC_INT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Document Intelligence for Compliance"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/doc_intelligence"

    # --- Redis / Celery ---
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"

    # --- spaCy ---
    spacy_model: str = "en_core_web_sm"

    # --- Upload limits ---
    max_upload_size_mb: int = 50

    # --- Risk thresholds (0-100 scale) ---
    risk_thresholds: dict[str, int] = Field(
        default={
            "critical": 75,
            "high": 50,
            "medium": 25,
        }
    )

    @property
    def max_upload_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
