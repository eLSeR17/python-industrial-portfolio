"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the Vision Inspector service.

    All fields can be overridden via environment variables prefixed with
    ``VISION_``.  For example ``VISION_DEBUG=true`` or
    ``VISION_CONFIDENCE_THRESHOLD=0.6``.
    """

    model_config = {"env_prefix": "VISION_"}

    app_name: str = "Vision Inspector"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str | None = None
    confidence_threshold: float = 0.5
    defect_severity_thresholds: dict[str, float] = {
        "scratch_width_minor_mm": 0.1,
        "scratch_width_critical_mm": 0.5,
        "dent_area_minor_pct": 0.5,
        "dent_area_major_pct": 2.0,
        "crack_length_minor_mm": 1.0,
        "crack_length_major_mm": 5.0,
        "stain_area_minor_pct": 0.3,
        "stain_area_major_pct": 1.0,
    }
    redis_url: str = "redis://redis:6379/0"
    max_image_size_mb: int = 10
    pixels_per_mm: float = 10.0
    alert_threshold_pct: float = 5.0


# Direct instantiation — simple and explicit
settings = Settings()
