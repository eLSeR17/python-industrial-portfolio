from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Provides database, Redis, and optimization engine settings with
    sensible defaults for local development.
    """

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+asyncpg://scuser:scpassword@localhost:5432/supplychain"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # Optimization defaults
    default_service_level: float = 0.95
    default_holding_cost_pct: float = 0.25
    default_ordering_cost: float = 50.0
    default_lead_time_days: float = 7.0
    solver_time_limit_seconds: int = 60
    default_vehicle_capacity: float = 1000.0
    default_vehicle_count: int = 5

    # Cost model defaults
    fuel_cost_per_km: float = 0.35
    driver_cost_per_hour: float = 25.0
    average_speed_kmh: float = 60.0
    warehouse_cost_per_unit_day: float = 0.15
    handling_cost_per_unit: float = 0.50
    duty_rate_pct: float = 0.0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False


settings = Settings()
