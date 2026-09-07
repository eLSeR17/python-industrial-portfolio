"""Application-wide settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    app_name: str = "Digital Twin Simulator"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883

    redis_url: str = "redis://localhost:6379/0"

    max_simulation_time: float = 10_000.0
    default_replications: int = 10

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}


# Minimal config — most tuning happens in simulation params
settings = Settings()
