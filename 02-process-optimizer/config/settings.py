"""Application configuration loaded from environment variables with sensible defaults.

All settings are read via pydantic-settings so they can be overridden with
environment variables or a .env file without touching code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the process optimizer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PROC_OPT_",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "Process Optimizer"
    debug: bool = False
    log_level: str = "INFO"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    redis_state_ttl_seconds: int = 300  # 5 min cache for process state
    redis_optimization_ttl_seconds: int = 60

    # --- Kafka ---
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "process-optimizer-group"
    kafka_sensor_topic: str = "sensor-readings"
    kafka_optimization_topic: str = "optimization-recommendations"
    kafka_auto_offset_reset: str = "latest"
    kafka_max_poll_records: int = 500

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://optimizer:optimizer_dev_2024@localhost:5432/process_optimizer"

    # --- Optimizer ---
    optimization_interval_seconds: float = 10.0
    nelder_mead_max_iterations: int = 200
    nelder_mead_xatol: float = 1e-6
    nelder_mead_fatol: float = 1e-6
    penalty_weight_initial: float = 100.0
    penalty_weight_growth: float = 10.0
    coordinate_descent_max_iterations: int = 100

    # --- PID Controller ---
    pid_default_kp: float = 1.0
    pid_default_ki: float = 0.1
    pid_default_kd: float = 0.01
    pid_derivative_filter_alpha: float = 0.1  # EMA alpha for derivative noise filter
    pid_anti_windup_clamp_min: float = -100.0
    pid_anti_windup_clamp_max: float = 100.0
    pid_ultimate_gain_samples: int = 500

    # --- Process Model (FOPDT) ---
    fopdt_min_gain: float = 0.01
    fopdt_max_gain: float = 100.0
    fopdt_min_time_constant: float = 0.1
    fopdt_max_time_constant: float = 3600.0
    fopdt_min_dead_time: float = 0.0
    fopdt_max_dead_time: float = 600.0

    # --- SPC (Statistical Process Control) ---
    spc_xbar_window_size: int = 25
    spc_r_chart_window_size: int = 25
    spc_cusum_threshold: float = 5.0
    spc_cusum_drift: float = 0.5
    spc_western_electric_zones: bool = True

    # --- OEE ---
    oee_planned_production_hours: float = 24.0
    oee_ideal_cycle_time_seconds: float = 1.0

    # --- WebSocket ---
    ws_heartbeat_interval_seconds: float = 15.0


# Instantiate once at import time — env vars won't change at runtime anyway
settings = Settings()
