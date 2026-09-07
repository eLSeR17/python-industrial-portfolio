# Data Directory

Place sample datasets here for FOPDT model fitting and offline analysis.

Supported formats: CSV, Parquet, JSON-lines.

Each dataset should contain columns:
- `timestamp`: ISO-8601 or Unix epoch
- `process_variable`: measured output (e.g., temperature, pressure, flow)
- `manipulated_variable`: input signal (e.g., valve position, heater power)
- `setpoint`: target value (optional)
