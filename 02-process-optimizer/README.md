# Real-Time Process Optimizer

## The Problem

In most chemical plants and continuous production lines, PID controllers are tuned once at commissioning and left alone. Setpoints are conservative by design — operators would rather run 10% below capacity than risk an excursion. Meanwhile, feedstock composition shifts, ambient conditions change, and equipment wears, but the control parameters never get updated. The result: 20–30% waste in heavy industry, mostly from running suboptimal setpoints nobody has time to recalculate.

This project replaces that static approach with a closed-loop optimizer that continuously re-tunes operating parameters using real sensor streams and first-principles process models.

## Architecture

```
Sensor Streams (Kafka)
        │
        ▼
┌───────────────────┐
│  Stream Processor  │  aiokafka consumer — ingests IoT PLC signals
└────────┬──────────┘
         │
         ▼
┌───────────────────┐    ┌────────────────┐
│  Process Model     │◄──►│  Redis Cache    │
│  (FOPDT digital)   │    └────────────────┘
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Optimizer         │  Nelder-Mead + Bayesian + penalty constraints
│  (gradient-free)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  PID Controller    │  Ziegler-Nichols auto-tune, anti-windup
└────────┬──────────┘
         │
         ├──► Kafka (optimization recommendations → PLC)
         ├──► WebSocket (live dashboard for operators)
         └──► SPC Engine (X-bar, R-charts, CUSUM, Western Electric)
```

## Key Technical Features

| Feature | Implementation |
|---|---|
| **Stream Processing** | aiokafka async consumer with configurable batching and backpressure |
| **Optimization** | Nelder-Mead simplex, coordinate descent, Bayesian surrogate with penalty methods for constraint handling |
| **PID Control** | Ziegler-Nichols auto-tuning, integral anti-windup (clamping + back-calculation), derivative first-order filter |
| **Process Modeling** | FOPDT (First-Order Plus Dead Time) model identification from step-response data via least-squares fitting |
| **Statistical Process Control** | X-bar and R-charts, CUSUM change detection, Western Electric rules (Zone A/B/C + run rules) |
| **OEE Calculation** | Availability × Performance × Quality with bottleneck stage identification |
| **Real-time Dashboard** | WebSocket push of optimization state, control actions, and SPC alarms |
| **Caching** | Redis-backed process state and optimization result cache with TTL |

## Quick Start

```bash
# Clone and enter the project
cd 02-process-optimizer

# Start infrastructure (Kafka/Redis/Postgres)
docker compose up -d

# Install dependencies and run
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000/docs` (Swagger UI).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process/update` | Ingest a new sensor reading |
| `GET` | `/optimize/{process_id}` | Trigger optimization for a process line |
| `GET` | `/spc/{process_id}` | Retrieve SPC charts and alarm state |
| `GET` | `/dashboard-data` | Full dashboard payload (all processes) |
| `WS` | `/ws/live` | Real-time streaming of optimization + SPC data |

### Example Requests

```bash
# Ingest sensor readings
curl -X POST http://localhost:8000/api/v1/process/update \
  -H "Content-Type: application/json" \
  -d '{
    "process_id": "reactor-01",
    "process_type": "chemical_reactor",
    "readings": [
      {
        "sensor_id": "temp-inlet-01",
        "process_id": "reactor-01",
        "value": 185.3,
        "unit": "°C",
        "quality": 0.98
      },
      {
        "sensor_id": "pressure-01",
        "process_id": "reactor-01",
        "value": 4.2,
        "unit": "bar",
        "quality": 0.95
      },
      {
        "sensor_id": "flow-rate-01",
        "process_id": "reactor-01",
        "value": 120.5,
        "unit": "L/min",
        "quality": 0.99
      }
    ],
    "setpoints": {
      "temperature": 190.0,
      "pressure": 4.0,
      "flow_rate": 125.0
    }
  }'

# Trigger optimization (Nelder-Mead, default)
curl http://localhost:8000/api/v1/optimize/reactor-01

# Trigger optimization with specific method and iteration limit
curl "http://localhost:8000/api/v1/optimize/reactor-01?method=coordinate_descent&max_iterations=500"

# Run SPC analysis
curl "http://localhost:8000/api/v1/spc/reactor-01?window=200&subgroup_size=5"

# Get full dashboard data
curl http://localhost:8000/api/v1/dashboard-data

# Health check
curl http://localhost:8000/api/v1/health
```

## Tests

```bash
pytest tests/ -v
```

All tests use NumPy-generated synthetic data — no external services required.

## Project Structure

```
config/             — Settings and environment configuration
src/
  models/           — Pydantic schemas and process state models
  services/         — Core business logic (optimizer, PID, SPC, modeling)
  api/              — FastAPI routes and WebSocket handler
  utils/            — NumPy signal processing and time-series utilities
tests/              — pytest suite with synthetic process data
data/               — Sample datasets for model fitting
```

## License

MIT
