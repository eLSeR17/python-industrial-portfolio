# 06 — Digital Twin Simulator

Before you move a single machine on the shop floor, you can simulate the
whole line in Python and see what actually happens. This project wraps
SimPy 4 in a FastAPI service so you can run what-if scenarios via REST:
change buffer sizes, swap scheduling rules, inject failures, compare
results side by side.

I built this because running physical experiments on a production line
costs real money and downtime. A simulation with the right parameters
gets you 80% of the answer in seconds.

## Physical vs Digital

| Physical experiment | Digital twin |
|---|---|
| $10K – $100K per trial | Runs on your laptop |
| Days / weeks of downtime | Seconds of wall-clock time |
| Risk of equipment damage | Zero physical risk |
| Limited scenarios | Unlimited what-if comparisons |

Questions like *"What if we double the buffer before Station 3?"* or
*"Does SPT beat FIFO here?"* get answered without touching the floor.

## Architecture

```
┌──────────────┐      ┌──────────────────────┐
│   FastAPI    │◄────►│   ScenarioRunner      │
│  REST API    │      │   (what-if engine)    │
└──────┬───────┘      └──────────┬───────────┘
       │                         │
       ▼                         ▼
┌──────────────┐      ┌──────────────────────┐
│   MQTT       │      │   SimPy Engine       │
│   Bridge     │      │   ┌─────┐ ┌───────┐ │
│  (optional)  │      │   │Mach.│►│Buffer │ │
└──────────────┘      │   └─────┘ └───────┘ │
                      │   ┌─────┐ ┌───────┐ │
                      │   │Mach.│►│Buffer │ │
                      │   └─────┘ └───────┘ │
                      └──────────────────────┘
```

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Simulation core | **SimPy 4** | Discrete-event simulation engine |
| Numerics | **NumPy / SciPy** | Distribution fitting, Weibull MLE |
| API | **FastAPI + Uvicorn** | Async REST interface |
| IoT bridge | **Paho MQTT** | Publish/subscribe to shop-floor SCADA |
| Config | **Pydantic v2** | Type-safe settings and schemas |
| Cache | **Redis** | Result caching (optional) |

## Quick Start

```bash
# Local
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker compose up --build
```

## API — Curl Examples

### Health check

```bash
curl http://localhost:8000/health
```

### Run a simulation

```bash
curl -s -X POST http://localhost:8000/api/v1/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "sim-001",
    "name": "Two-machine line",
    "duration": 1000,
    "warmup_period": 100,
    "machines": [
      {"id": "m1", "name": "CNC-1", "processing_time_mean": 10, "processing_time_std": 2,
       "failure_rate": 0.01, "repair_time_mean": 30, "repair_time_std": 5, "capacity": 1},
      {"id": "m2", "name": "CNC-2", "processing_time_mean": 12, "processing_time_std": 3,
       "failure_rate": 0.005, "repair_time_mean": 45, "repair_time_std": 10, "capacity": 1}
    ],
    "buffers": [
      {"id": "b1", "name": "WIP-Between", "capacity": 20, "initial_level": 0}
    ],
    "conveyors": [
      {"id": "c1", "name": "Belt-1", "from_station": "m1", "to_station": "m2",
       "speed": 1.0, "capacity": 5}
    ],
    "schedule_type": "SPT",
    "random_seed": 42
  }' | python -m json.tool
```

### Compare two scheduling rules

```bash
curl -s -X POST http://localhost:8000/api/v1/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "scenarios": [
      {"id": "fifo", "name": "FIFO", "duration": 500, "warmup_period": 50,
       "machines": [{"id": "m1", "name": "M1", "processing_time_mean": 10, "processing_time_std": 1,
                     "failure_rate": 0.01, "repair_time_mean": 20, "repair_time_std": 5, "capacity": 1}],
       "buffers": [{"id": "b1", "name": "B1", "capacity": 10, "initial_level": 0}],
       "conveyors": [{"id": "c1", "name": "C1", "from_station": "m1", "to_station": "m1",
                       "speed": 1.0, "capacity": 1}],
       "schedule_type": "FIFO", "random_seed": 1},
      {"id": "spt", "name": "SPT", "duration": 500, "warmup_period": 50,
       "machines": [{"id": "m1", "name": "M1", "processing_time_mean": 10, "processing_time_std": 1,
                     "failure_rate": 0.01, "repair_time_mean": 20, "repair_time_std": 5, "capacity": 1}],
       "buffers": [{"id": "b1", "name": "B1", "capacity": 10, "initial_level": 0}],
       "conveyors": [{"id": "c1", "name": "C1", "from_station": "m1", "to_station": "m1",
                       "speed": 1.0, "capacity": 1}],
       "schedule_type": "SPT", "random_seed": 1}
    ]
  }' | python -m json.tool
```

### Get cached results

```bash
curl http://localhost:8000/api/v1/results/sim-001
```

### Plant topology

```bash
curl http://localhost:8000/api/v1/plant/default/topology
```

## Scheduling Algorithms

| Strategy | Best for | Trade-off |
|---|---|---|
| **FIFO** | Stable flow, fairness | Ignores job urgency |
| **SPT** | Minimizing avg flow time | Can starve long jobs |
| **EDD** | Minimizing max tardiness | May increase WIP |
| **CRITICAL_RATIO** | Dynamic priority | Needs accurate due dates |

## Project Structure

```
src/
├── main.py               # FastAPI application
├── models/               # Pydantic schemas + dataclasses
├── simulation/           # SimPy engine, machine, buffer, conveyor, failures
├── services/             # Metrics, scenario runner, MQTT bridge
├── api/                  # REST routes
└── utils/                # Statistical distributions
```

## License

Internal — not for redistribution.
