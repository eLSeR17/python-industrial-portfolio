# 🏭 Python Industrial Portfolio

> Python engineering for industrial environments: automation, predictive
> analytics, computer vision, and process optimization. Built over 3+
> years of hands-on work with manufacturing plants, warehouses, and
> continuous production lines.
>
> **8 projects, ~190 Python files, full test suites**

## Portfolio Overview

| # | Project | Core Tech | Industry Impact | Tests |
|---|---------|-----------|-----------------|-------|
| 1 | [Predictive Maintenance Engine](./01-predictive-maintenance/) | FastAPI, scikit-learn, TimescaleDB | ↓ 25-30% unplanned downtime costs | 100 |
| 2 | [Real-Time Process Optimizer](./02-process-optimizer/) | NumPy, Kafka, Redis, asyncio | ↑ 15-20% throughput, ↓ waste | 51 |
| 3 | [Industrial Energy Auditor](./03-energy-auditor/) | Pandas, Dash, PostgreSQL | ↓ 10-15% energy costs | 66 |
| 4 | [Supply Chain Cost Analyzer](./04-supply-chain-optimizer/) | NetworkX, PuLP, FastAPI | ↓ 12-18% logistics costs | 96 |
| 5 | [Computer Vision Quality Inspector](./05-vision-inspector/) | OpenCV, YOLOv8, FastAPI | ↓ 90% inspection cost vs manual | 56 |
| 6 | [Digital Twin Simulator](./06-digital-twin/) | SimPy, MQTT, FastAPI | Risk-free plant simulation | 14 |
| 7 | [Document Intelligence](./07-doc-intelligence/) | spaCy, PyMuPDF, Celery | ↓ 85% compliance processing time | 38 |
| 8 | [Fleet & Asset Tracker](./08-fleet-tracker/) | SQLAlchemy, WebSocket, Leaflet | Real-time industrial asset visibility | 51 |

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Python Industrial Portfolio      │
                    └────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────▼─────┐              ┌───────▼───────┐              ┌────▼─────┐
   │ Predict  │              │   Optimize    │              │   See    │
   │  01, 06  │              │   02, 03, 04  │              │  05, 07  │
   │ ML/DL    │              │ Math/Graphs   │              │ CV/NLP   │
   └────┬─────┘              └───────┬───────┘              └────┬─────┘
        │                            │                            │
        │         ┌──────────────────┼──────────────────┐         │
        │         │                  │                  │         │
   ┌────▼─────────▼──┐    ┌─────────▼────────┐   ┌─────▼─────────▼──┐
   │    FastAPI      │    │   PostgreSQL /   │   │    WebSocket /   │
   │    REST APIs    │    │   TimescaleDB    │   │    Real-time     │
   └────────┬────────┘    └─────────┬────────┘   └─────┬────────────┘
            │                       │                    │
   ┌────────▼───────────────────────▼────────────────────▼────────┐
   │                     Docker Compose                           │
   │  ┌─────┐  ┌───────┐  ┌───────┐  ┌──────┐  ┌──────────┐    │
   │  │Redis│  │ Kafka │  │ MQTT  │  │Celery│  │ Ollama   │    │
   │  └─────┘  └───────┘  └───────┘  └──────┘  └──────────┘    │
   └─────────────────────────────────────────────────────────────┘
```

## Skills Demonstrated

| Domain | Technologies |
|--------|-------------|
| **Backend Engineering** | FastAPI, async Python, REST APIs, WebSockets, Celery |
| **Data Engineering** | Pandas, NumPy, TimescaleDB, Kafka, time-series pipelines |
| **Machine Learning** | scikit-learn, anomaly detection, classification, regression |
| **Computer Vision** | OpenCV, YOLOv8, image processing, defect detection, contour analysis |
| **Optimization** | Linear programming (PuLP), graph algorithms (NetworkX), Nelder-Mead |
| **Simulation** | Discrete-event simulation (SimPy), Weibull failure models, digital twins |
| **DevOps** | Docker, Docker Compose, healthchecks, multi-service architectures |
| **IoT/Industrial** | MQTT, sensor data simulation, SCADA data pipelines |
| **NLP** | spaCy NER, entity extraction, document parsing, compliance automation |
| **GIS** | Spatial queries, Haversine, point-in-polygon, real-time tracking |

## Tech Stack

```
Python 3.11+ | FastAPI | PostgreSQL/TimescaleDB | Redis | Kafka
scikit-learn | OpenCV | YOLOv8 | NumPy | Pandas | SciPy | SimPy
Docker | MQTT | WebSocket | Celery | spaCy | PyMuPDF | PuLP | NetworkX
Plotly/Dash | SQLAlchemy | Jinja2 | Pydantic v2 | pytest
```

## How to Use

Each project is self-contained with its own `README.md`, `requirements.txt`,
and `docker-compose.yml`. Projects run independently.

### Quick Start (any project)

```bash
cd 01-predictive-maintenance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

### Docker (any project)

```bash
cd 05-vision-inspector
docker compose up -d
curl http://localhost:8000/health
```

### Run Tests

```bash
cd 02-process-optimizer
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Project Summaries

### 01 — Predictive Maintenance Engine
Sensor data ingestion → feature engineering (time-domain, FFT, rolling stats) → Isolation Forest anomaly detection → Random Forest failure prediction → multi-tier alerting. WebSocket streaming for real-time dashboards. Physics-based fallback for cold-start.

### 02 — Real-Time Process Optimizer
Real-time process data stream → PID controller with auto-tuning (relay + step response) → SPC (X-bar/R charts, CUSUM, EWMA) → waste objective optimization (Nelder-Mead + coordinate descent) → OEE computation. Kafka consumer/producer pattern.

### 03 — Industrial Energy Auditor
Smart meter data ingestion → load profiling → TOU tariff calculation → anomaly detection (Z-score + IQR) → ISO 50001 Energy Performance Indicators → power factor analysis → savings opportunity identification. Plotly Dash dashboard.

### 04 — Supply Chain Cost Analyzer
NetworkX graph modeling → landed cost calculation → inventory optimization (EOQ + safety stock) → VRP route optimization (PuLP/MILP) → demand forecasting (exponential smoothing, Holt, seasonal) → supplier scoring → what-if simulation.

### 05 — Computer Vision Quality Inspector
Image preprocessing pipeline (CLAHE, denoise, ROI) → dual defect detection (YOLOv8 + classical CV with Canny/contour/blob analysis) → rule-based severity classification → annotated output images → SPC quality statistics with C-charts and Pareto analysis.

### 06 — Digital Twin Simulator
SimPy discrete-event simulation → configurable machine processes (Weibull failure, lognormal repair) → buffer/conveyor modeling → 4 scheduling algorithms (FIFO, SPT, EDD, Critical Ratio) → multi-replication scenario comparison with confidence intervals → MQTT bridge for real-time data.

### 07 — Document Intelligence for Compliance
PDF/DOCX/TXT parsing (PyMuPDF) → spaCy NER + custom EntityRuler (chemicals, hazards, PPE, concentrations) → GHS/OSHA compliance checking → weighted risk scoring (0-100) → Jinja2 HTML report generation → Celery async pipeline.

### 08 — Fleet & Asset Tracker
GPS ingestion → Haversine distance/bearing → geofence management (ray casting point-in-polygon) → utilization analysis (active/idle/maintenance classification) → predictive maintenance scheduling → route analysis with deviation detection → WebSocket real-time dashboard.

## Author

Python developer focused on industrial automation and cost reduction.
These projects came out of real work with manufacturing environments —
predicting failures before they happen, finding waste in processes,
and making supply chains actually work.
