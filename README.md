# 🏭 Python Industrial Portfolio

> Python engineering for industrial environments: automation, predictive
> analytics, computer vision, and process optimization. Built over 3+
> years of hands-on work with manufacturing plants, warehouses, and
> continuous production lines.
>
> **8 projects, ~190 Python files, full test suites**
[![CI](https://github.com/eLSeR17/python-industrial-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/eLSeR17/python-industrial-portfolio/actions)


## Portfolio Overview

| # | Project | Core Tech | Impact target † | Tests |
|---|---------|-----------|-----------------|-------|
| 1 | [Predictive Maintenance Engine](./01-predictive-maintenance/) | FastAPI, scikit-learn, TimescaleDB | ↓ 25-30% unplanned downtime costs † | 100 |
| 2 | [Real-Time Process Optimizer](./02-process-optimizer/) | NumPy, Kafka, Redis, asyncio | ↑ 15-20% throughput, ↓ waste † | 51 |
| 3 | [Industrial Energy Auditor](./03-energy-auditor/) | Pandas, Dash, PostgreSQL | ↓ 10-15% energy costs † | 66 |
| 4 | [Supply Chain Cost Analyzer](./04-supply-chain-optimizer/) | NetworkX, PuLP, FastAPI | ↓ 12-18% logistics costs † | 96 |
| 5 | [Computer Vision Quality Inspector](./05-vision-inspector/) | OpenCV, YOLOv8, FastAPI | ↓ 90% inspection cost vs manual † | 56 |
| 6 | [Digital Twin Simulator](./06-digital-twin/) | SimPy, MQTT, FastAPI | Risk-free plant simulation | 14 |
| 7 | [Document Intelligence](./07-doc-intelligence/) | spaCy, PyMuPDF, Celery | ↓ 85% compliance processing time † | 38 |
| 8 | [Fleet & Asset Tracker](./08-fleet-tracker/) | SQLAlchemy, WebSocket, Leaflet | Real-time industrial asset visibility | 51 |

> † **Impact targets** — design estimates based on industry benchmarks for each
> class of system (predictive maintenance programs, statistical process
> control, energy audits, automated visual inspection, document automation),
> not measurements from a specific plant. Each project shows the mechanism
> that produces the improvement. See
> [*How to read the impact numbers*](#how-to-read-the-impact-numbers).

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

## How to read the impact numbers

The impact column states **design targets, not measured results**: these
systems come from real industrial practice, but no single plant runs all
eight, so there is no single measurement to report. What you *can* verify is
the **mechanism** — each project shows in code and tests exactly how the
improvement is produced. The ranges are the ones the industry reports for each
class of system:

- **01 · Predictive Maintenance Engine → ↓ 25-30% unplanned downtime**. PdM
  programs consistently report 25-30% downtime reduction (reliability
  engineering literature). Mechanism: anomaly detection on streaming sensor
  data (Isolation Forest) plus failure prediction (Random Forest) with the
  lead time needed to intervene before the breakdown.
- **02 · Real-Time Process Optimizer → ↑ 15-20% throughput, ↓ waste**.
  Mechanism: SPC (X-bar/R, CUSUM, EWMA) + auto-tuned PID keep the process
  inside its target window — less drift, less scrap and rework. 15-20% is the
  documented range for SPC-driven process stabilization.
- **03 · Industrial Energy Auditor → ↓ 10-15% energy costs**. Energy audits
  typically identify 10-20% savings potential; this system operationalizes the
  audit: load profiling, TOU tariff analysis, anomaly detection and ISO 50001
  EnPIs turn that potential into a tracked program.
- **04 · Supply Chain Cost Analyzer → ↓ 12-18% logistics costs**. Mechanism:
  EOQ + safety stock cut inventory costs, VRP (MILP) cuts transport km — the
  combined range documented for inventory + routing optimization.
- **05 · Computer Vision Quality Inspector → ↓ 90% inspection cost vs
  manual**. Mechanism: automated visual inspection (YOLOv8 + classical CV)
  replaces a manual operator reading — the marginal cost of machine inspection
  is a small fraction of operator time. 80-95% cost reduction is the
  documented range for CV-based QA in manufacturing.
- **07 · Document Intelligence → ↓ 85% compliance processing time**. Mechanism:
  PDF parsing + spaCy NER + automated GHS/OSHA rule checks turn multi-hour
  manual reviews into minutes — the documented range for compliance-document
  automation.

That is the honest framing: **benchmarked targets with a code-visible
mechanism**, not unverifiable measurement claims. If someone asks *"how do you
know?"*, the answer starts with the mechanism, not the number.

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
