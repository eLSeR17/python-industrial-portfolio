# Industrial Energy Auditor

I built this after spending months looking at energy bills for a mid-size manufacturing client and realizing most of the savings opportunities were hiding in plain sight — poor power factor, equipment left running overnight, HVAC schedules that hadn't been updated since installation. This tool ingests smart meter data and runs the analysis an energy consultant would do, but in seconds instead of weeks.

## What it does

- **Smart Meter Ingestion** – Accepts 15-min / hourly readings with kWh, kVA, PF, voltage, temperature
- **Load Profiling** – 24-hour demand curves with TOU (time-of-use) period classification
- **Demand Analysis** – Contract utilization, peak detection, optimal contract demand sizing
- **Power Factor Analysis** – Penalty exposure calculation, capacitor bank sizing recommendations
- **TOU Cost Optimization** – Breakdown of energy costs by peak/shoulder/off-peak, load-shift savings estimation
- **Anomaly Detection** – Z-score spikes, baseline shifts, equipment left on, PF drops
- **Peak Shaving** – Analysis of demand reduction opportunities
- **HVAC Optimization** – Scheduling analysis, temperature overshoot detection
- **ISO 50001 Benchmarking** – EnPI calculation, cross-facility comparison, baseline tracking
- **Interactive Dashboard** – Plotly Dash with real-time charts (load profiles, demand, PF trends, anomalies)
- **HTML Audit Reports** – Professional Jinja2-templated reports with Plotly charts
- **REST API** – FastAPI with Swagger docs

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Smart Meters │────▶│  FastAPI      │────▶│  PostgreSQL  │
│  (kWh/kVA/PF) │     │  REST API     │     │  (time-series)│
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │  Analysis     │
                     │  Services     │
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
       ┌──────▼─────┐ ┌────▼──────┐ ┌────▼──────┐
       │  Load       │ │  Anomaly  │ │  Savings  │
       │  Profiling  │ │  Detector │ │  Optimizer │
       └────────────┘ └───────────┘ └───────────┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────┐
                     │  Plotly Dash  │     │  HTML Reports │
                     │  Dashboard    │     │  (Jinja2)     │
                     └──────────────┘     └──────────────┘
```

## Quick Start

### Docker (recommended)

```bash
docker compose up -d
```

The app starts at `http://localhost:8000` with a demo facility pre-loaded.

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/facilities` | Register a new facility |
| POST | `/api/v1/readings/ingest` | Ingest meter readings (batch) |
| POST | `/api/v1/readings/generate/{id}` | Generate synthetic demo data |
| GET | `/api/v1/dashboard/{id}` | Load profile + KPIs |
| GET | `/api/v1/analysis/demand/{id}` | Demand analysis |
| GET | `/api/v1/analysis/power-factor/{id}` | PF analysis + sizing |
| GET | `/api/v1/analysis/tou-cost/{id}` | TOU cost breakdown |
| GET | `/api/v1/analysis/anomalies/{id}` | Anomaly detection |
| GET | `/api/v1/audit/{id}` | Full audit HTML report |
| GET | `/api/v1/benchmark` | Cross-facility benchmark |
| GET | `/api/v1/savings-report/{id}` | Savings recommendations |
| GET | `/dashboard/` | Interactive Plotly dashboard |
| GET | `/docs` | Swagger API docs |

### Example Requests

**Health check:**
```bash
curl http://localhost:8000/health
```

**Register a facility:**
```bash
curl -X POST http://localhost:8000/api/v1/facilities \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Steel Foundry Plant",
    "code": "SFP-01",
    "address": "456 Industrial Way, Steel District",
    "facility_type": "manufacturing",
    "contract_demand_kva": 800,
    "tariff_profile": "tou_general",
    "timezone": "America/New_York"
  }'
```

**Ingest meter readings:**
```bash
curl -X POST http://localhost:8000/api/v1/readings/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {
        "meter_id": "MTR-MAIN",
        "facility_id": "10000000-0000-0000-0000-000000000001",
        "timestamp": "2026-01-15T08:00:00Z",
        "active_energy_kwh": 125.4,
        "reactive_energy_kvarh": 45.2,
        "apparent_energy_kvah": 133.5,
        "demand_kw": 320.0,
        "demand_kva": 340.0,
        "power_factor": 0.92,
        "voltage_avg": 480.0,
        "frequency_hz": 60.01,
        "current_a": 386.0,
        "thd_voltage_pct": 3.2,
        "temperature_c": 28.5
      }
    ]
  }'
```

**Generate synthetic data:**
```bash
curl -X POST "http://localhost:8000/api/v1/readings/generate/10000000-0000-0000-0000-000000000001?days=60"
```

**Get demand analysis:**
```bash
curl http://localhost:8000/api/v1/analysis/demand/10000000-0000-0000-0000-000000000001
```

**Run full audit (with production normalization):**
```bash
curl "http://localhost:8000/api/v1/audit/10000000-0000-0000-0000-000000000001?production_units=15000&floor_area_sqm=5000"
```

**Get savings report:**
```bash
curl "http://localhost:8000/api/v1/savings-report/10000000-0000-0000-0000-000000000001?production_units=15000"
```

### Demo Facility ID

```
10000000-0000-0000-0000-000000000001
```

## Key Algorithms

### Load Profiling
Groups meter readings into 24 hourly buckets, computing mean/min/max demand and power factor. Each bucket is classified into its TOU period (peak/shoulder/off-peak). The load factor (average/peak ratio) characterizes consumption efficiency.

### Anomaly Detection
- **Z-score spikes**: Rolling 7-day mean/std baseline; readings with |Z| > 3 flagged
- **Baseline shifts**: Comparison of recent 7-day average vs. prior 7-day average
- **Equipment left on**: Off-hours demand exceeding minimum threshold
- **PF drops**: Consecutive readings below penalty threshold

### Power Factor Correction
Capacitor bank sizing: `Qc = P × (tan(arccos(PF_old)) - tan(arccos(PF_new)))`. Eliminates penalty charges and reduces I²R losses.

### ISO 50001 EnPIs
Energy Performance Indicators normalize consumption against production output (kWh/unit), floor area (kWh/m²), or degree-days. Annual improvement target: 2%.

## Technical Stack

- **FastAPI** – async REST API with automatic OpenAPI docs
- **SQLAlchemy 2.0** – ORM for PostgreSQL time-series storage
- **Pandas** – resampling, rolling statistics, groupby aggregation
- **Plotly / Dash** – interactive charts (load profiles, demand curves, PF trends)
- **Jinja2** – HTML report templating with embedded Plotly charts
- **NumPy / SciPy** – statistical analysis (Z-score, linear regression)
- **Pydantic v2** – request/response validation with custom validators
- **Docker Compose** – PostgreSQL, Redis, app containers

## Project Structure

```
03-energy-auditor/
├── config/settings.py          # Pydantic settings (env vars)
├── src/
│   ├── main.py                 # FastAPI + Dash entry point
│   ├── models/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── db.py               # SQLAlchemy ORM (facilities, readings, equipment)
│   ├── services/
│   │   ├── meter_reader.py     # Smart meter data (real + synthetic generation)
│   │   ├── consumption_analyzer.py  # Load profiling, demand, PF, TOU cost
│   │   ├── anomaly_detector.py # 4 anomaly detectors (spikes, shifts, off-hours, PF)
│   │   ├── benchmark_engine.py # ISO 50001 EnPIs, cross-facility benchmarking
│   │   ├── optimizer.py        # Peak shaving, HVAC, savings report
│   │   └── report_generator.py # Jinja2 HTML audit reports
│   ├── api/routes.py           # FastAPI endpoints
│   ├── dashboard/plots.py      # Plotly Dash interactive dashboard
│   └── utils/
│       ├── tariff.py           # TOU tariff structures + bill calculation
│       └── units.py            # kW↔kVA, reactive power, capacitor sizing
├── tests/                      # pytest test suite
├── docker-compose.yml          # PostgreSQL + Redis + App
└── requirements.txt
```

## Business Context

Industrial energy costs represent 20-40% of manufacturing operating expenses. A systematic energy audit identifies:

1. **Power factor penalties** – Facilities with PF < 0.90 pay surcharges; correction pays back in 6-18 months
2. **Load shifting opportunities** – Moving 25% of peak load to off-peak saves 8-15% on energy charges
3. **Equipment waste** – Equipment left running overnight/weekends can add 10-20% to baseload
4. **HVAC inefficiency** – Poor scheduling and setpoints waste 5-15% of cooling/heating energy
5. **Peak demand** – Each kW of peak reduction saves $150-300/year in demand charges

Typical savings: **10-15% reduction** in total energy costs within the first year.

## Running Tests

```bash
pytest tests/ -v --tb=short
```
