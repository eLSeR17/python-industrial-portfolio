# Supply Chain Cost Analyzer & Optimizer

Supply chain typically eats 60–80% of COGS in manufacturing and retail. When margins are thin, that's where the money is. This tool gives you the standard optimization toolkit — network design, vehicle routing, inventory sizing, supplier scoring — wrapped in a REST API you can actually integrate into existing systems.

## What it covers

- **Network optimization** — facility location + flow assignment via MIP
- **Route optimization** — capacitated VRP with time windows
- **Inventory optimization** — EOQ, safety stock, multi-echelon bullwhip analysis
- **Supplier scoring** — PROMETHEE-style weighted criteria with risk index
- **Demand forecasting** — exponential smoothing, seasonal decomposition, auto-selection
- **Cost analysis** — total landed cost breakdown with sensitivity analysis
- **What-if simulation** — scenario planning with parameter perturbation

## Architecture

```
FastAPI (async) ──► Service Layer ──► NetworkX / PuLP / NumPy
                       │
                       ├── PostgreSQL (audit, historical data)
                       └── Redis (caching, session state)
```

## Quick Start

```bash
# Local development
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d

# Tests
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/network/create` | Build supply chain graph from nodes and edges |
| GET | `/optimize/route` | Optimize delivery routes (VRP) |
| GET | `/optimize/inventory` | Compute optimal inventory levels |
| GET | `/analyze/supplier` | Multi-criteria supplier scoring |
| GET | `/cost-breakdown` | Full landed cost decomposition |
| POST | `/what-if` | Scenario simulation with parameter changes |
| POST | `/forecast/demand` | Demand time-series forecasting |
| POST | `/analyze/supplier` | Score and rank suppliers |
| POST | `/cost-breakdown` | Full landed cost decomposition |
| POST | `/optimize/bullwhip` | Multi-echelon bullwhip analysis |

### Example Requests

```bash
# Create a supply chain network
curl -X POST http://localhost:8000/api/v1/network/create \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "SUP-1", "name": "Supplier A", "type": "supplier", "capacity": 5000, "variable_cost": 2.5},
      {"id": "WH-1", "name": "Warehouse", "type": "warehouse", "capacity": 8000, "fixed_cost": 5000},
      {"id": "CUS-1", "name": "Customer 1", "type": "customer", "demand": 1500}
    ],
    "edges": [
      {"source": "SUP-1", "target": "WH-1", "distance_km": 200, "cost_per_unit": 4.0, "transit_time_hours": 3, "capacity": 5000},
      {"source": "WH-1", "target": "CUS-1", "distance_km": 150, "cost_per_unit": 3.0, "transit_time_hours": 2, "capacity": 3000}
    ]
  }'

# Optimize inventory (EOQ + safety stock)
curl "http://localhost:8000/api/v1/optimize/inventory?annual_demand=10000&unit_cost=25&ordering_cost=75&holding_cost_pct=0.25&lead_time_days=14&demand_std_dev=5&service_level=0.95"

# Score suppliers
curl -X POST http://localhost:8000/api/v1/analyze/supplier \
  -H "Content-Type: application/json" \
  -d '{
    "suppliers": [
      {"supplier_id": "S1", "name": "Alpha Corp", "unit_price": 10, "defect_rate_ppm": 50, "lead_time_days": 7, "on_time_delivery_pct": 95},
      {"supplier_id": "S2", "name": "Beta Inc", "unit_price": 8, "defect_rate_ppm": 200, "lead_time_days": 12, "on_time_delivery_pct": 85}
    ]
  }'

# Landed cost breakdown
curl -X POST http://localhost:8000/api/v1/cost-breakdown \
  -H "Content-Type: application/json" \
  -d '{"material_cost": 5000, "transport_cost": 1200, "duty_cost": 250, "handling_cost": 100, "storage_cost": 50, "insurance_cost": 75, "quantity": 500, "markup_pct": 0.10}'

# Demand forecast (auto-select best method)
curl -X POST "http://localhost:8000/api/v1/forecast/demand?method=auto&periods_ahead=6" \
  -H "Content-Type: application/json" \
  -d '[120, 135, 140, 128, 150, 155, 142, 160, 170, 165, 180, 175, 190, 185, 200, 195]'

# What-if simulation
curl -X POST http://localhost:8000/api/v1/what-if \
  -H "Content-Type: application/json" \
  -d '{"parameters": [{"parameter": "fuel_cost", "factor": 1.30, "description": "30% fuel increase"}], "baseline_cost": 100000}'
```

## Tech Stack

- **Python 3.12** — type-hinted throughout
- **FastAPI** — async REST API with OpenAPI docs
- **NetworkX** — graph-based supply chain modeling
- **PuLP** — linear/mixed-integer programming (facility location, VRP, transportation)
- **NumPy/Pandas** — numerical computation and time series
- **PostgreSQL** — persistent storage
- **Redis** — caching layer
- **Docker Compose** — containerized deployment

## Cost Optimization Methods

### Network Design
Mixed-integer programming for facility location: minimize fixed + variable costs subject to capacity, demand, and service level constraints.

### Vehicle Routing
Capacitated VRP with time windows solved via integer programming. Supports fleet heterogeneous fleets and multi-depot scenarios.

### Inventory Management
Economic Order Quantity (EOQ) with demand uncertainty. Safety stock calculated for target service levels (95-99%). Multi-echelon analysis quantifies the bullwhip effect across supply chain tiers.

### Supplier Scoring
PROMETHEE-inspired weighted scoring: price competitiveness, quality (PPM defects), lead time, on-time delivery reliability, ESG risk index. Sensitivity analysis shows score robustness.

## Project Structure

```
├── config/settings.py          # Environment and app configuration
├── src/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── network.py          # Domain model for supply chain graph
│   ├── services/
│   │   ├── network_builder.py  # Construct supply chain graph
│   │   ├── cost_analyzer.py    # Landed cost, TCO, breakdown trees
│   │   ├── route_optimizer.py  # VRP, shortest path, fleet routing
│   │   ├── inventory_optimizer.py  # EOQ, safety stock, multi-echelon
│   │   ├── supplier_scorer.py  # Weighted scoring, risk analysis
│   │   └── demand_forecaster.py    # Time series forecasting
│   ├── api/routes.py           # FastAPI endpoint definitions
│   └── utils/
│       ├── graph_utils.py      # Centrality, bottleneck, critical path
│       └── cost_models.py      # Cost functions, sensitivity analysis
└── tests/                      # pytest test suite
```
