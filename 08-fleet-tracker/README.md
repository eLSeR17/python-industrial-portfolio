# Fleet & Asset Tracker

Knowing where your equipment is and whether it's being used properly
sounds simple until you have 50 forklifts across three warehouses. This
project tracks industrial assets in real time via WebSocket, enforces
geofences, and computes utilisation metrics — all through a FastAPI
backend with a Leaflet.js map UI.

Built for the case where you need to answer "where is Forklift Alpha
right now?" and also "which assets were idle most of last week?" without
stitching together three different tools.

```
┌──────────────┐  WebSocket / REST  ┌───────────────────┐
│  Leaflet.js  │◄──────────────────►│   FastAPI (app)   │
│   Map UI     │                    │                   │
└──────────────┘                    │  ┌─────────────┐  │
                                    │  │  Services   │  │
      GPS Devices / Trackers ──────►│  │  Location   │  │
                                    │  │  Geofence   │  │
                                    │  │  Utilizatn. │  │
                                    │  │  Maintenance│  │
                                    │  │  Route      │  │
                                    │  └──────┬──────┘  │
                                    │         │         │
                                    │  ┌──────▼──────┐  │
                                    │  │  SQLite /   │  │
                                    │  │  PostgreSQL  │  │
                                    │  └─────────────┘  │
                                    └───────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Real-time | WebSocket (FastAPI native) |
| Database | SQLite (aiosqlite) / PostgreSQL |
| Geospatial | Pure-Python Haversine, Ray-casting, Douglas-Peucker |
| Frontend | Leaflet.js + OpenStreetMap |
| Config | pydantic-settings (env-based) |
| Testing | pytest + pytest-asyncio + httpx |

## Setup

### Docker

```bash
cd 08-fleet-tracker
docker compose up --build
# Open http://localhost:8000
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### Tests

```bash
python -m pytest tests/ -v
```

## API

### Assets

```bash
# Register an asset
curl -X POST http://localhost:8000/api/v1/assets \
  -H "Content-Type: application/json" \
  -d '{"id":"FL-001","name":"Forklift Alpha","asset_type":"FORKLIFT","department":"Warehouse A"}'

# List all
curl http://localhost:8000/api/v1/assets

# Single asset
curl http://localhost:8000/api/v1/assets/FL-001

# GPS history (last 48h)
curl "http://localhost:8000/api/v1/assets/FL-001/history?hours=48"
```

### Location

```bash
# Ingest a GPS update
curl -X POST http://localhost:8000/api/v1/location \
  -H "Content-Type: application/json" \
  -d '{"asset_id":"FL-001","latitude":40.7128,"longitude":-74.006,"speed_kmh":8.5}'

# Batch ingest
curl -X POST http://localhost:8000/api/v1/location/batch \
  -H "Content-Type: application/json" \
  -d '[{"asset_id":"FL-001","latitude":40.7129,"longitude":-74.0061,"speed_kmh":9.0}]'
```

### Geofences

```bash
# Create a geofence
curl -X POST http://localhost:8000/api/v1/geofences \
  -H "Content-Type: application/json" \
  -d '{"name":"Zone A","fence_type":"POLYGON","coordinates":[[40.713,-74.007],[40.713,-74.005],[40.712,-74.005],[40.712,-74.007]],"zone_type":"WAREHOUSE","alert_on_entry":true}'

# List geofences
curl http://localhost:8000/api/v1/geofences

# Geofence events for an asset
curl "http://localhost:8000/api/v1/geofences/events?asset_id=FL-001&hours=24"
```

### Utilisation & Maintenance

```bash
# Fleet utilisation for a time window
curl "http://localhost:8000/api/v1/utilization?start=2025-01-01T00:00:00Z&end=2025-01-02T00:00:00Z"

# Maintenance schedule
curl http://localhost:8000/api/v1/maintenance

# Record a completed service
curl -X POST "http://localhost:8000/api/v1/maintenance/FL-001/service?service_type=oil_change&notes=Routine"
```

### Dashboard & Routes

```bash
# Dashboard KPIs
curl http://localhost:8000/api/v1/dashboard

# Route history
curl "http://localhost:8000/api/v1/routes/FL-001?days=7"
```

### WebSocket — Real-time tracking

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tracking');
ws.onopen = () => ws.send(JSON.stringify({ subscribe: ['FL-001'] }));
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  console.log(`${data.asset_id}: ${data.latitude}, ${data.longitude}`);
};
```

## Project Structure

```
08-fleet-tracker/
├── config/            # pydantic-settings configuration
├── src/
│   ├── main.py        # FastAPI app + lifespan
│   ├── db.py          # SQLAlchemy async engine & sessions
│   ├── models/        # ORM + Pydantic schemas
│   ├── services/      # Business logic (location, geofence, utilisation, maintenance, routes)
│   ├── api/           # REST routes + WebSocket
│   ├── websocket/     # Connection manager
│   └── utils/         # Geo & time helpers (pure Python)
├── static/            # Leaflet.js map UI
├── tests/             # Async test suite
├── data/              # Sample geofences
└── docker-compose.yml
```

## License

Private — for demonstration purposes only.
