# 05 — Vision Inspector

Quality inspection on manufacturing lines is tedious, error-prone, and
expensive. Inspectors catch maybe 85% of defects on a good day — fatigue
sets in after a few hours and standards drift between shifts. This project
builds a FastAPI service that takes camera frames, runs them through a
detection pipeline (OpenCV classical CV or YOLOv8), classifies severity,
and tracks quality metrics over time.

The goal was to have something that could sit between a camera and an
alert system, with enough SPC (Statistical Process Control) built in to
be useful beyond just "defect / no defect".

```
┌──────────────────────────────────────────────────────────────────┐
│                    Vision Inspector Pipeline                      │
├──────────┬──────────┬────────────┬──────────────┬───────────────┤
│  Camera  │  FastAPI │  OpenCV /  │  Severity    │  Statistics   │
│  Frame   │  intake  │  YOLOv8    │  Classifier  │  & SPC        │
│ ──────── │ ──────── │ ────────── │ ──────────── │ ───────────── │
│  Base64  │  preprocess → detect → classify    →  record &      │
│  image   │  resize   │ contour   │  rules-based │  alert        │
│          │  enhance  │ analysis  │  thresholds  │  C-chart      │
│          │           │ /YOLO     │              │  Pareto       │
└──────────┴──────────┴────────────┴──────────────┴───────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Computer Vision | OpenCV (CLAHE, Canny, contour analysis, blob detection) |
| Deep Learning | YOLOv8 (ultralytics) — optional |
| Numerics | NumPy |
| Modelling | Pydantic v2 + pydantic-settings |
| Storage | In-memory (Redis-ready) |
| Container | Docker + docker-compose |

## Setup

### Docker

```bash
docker compose up --build
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `POST /api/v1/inspect` — Send an image for inspection

```bash
# Generate a synthetic test image and send it
python -c "
import cv2, base64, json, requests
import numpy as np
img = np.full((480, 640, 3), 180, dtype=np.uint8)
cv2.ellipse(img, (320, 240), (40, 25), 0, 0, 360, (100, 100, 100), -1)
_, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf).decode()
r = requests.post('http://localhost:8000/api/v1/inspect', json={
    'image': b64, 'line_id': 'LINE-A', 'product_type': 'panel'
})
print(json.dumps(r.json(), indent=2))
"
```

### `GET /api/v1/statistics/{line_id}` — Quality stats for a line

```bash
curl http://localhost:8000/api/v1/statistics/LINE-A?period_minutes=60
```

### `GET /api/v1/statistics/{line_id}/pareto` — Pareto breakdown of defect types

```bash
curl http://localhost:8000/api/v1/statistics/LINE-A/pareto
```

### `GET /api/v1/alerts/{line_id}` — Active quality alerts

```bash
curl http://localhost:8000/api/v1/alerts/LINE-A
```

### `GET /api/v1/defect-types` — List of defect types

```bash
curl http://localhost:8000/api/v1/defect-types
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

## Project Structure

```
05-vision-inspector/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── config/
│   ├── __init__.py
│   └── settings.py              # pydantic-settings (VISION_ env prefix)
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app + lifespan + CORS
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # DefectType, SeverityLevel, request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processor.py   # Preprocessing, ROI, enhance, augment
│   │   ├── defect_detector.py   # YOLOv8 + classical CV fallback
│   │   ├── defect_classifier.py # Severity rules + shape → type
│   │   ├── annotation_service.py# Draw boxes, labels, stats overlay
│   │   └── statistics_service.py# Pareto, SPC, C-chart, alerts
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # All REST endpoints
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py       # Base64, histograms, synthetic images
├── tests/
│   ├── test_image_processor.py
│   ├── test_defect_detector.py
│   └── test_statistics_service.py
├── models/                      # YOLOv8 .pt weights (git-ignored)
│   └── README.md
└── data/                        # Runtime data (git-ignored)
    └── README.md
```

## Tests

All tests run on synthetic images generated with NumPy and OpenCV — no
external files or model weights needed.

```bash
python -m pytest tests/ -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_image_processor.py` | Resize, normalise, colour conversion, ROI, CLAHE enhance, augment, JPEG/PNG decode |
| `test_defect_detector.py` | Classical detection on drawn shapes, classify_defect geometry, clean-image rejection, severity sorting |
| `test_statistics_service.py` | Record/query lifecycle, Pareto sorting, C-chart math, trend direction, alert thresholds |

## Configuration

Env vars use the `VISION_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL_PATH` | `None` | Path to YOLOv8 `.pt` weights |
| `VISION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `VISION_DEBUG` | `false` | Enable debug logging |
| `VISION_HOST` | `0.0.0.0` | Bind address |
| `VISION_PORT` | `8000` | Bind port |
| `VISION_MAX_IMAGE_SIZE_MB` | `10` | Maximum upload size |
| `VISION_REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `VISION_ALERT_THRESHOLD_PCT` | `5.0` | Defect rate % that triggers alerts |
