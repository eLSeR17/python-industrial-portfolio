# Sensor Data Format Specification

## Overview

This directory stores raw and processed sensor data from industrial equipment.
The system accepts data in the format defined by the `SensorReading` Pydantic model.

## Expected Format (JSON)

```json
{
  "asset_id": "MOTOR-001",
  "asset_type": "motor",
  "timestamp": "2026-01-15T10:30:00Z",
  "vibration_x": 2.34,
  "vibration_y": 1.87,
  "vibration_z": 3.12,
  "temperature": 78.5,
  "pressure": 4.2,
  "current": 15.3,
  "rpm": 1750.0
}
```

## Field Descriptions

| Field | Type | Unit | Range | Description |
|-------|------|------|-------|-------------|
| `asset_id` | string | — | 1–64 chars | Unique equipment identifier |
| `asset_type` | enum | — | motor/pump/compressor | Equipment class |
| `timestamp` | ISO 8601 | — | — | UTC measurement time |
| `vibration_x` | float | mm/s | ±50 | Triaxial accelerometer X |
| `vibration_y` | float | mm/s | ±50 | Triaxial accelerometer Y |
| `vibration_z` | float | mm/s | ±50 | Triaxial accelerometer Z |
| `temperature` | float | °C | -40 to 300 | Surface/winding temperature |
| `pressure` | float | bar | 0–100 | Process/hydraulic pressure |
| `current` | float | Amps | 0–500 | Motor current draw |
| `rpm` | float | rev/min | 0–50000 | Rotational speed |

## Sample Rate

- **Recommended**: 100 Hz (one reading per 10ms)
- **Minimum**: 1 Hz (one reading per second)
- **Maximum**: 1000 Hz (requires edge preprocessing)

## Batch Format

For high-throughput ingestion, wrap readings in a batch:

```json
{
  "readings": [
    { ... reading 1 ... },
    { ... reading 2 ... }
  ]
}
```

Maximum batch size: 10,000 readings per request.

## Physical Plausibility

The API validates readings against physical limits:
- Vibration ±50 mm/s (values outside are sensor faults)
- Temperature -40°C to 300°C
- Pressure 0–100 bar (negative = sensor fault)
- Current 0–500 Amps

Readings outside these ranges are rejected with a validation error.
