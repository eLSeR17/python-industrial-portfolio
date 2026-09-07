"""WebSocket endpoint for real-time dashboard streaming.

Provides a WebSocket interface for live sensor data visualization.
Clients subscribe to an asset ID and receive a stream of:
- Raw sensor readings (compressed to essential fields)
- Computed health scores
- Active alerts
- Failure predictions

Designed for industrial dashboard displays on the plant floor.
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections grouped by asset ID.

    Supports multiple dashboard clients subscribing to the same asset.
    When new data arrives, it's broadcast to all connected clients
    for that asset.
    """

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, asset_id: str) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if asset_id not in self._connections:
                self._connections[asset_id] = []
            self._connections[asset_id].append(websocket)
        logger.info("WebSocket connected for asset %s (total: %d)",
                     asset_id, len(self._connections.get(asset_id, [])))

    async def disconnect(self, websocket: WebSocket, asset_id: str) -> None:
        """Remove a disconnected WebSocket."""
        async with self._lock:
            conns = self._connections.get(asset_id, [])
            if websocket in conns:
                conns.remove(websocket)
            if not conns and asset_id in self._connections:
                del self._connections[asset_id]
        logger.info("WebSocket disconnected for asset %s", asset_id)

    async def broadcast_to_asset(self, asset_id: str, data: dict[str, Any]) -> None:
        """Send data to all clients subscribed to an asset.

        Failed connections are automatically cleaned up to prevent
        memory leaks from zombie sockets.
        """
        conns = self._connections.get(asset_id, [])
        if not conns:
            return

        message = json.dumps(data, default=str)
        disconnected: list[WebSocket] = []

        for ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        # Clean up failed connections
        for ws in disconnected:
            await self.disconnect(ws, asset_id)

    async def broadcast_all(self, data: dict[str, Any]) -> None:
        """Send data to all connected clients (for global events)."""
        all_asset_ids = list(self._connections.keys())
        for asset_id in all_asset_ids:
            await self.broadcast_to_asset(asset_id, data)

    @property
    def total_connections(self) -> int:
        """Return total number of active connections across all assets."""
        return sum(len(conns) for conns in self._connections.values())


# Global connection manager instance
manager = ConnectionManager()


@ws_router.websocket("/ws/{asset_id}")
async def websocket_endpoint(websocket: WebSocket, asset_id: str) -> None:
    """WebSocket endpoint for real-time asset monitoring.

    Clients connect to `/ws/{asset_id}` to receive a live stream of
    sensor data, health scores, and alerts for the specified asset.

    Protocol: JSON messages, one per update cycle. Each message is a
    JSON object with fields such as type (e.g. "sensor_update"),
    asset_id, timestamp, vibration (x/y/z axes), temperature, pressure,
    health_score, and an alerts array. An HMI display on the plant
    floor can show live vibration waveforms and health gauges for
    critical motors from this stream.

    Args:
        websocket: FastAPI WebSocket instance.
        asset_id: Asset identifier to subscribe to.
    """
    await manager.connect(websocket, asset_id)
    try:
        while True:
            # Keep connection alive; client can send config messages
            data = await websocket.receive_text()
            try:
                config = json.loads(data)
                # Handle client configuration (e.g., update interval)
                if config.get("type") == "config":
                    await websocket.send_text(json.dumps({
                        "type": "config_ack",
                        "asset_id": asset_id,
                        "status": "subscribed",
                    }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON. Expected format: {\"type\": \"config\", ...}",
                }))
    except WebSocketDisconnect:
        await manager.disconnect(websocket, asset_id)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", asset_id, e)
        await manager.disconnect(websocket, asset_id)


async def push_sensor_update(
    asset_id: str,
    reading: dict[str, Any],
    health_score: float | None = None,
    alerts: list[dict[str, Any]] | None = None,
) -> None:
    """Push a sensor update to all connected dashboard clients.

    Called by the data processing pipeline after each reading is
    ingested and analyzed. The update includes the raw reading
    plus any computed health scores and active alerts.

    Args:
        asset_id: Asset that generated the data.
        reading: Serialized sensor reading.
        health_score: Latest health score (if computed).
        alerts: Any new alerts triggered by this reading.
    """
    from src.utils.helpers import utc_now

    message = {
        "type": "sensor_update",
        "asset_id": asset_id,
        "timestamp": utc_now().isoformat(),
        "vibration": {
            "x": reading.get("vibration_x", 0),
            "y": reading.get("vibration_y", 0),
            "z": reading.get("vibration_z", 0),
        },
        "temperature": reading.get("temperature", 0),
        "pressure": reading.get("pressure", 0),
        "current": reading.get("current", 0),
        "rpm": reading.get("rpm", 0),
        "health_score": health_score,
        "alerts": alerts or [],
    }
    await manager.broadcast_to_asset(asset_id, message)


async def push_alert_notification(alert: dict[str, Any]) -> None:
    """Push an alert notification to all connected clients.

    Alerts are broadcast globally since operators may need to see
    alerts across all assets from a single dashboard.
    """
    message = {
        "type": "alert",
        "data": alert,
    }
    await manager.broadcast_all(message)
