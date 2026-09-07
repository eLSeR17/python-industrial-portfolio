"""WebSocket handler for real-time optimization dashboard streaming.

The WebSocket endpoint pushes periodic updates to connected clients:
    - Current process states
    - Latest optimization results
    - SPC alarms as they occur
    - OEE metrics

Clients can also send commands:
    - ``{"action": "subscribe", "process_id": "..."}`` — filter updates
    - ``{"action": "optimize", "process_id": "..."}`` — trigger optimization

Updates are broadcast at a configurable interval (default 2 seconds) to
avoid overwhelming clients with high-frequency data.
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings
from src.models.process_state import holder
from src.models.schemas import WSMessage
from src.services.optimizer import optimizer
from src.services.stream_processor import stream_processor
from src.services.waste_analyzer import waste_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages active WebSocket connections and subscriptions.

    Each connected client is tracked with an optional filter for specific
    process IDs. When no filter is set, the client receives all updates.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._subscriptions: dict[WebSocket, set[str]] = {}
        self._last_broadcast: float = 0.0

    async def connect(self, ws: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await ws.accept()
        self._connections.append(ws)
        self._subscriptions[ws] = set()
        logger.info("WebSocket connected — %d active", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a disconnected client."""
        if ws in self._connections:
            self._connections.remove(ws)
        self._subscriptions.pop(ws, None)
        logger.info("WebSocket disconnected — %d active", len(self._connections))

    def subscribe(self, ws: WebSocket, process_id: str) -> None:
        """Subscribe a client to updates for a specific process."""
        if ws in self._subscriptions:
            self._subscriptions[ws].add(process_id)

    def unsubscribe(self, ws: WebSocket, process_id: str) -> None:
        """Unsubscribe a client from a process."""
        if ws in self._subscriptions:
            self._subscriptions[ws].discard(process_id)

    def _is_subscribed(self, ws: WebSocket, process_id: str) -> bool:
        """Check if a client should receive an update for a process."""
        subs = self._subscriptions.get(ws, set())
        return len(subs) == 0 or process_id in subs

    async def broadcast(self, message: WSMessage) -> None:
        """Send a message to all connected (and subscribed) clients.

        Failed sends result in client disconnection — no retry to avoid
        cascading failures.
        """
        dead: list[WebSocket] = []
        payload = message.model_dump_json()

        for ws in self._connections:
            try:
                # Check subscription filter for process-specific messages.
                process_id = message.data.get("process_id")
                if process_id and not self._is_subscribed(ws, process_id):
                    continue
                await ws.send_text(payload)
            except Exception:
                logger.debug("Failed to send to WebSocket, marking for disconnect")
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)

    @property
    def should_broadcast(self) -> bool:
        """Rate-limit broadcasts to once per configured interval."""
        now = time.monotonic()
        if now - self._last_broadcast >= settings.ws_heartbeat_interval_seconds:
            self._last_broadcast = now
            return True
        return False


manager = ConnectionManager()


# -----------------------------------------------------------------------
# WebSocket Endpoint
# -----------------------------------------------------------------------

@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket) -> None:
    """WebSocket endpoint for the real-time optimization dashboard.

    Protocol:
        Server → Client (every N seconds):
            {"channel": "state", "data": {...}, "timestamp": "..."}
            {"channel": "optimization", "data": {...}, "timestamp": "..."}
            {"channel": "alarm", "data": {...}, "timestamp": "..."}

        Client → Server:
            {"action": "subscribe", "process_id": "reactor-01"}
            {"action": "unsubscribe", "process_id": "reactor-01"}
            {"action": "optimize", "process_id": "reactor-01"}
    """
    await manager.connect(ws)
    try:
        while True:
            # Receive client commands (non-blocking with timeout).
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                await _handle_client_command(ws, raw)
            except asyncio.TimeoutError:
                pass  # No command received — continue with broadcast cycle.

            # Periodic broadcast of current state.
            if manager.should_broadcast:
                await _broadcast_state()

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        manager.disconnect(ws)


async def _handle_client_command(ws: WebSocket, raw: str) -> None:
    """Parse and execute a client command received over WebSocket."""
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError:
        await ws.send_text(json.dumps({"error": "Invalid JSON"}))
        return

    action = cmd.get("action")
    process_id = cmd.get("process_id", "")

    if action == "subscribe" and process_id:
        manager.subscribe(ws, process_id)
        await ws.send_text(
            WSMessage(
                channel="ack",
                data={"action": "subscribe", "process_id": process_id},
            ).model_dump_json()
        )
    elif action == "unsubscribe" and process_id:
        manager.unsubscribe(ws, process_id)
        await ws.send_text(
            WSMessage(
                channel="ack",
                data={"action": "unsubscribe", "process_id": process_id},
            ).model_dump_json()
        )
    elif action == "optimize" and process_id:
        # Trigger optimization in the background.
        asyncio.create_task(_trigger_optimization_for_ws(ws, process_id))
    else:
        await ws.send_text(json.dumps({"error": f"Unknown action: {action}"}))


async def _trigger_optimization_for_ws(ws: WebSocket, process_id: str) -> None:
    """Run optimization and push the result to the requesting client."""
    try:
        state = await holder.get_state(process_id)
        if state is None:
            await ws.send_text(
                WSMessage(channel="error", data={"message": f"Process '{process_id}' not found"}).model_dump_json()
            )
            return

        current_setpoints = {v.name: v.value for v in state.variables}
        variable_names = list(current_setpoints.keys())
        variable_limits = {v.name: (v.min_limit, v.max_limit) for v in state.variables}

        result = optimizer.optimize(
            process_id=process_id,
            current_setpoints=current_setpoints,
            variable_names=variable_names,
            variable_limits=variable_limits,
        )

        await stream_processor.publish_recommendation(
            process_id=process_id,
            setpoints=result.recommended_setpoints,
        )

        msg = WSMessage(
            channel="optimization",
            data=result.model_dump(),
        )
        await ws.send_text(msg.model_dump_json())

    except Exception:
        logger.exception("Optimization failed for WS client, process=%s", process_id)
        await ws.send_text(
            WSMessage(channel="error", data={"message": "Optimization failed"}).model_dump_json()
        )


async def _broadcast_state() -> None:
    """Broadcast the current process state to all connected clients."""
    import numpy as np

    states = await holder.get_all_states()
    for state in states:
        pid = state.process_id

        # Process state.
        await manager.broadcast(
            WSMessage(
                channel="state",
                data={
                    "process_id": pid,
                    "variables": {v.name: v.value for v in state.variables},
                    "timestamp": state.timestamp.isoformat(),
                },
            )
        )

        # Quick SPC check — only broadcast if there are alarms.
        variable_data: dict[str, np.ndarray] = {}
        for var in state.variables:
            history = await holder.get_history(pid, var.name, last_n=100)
            if len(history) >= 10:
                variable_data[var.name] = np.array(
                    [h["value"] for h in history], dtype=np.float64
                )

        if variable_data:
            spc = waste_analyzer.analyze(pid, variable_data)
            for alarm in spc.alarms:
                await manager.broadcast(
                    WSMessage(
                        channel="alarm",
                        data=alarm.model_dump(),
                    )
                )
