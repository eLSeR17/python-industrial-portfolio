"""WebSocket connection manager for real-time asset tracking."""

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and per-asset subscriptions."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}  # asset_id -> {client_ids}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new WebSocket connection and register it."""
        await websocket.accept()
        self._connections[client_id] = websocket
        logger.info("Client %s connected (%d total)", client_id, len(self._connections))

    def disconnect(self, client_id: str) -> None:
        """Remove a client and clean up its subscriptions."""
        self._connections.pop(client_id, None)
        for asset_subs in self._subscriptions.values():
            asset_subs.discard(client_id)
        logger.info("Client %s disconnected (%d remaining)", client_id, len(self._connections))

    async def broadcast_to_asset(self, asset_id: str, data: dict[str, Any]) -> None:
        """Send *data* to every client subscribed to *asset_id*."""
        subscribers = self._subscriptions.get(asset_id, set())
        payload = json.dumps(data)
        dead: list[str] = []
        for cid in subscribers:
            ws = self._connections.get(cid)
            if ws is None:
                dead.append(cid)
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    async def broadcast_all(self, data: dict[str, Any]) -> None:
        """Send *data* to every connected client."""
        payload = json.dumps(data)
        dead: list[str] = []
        for cid, ws in self._connections.items():
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    def get_connected_count(self) -> int:
        """Return the number of active connections."""
        return len(self._connections)

    def subscribe_to_assets(self, client_id: str, asset_ids: list[str]) -> None:
        """Register *client_id* to receive updates for *asset_ids*."""
        for aid in asset_ids:
            self._subscriptions.setdefault(aid, set()).add(client_id)
