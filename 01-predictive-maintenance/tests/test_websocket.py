"""Tests for the WebSocket ConnectionManager.

Verifies connection tracking, multi-client broadcast, disconnect
cleanup, and the total_connections counter. All tests mock the
WebSocket object — no real network I/O.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.websocket import ConnectionManager, manager


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_mock_ws() -> MagicMock:
    """Create a mock WebSocket that simulates send_text and accept."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.accept = AsyncMock()
    return ws


def _make_failing_ws() -> MagicMock:
    """Create a mock WebSocket that raises on send_text but accepts."""
    ws = MagicMock()
    ws.send_text = AsyncMock(side_effect=ConnectionError("broken pipe"))
    ws.accept = AsyncMock()
    return ws


# ── Connection tracking tests ───────────────────────────────────────────


class TestConnectionTracking:
    """Tests for connect/disconnect and connection bookkeeping."""

    @pytest.mark.asyncio
    async def test_connect_increments_count(self) -> None:
        """Connecting a client should increase total_connections."""
        mgr = ConnectionManager()
        ws = _make_mock_ws()
        await mgr.connect(ws, "MOTOR-001")
        assert mgr.total_connections == 1

    @pytest.mark.asyncio
    async def test_multiple_clients_same_asset(self) -> None:
        """Multiple clients for the same asset should all be tracked."""
        mgr = ConnectionManager()
        for _ in range(3):
            await mgr.connect(_make_mock_ws(), "MOTOR-001")
        assert mgr.total_connections == 3

    @pytest.mark.asyncio
    async def test_multiple_assets(self) -> None:
        """Connections to different assets should be tracked independently."""
        mgr = ConnectionManager()
        await mgr.connect(_make_mock_ws(), "MOTOR-001")
        await mgr.connect(_make_mock_ws(), "PUMP-002")
        assert mgr.total_connections == 2

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self) -> None:
        """Disconnecting a client should remove it from tracking."""
        mgr = ConnectionManager()
        ws = _make_mock_ws()
        await mgr.connect(ws, "MOTOR-001")
        await mgr.disconnect(ws, "MOTOR-001")
        assert mgr.total_connections == 0

    @pytest.mark.asyncio
    async def test_disconnect_only_removes_target(self) -> None:
        """Disconnecting one client should not affect others on the same asset."""
        mgr = ConnectionManager()
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        await mgr.connect(ws1, "MOTOR-001")
        await mgr.connect(ws2, "MOTOR-001")
        await mgr.disconnect(ws1, "MOTOR-001")
        assert mgr.total_connections == 1

    @pytest.mark.asyncio
    async def test_disconnect_unknown_ws_no_error(self) -> None:
        """Disconnecting a WebSocket that was never connected should be safe."""
        mgr = ConnectionManager()
        ws = _make_mock_ws()
        # Should not raise
        await mgr.disconnect(ws, "MOTOR-001")
        assert mgr.total_connections == 0

    @pytest.mark.asyncio
    async def test_disconnect_cleans_empty_asset_entry(self) -> None:
        """When the last client disconnects, the asset key should be removed."""
        mgr = ConnectionManager()
        ws = _make_mock_ws()
        await mgr.connect(ws, "MOTOR-001")
        await mgr.disconnect(ws, "MOTOR-001")
        assert "MOTOR-001" not in mgr._connections


# ── Broadcast tests ─────────────────────────────────────────────────────


class TestBroadcast:
    """Tests for broadcast_to_asset and broadcast_all."""

    @pytest.mark.asyncio
    async def test_broadcast_reaches_all_subscribers(self) -> None:
        """All clients subscribed to an asset should receive the broadcast."""
        mgr = ConnectionManager()
        wss = [_make_mock_ws() for _ in range(3)]
        for ws in wss:
            await mgr.connect(ws, "MOTOR-001")

        data = {"type": "sensor_update", "vibration_x": 2.5}
        await mgr.broadcast_to_asset("MOTOR-001", data)

        for ws in wss:
            ws.send_text.assert_awaited_once()
            sent = json.loads(ws.send_text.call_args[0][0])
            assert sent["type"] == "sensor_update"

    @pytest.mark.asyncio
    async def test_broadcast_empty_asset_no_error(self) -> None:
        """Broadcasting to an asset with no subscribers should not error."""
        mgr = ConnectionManager()
        await mgr.broadcast_to_asset("NONEXISTENT", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_all_reaches_everyone(self) -> None:
        """broadcast_all should send to every connected client."""
        mgr = ConnectionManager()
        ws1 = _make_mock_ws()
        ws2 = _make_mock_ws()
        await mgr.connect(ws1, "MOTOR-001")
        await mgr.connect(ws2, "PUMP-002")

        await mgr.broadcast_all({"type": "global_alert", "message": "test"})

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_serializes_datetime_as_string(self) -> None:
        """Default=str serializer should handle datetime objects in data."""
        from datetime import datetime, timezone

        mgr = ConnectionManager()
        ws = _make_mock_ws()
        await mgr.connect(ws, "MOTOR-001")

        data = {"timestamp": datetime.now(timezone.utc)}
        await mgr.broadcast_to_asset("MOTOR-001", data)
        ws.send_text.assert_awaited_once()
        # Should not raise JSON serialization error
        json.loads(ws.send_text.call_args[0][0])


# ── Disconnect cleanup on failed send ───────────────────────────────────


class TestFailedSendCleanup:
    """Tests that failed sends automatically disconnect the client."""

    @pytest.mark.asyncio
    async def test_failing_client_removed_after_broadcast(self) -> None:
        """A client that fails during broadcast should be disconnected."""
        mgr = ConnectionManager()
        good_ws = _make_mock_ws()
        bad_ws = _make_failing_ws()
        await mgr.connect(good_ws, "MOTOR-001")
        await mgr.connect(bad_ws, "MOTOR-001")
        assert mgr.total_connections == 2

        await mgr.broadcast_to_asset("MOTOR-001", {"type": "test"})

        # bad_ws should have been removed
        assert mgr.total_connections == 1
        good_ws.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_clients_fail_cleans_asset(self) -> None:
        """If all clients for an asset fail, the asset entry is removed."""
        mgr = ConnectionManager()
        bad_ws = _make_failing_ws()
        await mgr.connect(bad_ws, "MOTOR-001")

        await mgr.broadcast_to_asset("MOTOR-001", {"type": "test"})
        assert mgr.total_connections == 0
        assert "MOTOR-001" not in mgr._connections
