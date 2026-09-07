"""MQTT bridge for publishing simulation state to shop-floor SCADA systems.

All MQTT operations are optional: if the broker is unavailable the bridge
degrades silently and logs a warning.
"""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Lazy import – paho.mqtt may not be installed in test environments
_mqtt_client_mod = None


def _get_mqtt():  # type: ignore[no-untyped-def]
    """Return the ``paho.mqtt.client`` module, importing lazily."""
    global _mqtt_client_mod  # noqa: PLW0603
    if _mqtt_client_mod is None:
        import paho.mqtt.client as mqtt  # type: ignore[import-untyped]
        _mqtt_client_mod = mqtt
    return _mqtt_client_mod


class MQTTBridge:
    """Publish/subscribe bridge between the simulation and an MQTT broker.

    All methods catch connection errors and log warnings so the simulation
    is never blocked by a missing broker.

    Args:
        broker: Hostname or IP of the MQTT broker.
        port: Port of the MQTT broker.
    """

    def __init__(self, broker: str = "localhost", port: int = 1883) -> None:
        self.broker = broker
        self.port = port
        self._client: Any = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the MQTT broker.

        Falls back to no-op if the broker is unreachable.
        """
        try:
            mqtt_mod = _get_mqtt()
            # paho-mqtt 2.x requires callback_api_version
            try:
                self._client = mqtt_mod.Client(
                    callback_api_version=mqtt_mod.CallbackAPIVersion.VERSION2,
                    client_id="digital-twin-sim",
                    protocol=mqtt_mod.MQTTv311,
                )
            except (AttributeError, TypeError):
                # Fallback for paho-mqtt 1.x
                self._client = mqtt_mod.Client(
                    client_id="digital-twin-sim", protocol=mqtt_mod.MQTTv311
                )
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            self._connected = True
            logger.info("MQTT connected to %s:%d", self.broker, self.port)
        except Exception as exc:
            logger.warning("MQTT connection failed (degraded mode): %s", exc)
            self._connected = False

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        if self._client and self._connected:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False

    def publish_simulation_state(self, sim_id: str, state: dict[str, Any]) -> None:
        """Publish the current simulation state.

        Topic: ``sim/{sim_id}/state``

        Args:
            sim_id: Simulation identifier.
            state: JSON-serialisable state dict.
        """
        self._safe_publish(f"sim/{sim_id}/state", state)

    def publish_metrics(self, sim_id: str, metrics: dict[str, Any]) -> None:
        """Publish aggregated simulation metrics.

        Topic: ``sim/{sim_id}/metrics``

        Args:
            sim_id: Simulation identifier.
            metrics: JSON-serialisable metrics dict.
        """
        self._safe_publish(f"sim/{sim_id}/metrics", metrics)

    def publish_machine_status(
        self, sim_id: str, machine_id: str, status: str
    ) -> None:
        """Publish a machine status change.

        Topic: ``sim/{sim_id}/machine/{machine_id}/status``

        Args:
            sim_id: Simulation identifier.
            machine_id: Machine identifier.
            status: New status string.
        """
        self._safe_publish(
            f"sim/{sim_id}/machine/{machine_id}/status",
            {"status": "status", "machine_id": machine_id},
        )

    def subscribe_to_commands(self, callback: Callable[[str, dict], None]) -> None:
        """Subscribe to parameter-change commands.

        Topic: ``sim/+/commands``

        Args:
            callback: ``(topic, payload_dict) -> None`` called for each message.
        """
        if not self._client or not self._connected:
            logger.warning("Cannot subscribe: MQTT not connected.")
            return
        try:
            self._callback = callback
            self._client.subscribe("sim/+/commands")
            self._client.on_message = self._on_message  # type: ignore[assignment]
        except Exception as exc:
            logger.warning("MQTT subscribe failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish JSON payload, swallowing errors."""
        if not self._client or not self._connected:
            logger.debug("MQTT not connected, skipping publish to %s", topic)
            return
        try:
            self._client.publish(topic, json.dumps(payload, default=str))
        except Exception as exc:
            logger.warning("MQTT publish failed for %s: %s", topic, exc)

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:  # noqa: ARG002
        """Callback when the broker accepts the connection."""
        if rc == 0:
            logger.info("MQTT on_connect: rc=%d", rc)
        else:
            logger.warning("MQTT on_connect: non-zero rc=%d", rc)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:  # noqa: ARG002
        """Callback for incoming messages (dispatches to registered handler)."""
        try:
            payload = json.loads(msg.payload.decode())
            if hasattr(self, "_callback") and self._callback is not None:  # type: ignore[attr-defined]
                self._callback(msg.topic, payload)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Failed to process MQTT message on %s: %s", msg.topic, exc)
