"""Real-time Kafka stream processor for IoT sensor data.

This module consumes sensor readings from Kafka topics, validates them,
and feeds them into the ProcessStateHolder. It also publishes optimization
recommendations back to Kafka for downstream consumers (PLC controllers,
historians, dashboards).

Architecture notes:
    - Uses aiokafka for non-blocking async consumption.
    - Consumer runs in a background asyncio task started by ``StreamProcessor.start()``.
    - Backpressure is handled by ``max_poll_records`` and internal buffering.
    - A separate producer publishes optimization results to the recommendations topic.
"""

import asyncio
import json
import logging
import time
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.settings import settings
from src.models.process_state import holder
from src.models.schemas import ProcessUpdateRequest, SensorReading

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Async Kafka consumer and producer for the process optimizer.

    Lifecycle:
        1. ``await processor.start()`` — creates the consumer and producer,
           subscribes to the sensor topic.
        2. The ``_consume_loop`` runs in the background, processing messages
           until ``await processor.stop()`` is called.
        3. ``await processor.stop()`` — commits offsets and closes connections.

    The processor is designed to run as a long-lived background task alongside
    the FastAPI server.
    """

    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._messages_processed = 0
        self._errors: list[str] = []
        self._last_message_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize the Kafka consumer and producer, then start the consume loop."""
        if self._running:
            logger.warning("StreamProcessor is already running")
            return

        try:
            self._consumer = AIOKafkaConsumer(
                settings.kafka_sensor_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group,
                auto_offset_reset=settings.kafka_auto_offset_reset,
                enable_auto_commit=False,
                max_poll_records=settings.kafka_max_poll_records,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
            await self._consumer.start()

            self._producer = AIOKafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()

            self._running = True
            self._task = asyncio.create_task(self._consume_loop(), name="kafka-consume-loop")
            logger.info(
                "StreamProcessor started — consuming from '%s', producing to '%s'",
                settings.kafka_sensor_topic,
                settings.kafka_optimization_topic,
            )
        except Exception:
            logger.exception("Failed to start StreamProcessor")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully shut down the consumer and producer."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

        logger.info(
            "StreamProcessor stopped — processed %d messages",
            self._messages_processed,
        )

    # ------------------------------------------------------------------
    # Consume Loop
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """Main consume loop — reads messages from Kafka and processes them.

        Messages are expected to have the structure:
            {
                "process_id": str,
                "readings": [{"sensor_id": str, "value": float, "unit": str, ...}],
                "setpoints": {"var_name": float, ...},
                "process_type": str  # optional, defaults to chemical_reactor
            }
        """
        logger.info("Consume loop started")
        assert self._consumer is not None

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                try:
                    payload = msg.value
                    if not isinstance(payload, dict):
                        logger.warning("Ignoring non-dict message on offset %d", msg.offset)
                        continue

                    await self._process_message(payload)
                    self._messages_processed += 1
                    self._last_message_time = time.monotonic()

                except Exception:
                    logger.exception("Error processing message at offset %d", msg.offset)
                    self._errors.append(f"offset={msg.offset}")
                    # Keep processing — one bad message should not crash the pipeline.

                # Commit offsets periodically (every message for correctness;
                # in production, batch commits are more efficient).
                await self._consumer.commit()

        except asyncio.CancelledError:
            logger.info("Consume loop cancelled")
        except Exception:
            logger.exception("Consume loop crashed unexpectedly")
            self._running = False

    async def _process_message(self, payload: dict[str, Any]) -> None:
        """Validate and apply a single sensor message to the process state.

        Args:
            payload: Decoded JSON message from Kafka.
        """
        process_id = payload.get("process_id")
        if not process_id:
            logger.warning("Message missing process_id, skipping")
            return

        raw_readings = payload.get("readings", [])
        if not raw_readings:
            logger.debug("Empty readings for process %s, skipping", process_id)
            return

        # Build Pydantic models with validation.
        readings: list[SensorReading] = []
        for raw in raw_readings:
            try:
                readings.append(
                    SensorReading(
                        sensor_id=raw["sensor_id"],
                        process_id=process_id,
                        value=float(raw["value"]),
                        unit=raw.get("unit", ""),
                        quality=float(raw.get("quality", 1.0)),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("Invalid reading in process %s: %s", process_id, e)

        if not readings:
            return

        # Determine process type.
        from src.models.schemas import ProcessType

        try:
            process_type = ProcessType(payload.get("process_type", "chemical_reactor"))
        except ValueError:
            process_type = ProcessType.CHEMICAL_REACTOR

        request = ProcessUpdateRequest(
            process_id=process_id,
            process_type=process_type,
            readings=readings,
            setpoints=payload.get("setpoints", {}),
        )

        # Apply to the state holder.
        state = await holder.update(request)
        logger.debug(
            "Updated process %s: %d variables, %d total readings",
            process_id,
            len(state.variables),
            self._messages_processed,
        )

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    async def publish_recommendation(
        self,
        process_id: str,
        setpoints: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish an optimization recommendation to Kafka.

        The recommendation is consumed by PLC controllers or historians
        that apply the new setpoints to the physical process.

        Args:
            process_id: Target process line.
            setpoints: Recommended variable values.
            metadata: Optional extra fields (objective_value, method, etc.).
        """
        if self._producer is None:
            logger.warning("Producer not initialized — cannot publish recommendation")
            return

        message = {
            "process_id": process_id,
            "setpoints": setpoints,
            "source": "process-optimizer",
            "timestamp": time.time(),
        }
        if metadata:
            message["metadata"] = metadata

        try:
            await self._producer.send_and_wait(
                topic=settings.kafka_optimization_topic,
                key=process_id.encode("utf-8"),
                value=message,
            )
            logger.info("Published recommendation for %s: %s", process_id, setpoints)
        except Exception:
            logger.exception("Failed to publish recommendation for %s", process_id)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def messages_processed(self) -> int:
        return self._messages_processed

    @property
    def last_message_age_seconds(self) -> float:
        """Seconds since the last message was processed. 0 if no messages yet."""
        if self._last_message_time == 0:
            return 0.0
        return time.monotonic() - self._last_message_time

    @property
    def recent_errors(self) -> list[str]:
        return self._errors[-20:]

    def status(self) -> dict[str, Any]:
        """Return a summary of the processor state for the dashboard."""
        return {
            "running": self._running,
            "messages_processed": self._messages_processed,
            "last_message_age_seconds": round(self.last_message_age_seconds, 1),
            "recent_errors": len(self._errors),
            "topic": settings.kafka_sensor_topic,
            "group_id": settings.kafka_consumer_group,
        }


# Module-level singleton.
stream_processor = StreamProcessor()
