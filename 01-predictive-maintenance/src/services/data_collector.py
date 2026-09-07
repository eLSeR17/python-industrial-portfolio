"""Sensor data collector and simulator for industrial equipment.

Generates realistic multi-sensor data streams that mimic real-world
degradation patterns for motors, pumps, and compressors. Includes:

- Baseline noise modeling (Gaussian sensor noise)
- Degradation signal injection (bearing wear, thermal drift)
- Intermittent fault simulation (stuck sensors, spikes)
- Configurable sample rates and asset inventories

The simulator produces data indistinguishable from real industrial
sensor feeds, enabling full-pipeline testing without hardware.
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.models.schemas import AssetType, SensorReading
from src.utils.helpers import degradation_curve, utc_now


@dataclass
class AssetState:
    """Internal state tracking for a simulated asset.

    Each asset maintains its own degradation trajectory and noise
    characteristics to produce realistic per-unit variation.
    """

    asset_id: str
    asset_type: AssetType
    operating_hours: float = 0.0
    degradation_start_hour: float = 72.0
    failure_hour: float = 96.0
    base_vibration: float = 1.5
    base_temperature: float = 65.0
    base_pressure: float = 4.0
    base_current: float = 12.0
    rated_rpm: float = 1750.0
    noise_level: float = 0.15
    fault_active: bool = False
    fault_type: str | None = None
    reading_count: int = 0

    # Randomized per-asset variation (set at creation)
    temp_offset: float = field(default_factory=lambda: random.gauss(0, 3))
    vibration_offset: float = field(default_factory=lambda: random.gauss(0, 0.3))
    pressure_offset: float = field(default_factory=lambda: random.gauss(0, 0.2))


class DataCollector:
    """Manages sensor data collection and simulation for a fleet of assets.

    Usage:
        collector = DataCollector()
        collector.register_asset("MOTOR-001", AssetType.MOTOR)
        reading = collector.collect_reading("MOTOR-001")
    """

    def __init__(self, sample_rate_hz: float = 100.0) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._assets: dict[str, AssetState] = {}
        self._history: dict[str, list[SensorReading]] = defaultdict(list)
        self._max_history_per_asset: int = 10000

    @property
    def asset_ids(self) -> list[str]:
        """Return IDs of all registered assets."""
        return list(self._assets.keys())

    def register_asset(
        self,
        asset_id: str,
        asset_type: AssetType = AssetType.MOTOR,
        degradation_start_hour: float = 72.0,
        failure_hour: float = 96.0,
    ) -> None:
        """Register a new asset for monitoring.

        Each asset gets randomized baseline parameters to simulate
        manufacturing variation between identical equipment models.

        Args:
            asset_id: Unique identifier (e.g., "MOTOR-001").
            asset_type: Equipment class (motor/pump/compressor).
            degradation_start_hour: Operating hour degradation begins.
            failure_hour: Expected failure hour for simulation.
        """
        # Per-type base parameters
        type_params: dict[AssetType, dict[str, float]] = {
            AssetType.MOTOR: {
                "base_vibration": 1.5,
                "base_temperature": 65.0,
                "base_pressure": 4.0,
                "base_current": 12.0,
                "rated_rpm": 1750.0,
            },
            AssetType.PUMP: {
                "base_vibration": 2.0,
                "base_temperature": 55.0,
                "base_pressure": 5.0,
                "base_current": 8.0,
                "rated_rpm": 1450.0,
            },
            AssetType.COMPRESSOR: {
                "base_vibration": 1.8,
                "base_temperature": 72.0,
                "base_pressure": 6.0,
                "base_current": 20.0,
                "rated_rpm": 3000.0,
            },
        }
        params = type_params.get(asset_type, type_params[AssetType.MOTOR])
        self._assets[asset_id] = AssetState(
            asset_id=asset_id,
            asset_type=asset_type,
            degradation_start_hour=degradation_start_hour,
            failure_hour=failure_hour,
            **params,
        )

    def register_fleet(
        self,
        count: int = 10,
        asset_type: AssetType = AssetType.MOTOR,
        prefix: str = "MOTOR",
    ) -> list[str]:
        """Register a fleet of assets with randomized parameters.

        Creates `count` assets with varied degradation profiles,
        simulating a realistic plant floor with mixed-age equipment.

        Returns:
            List of registered asset IDs.
        """
        asset_ids: list[str] = []
        for i in range(count):
            asset_id = f"{prefix}-{i + 1:03d}"
            # Vary degradation start and failure times across fleet
            deg_start = random.uniform(48.0, 96.0)
            fail_hour = deg_start + random.uniform(24.0, 48.0)
            self.register_asset(
                asset_id,
                asset_type,
                degradation_start_hour=deg_start,
                failure_hour=fail_hour,
            )
            asset_ids.append(asset_id)
        return asset_ids

    def collect_reading(self, asset_id: str) -> SensorReading:
        """Generate a single sensor reading for an asset.

        Applies the following signal model:
        1. Baseline values (per-type defaults + per-unit offset)
        2. Degradation curve (exponential after degradation_start_hour)
        3. Gaussian sensor noise
        4. Occasional spikes / intermittent faults
        5. RPM jitter around rated speed

        Args:
            asset_id: ID of the asset to read.

        Returns:
            A single SensorReading with realistic sensor values.

        Raises:
            KeyError: If asset_id is not registered.
        """
        state = self._assets.get(asset_id)
        if state is None:
            raise KeyError(f"Asset '{asset_id}' not registered. Call register_asset() first.")

        state.reading_count += 1
        # Advance operating time slightly per reading (simulated clock)
        state.operating_hours += 1.0 / 3600.0  # Each reading = 1 second

        # Compute degradation multiplier
        deg = degradation_curve(
            state.operating_hours,
            state.degradation_start_hour,
            state.failure_hour,
        )

        # Degradation factor: 1.0 at baseline, up to ~55x at failure
        vib_deg = deg
        temp_deg = 1.0 + (deg - 1.0) * 0.3  # Temperature degrades slower
        pressure_deg = 1.0 - (deg - 1.0) * 0.05  # Pressure drops with wear

        # Apply noise and offsets, clamp to Pydantic-valid ranges
        noise = state.noise_level
        reading = SensorReading(
            asset_id=asset_id,
            asset_type=state.asset_type,
            timestamp=utc_now(),
            vibration_x=round(min(50.0, max(0, (state.base_vibration + state.vibration_offset) * vib_deg
                + random.gauss(0, noise * state.base_vibration))),
                3,
            ),
            vibration_y=round(min(50.0, max(0, (state.base_vibration * 0.8 + state.vibration_offset) * vib_deg
                + random.gauss(0, noise * state.base_vibration * 0.8))),
                3,
            ),
            vibration_z=round(min(50.0, max(0, (state.base_vibration * 1.1 + state.vibration_offset) * vib_deg
                + random.gauss(0, noise * state.base_vibration * 1.1))),
                3,
            ),
            temperature=round(
                min(300.0, max(-40.0,
                state.base_temperature + state.temp_offset + (temp_deg - 1.0) * 30.0
                + random.gauss(0, noise * 5))),
                1,
            ),
            pressure=round(
                min(100.0, max(0.1, (state.base_pressure + state.pressure_offset) * pressure_deg
                + random.gauss(0, noise * 0.5))),
                2,
            ),
            current=round(
                min(500.0, max(0, state.base_current * (0.9 + vib_deg * 0.1)
                + random.gauss(0, noise * state.base_current * 0.1))),
                2,
            ),
            rpm=round(
                state.rated_rpm + random.gauss(0, state.rated_rpm * 0.02),
                1,
            ),
        )

        # Inject occasional spikes (2% probability) — clamp to valid ranges
        if random.random() < 0.02:
            spike_channel = random.choice(["vibration_x", "vibration_y", "vibration_z"])
            spike_value = reading.model_dump()
            spike_value[spike_channel] = round(
                min(50.0, spike_value[spike_channel] * random.uniform(3.0, 8.0)), 3
            )
            reading = SensorReading(**spike_value)

        # Store in history (with eviction)
        self._history[asset_id].append(reading)
        if len(self._history[asset_id]) > self._max_history_per_asset:
            self._history[asset_id] = self._history[asset_id][-self._max_history_per_asset:]

        return reading

    def collect_batch(
        self, asset_ids: list[str] | None = None, count: int = 1
    ) -> list[SensorReading]:
        """Collect multiple readings from one or more assets.

        Args:
            asset_ids: Assets to collect from. None = all registered.
            count: Number of readings per asset.

        Returns:
            List of SensorReading objects.
        """
        targets = asset_ids or self.asset_ids
        readings: list[SensorReading] = []
        for _ in range(count):
            for aid in targets:
                try:
                    readings.append(self.collect_reading(aid))
                except KeyError:
                    continue
        return readings

    def get_history(self, asset_id: str, last_n: int | None = None) -> list[SensorReading]:
        """Retrieve stored readings for an asset.

        Args:
            asset_id: Asset identifier.
            last_n: Return only the last N readings. None = all.

        Returns:
            List of SensorReading objects (oldest first).
        """
        history = self._history.get(asset_id, [])
        if last_n is not None:
            return history[-last_n:]
        return list(history)

    def get_vibration_array(
        self, asset_id: str, channel: str = "vibration_x", last_n: int = 200
    ) -> NDArray[np.float64]:
        """Extract a single vibration channel as a numpy array.

        Useful for FFT and spectral analysis that requires contiguous
        signal data rather than individual readings.

        Args:
            asset_id: Asset identifier.
            channel: Sensor channel name.
            last_n: Number of most recent samples.

        Returns:
            1-D numpy array of vibration values.
        """
        history = self._history.get(asset_id, [])[-last_n:]
        values = [getattr(r, channel, 0.0) for r in history]
        return np.array(values, dtype=np.float64)
