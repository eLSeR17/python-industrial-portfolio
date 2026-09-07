"""Feature engineering pipeline for industrial sensor data.

Transforms raw multi-sensor readings into a rich feature vector suitable
for anomaly detection and failure prediction. Features are organized in
three domains:

1. Time-domain: rolling statistics (mean, std, min, max), crest factor,
   zero-crossing rate, kurtosis, skewness
2. Frequency-domain: FFT peak frequency, spectral energy, spectral entropy,
   harmonics analysis
3. Cross-sensor: vibration-to-temperature ratio, pressure-current correlation

The feature set follows ISO 13373 (condition monitoring) and ISO 10816
(vibration severity) standards used in industrial maintenance.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from src.models.schemas import SensorReading
from src.utils.helpers import (
    compute_crest_factor,
    compute_peak,
    compute_rms,
    utc_now,
    zero_crossing_rate,
)


@dataclass
class FeatureVector:
    """Computed feature vector for a single asset at a point in time.

    Attributes are organized by domain for traceability. Total dimensionality
    is ~65 features, providing rich signal representation for ML models.
    """

    asset_id: str
    timestamp: object  # datetime

    # ── Time-domain features (per channel) ──
    vib_x_rms: float = 0.0
    vib_x_peak: float = 0.0
    vib_x_crest: float = 0.0
    vib_x_zcr: float = 0.0
    vib_x_kurtosis: float = 0.0
    vib_x_skewness: float = 0.0
    vib_y_rms: float = 0.0
    vib_y_peak: float = 0.0
    vib_y_crest: float = 0.0
    vib_y_zcr: float = 0.0
    vib_z_rms: float = 0.0
    vib_z_peak: float = 0.0
    vib_z_crest: float = 0.0
    vib_z_zcr: float = 0.0

    # ── Rolling statistics (vibration composite) ──
    vib_composite_rms: float = 0.0
    rolling_mean_10: float = 0.0
    rolling_std_10: float = 0.0
    rolling_max_10: float = 0.0
    rolling_min_10: float = 0.0
    rolling_mean_50: float = 0.0
    rolling_std_50: float = 0.0
    rolling_max_50: float = 0.0
    rolling_min_50: float = 0.0
    rolling_mean_200: float = 0.0
    rolling_std_200: float = 0.0
    rolling_max_200: float = 0.0
    rolling_min_200: float = 0.0

    # ── Thermal features ──
    temperature_current: float = 0.0
    temperature_rolling_mean_10: float = 0.0
    temperature_rolling_std_10: float = 0.0
    temperature_rate_of_change: float = 0.0

    # ── Pressure features ──
    pressure_current: float = 0.0
    pressure_rolling_mean_10: float = 0.0
    pressure_rolling_std_10: float = 0.0

    # ── Frequency-domain features ──
    fft_peak_frequency: float = 0.0
    fft_peak_magnitude: float = 0.0
    spectral_energy: float = 0.0
    spectral_entropy: float = 0.0
    spectral_centroid: float = 0.0
    harmonic_ratio: float = 0.0

    # ── Cross-sensor features ──
    vib_temp_ratio: float = 0.0
    pressure_current_ratio: float = 0.0
    vibration_angle: float = 0.0

    # ── Operational ──
    rpm_current: float = 0.0
    power_estimate: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        """Convert all numeric attributes to a flat numpy array.

        Excludes asset_id and timestamp which are identifiers, not features.
        This vector is the input to ML models (anomaly detection, prediction).
        """
        fields = [
            v for k, v in self.__dict__.items()
            if k not in ("asset_id", "timestamp") and isinstance(v, (int, float))
        ]
        return np.array(fields, dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered list of feature names matching to_array()."""
        exclude = {"asset_id", "timestamp"}
        return [k for k in FeatureVector.__dataclass_fields__ if k not in exclude]


class FeatureEngineer:
    """Stateful feature engineering engine for streaming sensor data.

    Maintains per-asset signal buffers for rolling and FFT computations.
    Call `compute_features()` with each new reading to get an updated
    feature vector.

    The buffer size is configurable but defaults to 200 samples, which
    at 100 Hz provides a 2-second window — sufficient for both rolling
    statistics and FFT analysis of industrial machinery.
    """

    def __init__(self, buffer_size: int = 200, fft_window: int = 256) -> None:
        """Initialize the feature engineering engine.

        Args:
            buffer_size: Number of recent readings to keep per asset.
            fft_window: FFT window size (zero-padded if buffer smaller).
        """
        self._buffer_size = buffer_size
        self._fft_window = fft_window
        # Per-asset circular buffers for each channel
        self._buffers: dict[str, dict[str, list[float]]] = {}

    def _get_buffer(self, asset_id: str) -> dict[str, list[float]]:
        """Get or initialize signal buffers for an asset."""
        if asset_id not in self._buffers:
            self._buffers[asset_id] = {
                "vibration_x": [],
                "vibration_y": [],
                "vibration_z": [],
                "temperature": [],
                "pressure": [],
                "current": [],
                "rpm": [],
            }
        return self._buffers[asset_id]

    def _append_to_buffer(self, asset_id: str, reading: SensorReading) -> None:
        """Append reading values to the circular buffer, evicting oldest."""
        buf = self._get_buffer(asset_id)
        for channel in ["vibration_x", "vibration_y", "vibration_z",
                        "temperature", "pressure", "current", "rpm"]:
            values = buf[channel]
            values.append(getattr(reading, channel))
            if len(values) > self._buffer_size:
                buf[channel] = values[-self._buffer_size:]

    def _rolling_stats(
        self, values: list[float], window: int
    ) -> tuple[float, float, float, float]:
        """Compute rolling mean, std, max, min over the last `window` samples."""
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(values[-window:], dtype=np.float64)
        return (
            float(np.mean(arr)),
            float(np.std(arr)) if arr.size > 1 else 0.0,
            float(np.max(arr)),
            float(np.min(arr)),
        )

    def _compute_spectral_features(
        self, signal_data: NDArray[np.float64], sample_rate: float = 100.0
    ) -> tuple[float, float, float, float, float]:
        """Compute FFT-based frequency-domain features.

        Features extracted:
        - Peak frequency: dominant vibration frequency (bearing defect freq?)
        - Peak magnitude: strength of dominant frequency
        - Spectral energy: total signal energy in frequency domain
        - Spectral entropy: complexity/uniformity of the spectrum
        - Spectral centroid: weighted mean frequency (brightness)

        Args:
            signal_data: 1-D time-domain signal.
            sample_rate: Sampling rate in Hz.

        Returns:
            Tuple of (peak_freq, peak_magnitude, spectral_energy,
                      spectral_entropy, spectral_centroid).
        """
        if signal_data.size < 4:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Apply Hanning window to reduce spectral leakage
        windowed = signal_data * np.hanning(signal_data.size)
        n_fft = max(signal_data.size, self._fft_window)
        fft_result = np.fft.rfft(windowed, n=n_fft)
        magnitudes = np.abs(fft_result)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        if magnitudes.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Peak frequency (exclude DC component at index 0)
        peak_idx = int(np.argmax(magnitudes[1:])) + 1
        peak_freq = float(freqs[peak_idx])
        peak_magnitude = float(magnitudes[peak_idx])

        # Spectral energy
        spectral_energy = float(np.sum(magnitudes ** 2))

        # Spectral entropy (normalized)
        total = float(np.sum(magnitudes))
        if total > 1e-12:
            probs = magnitudes / total
            probs = probs[probs > 0]
            spectral_entropy = float(-np.sum(probs * np.log2(probs)))
            # Normalize by log2 of number of bins
            spectral_entropy /= float(np.log2(probs.size)) if probs.size > 1 else 1.0
        else:
            spectral_entropy = 0.0

        # Spectral centroid
        if spectral_energy > 1e-12:
            spectral_centroid = float(np.sum(freqs * magnitudes ** 2) / spectral_energy)
        else:
            spectral_centroid = 0.0

        return peak_freq, peak_magnitude, spectral_energy, spectral_entropy, spectral_centroid

    def _compute_harmonic_ratio(
        self, signal_data: NDArray[np.float64], fundamental_freq: float,
        sample_rate: float = 100.0,
    ) -> float:
        """Compute the ratio of harmonic energy to total energy.

        High harmonic content relative to the fundamental indicates
        mechanical faults: misalignment produces 2x RPM peaks,
        bearing defects produce characteristic defect frequencies.

        Args:
            signal_data: Time-domain vibration signal.
            fundamental_freq: Expected fundamental frequency (e.g., RPM/60).
            sample_rate: Sampling rate in Hz.

        Returns:
            Ratio of harmonic energy to total spectral energy (0.0–1.0).
        """
        if signal_data.size < 4 or fundamental_freq <= 0:
            return 0.0

        n_fft = max(signal_data.size, self._fft_window)
        fft_result = np.fft.rfft(signal_data * np.hanning(signal_data.size), n=n_fft)
        magnitudes = np.abs(fft_result)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        total_energy = float(np.sum(magnitudes ** 2))
        if total_energy < 1e-12:
            return 0.0

        # Sum energy at harmonic frequencies (2x, 3x, 4x, 5x fundamental)
        harmonic_energy = 0.0
        for harmonic in range(2, 6):
            target_freq = fundamental_freq * harmonic
            # Find closest FFT bin
            idx = int(np.argmin(np.abs(freqs - target_freq)))
            # Sum energy in a small window around the harmonic (±2 bins)
            low = max(0, idx - 2)
            high = min(magnitudes.size, idx + 3)
            harmonic_energy += float(np.sum(magnitudes[low:high] ** 2))

        return min(1.0, harmonic_energy / total_energy)

    def compute_features(self, reading: SensorReading) -> FeatureVector:
        """Compute the full feature vector from a new sensor reading.

        This is the main entry point. Each call:
        1. Appends the reading to per-asset buffers
        2. Computes time-domain features from buffered data
        3. Computes frequency-domain features via FFT
        4. Computes cross-sensor derived features
        5. Returns a complete FeatureVector

        Args:
            reading: New sensor reading from the data collector.

        Returns:
            FeatureVector with ~65 engineered features.
        """
        self._append_to_buffer(reading.asset_id, reading)
        buf = self._get_buffer(reading.asset_id)

        # Convert buffers to numpy arrays
        vib_x = np.array(buf["vibration_x"], dtype=np.float64)
        vib_y = np.array(buf["vibration_y"], dtype=np.float64)
        vib_z = np.array(buf["vibration_z"], dtype=np.float64)
        temp = np.array(buf["temperature"], dtype=np.float64)
        pres = np.array(buf["pressure"], dtype=np.float64)
        curr = np.array(buf["current"], dtype=np.float64)
        rpm = np.array(buf["rpm"], dtype=np.float64)

        # Composite vibration (RMS of triaxial)
        vib_composite = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)

        # ── Time-domain features ──
        fv = FeatureVector(asset_id=reading.asset_id, timestamp=reading.timestamp)

        # Per-channel time-domain
        for prefix, arr in [("vib_x", vib_x), ("vib_y", vib_y), ("vib_z", vib_z)]:
            setattr(fv, f"{prefix}_rms", compute_rms(arr))
            setattr(fv, f"{prefix}_peak", compute_peak(arr))
            setattr(fv, f"{prefix}_crest", compute_crest_factor(arr))
            if prefix in ("vib_x", "vib_y"):  # ZCR only for x and y
                setattr(fv, f"{prefix}_zcr", zero_crossing_rate(arr))
        fv.vib_x_kurtosis = float(scipy_stats.kurtosis(vib_x)) if vib_x.size > 3 else 0.0
        fv.vib_x_skewness = float(scipy_stats.skew(vib_x)) if vib_x.size > 3 else 0.0
        fv.vib_y_zcr = zero_crossing_rate(vib_y)
        fv.vib_z_zcr = zero_crossing_rate(vib_z)

        # Composite vibration RMS
        fv.vib_composite_rms = compute_rms(vib_composite)

        # Rolling statistics over multiple windows
        for window in [10, 50, 200]:
            mean, std, maxv, minv = self._rolling_stats(list(vib_composite), window)
            setattr(fv, f"rolling_mean_{window}", mean)
            setattr(fv, f"rolling_std_{window}", std)
            setattr(fv, f"rolling_max_{window}", maxv)
            setattr(fv, f"rolling_min_{window}", minv)

        # ── Thermal features ──
        fv.temperature_current = float(temp[-1]) if temp.size > 0 else 0.0
        t_mean, t_std, _, _ = self._rolling_stats(list(temp), 10)
        fv.temperature_rolling_mean_10 = t_mean
        fv.temperature_rolling_std_10 = t_std
        # Rate of change (last sample vs 10 samples ago)
        if temp.size >= 10:
            fv.temperature_rate_of_change = float(temp[-1] - temp[-10])
        else:
            fv.temperature_rate_of_change = 0.0

        # ── Pressure features ──
        fv.pressure_current = float(pres[-1]) if pres.size > 0 else 0.0
        p_mean, p_std, _, _ = self._rolling_stats(list(pres), 10)
        fv.pressure_rolling_mean_10 = p_mean
        fv.pressure_rolling_std_10 = p_std

        # ── Frequency-domain features (on vibration_x) ──
        peak_f, peak_m, spec_e, spec_ent, spec_c = self._compute_spectral_features(vib_x)
        fv.fft_peak_frequency = peak_f
        fv.fft_peak_magnitude = peak_m
        fv.spectral_energy = spec_e
        fv.spectral_entropy = spec_ent
        fv.spectral_centroid = spec_c

        # Harmonic ratio using RPM as fundamental frequency
        current_rpm = float(rpm[-1]) if rpm.size > 0 else 1750.0
        fundamental = current_rpm / 60.0  # Convert RPM to Hz
        fv.harmonic_ratio = self._compute_harmonic_ratio(vib_x, fundamental)

        # ── Cross-sensor features ──
        temp_val = float(temp[-1]) if temp.size > 0 else 25.0
        vib_val = fv.vib_composite_rms
        pres_val = float(pres[-1]) if pres.size > 0 else 1.0
        curr_val = float(curr[-1]) if curr.size > 0 else 1.0

        fv.vib_temp_ratio = vib_val / max(temp_val, 1.0)
        fv.pressure_current_ratio = pres_val / max(curr_val, 0.1)
        # Vibration angle in XZ plane (indicates direction of dominant force)
        fv.vibration_angle = float(np.arctan2(
            float(np.mean(vib_z)) if vib_z.size > 0 else 0.0,
            float(np.mean(vib_x)) if vib_x.size > 0 else 1e-12,
        ))

        # ── Operational features ──
        fv.rpm_current = current_rpm
        # Rough power estimate: P ≈ V × I × power_factor (0.85 assumed)
        fv.power_estimate = curr_val * 230.0 * 0.85 / 1000.0  # kW

        return fv

    def compute_features_batch(
        self, readings: list[SensorReading]
    ) -> list[FeatureVector]:
        """Compute features for a batch of readings in sequence.

        Maintains buffer state across the batch, so earlier readings
        contribute to rolling statistics of later ones.
        """
        return [self.compute_features(r) for r in readings]

    def get_feature_matrix(self, asset_id: str, last_n: int = 100) -> NDArray[np.float64] | None:
        """Build a feature matrix from the asset's buffered sensor data.

        Snapshots the current per-channel buffers, replays each historical
        reading through the full feature pipeline (time-domain, frequency-
        domain, rolling windows, cross-sensor), collects the resulting
        FeatureVectors, and stacks them into a 2-D numpy array. The
        original buffer state is restored after the replay so callers
        are unaffected.

        Args:
            asset_id: Target asset.
            last_n: How many historical readings to process (from the
                     tail of the buffer).

        Returns:
            Feature matrix of shape (n_samples, n_features) or None
            if the asset has no buffered data.
        """
        if asset_id not in self._buffers:
            return None

        buf = self._buffers[asset_id]
        channels = [
            "vibration_x", "vibration_y", "vibration_z",
            "temperature", "pressure", "current", "rpm",
        ]

        # Verify every channel has data
        for ch in channels:
            if not buf[ch]:
                return None

        n_available = min(len(buf[ch]) for ch in channels)
        if n_available == 0:
            return None

        n_readings = min(last_n, n_available)

        # Snapshot current buffer state
        snapshot = {ch: list(buf[ch]) for ch in channels}

        try:
            vectors: list[FeatureVector] = []
            for i in range(n_readings):
                # Restore the snapshot before each replay step so that
                # compute_features (which mutates the buffer via
                # _append_to_buffer) always starts from the same base
                for ch in channels:
                    buf[ch] = list(snapshot[ch])

                idx = -n_readings + i
                reading = SensorReading(
                    asset_id=asset_id,
                    timestamp=utc_now(),
                    vibration_x=snapshot["vibration_x"][idx],
                    vibration_y=snapshot["vibration_y"][idx],
                    vibration_z=snapshot["vibration_z"][idx],
                    temperature=snapshot["temperature"][idx],
                    pressure=snapshot["pressure"][idx],
                    current=snapshot["current"][idx],
                    rpm=snapshot["rpm"][idx],
                )
                vectors.append(self.compute_features(reading))

            # Restore the original buffer state
            for ch in channels:
                buf[ch] = snapshot[ch]

            if not vectors:
                return None

            return np.array([v.to_array() for v in vectors], dtype=np.float64)
        except Exception:
            # Restore buffer on any failure
            for ch in channels:
                buf[ch] = snapshot[ch]
            return None

    def clear_buffer(self, asset_id: str) -> None:
        """Clear all buffered data for an asset.

        Useful when an asset is decommissioned or when resetting
        after a long period of inactivity.
        """
        self._buffers.pop(asset_id, None)
