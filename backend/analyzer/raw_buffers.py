"""Reusable SAN-90 uint8 acquisition and bounded display snapshot buffers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time

import numpy as np
from numpy.typing import NDArray


UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RawAmplitudeMapping:
    scale_db_per_code: float
    offset_dbm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_db_per_code) or self.scale_db_per_code <= 0:
            raise ValueError("SAN-90 raw amplitude scale must be finite and positive")
        if not np.isfinite(self.offset_dbm):
            raise ValueError("SAN-90 raw amplitude offset must be finite")

    def convert(self, raw: UInt8Array, output: Float32Array) -> None:
        if raw.shape != output.shape:
            raise ValueError("raw and output buffers must have matching shapes")
        np.multiply(raw, np.float32(self.scale_db_per_code), out=output, casting="unsafe")
        np.add(output, np.float32(self.offset_dbm), out=output)


@dataclass(frozen=True, slots=True)
class RawTraceMetadata:
    sequence: int
    device_timestamp_ns: int
    host_timestamp_ns: int
    receipt_monotonic_ns: int
    start_frequency_hz: float
    center_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    rbw_hz: float
    reference_level_dbm: float
    mapping: RawAmplitudeMapping
    configuration_generation: int = 0


@dataclass(frozen=True, slots=True)
class DisplaySnapshot:
    metadata: RawTraceMetadata
    spectrum_float32: Float32Array
    waterfall_uint8: UInt8Array
    spectrum_ready: bool
    waterfall_ready: bool
    generation: int


class RawRtaAccumulator:
    """Single-owner reusable latest/max buffers for native RTA codes."""

    def __init__(self, point_count: int) -> None:
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        self.point_count = point_count
        self.latest_raw = np.empty(point_count, dtype=np.uint8)
        self.interval_max_raw = np.empty(point_count, dtype=np.uint8)
        self._packet_max_raw = np.empty(point_count, dtype=np.uint8)
        self._has_latest = False
        self._has_interval = False
        self.metadata: RawTraceMetadata | None = None

    def update(self, packet: UInt8Array, metadata: RawTraceMetadata, *, accumulate_interval: bool = True) -> None:
        if packet.ndim != 2 or packet.shape[1] != self.point_count or packet.shape[0] == 0:
            raise ValueError(f"invalid native RTA packet shape {packet.shape}")
        if packet.dtype != np.uint8 or not packet.flags.c_contiguous:
            raise ValueError("native RTA packet must be contiguous uint8")
        mapping_changed = self.metadata is not None and self.metadata.mapping != metadata.mapping
        np.copyto(self.latest_raw, packet[-1], casting="no")
        if accumulate_interval:
            np.max(packet, axis=0, out=self._packet_max_raw)
            if not self._has_interval or mapping_changed:
                np.copyto(self.interval_max_raw, self._packet_max_raw, casting="no")
            else:
                np.maximum(self.interval_max_raw, self._packet_max_raw, out=self.interval_max_raw)
            self._has_interval = True
        self._has_latest = True
        self.metadata = metadata

    @property
    def ready(self) -> bool:
        return self._has_latest and self.metadata is not None

    def copy_latest_dbm(self) -> Float32Array:
        if not self.ready or self.metadata is None:
            raise RuntimeError("no RTA trace is available")
        output = np.empty(self.point_count, dtype=np.float32)
        self.metadata.mapping.convert(self.latest_raw, output)
        return output

    def copy_interval_max_dbm(self) -> Float32Array:
        if not self._has_interval or self.metadata is None:
            raise RuntimeError("no RTA interval max is available")
        output = np.empty(self.point_count, dtype=np.float32)
        self.metadata.mapping.convert(self.interval_max_raw, output)
        return output

    def reset_interval(self) -> None:
        self._has_interval = False


@dataclass(slots=True)
class _SnapshotSlot:
    spectrum: Float32Array
    waterfall: UInt8Array
    metadata: RawTraceMetadata | None = None
    spectrum_ready: bool = False
    waterfall_ready: bool = False
    generation: int = 0


class DisplaySnapshotExchange:
    """Two reusable producer slots with newest-only copied consumption."""

    def __init__(self, point_count: int) -> None:
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        self.point_count = point_count
        self._slots = [
            _SnapshotSlot(np.empty(point_count, np.float32), np.empty(point_count, np.uint8)),
            _SnapshotSlot(np.empty(point_count, np.float32), np.empty(point_count, np.uint8)),
        ]
        self._lock = Lock()
        self._write_index = 0
        self._published_index: int | None = None
        self._generation = 0
        self._consumed_generation = 0
        self._replaced = 0
        self.last_conversion_s = 0.0
        self.last_publish_s = 0.0

    @property
    def replaced(self) -> int:
        with self._lock:
            return self._replaced

    def publish(self, accumulator: RawRtaAccumulator, *, spectrum: bool, waterfall: bool) -> bool:
        if not accumulator.ready or accumulator.metadata is None or not (spectrum or waterfall):
            return False
        publish_started = time.perf_counter()
        slot = self._slots[self._write_index]
        conversion_started = time.perf_counter()
        if spectrum:
            accumulator.metadata.mapping.convert(accumulator.latest_raw, slot.spectrum)
        self.last_conversion_s = time.perf_counter() - conversion_started
        if waterfall:
            if not accumulator._has_interval:
                return False
            np.copyto(slot.waterfall, accumulator.interval_max_raw, casting="no")
        slot.metadata = accumulator.metadata
        slot.spectrum_ready = spectrum
        slot.waterfall_ready = waterfall
        with self._lock:
            if self._generation > self._consumed_generation:
                self._replaced += 1
            self._generation += 1
            slot.generation = self._generation
            self._published_index = self._write_index
            self._write_index = 1 - self._write_index
        if waterfall:
            accumulator.reset_interval()
        self.last_publish_s = time.perf_counter() - publish_started
        return True

    def take_latest(self) -> DisplaySnapshot | None:
        with self._lock:
            if self._published_index is None or self._generation == self._consumed_generation:
                return None
            slot = self._slots[self._published_index]
            assert slot.metadata is not None
            snapshot = DisplaySnapshot(
                metadata=slot.metadata,
                spectrum_float32=np.array(slot.spectrum, dtype=np.float32, order="C", copy=True),
                waterfall_uint8=np.array(slot.waterfall, dtype=np.uint8, order="C", copy=True),
                spectrum_ready=slot.spectrum_ready,
                waterfall_ready=slot.waterfall_ready,
                generation=slot.generation,
            )
            self._consumed_generation = self._generation
            return snapshot
