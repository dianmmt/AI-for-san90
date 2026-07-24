from __future__ import annotations

from dataclasses import replace
import math
from threading import Lock
import time

import numpy as np
from numpy.typing import NDArray

from .models import SpectrumTemporalFrame
from .raw_buffers import RawAmplitudeMapping, RawTraceMetadata

UInt8Array = NDArray[np.uint8]


class NativeSpectrumTemporalAccumulator:
    """Accumulate every native uint8 trace into one 60 Hz temporal frame."""

    def __init__(self, point_count: int, interval_ns: int = 16_666_667) -> None:
        if point_count < 2 or interval_ns < 1:
            raise ValueError("invalid temporal accumulator dimensions")
        self.point_count = point_count
        self.interval_ns = interval_ns
        self._latest = np.empty(point_count, np.uint8)
        self._maximum = np.empty(point_count, np.uint8)
        self._packet_max = np.empty(point_count, np.uint8)
        self._latest_float = np.empty(point_count, np.float32)
        self._maximum_float = np.empty(point_count, np.float32)
        self._started = False
        self._resume_start_ns: int | None = None
        self._start_ns = self._deadline_ns = self._traces = self._sequence = 0
        self._generation = 0
        self._metadata: RawTraceMetadata | None = None
        self._first_receipt_ns: int | None = None
        self._last_receipt_ns: int | None = None
        self.discarded_incomplete_intervals = 0
        self.completed_intervals = 0
        self.missed_interval_deadlines = 0
        self.total_traces_integrated = 0
        self.minimum_traces_integrated = 0
        self.maximum_traces_integrated = 0
        self.max_hold_update_total_ns = 0
        self.conversion_total_ns = 0
        self.conversion_max_ns = 0
        self.finalization_total_ns = 0
        self.finalization_max_ns = 0
        self.receipt_span_total_ns = 0
        self.receipt_span_min_ns = 0
        self.receipt_span_max_ns = 0

    @property
    def traces_integrated(self) -> int:
        return self._traces

    def reset(self, *, generation: int | None = None) -> None:
        if self._started and self._traces:
            self.discarded_incomplete_intervals += 1
        self._started = False
        self._resume_start_ns = None
        self._traces = 0
        self._metadata = None
        self._first_receipt_ns = None
        self._last_receipt_ns = None
        if generation is not None:
            self._generation = generation

    def add_packet(self, packet: UInt8Array, metadata: RawTraceMetadata) -> SpectrumTemporalFrame | None:
        if packet.dtype != np.uint8 or packet.ndim != 2 or packet.shape[0] < 1:
            raise ValueError("packet must be a non-empty two-dimensional uint8 array")
        if packet.shape[1] != self.point_count or not packet.flags.c_contiguous:
            raise ValueError("packet dimensions or storage do not match the accumulator")
        if self._started and metadata.configuration_generation != self._generation:
            self.reset(generation=metadata.configuration_generation)
            completed = None
        else:
            completed = self.finalize_if_due(metadata.receipt_monotonic_ns)
        if not self._started:
            self._start_ns = self._resume_start_ns if self._resume_start_ns is not None else metadata.receipt_monotonic_ns
            self._resume_start_ns = None
            self._deadline_ns = self._start_ns + self.interval_ns
            self._generation = metadata.configuration_generation
            self._maximum.fill(0)
            self._started = True
            self._first_receipt_ns = metadata.receipt_monotonic_ns
        update_started = time.perf_counter_ns()
        np.copyto(self._latest, packet[-1])
        np.max(packet, axis=0, out=self._packet_max)
        np.maximum(self._maximum, self._packet_max, out=self._maximum)
        self.max_hold_update_total_ns += time.perf_counter_ns() - update_started
        self._traces += int(packet.shape[0])
        self._metadata = metadata
        self._last_receipt_ns = metadata.receipt_monotonic_ns
        return completed

    def finalize_if_due(self, now_ns: int) -> SpectrumTemporalFrame | None:
        if not self._started or not self._traces or now_ns < self._deadline_ns:
            return None
        deadline = self._deadline_ns
        frame = self._finalize(deadline)
        # Skip empty elapsed slots but preserve the original deadline grid.
        # The next packet begins in the current grid interval; no duplicate
        # temporal frame is fabricated to fill missed slots.
        missed = max(0, (now_ns - deadline) // self.interval_ns)
        self.missed_interval_deadlines += missed
        self._resume_start_ns = deadline + missed * self.interval_ns
        return frame

    def flush(self, now_ns: int) -> SpectrumTemporalFrame | None:
        if not self._started or not self._traces:return None
        frame=self._finalize(now_ns);self._resume_start_ns=None;return frame

    def _finalize(self, end_ns: int) -> SpectrumTemporalFrame:
        finalization_started = time.perf_counter_ns()
        assert self._metadata is not None
        metadata = self._metadata
        conversion_started = time.perf_counter_ns()
        metadata.mapping.convert(self._latest, self._latest_float)
        metadata.mapping.convert(self._maximum, self._maximum_float)
        conversion_ns = time.perf_counter_ns() - conversion_started
        self.conversion_total_ns += conversion_ns
        self.conversion_max_ns = max(self.conversion_max_ns, conversion_ns)
        self._sequence += 1
        self.completed_intervals += 1
        self.total_traces_integrated += self._traces
        self.minimum_traces_integrated = self._traces if self.minimum_traces_integrated == 0 else min(self.minimum_traces_integrated, self._traces)
        self.maximum_traces_integrated = max(self.maximum_traces_integrated, self._traces)
        frame = SpectrumTemporalFrame(
            generation=self._generation, sequence=self._sequence, point_count=self.point_count,
            interval_start_monotonic_ns=self._start_ns, interval_end_monotonic_ns=end_ns,
            host_timestamp_ns=metadata.host_timestamp_ns,
            device_timestamp_ns=metadata.device_timestamp_ns or None,
            traces_integrated=self._traces,
            latest_trace_float32=self._latest_float.copy(),
            interval_max_trace_float32=self._maximum_float.copy(),
            start_frequency_hz=metadata.start_frequency_hz,
            center_frequency_hz=metadata.center_frequency_hz,
            stop_frequency_hz=metadata.stop_frequency_hz, span_hz=metadata.span_hz,
            rbw_hz=metadata.rbw_hz, reference_level_dbm=metadata.reference_level_dbm,
            scale_to_dbm=metadata.mapping.scale_db_per_code,
            offset_to_dbm=metadata.mapping.offset_dbm,
            first_receipt_monotonic_ns=self._first_receipt_ns,
            last_receipt_monotonic_ns=self._last_receipt_ns,
        )
        if self._first_receipt_ns is not None and self._last_receipt_ns is not None:
            receipt_span = self._last_receipt_ns - self._first_receipt_ns
            self.receipt_span_total_ns += receipt_span
            self.receipt_span_min_ns = receipt_span if self.receipt_span_min_ns == 0 else min(self.receipt_span_min_ns, receipt_span)
            self.receipt_span_max_ns = max(self.receipt_span_max_ns, receipt_span)
        self._started = False
        self._traces = 0
        self._metadata = None
        self._first_receipt_ns = None
        self._last_receipt_ns = None
        finalization_ns = time.perf_counter_ns() - finalization_started
        self.finalization_total_ns += finalization_ns
        self.finalization_max_ns = max(self.finalization_max_ns, finalization_ns)
        return frame


class LatestSpectrumTemporalExchange:
    """Newest-only exchange; replacements merge maxima instead of forming a queue."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: SpectrumTemporalFrame | None = None
        self.frames_replaced = 0
        self.frames_published = 0
        self.frames_displaced = 0
        self.compatible_maximum_merges = 0
        self.incompatible_merge_rejections = 0
        self.traces_preserved_by_merges = 0

    @staticmethod
    def _compatible(previous: SpectrumTemporalFrame, frame: SpectrumTemporalFrame) -> bool:
        return (
            previous.generation == frame.generation
            and previous.point_count == frame.point_count
            and math.isclose(previous.start_frequency_hz, frame.start_frequency_hz, rel_tol=0.0, abs_tol=1e-3)
            and math.isclose(previous.stop_frequency_hz, frame.stop_frequency_hz, rel_tol=0.0, abs_tol=1e-3)
            and math.isclose(previous.scale_to_dbm, frame.scale_to_dbm, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(previous.offset_to_dbm, frame.offset_to_dbm, rel_tol=0.0, abs_tol=1e-9)
        )

    def publish(self, frame: SpectrumTemporalFrame) -> None:
        with self._lock:
            self.frames_published += 1
            previous = self._pending
            if previous is not None:
                self.frames_displaced += 1
            if previous is not None and self._compatible(previous, frame):
                merged = np.maximum(previous.interval_max_trace_float32, frame.interval_max_trace_float32)
                frame = replace(
                    frame,
                    interval_start_monotonic_ns=previous.interval_start_monotonic_ns,
                    first_receipt_monotonic_ns=previous.first_receipt_monotonic_ns,
                    traces_integrated=previous.traces_integrated + frame.traces_integrated,
                    interval_max_trace_float32=np.array(merged, dtype=np.float32, order="C", copy=True),
                )
                self.frames_replaced += 1
                self.compatible_maximum_merges += 1
                self.traces_preserved_by_merges += previous.traces_integrated
            elif previous is not None:
                self.incompatible_merge_rejections += 1
            self._pending = frame

    def take(self) -> SpectrumTemporalFrame | None:
        with self._lock:
            frame, self._pending = self._pending, None
            return frame

    def clear(self) -> None:
        with self._lock:
            self._pending = None
