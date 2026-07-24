"""Timed native-domain waterfall integration and bounded batch exchange."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .models import SpectrumFrame, WaterfallBatch
from .raw_buffers import RawTraceMetadata
from .tradeoff import SAN90_RESOLUTION_TRADEOFF_STEPS, match_actual_tradeoff_step


def waterfall_rate_override_from_environment() -> WaterfallRateConfig | None:
    """Return an explicit operator override, or None for profile-aware defaults."""
    rows_text = os.getenv("SAN90_WATERFALL_ROWS_PER_SECOND") or os.getenv("SAN90_WATERFALL_FPS")
    batches_text = os.getenv("SAN90_WATERFALL_BATCHES_PER_SECOND")
    batch_rows_text = os.getenv("SAN90_WATERFALL_ROWS_PER_BATCH")
    if rows_text is None and batches_text is None and batch_rows_text is None:
        return None
    rows = None if rows_text is None else float(rows_text)
    batches = 60.0 if batches_text is None else float(batches_text)
    if not math.isfinite(batches) or batches <= 0:
        raise ValueError("SAN90_WATERFALL_BATCHES_PER_SECOND must be finite and positive")
    if batch_rows_text is None:
        if rows is None:
            rows_per_batch = 1
        else:
            ratio = rows / batches
            rows_per_batch = round(ratio)
            if rows_per_batch <= 0 or not math.isclose(ratio, rows_per_batch, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("waterfall rows/s divided by batches/s must be a positive integer")
    else:
        rows_per_batch = int(batch_rows_text)
    if rows is None:
        rows = batches * rows_per_batch
    return WaterfallRateConfig(rows, batches, rows_per_batch)


def waterfall_rate_for_profile(
    actual_rbw_hz: float,
    point_count: int,
    override: WaterfallRateConfig | None = None,
) -> WaterfallRateConfig:
    """Select the verified safe/fast display rate from actual SDK output."""
    if override is not None:
        return override
    measured = match_actual_tradeoff_step(
        SAN90_RESOLUTION_TRADEOFF_STEPS,
        actual_rbw_hz=actual_rbw_hz,
        point_count=point_count,
        fft_size=None,
    )
    if measured is not None:
        return WaterfallRateConfig(
            measured.waterfall_rows_per_second,
            measured.waterfall_batches_per_second,
            measured.waterfall_rows_per_batch,
        )
    fast_profile = actual_rbw_hz >= 200_000 and point_count < 2_000
    return WaterfallRateConfig(240.0, 60.0, 4) if fast_profile else WaterfallRateConfig(60.0, 60.0, 1)


@dataclass(frozen=True, slots=True)
class WaterfallRateConfig:
    rows_per_second: float
    batches_per_second: float
    rows_per_batch: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.rows_per_second) or self.rows_per_second <= 0:
            raise ValueError("waterfall rows_per_second must be finite and positive")
        if not math.isfinite(self.batches_per_second) or self.batches_per_second <= 0:
            raise ValueError("waterfall batches_per_second must be finite and positive")
        if self.rows_per_batch <= 0:
            raise ValueError("waterfall rows_per_batch must be positive")
        expected = self.batches_per_second * self.rows_per_batch
        if not math.isclose(self.rows_per_second, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "waterfall rows_per_second must equal batches_per_second × rows_per_batch "
                f"({self.rows_per_second} != {self.batches_per_second} × {self.rows_per_batch})"
            )

    @property
    def nominal_row_period_ns(self) -> int:
        return max(1, round(1_000_000_000 / self.rows_per_second))


class LatestWaterfallBatchExchange:
    """One-slot newest-data-first exchange; acquisition never waits for a reader."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._batch: WaterfallBatch | None = None
        self._published_generation = 0
        self._read_generation = 0
        self.replaced_batches = 0
        self.replaced_rows = 0

    def publish(self, batch: WaterfallBatch) -> None:
        with self._lock:
            if self._batch is not None and self._read_generation < self._published_generation:
                self.replaced_batches += 1
                self.replaced_rows += self._batch.row_count
            self._batch = batch
            self._published_generation += 1

    def take_latest(self) -> WaterfallBatch | None:
        with self._lock:
            if self._batch is None or self._read_generation == self._published_generation:
                return None
            self._read_generation = self._published_generation
            return self._batch

    def clear(self) -> None:
        with self._lock:
            self._batch = None
            self._read_generation = self._published_generation


@dataclass(frozen=True, slots=True)
class WaterfallProducerMetrics:
    target_rows_per_second: float
    target_batches_per_second: float
    rows_per_batch: int
    nominal_row_period_ns: int
    actual_rows_per_second: float
    actual_batches_per_second: float
    completed_rows: int
    completed_batches: int
    traces_integrated: int
    minimum_traces_per_row: int
    maximum_traces_per_row: int
    mean_traces_per_row: float
    mean_row_jitter_ns: float
    maximum_row_jitter_ns: int
    mean_trace_span_per_row_ns: float
    maximum_trace_span_per_row_ns: int
    missed_row_deadlines: int
    empty_rows: int
    discarded_incomplete_rows: int
    discarded_incomplete_batch_rows: int
    replaced_batches: int
    replaced_rows: int
    mean_max_hold_update_ns_per_trace: float
    mean_row_finalization_ns: float
    maximum_row_finalization_ns: int
    mean_batch_finalization_ns: float
    maximum_batch_finalization_ns: int


@dataclass(frozen=True, slots=True)
class _RowMetadata:
    receipt_monotonic_ns: int
    device_timestamp_ns: int
    host_timestamp_ns: int
    start_frequency_hz: float
    center_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    rbw_hz: float
    reference_level_dbm: float


class TimedWaterfallBatchProducer:
    """Closes monotonic max-hold windows and assembles fixed-size uint8 batches."""

    def __init__(self, point_count: int, configuration_generation: int, config: WaterfallRateConfig) -> None:
        self.exchange = LatestWaterfallBatchExchange()
        self._config = config
        self._point_count = point_count
        self._configuration_generation = configuration_generation
        self._period_ns = config.nominal_row_period_ns
        self._deadline_ns: int | None = None
        self._row_max = np.empty(point_count, dtype=np.uint8)
        self._row_trace_count = 0
        self._row_first_receipt_ns = 0
        self._last_metadata: _RowMetadata | None = None
        self._segment_max = np.empty(point_count, dtype=np.uint8)
        self._batch_values = np.empty((config.rows_per_batch, point_count), dtype=np.uint8)
        self._batch_fill = 0
        self._batch_first_row_sequence = 0
        self._batch_first_host_timestamp_ns = 0
        self._batch_first_device_timestamp_ns = 0
        self._row_sequence = 0
        self._batch_sequence = 0
        self._completed_rows = 0
        self._completed_batches = 0
        self._traces_integrated = 0
        self._completed_trace_total = 0
        self._min_traces = 0
        self._max_traces = 0
        self._jitter_total_ns = 0
        self._max_jitter_ns = 0
        self._trace_span_total_ns = 0
        self._trace_span_max_ns = 0
        self._missed_deadlines = 0
        self._empty_rows = 0
        self._discarded_rows = 0
        self._discarded_batch_rows = 0
        self._metrics_started_ns = time.monotonic_ns()
        self._max_hold_update_total_ns = 0
        self._row_finalization_total_ns = 0
        self._row_finalization_max_ns = 0
        self._batch_finalization_total_ns = 0
        self._batch_finalization_max_ns = 0

    @property
    def config(self) -> WaterfallRateConfig:
        return self._config

    def reconfigure(self, point_count: int, configuration_generation: int, config: WaterfallRateConfig) -> None:
        """Reset on the acquisition owner thread; never races with add_trace."""
        if point_count <= 0:
            raise ValueError("waterfall point_count must be positive")
        if self._row_trace_count:
            self._discarded_rows += 1
        if self._batch_fill:
            self._discarded_batch_rows += self._batch_fill
        self._config = config
        self._point_count = point_count
        self._configuration_generation = configuration_generation
        self._period_ns = config.nominal_row_period_ns
        self._deadline_ns = None
        self._row_max = np.empty(point_count, dtype=np.uint8)
        self._row_trace_count = 0
        self._row_first_receipt_ns = 0
        self._last_metadata = None
        self._segment_max = np.empty(point_count, dtype=np.uint8)
        self._batch_values = np.empty((config.rows_per_batch, point_count), dtype=np.uint8)
        self._batch_fill = 0
        self._row_sequence = 0
        self._batch_sequence = 0
        self._completed_rows = 0
        self._completed_batches = 0
        self._traces_integrated = 0
        self._completed_trace_total = 0
        self._min_traces = 0
        self._max_traces = 0
        self._jitter_total_ns = 0
        self._max_jitter_ns = 0
        self._trace_span_total_ns = 0
        self._trace_span_max_ns = 0
        self._missed_deadlines = 0
        self._empty_rows = 0
        self._metrics_started_ns = time.monotonic_ns()
        self._max_hold_update_total_ns = 0
        self._row_finalization_total_ns = 0
        self._row_finalization_max_ns = 0
        self._batch_finalization_total_ns = 0
        self._batch_finalization_max_ns = 0
        self.exchange.clear()

    def add_trace(
        self,
        values: np.ndarray,
        frame: SpectrumFrame,
        *,
        receipt_monotonic_ns: int,
        host_timestamp_ns: int,
    ) -> None:
        if values.dtype != np.uint8 or values.ndim != 1 or values.size != self._point_count or not values.flags.c_contiguous:
            raise ValueError("waterfall trace must be contiguous uint8 matching point_count")
        if frame.configuration_generation != self._configuration_generation or frame.point_count != self._point_count:
            raise ValueError("waterfall trace does not match the active configuration generation")
        metadata = _RowMetadata(
            receipt_monotonic_ns, frame.timestamp_ns, host_timestamp_ns, frame.start_frequency_hz, frame.center_frequency_hz,
            frame.stop_frequency_hz, frame.span_hz, frame.rbw_hz, frame.reference_level_dbm,
        )
        self._advance_deadline(receipt_monotonic_ns)
        self._accumulate_segment(values, 1, metadata, receipt_monotonic_ns)

    def add_packet(
        self,
        packet: np.ndarray,
        metadata: RawTraceMetadata,
        *,
        trace_timestamp_step_ns: int,
    ) -> None:
        """Vectorize native packet max-hold while preserving row boundaries."""
        if packet.dtype != np.uint8 or packet.ndim != 2 or packet.shape[1] != self._point_count or packet.shape[0] <= 0 or not packet.flags.c_contiguous:
            raise ValueError("waterfall packet must be contiguous uint8 with active point_count")
        if metadata.configuration_generation != self._configuration_generation:
            raise ValueError("waterfall packet does not match the active configuration generation")
        if trace_timestamp_step_ns < 0:
            raise ValueError("trace timestamp step must not be negative")
        count = packet.shape[0]
        if trace_timestamp_step_ns == 0:
            self._advance_deadline(metadata.receipt_monotonic_ns)
            row_metadata = _RowMetadata(
                metadata.receipt_monotonic_ns, metadata.device_timestamp_ns, metadata.host_timestamp_ns, metadata.start_frequency_hz,
                metadata.center_frequency_hz, metadata.stop_frequency_hz, metadata.span_hz,
                metadata.rbw_hz, metadata.reference_level_dbm,
            )
            self._accumulate_segment(packet, count, row_metadata, metadata.receipt_monotonic_ns)
            return
        first_receipt_ns = metadata.receipt_monotonic_ns - (count - 1) * trace_timestamp_step_ns
        first_device_ns = metadata.device_timestamp_ns - (count - 1) * trace_timestamp_step_ns
        first_host_ns = metadata.host_timestamp_ns - (count - 1) * trace_timestamp_step_ns
        index = 0
        while index < count:
            receipt_ns = first_receipt_ns + index * trace_timestamp_step_ns
            self._advance_deadline(receipt_ns)
            assert self._deadline_ns is not None
            traces_before_deadline = max(1, (self._deadline_ns - 1 - receipt_ns) // trace_timestamp_step_ns + 1)
            end = min(count, index + traces_before_deadline)
            last = end - 1
            row_metadata = _RowMetadata(
                first_receipt_ns + last * trace_timestamp_step_ns,
                first_device_ns + last * trace_timestamp_step_ns,
                first_host_ns + last * trace_timestamp_step_ns,
                metadata.start_frequency_hz,
                metadata.center_frequency_hz,
                metadata.stop_frequency_hz,
                metadata.span_hz,
                metadata.rbw_hz,
                metadata.reference_level_dbm,
            )
            self._accumulate_segment(packet[index:end], end - index, row_metadata, receipt_ns)
            index = end

    def _advance_deadline(self, receipt_monotonic_ns: int) -> None:
        if self._deadline_ns is None:
            self._deadline_ns = receipt_monotonic_ns + self._period_ns
        elif receipt_monotonic_ns >= self._deadline_ns:
            jitter = receipt_monotonic_ns - self._deadline_ns
            if self._row_trace_count:
                self._finalize_row(jitter)
            while receipt_monotonic_ns >= self._deadline_ns:
                self._deadline_ns += self._period_ns
                if receipt_monotonic_ns >= self._deadline_ns:
                    self._missed_deadlines += 1
                    self._empty_rows += 1

    def _accumulate_segment(
        self,
        values: np.ndarray,
        trace_count: int,
        metadata: _RowMetadata,
        first_receipt_ns: int,
    ) -> None:
        started = time.perf_counter_ns()
        if values.ndim == 1:
            segment_max = values
        else:
            np.max(values, axis=0, out=self._segment_max)
            segment_max = self._segment_max
        if self._row_trace_count == 0:
            np.copyto(self._row_max, segment_max)
            self._row_first_receipt_ns = first_receipt_ns
        else:
            np.maximum(self._row_max, segment_max, out=self._row_max)
        self._max_hold_update_total_ns += time.perf_counter_ns() - started
        self._row_trace_count += trace_count
        self._traces_integrated += trace_count
        self._last_metadata = metadata

    def _finalize_row(self, jitter_ns: int) -> None:
        started = time.perf_counter_ns()
        metadata = self._last_metadata
        if metadata is None or self._row_trace_count == 0:
            return
        if self._batch_fill == 0:
            self._batch_first_row_sequence = self._row_sequence
            self._batch_first_host_timestamp_ns = metadata.host_timestamp_ns
            self._batch_first_device_timestamp_ns = metadata.device_timestamp_ns
        np.copyto(self._batch_values[self._batch_fill], self._row_max)
        self._batch_fill += 1
        traces = self._row_trace_count
        trace_span_ns = max(0, metadata.receipt_monotonic_ns - self._row_first_receipt_ns)
        self._completed_rows += 1
        self._completed_trace_total += traces
        self._min_traces = traces if self._min_traces == 0 else min(self._min_traces, traces)
        self._max_traces = max(self._max_traces, traces)
        self._jitter_total_ns += jitter_ns
        self._max_jitter_ns = max(self._max_jitter_ns, jitter_ns)
        self._trace_span_total_ns += trace_span_ns
        self._trace_span_max_ns = max(self._trace_span_max_ns, trace_span_ns)
        self._row_sequence += 1
        self._row_trace_count = 0
        self._last_metadata = None
        if self._batch_fill == self._config.rows_per_batch:
            batch_started = time.perf_counter_ns()
            values = np.array(self._batch_values, dtype=np.uint8, order="C", copy=True)
            batch = WaterfallBatch(
                configuration_generation=self._configuration_generation,
                batch_sequence=self._batch_sequence,
                first_row_sequence=self._batch_first_row_sequence,
                row_count=self._config.rows_per_batch,
                point_count=self._point_count,
                first_host_timestamp_ns=self._batch_first_host_timestamp_ns,
                first_device_timestamp_ns=self._batch_first_device_timestamp_ns,
                nominal_row_period_ns=self._period_ns,
                start_frequency_hz=metadata.start_frequency_hz,
                center_frequency_hz=metadata.center_frequency_hz,
                stop_frequency_hz=metadata.stop_frequency_hz,
                span_hz=metadata.span_hz,
                rbw_hz=metadata.rbw_hz,
                reference_level_dbm=metadata.reference_level_dbm,
                values=values,
            )
            self.exchange.publish(batch)
            self._completed_batches += 1
            self._batch_sequence += 1
            self._batch_fill = 0
            batch_elapsed = time.perf_counter_ns() - batch_started
            self._batch_finalization_total_ns += batch_elapsed
            self._batch_finalization_max_ns = max(self._batch_finalization_max_ns, batch_elapsed)
        row_elapsed = time.perf_counter_ns() - started
        self._row_finalization_total_ns += row_elapsed
        self._row_finalization_max_ns = max(self._row_finalization_max_ns, row_elapsed)

    def metrics(self) -> WaterfallProducerMetrics:
        rows = self._completed_rows
        elapsed = max((time.monotonic_ns() - self._metrics_started_ns) / 1e9, 1e-9)
        return WaterfallProducerMetrics(
            target_rows_per_second=self._config.rows_per_second,
            target_batches_per_second=self._config.batches_per_second,
            rows_per_batch=self._config.rows_per_batch,
            nominal_row_period_ns=self._period_ns,
            actual_rows_per_second=rows / elapsed,
            actual_batches_per_second=self._completed_batches / elapsed,
            completed_rows=rows,
            completed_batches=self._completed_batches,
            traces_integrated=self._traces_integrated,
            minimum_traces_per_row=self._min_traces,
            maximum_traces_per_row=self._max_traces,
            mean_traces_per_row=self._completed_trace_total / rows if rows else 0.0,
            mean_row_jitter_ns=self._jitter_total_ns / rows if rows else 0.0,
            maximum_row_jitter_ns=self._max_jitter_ns,
            mean_trace_span_per_row_ns=self._trace_span_total_ns / rows if rows else 0.0,
            maximum_trace_span_per_row_ns=self._trace_span_max_ns,
            missed_row_deadlines=self._missed_deadlines,
            empty_rows=self._empty_rows,
            discarded_incomplete_rows=self._discarded_rows,
            discarded_incomplete_batch_rows=self._discarded_batch_rows,
            replaced_batches=self.exchange.replaced_batches,
            replaced_rows=self.exchange.replaced_rows,
            mean_max_hold_update_ns_per_trace=self._max_hold_update_total_ns / self._traces_integrated if self._traces_integrated else 0.0,
            mean_row_finalization_ns=self._row_finalization_total_ns / rows if rows else 0.0,
            maximum_row_finalization_ns=self._row_finalization_max_ns,
            mean_batch_finalization_ns=self._batch_finalization_total_ns / self._completed_batches if self._completed_batches else 0.0,
            maximum_batch_finalization_ns=self._batch_finalization_max_ns,
        )
