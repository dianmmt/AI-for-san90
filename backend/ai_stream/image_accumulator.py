"""Preallocated, non-blocking chronological 640-trace image accumulation."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.analyzer.raw_buffers import RawAmplitudeMapping, RawTraceMetadata

from .metrics import AiStreamMetrics
from .power_profiles import PowerProfile
from .protocol import CaptureMetadata


@dataclass(slots=True)
class ImageBuffer:
    identifier: int
    dbm: np.ndarray
    gray8: np.ndarray
    workspace: np.ndarray


@dataclass(slots=True)
class CompletedImage:
    buffer: ImageBuffer
    capture: CaptureMetadata


def raw_packet_to_dbm(
    raw: np.ndarray,
    mapping: RawAmplitudeMapping,
    *,
    output: np.ndarray | None = None,
) -> np.ndarray:
    if raw.ndim != 2 or raw.dtype != np.uint8 or not raw.flags.c_contiguous:
        raise ValueError("raw AI packet must be a contiguous two-dimensional uint8 array")
    if output is None:
        output = np.empty(raw.shape, dtype=np.float32)
    if output.shape != raw.shape or output.dtype != np.float32 or not output.flags.c_contiguous:
        raise ValueError("raw-to-dBm output must match the packet as contiguous float32")
    mapping.convert(raw, output)
    return output


def resize_frequency_linear(
    source: np.ndarray,
    width: int = 640,
    *,
    output: np.ndarray | None = None,
) -> np.ndarray:
    if source.ndim != 2 or source.dtype != np.float32 or not source.flags.c_contiguous or source.shape[1] < 1:
        raise ValueError("frequency resize source must be contiguous two-dimensional float32")
    if width <= 0:
        raise ValueError("frequency resize width must be positive")
    rows, source_width = source.shape
    if output is None:
        output = np.empty((rows, width), dtype=np.float32)
    if output.shape != (rows, width) or output.dtype != np.float32 or not output.flags.c_contiguous:
        raise ValueError("frequency resize output has invalid dimensions")
    if source_width == width:
        np.copyto(output, source)
        return output
    if source_width == 1:
        output[:] = source[:, :1]
        return output
    positions = np.linspace(0.0, source_width - 1, width, dtype=np.float32)
    left = np.floor(positions).astype(np.intp)
    right = np.minimum(left + 1, source_width - 1)
    fraction = positions - left
    # Advanced indexing is block-vectorized: there is no Python frequency-bin loop.
    np.multiply(source[:, left], 1.0 - fraction, out=output)
    output += source[:, right] * fraction
    return output


class AiImageAccumulator:
    """Single acquisition-thread writer with a bounded worker handoff."""

    def __init__(
        self,
        *,
        target_images_per_second: float,
        queue_size: int,
        buffer_pool_size: int,
        profile_provider: Callable[[], PowerProfile],
        metrics: AiStreamMetrics,
    ) -> None:
        if not 7.0 <= target_images_per_second <= 10.0:
            raise ValueError("AI image target must be between 7 and 10 images/s")
        self.target_images_per_second = target_images_per_second
        self._period_ns = round(1e9 / target_images_per_second)
        self._profile_provider = profile_provider
        self.metrics = metrics
        self.completed: queue.Queue[CompletedImage] = queue.Queue(maxsize=queue_size)
        self.free: queue.Queue[ImageBuffer] = queue.Queue(maxsize=buffer_pool_size)
        for identifier in range(buffer_pool_size):
            self.free.put_nowait(ImageBuffer(
                identifier,
                np.empty((640, 640), dtype=np.float32),
                np.empty((640, 640), dtype=np.uint8),
                np.empty((640, 640), dtype=np.float32),
            ))
        self._packet_dbm: np.ndarray | None = None
        self._resize_scratch: np.ndarray | None = None
        self._resize_left: np.ndarray | None = None
        self._resize_right: np.ndarray | None = None
        self._resize_fraction: np.ndarray | None = None
        self._configured_shape: tuple[int, int] | None = None
        self._generation: int | None = None
        self._window_position = 0
        self._window_selected = False
        self._active: ImageBuffer | None = None
        self._active_profile: PowerProfile | None = None
        self._active_first_sequence = 0
        self._active_start_timestamp_ns = 0
        self._next_selected_start_ns: int | None = None
        self._image_sequence = 0
        self._enabled = True

    def configure(self, packet_frames: int, frame_width: int, generation: int) -> None:
        if packet_frames < 1 or frame_width < 1:
            raise ValueError("AI packet dimensions must be positive")
        self._return_active()
        self._packet_dbm = np.empty((packet_frames, frame_width), dtype=np.float32)
        self._resize_scratch = np.empty((packet_frames, 640), dtype=np.float32)
        if frame_width == 640:
            self._resize_left = self._resize_right = self._resize_fraction = None
        elif frame_width == 1:
            self._resize_left = np.zeros(640, dtype=np.intp)
            self._resize_right = np.zeros(640, dtype=np.intp)
            self._resize_fraction = np.zeros(640, dtype=np.float32)
        else:
            positions = np.linspace(0.0, frame_width - 1, 640, dtype=np.float32)
            self._resize_left = np.floor(positions).astype(np.intp)
            self._resize_right = np.minimum(self._resize_left + 1, frame_width - 1)
            self._resize_fraction = positions - self._resize_left
        self._configured_shape = (packet_frames, frame_width)
        self._generation = generation
        self._window_position = 0
        self._window_selected = False
        self._next_selected_start_ns = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._return_active()
            self._window_position = 0
            self._window_selected = False
            self._next_selected_start_ns = None

    def offer_packet(
        self,
        raw: np.ndarray,
        metadata: RawTraceMetadata,
        *,
        trace_timestamp_step_ns: int,
    ) -> None:
        """Process one SDK packet without any blocking queue operation."""
        count, width = raw.shape
        self.metrics.increment("ai_traces_received_total", count)
        if not self._enabled:
            return
        if self._configured_shape != (count, width) or self._generation != metadata.configuration_generation:
            self.configure(count, width, metadata.configuration_generation)
        assert self._packet_dbm is not None
        raw_packet_to_dbm(raw, metadata.mapping, output=self._packet_dbm)
        packet_first_sequence = metadata.sequence - count + 1
        # The installed firmware's auxiliary nsSinceEpoch value can move
        # backwards between packets. Anchor capture metadata to the host epoch
        # timestamp and the SDK-derived per-trace interval; rate selection uses
        # the independent monotonic receipt timeline below.
        packet_first_capture_timestamp = metadata.host_timestamp_ns - (count - 1) * trace_timestamp_step_ns
        packet_first_schedule_timestamp = metadata.receipt_monotonic_ns - (count - 1) * trace_timestamp_step_ns
        index = 0
        while index < count:
            if self._window_position == 0:
                window_schedule_ns = packet_first_schedule_timestamp + index * trace_timestamp_step_ns
                window_capture_ns = packet_first_capture_timestamp + index * trace_timestamp_step_ns
                due = self._next_selected_start_ns is None or window_schedule_ns >= self._next_selected_start_ns
                if due:
                    try:
                        self._active = self.free.get_nowait()
                    except queue.Empty:
                        self._active = None
                        self.metrics.increment("ai_images_dropped_no_buffer_total")
                    self._window_selected = self._active is not None
                    if self._window_selected:
                        self._active_profile = self._profile_provider()
                        self._active_first_sequence = packet_first_sequence + index
                        self._active_start_timestamp_ns = window_capture_ns
                    if self._next_selected_start_ns is None:
                        self._next_selected_start_ns = window_schedule_ns + self._period_ns
                    else:
                        while self._next_selected_start_ns <= window_schedule_ns:
                            self._next_selected_start_ns += self._period_ns
                else:
                    self._window_selected = False

            take = min(640 - self._window_position, count - index)
            if self._window_selected and self._active is not None:
                destination = self._active.dbm[self._window_position:self._window_position + take]
                self._resize_selected(self._packet_dbm[index:index + take], destination)
                self.metrics.increment("ai_traces_used_total", take)
            else:
                self.metrics.increment("ai_traces_skipped_rate_limit_total", take)
            self._window_position += take
            index += take

            if self._window_position == 640:
                self.metrics.increment("ai_images_completed_total")
                if self._window_selected and self._active is not None and self._active_profile is not None:
                    last_sequence = packet_first_sequence + index - 1
                    last_timestamp = packet_first_capture_timestamp + (index - 1) * trace_timestamp_step_ns
                    capture = CaptureMetadata(
                        sequence=self._image_sequence,
                        first_trace_sequence=self._active_first_sequence,
                        last_trace_sequence=last_sequence,
                        capture_start_timestamp_ns=self._active_start_timestamp_ns,
                        capture_end_timestamp_ns=last_timestamp,
                        center_frequency_hz=metadata.center_frequency_hz,
                        start_frequency_hz=metadata.start_frequency_hz,
                        stop_frequency_hz=metadata.stop_frequency_hz,
                        frame_width_source=width,
                        configuration_generation=metadata.configuration_generation,
                        power_profile=self._active_profile,
                    )
                    self._image_sequence += 1
                    self.metrics.increment("ai_images_created_total")
                    self._publish_nonblocking(CompletedImage(self._active, capture))
                self._active = None
                self._active_profile = None
                self._window_position = 0
                self._window_selected = False

    def _resize_selected(self, source: np.ndarray, output: np.ndarray) -> None:
        if source.shape[1] == 640:
            np.copyto(output, source)
            return
        assert self._resize_scratch is not None
        assert self._resize_left is not None and self._resize_right is not None and self._resize_fraction is not None
        rows = source.shape[0]
        scratch = self._resize_scratch[:rows]
        np.take(source, self._resize_left, axis=1, out=output)
        np.multiply(output, 1.0 - self._resize_fraction, out=output)
        np.take(source, self._resize_right, axis=1, out=scratch)
        np.multiply(scratch, self._resize_fraction, out=scratch)
        np.add(output, scratch, out=output)

    def _publish_nonblocking(self, image: CompletedImage) -> None:
        try:
            self.completed.put_nowait(image)
            return
        except queue.Full:
            pass
        try:
            oldest = self.completed.get_nowait()
        except queue.Empty:
            self.metrics.increment("ai_images_dropped_queue_total")
            self.release(image.buffer)
            return
        self.metrics.increment("ai_images_dropped_queue_total")
        self.release(oldest.buffer)
        try:
            self.completed.put_nowait(image)
        except queue.Full:
            self.metrics.increment("ai_images_dropped_queue_total")
            self.release(image.buffer)

    def release(self, buffer: ImageBuffer) -> None:
        try:
            self.free.put_nowait(buffer)
        except queue.Full as error:
            raise RuntimeError("AI buffer was returned more than once") from error

    def _return_active(self) -> None:
        if self._active is not None:
            self.release(self._active)
        self._active = None
        self._active_profile = None

    def drain(self) -> None:
        self._return_active()
        while True:
            try:
                self.release(self.completed.get_nowait().buffer)
            except queue.Empty:
                break

    @property
    def queue_depth(self) -> int:
        return self.completed.qsize()

    @property
    def free_buffer_count(self) -> int:
        return self.free.qsize()
