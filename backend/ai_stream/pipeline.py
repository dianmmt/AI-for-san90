"""Facade joining acquisition-thread accumulation to the publisher worker."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from backend.analyzer.raw_buffers import RawTraceMetadata

from .config import AiStreamConfig
from .image_accumulator import AiImageAccumulator
from .image_publisher import AiImagePublisher
from .metrics import AiStreamMetrics
from .power_profiles import PowerProfile, require_power_profile


logger = logging.getLogger("san90.ai_stream")


class AiStreamPipeline:
    def __init__(self, config: AiStreamConfig | None = None) -> None:
        self.config = config or AiStreamConfig.from_environment()
        self.metrics = AiStreamMetrics()
        self._profile_lock = threading.Lock()
        self._profile = require_power_profile(self.config.power_profile)
        self._enabled = self.config.enabled
        self.accumulator = AiImageAccumulator(
            target_images_per_second=self.config.target_images_per_second,
            queue_size=self.config.queue_size,
            buffer_pool_size=self.config.buffer_pool_size,
            profile_provider=self._current_profile,
            metrics=self.metrics,
        )
        self.accumulator.set_enabled(self._enabled)
        self.publisher = AiImagePublisher(self.config, self.accumulator, self.metrics)
        self._last_ingest_error_log = 0.0

    def start(self) -> None:
        if self._enabled:
            self.publisher.start()

    def stop(self) -> None:
        self.accumulator.set_enabled(False)
        self.publisher.stop()
        self.accumulator.drain()

    def configure(self, packet_frames: int, frame_width: int, generation: int) -> None:
        self.accumulator.configure(packet_frames, frame_width, generation)

    def offer_packet(self, raw: np.ndarray, metadata: RawTraceMetadata, *, trace_timestamp_step_ns: int) -> None:
        if self._enabled:
            try:
                self.accumulator.offer_packet(raw, metadata, trace_timestamp_step_ns=trace_timestamp_step_ns)
            # This is the fault boundary protecting the sole SDK acquisition
            # loop. Failures are counted and rate-limited in logs, never hidden.
            except Exception as error:
                self.metrics.update_latest(last_error=str(error))
                now = time.monotonic()
                if now - self._last_ingest_error_log >= self.config.recurring_log_interval_seconds:
                    logger.error("AI packet branch dropped data without stopping acquisition: %s", error)
                    self._last_ingest_error_log = now

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self.accumulator.set_enabled(enabled)
        if enabled:
            self.publisher.start()
        else:
            self.publisher.stop()
            self.accumulator.drain()

    def start_publisher(self) -> None:
        self.publisher.start()

    def stop_publisher(self) -> None:
        self.publisher.stop()
        self.accumulator.drain()

    def set_accepting(self, enabled: bool) -> None:
        """Change acquisition acceptance on the single SDK owner thread."""
        self._enabled = enabled
        self.accumulator.set_enabled(enabled)

    def set_power_profile(self, name: str) -> PowerProfile:
        profile = require_power_profile(name)
        with self._profile_lock:
            self._profile = profile
        return profile

    def _current_profile(self) -> PowerProfile:
        with self._profile_lock:
            return self._profile

    def status(self) -> dict[str, object]:
        profile = self._current_profile()
        return {
            "enabled": self._enabled,
            "bound": self.publisher.bound,
            "connected_or_sending": self.publisher.connected_or_sending,
            "active_power_profile": profile.name,
            "power_min_dbm": profile.min_dbm,
            "power_max_dbm": profile.max_dbm,
            "target_images_per_second": self.config.target_images_per_second,
            "image_width": 640,
            "image_height": 640,
            "pixel_format": "GRAY8",
            "payload_size_bytes": 409600,
            "bind": self.config.bind,
            **self.metrics.snapshot(
                queue_depth=self.accumulator.queue_depth,
                free_buffer_count=self.accumulator.free_buffer_count,
            ),
        }

    def latest_preview_png(self) -> bytes | None:
        return self.publisher.latest_preview_png()
