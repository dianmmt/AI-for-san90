"""Independent ZeroMQ/preview worker; never runs on the SDK owner thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .config import AiStreamConfig
from .image_accumulator import AiImageAccumulator, CompletedImage
from .metrics import AiStreamMetrics
from .power_profiles import dbm_to_gray8
from .preview import PreviewWriter
from .protocol import build_metadata, encode_metadata


logger = logging.getLogger("san90.ai_stream")


class AiImagePublisher:
    def __init__(self, config: AiStreamConfig, accumulator: AiImageAccumulator, metrics: AiStreamMetrics) -> None:
        self.config = config
        self.accumulator = accumulator
        self.metrics = metrics
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._preview = PreviewWriter(
            config.preview_directory,
            config.preview_save_interval_seconds,
            config.preview_max_files,
        ) if config.preview_enabled else None
        self._last_error_log = 0.0
        self._last_clip_log = 0.0
        self._bound = False
        self._last_sent_monotonic: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="san90-ai-publisher", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1.0)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("AI publisher did not stop within %.1f seconds", timeout)
        self._thread = None
        self._bound = False

    def _run(self) -> None:
        socket: Any = None
        context: Any = None
        try:
            try:
                import zmq
            except ImportError as error:
                self.metrics.update_latest(last_error="pyzmq is required when AI_STREAM_ENABLED=true")
                self._ready.set()
                self._drain_without_transport()
                return
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, self.config.send_high_water_mark)
            socket.setsockopt(zmq.SNDTIMEO, self.config.send_timeout_ms)
            socket.setsockopt(zmq.LINGER, self.config.socket_linger_ms)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.bind(self.config.bind)
            self._bound = True
            self.metrics.update_latest(last_error=None)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    image = self.accumulator.completed.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._publish_one(socket, zmq, image)
        except Exception as error:
            self.metrics.update_latest(last_error=str(error))
            self._rate_limited_error("AI publisher failed", error)
            self._ready.set()
            self._drain_without_transport()
        finally:
            self._ready.set()
            if socket is not None:
                socket.close(linger=self.config.socket_linger_ms)
            if context is not None:
                context.term()

    def _publish_one(self, socket: Any, zmq: Any, image: CompletedImage) -> None:
        try:
            normalize_started = time.perf_counter()
            metadata = build_metadata(image.capture, image.buffer.dbm)
            dbm_to_gray8(
                image.buffer.dbm,
                image.capture.power_profile.min_dbm,
                image.capture.power_profile.max_dbm,
                output=image.buffer.gray8,
                workspace=image.buffer.workspace,
            )
            normalize_ms = (time.perf_counter() - normalize_started) * 1000.0
            self.metrics.update_latest(
                ai_normalize_time_ms=normalize_ms,
                ai_latest_clipped_low_ratio=metadata["clipped_low_ratio"],
                ai_latest_clipped_high_ratio=metadata["clipped_high_ratio"],
                ai_latest_image_min_dbm=metadata["image_min_dbm"],
                ai_latest_image_max_dbm=metadata["image_max_dbm"],
            )
            if float(metadata["clipped_high_ratio"]) > self.config.clipped_high_warning_ratio:
                now = time.monotonic()
                if now - self._last_clip_log >= self.config.recurring_log_interval_seconds:
                    logger.warning(
                        "AI image high clipping sequence=%s ratio=%.6f profile=%s; profile is unchanged",
                        metadata["sequence"], metadata["clipped_high_ratio"], metadata["power_profile"],
                    )
                    self._last_clip_log = now

            send_started = time.perf_counter()
            try:
                socket.send_multipart(
                    [encode_metadata(metadata), memoryview(image.buffer.gray8)],
                    flags=zmq.NOBLOCK,
                    copy=True,
                )
                self.metrics.increment("ai_images_sent_total")
                self._last_sent_monotonic = time.monotonic()
            except zmq.Again:
                self.metrics.increment("ai_images_dropped_send_total")
            self.metrics.update_latest(ai_send_time_ms=(time.perf_counter() - send_started) * 1000.0)

            if self._preview is not None:
                saved, elapsed_ms = self._preview.maybe_save(image.buffer.gray8, metadata)
                if saved:
                    self.metrics.increment("ai_preview_images_saved_total")
                    self.metrics.update_latest(ai_preview_save_time_ms=elapsed_ms)
        except Exception as error:
            self.metrics.increment("ai_images_dropped_send_total")
            self.metrics.update_latest(last_error=str(error))
            self._rate_limited_error("AI image processing failed", error)
        finally:
            self.accumulator.release(image.buffer)

    def _drain_without_transport(self) -> None:
        while not self._stop.is_set():
            try:
                image = self.accumulator.completed.get(timeout=0.1)
            except queue.Empty:
                continue
            self.metrics.increment("ai_images_dropped_send_total")
            self.accumulator.release(image.buffer)

    def _rate_limited_error(self, message: str, error: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= self.config.recurring_log_interval_seconds:
            logger.error("%s: %s", message, error)
            self._last_error_log = now

    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def connected_or_sending(self) -> bool:
        return self._last_sent_monotonic is not None and time.monotonic() - self._last_sent_monotonic < 2.0

    def latest_preview_png(self) -> bytes | None:
        return None if self._preview is None else self._preview.latest_png()
