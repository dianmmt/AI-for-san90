"""Small thread-safe metric registry for the acquisition and publisher threads."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, asdict


COUNTERS = (
    "ai_traces_received_total",
    "ai_traces_used_total",
    "ai_traces_skipped_rate_limit_total",
    "ai_images_completed_total",
    "ai_images_created_total",
    "ai_images_sent_total",
    "ai_images_dropped_queue_total",
    "ai_images_dropped_no_buffer_total",
    "ai_images_dropped_send_total",
    "ai_preview_images_saved_total",
)


@dataclass(frozen=True, slots=True)
class AiLatestMetrics:
    ai_normalize_time_ms: float = 0.0
    ai_send_time_ms: float = 0.0
    ai_preview_save_time_ms: float = 0.0
    ai_latest_clipped_low_ratio: float | None = None
    ai_latest_clipped_high_ratio: float | None = None
    ai_latest_image_min_dbm: float | None = None
    ai_latest_image_max_dbm: float | None = None
    last_error: str | None = None


class AiStreamMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._counters = {name: 0 for name in COUNTERS}
        self._latest = AiLatestMetrics()
        self._sent_times: deque[float] = deque(maxlen=512)
        self._created_times: deque[float] = deque(maxlen=512)

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self._counters:
            raise KeyError(name)
        with self._lock:
            self._counters[name] += amount
            now = time.monotonic()
            if name == "ai_images_sent_total":
                self._sent_times.extend([now] * amount)
            elif name == "ai_images_created_total":
                self._created_times.extend([now] * amount)

    def update_latest(self, **values: float | str | None) -> None:
        with self._lock:
            current = asdict(self._latest)
            current.update(values)
            self._latest = AiLatestMetrics(**current)

    def snapshot(self, *, queue_depth: int, free_buffer_count: int) -> dict[str, object]:
        with self._lock:
            now = time.monotonic()
            while self._sent_times and self._sent_times[0] < now - 1.0:
                self._sent_times.popleft()
            while self._created_times and self._created_times[0] < now - 1.0:
                self._created_times.popleft()
            counters = dict(self._counters)
            latest = asdict(self._latest)
            elapsed = max(1e-9, time.monotonic() - self._started)
        return {
            **counters,
            **latest,
            "ai_queue_depth": queue_depth,
            "ai_free_buffer_count": free_buffer_count,
            "ai_actual_output_fps": float(len(self._sent_times)),
            "ai_created_fps": float(len(self._created_times)),
            "ai_lifetime_sent_fps": counters["ai_images_sent_total"] / elapsed,
            "ai_lifetime_created_fps": counters["ai_images_created_total"] / elapsed,
        }
