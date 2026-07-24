"""Bounded latest-frame and interval max-hold buffers."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock

import numpy as np

from .models import FrameType, SpectrumFrame


def clone_frame(frame: SpectrumFrame, *, frame_type: FrameType | None = None) -> SpectrumFrame:
    """Detach a frame from an acquisition buffer owned by another thread."""
    values = np.array(frame.values, dtype=np.float32, order="C", copy=True)
    return replace(frame, values=values, frame_type=frame_type or frame.frame_type)


class LatestFrameBuffer:
    """One-slot buffer where newer data replaces unpublished old data."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: SpectrumFrame | None = None
        self._generation = 0
        self._last_read_generation = 0
        self._replaced = 0

    @property
    def replaced(self) -> int:
        with self._lock:
            return self._replaced

    def publish(self, frame: SpectrumFrame) -> None:
        detached = clone_frame(frame)
        with self._lock:
            if self._frame is not None and self._last_read_generation < self._generation:
                self._replaced += 1
            self._frame = detached
            self._generation += 1

    def read(self) -> SpectrumFrame | None:
        with self._lock:
            if self._frame is None or self._last_read_generation == self._generation:
                return None
            self._last_read_generation = self._generation
            return clone_frame(self._frame)


class IntervalMaxHoldBuffer:
    """Accumulates element-wise maxima until a consumer takes the interval."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._template: SpectrumFrame | None = None
        self._values: np.ndarray | None = None

    def accumulate(self, frame: SpectrumFrame) -> None:
        with self._lock:
            if self._values is None or self._values.size != frame.point_count:
                self._values = np.array(frame.values, dtype=np.float32, order="C", copy=True)
            else:
                np.maximum(self._values, frame.values, out=self._values)
            self._template = frame

    def take(self) -> SpectrumFrame | None:
        with self._lock:
            if self._template is None or self._values is None:
                return None
            values = np.array(self._values, dtype=np.float32, order="C", copy=True)
            result = replace(self._template, values=values, frame_type=FrameType.MAX_HOLD)
            self._template = None
            self._values = None
            return result
