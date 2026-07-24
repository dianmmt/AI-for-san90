"""Lossless, rate-limited diagnostic PNG/JSON preview output."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image


class PreviewWriter:
    def __init__(self, directory: Path, interval_seconds: float, max_files: int) -> None:
        self.directory = directory
        self.interval_seconds = interval_seconds
        self.max_files = max_files
        self._last_saved = 0.0
        self._latest_png: bytes | None = None
        self._lock = Lock()

    def maybe_save(self, image: np.ndarray, metadata: dict[str, Any]) -> tuple[bool, float]:
        now = time.monotonic()
        if now - self._last_saved < self.interval_seconds:
            return False, 0.0
        if image.shape != (640, 640) or image.dtype != np.uint8:
            raise ValueError("preview image must be 640x640 GRAY8")
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        center_mhz = float(metadata["center_frequency_hz"]) / 1e6
        stem = (
            f"san90_gray8_{int(metadata['sequence']):012d}_{metadata['power_profile']}_"
            f"{center_mhz:.3f}MHz_{int(metadata['timestamp_ns'])}"
        )
        png_path = self.directory / f"{stem}.png"
        json_path = self.directory / f"{stem}.json"
        pil_image = Image.fromarray(image, mode="L")
        encoded = io.BytesIO()
        pil_image.save(encoded, format="PNG")
        png_bytes = encoded.getvalue()
        png_path.write_bytes(png_bytes)
        json_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with self._lock:
            self._latest_png = png_bytes
        self._last_saved = now
        self._rotate()
        return True, (time.perf_counter() - started) * 1000.0

    def latest_png(self) -> bytes | None:
        with self._lock:
            return self._latest_png

    def _rotate(self) -> None:
        images = sorted(self.directory.glob("san90_gray8_*.png"), key=lambda path: path.stat().st_mtime_ns)
        for path in images[:-self.max_files]:
            json_path = path.with_suffix(".json")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            try:
                json_path.unlink()
            except FileNotFoundError:
                pass
