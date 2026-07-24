"""Validated environment configuration for the AI image publisher."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .power_profiles import DEFAULT_POWER_PROFILE, require_power_profile


def _boolean(name: str, default: bool) -> bool:
    text = os.getenv(name, "true" if default else "false").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class AiStreamConfig:
    enabled: bool = True
    target_images_per_second: float = 10.0
    image_width: int = 640
    image_height: int = 640
    traces_per_image: int = 640
    power_profile: str = DEFAULT_POWER_PROFILE
    bind: str = "tcp://0.0.0.0:5557"
    queue_size: int = 2
    buffer_pool_size: int = 4
    drop_policy: str = "drop_oldest"
    preview_enabled: bool = False
    preview_directory: Path = Path("data/ai_preview")
    preview_save_interval_seconds: float = 1.0
    preview_max_files: int = 20
    send_high_water_mark: int = 2
    send_timeout_ms: int = 5
    socket_linger_ms: int = 0
    clipped_high_warning_ratio: float = 0.001
    recurring_log_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (self.image_width, self.image_height, self.traces_per_image) != (640, 640, 640):
            raise ValueError("AI image width, height, and traces per image must all be exactly 640")
        if not math.isfinite(self.target_images_per_second) or not 7.0 <= self.target_images_per_second <= 10.0:
            raise ValueError("AI_TARGET_IMAGES_PER_SECOND must be between 7 and 10")
        require_power_profile(self.power_profile)
        if self.queue_size < 1:
            raise ValueError("AI_QUEUE_SIZE must be positive")
        if self.buffer_pool_size < self.queue_size + 2:
            raise ValueError("AI_BUFFER_POOL_SIZE must be at least AI_QUEUE_SIZE + 2")
        if self.drop_policy != "drop_oldest":
            raise ValueError("AI_DROP_POLICY currently supports only 'drop_oldest'")
        if not self.bind.startswith("tcp://"):
            raise ValueError("AI_STREAM_BIND must be a tcp:// endpoint")
        if self.preview_save_interval_seconds <= 0 or self.preview_max_files < 1:
            raise ValueError("AI preview interval and file limit must be positive")
        if self.send_high_water_mark < 1 or self.send_timeout_ms < 0 or self.socket_linger_ms < 0:
            raise ValueError("AI ZeroMQ limits must be non-negative with a positive HWM")
        if not 0 <= self.clipped_high_warning_ratio <= 1:
            raise ValueError("AI clipping warning ratio must be between zero and one")

    @classmethod
    def from_environment(cls) -> "AiStreamConfig":
        return cls(
            enabled=_boolean("AI_STREAM_ENABLED", True),
            target_images_per_second=float(os.getenv("AI_TARGET_IMAGES_PER_SECOND", "10.0")),
            image_width=int(os.getenv("AI_IMAGE_WIDTH", "640")),
            image_height=int(os.getenv("AI_IMAGE_HEIGHT", "640")),
            traces_per_image=int(os.getenv("AI_TRACES_PER_IMAGE", "640")),
            power_profile=os.getenv("AI_POWER_PROFILE", DEFAULT_POWER_PROFILE),
            bind=os.getenv("AI_STREAM_BIND", "tcp://0.0.0.0:5557"),
            queue_size=int(os.getenv("AI_QUEUE_SIZE", "2")),
            buffer_pool_size=int(os.getenv("AI_BUFFER_POOL_SIZE", "4")),
            drop_policy=os.getenv("AI_DROP_POLICY", "drop_oldest"),
            preview_enabled=_boolean("AI_PREVIEW_ENABLED", False),
            preview_directory=Path(os.getenv("AI_PREVIEW_DIRECTORY", "data/ai_preview")),
            preview_save_interval_seconds=float(os.getenv("AI_PREVIEW_SAVE_INTERVAL_SECONDS", "1.0")),
            preview_max_files=int(os.getenv("AI_PREVIEW_MAX_FILES", "20")),
            send_high_water_mark=int(os.getenv("AI_SEND_HIGH_WATER_MARK", "2")),
            send_timeout_ms=int(os.getenv("AI_SEND_TIMEOUT_MS", "5")),
            socket_linger_ms=int(os.getenv("AI_SOCKET_LINGER_MS", "0")),
            clipped_high_warning_ratio=float(os.getenv("AI_CLIPPED_HIGH_WARNING_RATIO", "0.001")),
            recurring_log_interval_seconds=float(os.getenv("AI_LOG_INTERVAL_SECONDS", "5.0")),
        )
