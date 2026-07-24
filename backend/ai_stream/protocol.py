"""Versioned two-part JSON + raw GRAY8 ZeroMQ protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .power_profiles import PowerProfile


PROTOCOL_VERSION = 1
MESSAGE_TYPE = "san90_gray8_waterfall"
PIXEL_FORMAT = "GRAY8"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
PAYLOAD_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT


@dataclass(frozen=True, slots=True)
class CaptureMetadata:
    sequence: int
    first_trace_sequence: int
    last_trace_sequence: int
    capture_start_timestamp_ns: int
    capture_end_timestamp_ns: int
    center_frequency_hz: float
    start_frequency_hz: float
    stop_frequency_hz: float
    frame_width_source: int
    configuration_generation: int
    power_profile: PowerProfile


def build_metadata(capture: CaptureMetadata, waterfall_dbm: np.ndarray) -> dict[str, Any]:
    if waterfall_dbm.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or waterfall_dbm.dtype != np.float32:
        raise ValueError("AI metadata requires a 640x640 float32 image")
    profile = capture.power_profile
    minimum = float(np.min(waterfall_dbm))
    maximum = float(np.max(waterfall_dbm))
    clipped_low = float(np.count_nonzero(waterfall_dbm <= profile.min_dbm) / PAYLOAD_SIZE)
    clipped_high = float(np.count_nonzero(waterfall_dbm >= profile.max_dbm) / PAYLOAD_SIZE)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": MESSAGE_TYPE,
        "source": "SAN-90",
        "sequence": capture.sequence,
        "timestamp_ns": capture.capture_end_timestamp_ns,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "channels": 1,
        "pixel_format": PIXEL_FORMAT,
        "dtype": "uint8",
        "memory_order": "C",
        "payload_size_bytes": PAYLOAD_SIZE,
        "center_frequency_hz": capture.center_frequency_hz,
        "start_frequency_hz": capture.start_frequency_hz,
        "stop_frequency_hz": capture.stop_frequency_hz,
        "power_profile": profile.name,
        "power_min_dbm": profile.min_dbm,
        "power_max_dbm": profile.max_dbm,
        "db_per_gray_level": profile.db_per_gray_level,
        "trace_count": IMAGE_HEIGHT,
        "first_trace_sequence": capture.first_trace_sequence,
        "last_trace_sequence": capture.last_trace_sequence,
        "capture_start_timestamp_ns": capture.capture_start_timestamp_ns,
        "capture_end_timestamp_ns": capture.capture_end_timestamp_ns,
        "frame_width_source": capture.frame_width_source,
        "frame_width_output": IMAGE_WIDTH,
        "configuration_generation": capture.configuration_generation,
        "clipped_low_ratio": clipped_low,
        "clipped_high_ratio": clipped_high,
        "image_min_dbm": minimum,
        "image_max_dbm": maximum,
    }


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    validate_metadata(metadata)
    return json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "protocol_version", "message_type", "sequence", "timestamp_ns", "width", "height",
        "channels", "pixel_format", "dtype", "memory_order", "payload_size_bytes",
        "center_frequency_hz", "start_frequency_hz", "stop_frequency_hz", "power_profile",
        "power_min_dbm", "power_max_dbm", "db_per_gray_level", "trace_count",
        "first_trace_sequence", "last_trace_sequence", "capture_start_timestamp_ns",
        "capture_end_timestamp_ns", "frame_width_source", "frame_width_output",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"AI metadata is missing fields: {sorted(missing)}")
    if metadata["protocol_version"] != PROTOCOL_VERSION or metadata["message_type"] != MESSAGE_TYPE:
        raise ValueError("unsupported AI image protocol")
    if (metadata["width"], metadata["height"], metadata["channels"]) != (640, 640, 1):
        raise ValueError("AI image dimensions must be 640x640x1")
    if metadata["pixel_format"] != PIXEL_FORMAT or metadata["dtype"] != "uint8" or metadata["memory_order"] != "C":
        raise ValueError("AI image must be row-major GRAY8 uint8")
    if metadata["payload_size_bytes"] != PAYLOAD_SIZE or metadata["trace_count"] != 640:
        raise ValueError("AI image payload dimensions are invalid")
    numeric = (
        "center_frequency_hz", "start_frequency_hz", "stop_frequency_hz", "power_min_dbm",
        "power_max_dbm", "db_per_gray_level",
    )
    if not all(math.isfinite(float(metadata[name])) for name in numeric):
        raise ValueError("AI metadata contains non-finite numeric values")


def validate_multipart(metadata_bytes: bytes, payload: bytes | memoryview) -> tuple[dict[str, Any], np.ndarray]:
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid AI metadata JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("AI metadata JSON must be an object")
    validate_metadata(metadata)
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"AI payload must contain exactly {PAYLOAD_SIZE} bytes")
    image = np.frombuffer(payload, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    return metadata, image
