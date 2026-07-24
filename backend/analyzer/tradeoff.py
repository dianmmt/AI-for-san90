"""Measured SAN-90 resolution steps and source-independent matching helpers."""

from __future__ import annotations

import math
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .models import ResolutionTradeoffStep


VISIBLE_TIME_SPAN_SECONDS = 5.0


_TABLE_PATH = Path(__file__).resolve().parents[2] / "config" / "san90-resolution-tradeoff.json"
with _TABLE_PATH.open(encoding="utf-8") as _table_file:
    SAN90_RESOLUTION_TRADEOFF_STEPS = tuple(
        ResolutionTradeoffStep(**item) for item in json.load(_table_file)
    )


def visible_rows(rows_per_second: float, visible_time_span_seconds: float = VISIBLE_TIME_SPAN_SECONDS) -> int:
    if not math.isfinite(rows_per_second) or rows_per_second <= 0:
        raise ValueError("rows_per_second must be finite and positive")
    if not math.isfinite(visible_time_span_seconds) or visible_time_span_seconds <= 0:
        raise ValueError("visible_time_span_seconds must be finite and positive")
    return max(1, round(rows_per_second * visible_time_span_seconds))


def validate_tradeoff_index(steps: tuple[ResolutionTradeoffStep, ...], index: int) -> ResolutionTradeoffStep:
    if isinstance(index, bool) or index < 0 or index >= len(steps):
        raise ValueError(f"resolution trade-off index {index} is outside 0..{len(steps)-1}")
    step = steps[index]
    if step.index != index:
        raise ValueError("resolution trade-off capability indices are not contiguous")
    return step


def sort_and_deduplicate_steps(steps: Iterable[ResolutionTradeoffStep]) -> tuple[ResolutionTradeoffStep, ...]:
    unique: dict[tuple[float, int, int | None], ResolutionTradeoffStep] = {}
    for step in steps:
        key = (round(step.actual_rbw_hz, 3), step.point_count, step.fft_size)
        unique.setdefault(key, step)
    ordered = sorted(unique.values(), key=lambda step: (-step.actual_rbw_hz, step.point_count))
    return tuple(replace(step, id=f"rbw-step-{index}", index=index) for index, step in enumerate(ordered))


def match_actual_tradeoff_step(
    steps: tuple[ResolutionTradeoffStep, ...],
    *,
    actual_rbw_hz: float,
    point_count: int,
    fft_size: int | None,
    actual_span_hz: float | None = None,
    rbw_relative_tolerance: float = 1e-4,
) -> ResolutionTradeoffStep | None:
    for step in steps:
        if step.point_count != point_count or (fft_size is not None and step.fft_size != fft_size):
            continue
        if actual_span_hz is not None and not math.isclose(step.actual_span_hz, actual_span_hz, rel_tol=1e-6, abs_tol=2.0):
            continue
        if math.isclose(step.actual_rbw_hz, actual_rbw_hz, rel_tol=rbw_relative_tolerance, abs_tol=2.0):
            return step
    return None
