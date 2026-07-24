"""Typed application data exchanged by all analyzer sources."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


Float32Array = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]


class FrameType(str, Enum):
    CURRENT = "current"
    MAX_HOLD = "max_hold"
    DENSITY = "density"
    WATERFALL = "waterfall"


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: float
    maximum: float
    step: float | None = None


@dataclass(frozen=True, slots=True)
class ResolutionTradeoffStep:
    """One hardware-verified manual-RBW operating point."""

    id: str
    index: int
    requested_rbw_hz: float
    actual_rbw_hz: float
    point_count: int
    fft_size: int | None
    measured_trace_rate_hz: float | None
    spectrum_publish_fps: float
    spectrum_render_fps: float
    webgl_target_fps: float
    waterfall_rows_per_second: float
    waterfall_batches_per_second: float
    waterfall_rows_per_batch: int
    frequency_bin_spacing_hz: float
    nominal_time_per_row_s: float
    actual_span_hz: float
    label: str | None = None
    poi_s: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or self.point_count <= 0 or self.waterfall_rows_per_batch <= 0:
            raise ValueError("trade-off index and dimensions are invalid")
        if self.fft_size is not None and self.fft_size <= 0:
            raise ValueError("trade-off FFT size must be positive")
        numeric = (
            self.requested_rbw_hz, self.actual_rbw_hz, self.spectrum_publish_fps,
            self.spectrum_render_fps, self.webgl_target_fps, self.waterfall_rows_per_second,
            self.waterfall_batches_per_second, self.frequency_bin_spacing_hz,
            self.nominal_time_per_row_s, self.actual_span_hz,
        )
        if not all(np.isfinite(value) and value > 0 for value in numeric):
            raise ValueError("trade-off numeric values must be finite and positive")
        if self.measured_trace_rate_hz is not None and (
            not np.isfinite(self.measured_trace_rate_hz) or self.measured_trace_rate_hz <= 0
        ):
            raise ValueError("trade-off trace rate must be finite and positive")
        if self.spectrum_publish_fps != 60:
            raise ValueError("spectrum publish target must be fixed at 60 FPS")
        if self.spectrum_render_fps != 60:
            raise ValueError("spectrum render target must be fixed at 60 FPS")
        if self.webgl_target_fps != 60:
            raise ValueError("WebGL target must be fixed at 60 FPS")
        if not np.isclose(
            self.waterfall_rows_per_second,
            self.waterfall_batches_per_second * self.waterfall_rows_per_batch,
        ):
            raise ValueError("trade-off waterfall rate and batch dimensions are inconsistent")


@dataclass(frozen=True, slots=True)
class AnalyzerCapabilities:
    source: str
    measurement_modes: tuple[str, ...]
    supported_controls: frozenset[str]
    numeric_ranges: Mapping[str, NumericRange] = field(default_factory=dict)
    enum_values: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    native_point_counts: tuple[int, ...] = ()
    supports_density: bool = False
    center_frequency_min_hz: float | None = None
    center_frequency_max_hz: float | None = None
    center_frequency_step_hz: float | None = None
    reference_level_min_dbm: float | None = None
    reference_level_max_dbm: float | None = None
    reference_level_step_db: float | None = None
    supported_attenuation_values_db: tuple[int, ...] | None = None
    supports_automatic_attenuation: bool = False
    preamplifier_modes: tuple[str, ...] = ()
    gain_strategy_modes: tuple[str, ...] = ()
    supports_live_frequency_change: bool = False
    supports_live_amplitude_change: bool = False
    requires_restart_for_frequency: bool = True
    requires_restart_for_amplitude: bool = True
    supports_rbw_control: bool = False
    rbw_control_mode: str | None = None
    supported_rbw_values_hz: tuple[float, ...] | None = None
    rbw_min_hz: float | None = None
    rbw_max_hz: float | None = None
    rbw_is_discrete: bool | None = None
    rbw_is_profile_based: bool | None = None
    rbw_changes_point_count: bool | None = None
    rbw_changes_span: bool | None = None
    rbw_requires_restart: bool = True
    window_modes: tuple[str, ...] = ()
    detector_modes: tuple[str, ...] = ()
    window_requires_restart: bool = True
    detector_requires_restart: bool = True
    supports_resolution_tradeoff: bool = False
    resolution_tradeoff_steps: tuple[ResolutionTradeoffStep, ...] = ()
    resolution_tradeoff_min_index: int | None = None
    resolution_tradeoff_max_index: int | None = None
    resolution_tradeoff_direction: Mapping[str, str] = field(default_factory=dict)
    default_resolution_tradeoff_index: int | None = None
    supports_auto_rbw: bool = False

    def supports(self, control: str) -> bool:
        return control in self.supported_controls


@dataclass(frozen=True, slots=True)
class AnalyzerSettings:
    mode: str = "rta"
    center_frequency_hz: float = 1.0e9
    span_hz: float | None = None
    rbw_hz: float | None = None
    rbw_mode: str = "auto"
    vbw_hz: float | None = None
    reference_level_dbm: float = 0.0
    attenuation_db: int | None = None
    preamplifier: str | None = None
    gain_strategy: str | None = None
    if_agc_enabled: bool | None = None
    sweep_time_s: float | None = None
    window: str | None = None
    detector: str | None = None

    def updated(self, **changes: object) -> "AnalyzerSettings":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class AnalyzerActualSettings:
    center_frequency_hz: float
    start_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    reference_level_dbm: float
    attenuation_db: int | None
    attenuation_automatic: bool
    preamplifier: str | None
    gain_strategy: str | None
    rbw_hz: float
    rbw_mode: str
    window: str | None
    detector: str | None
    fft_size: int
    scale_to_dbm: float | None
    offset_to_dbm: float | None
    point_count: int
    resolution_tradeoff_index: int | None = None
    resolution_tradeoff_state: str = "auto"
    resolution_tradeoff_step_id: str | None = None
    frequency_bin_spacing_hz: float | None = None


@dataclass(frozen=True, slots=True)
class AnalyzerSettingsState:
    requested: AnalyzerSettings
    actual: AnalyzerActualSettings
    configuration_generation: int


@dataclass(frozen=True, slots=True)
class SpectrumFrame:
    sequence: int
    timestamp_ns: int
    values: Float32Array
    point_count: int
    start_frequency_hz: float
    center_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    rbw_hz: float
    vbw_hz: float | None
    reference_level_dbm: float
    sweep_time_s: float | None
    frame_type: FrameType = FrameType.CURRENT
    overload: bool = False
    dropped_frames: int = 0
    configuration_generation: int = 0

    def __post_init__(self) -> None:
        if self.values.dtype != np.float32:
            raise TypeError("SpectrumFrame.values must have dtype float32")
        if not self.values.flags.c_contiguous:
            raise ValueError("SpectrumFrame.values must be C-contiguous")
        if self.values.ndim != 1 or self.values.size != self.point_count:
            raise ValueError("SpectrumFrame point_count must match its one-dimensional values")
        if self.point_count <= 0:
            raise ValueError("SpectrumFrame point_count must be positive")
        if not self.values.flags.owndata:
            raise ValueError("SpectrumFrame values must own their storage")


@dataclass(frozen=True, slots=True)
class SpectrumTemporalFrame:
    """One bounded 60 Hz display interval with newest and peak information."""

    generation: int
    sequence: int
    point_count: int
    interval_start_monotonic_ns: int
    interval_end_monotonic_ns: int
    host_timestamp_ns: int
    device_timestamp_ns: int | None
    traces_integrated: int
    latest_trace_float32: Float32Array
    interval_max_trace_float32: Float32Array
    start_frequency_hz: float
    center_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    rbw_hz: float
    reference_level_dbm: float
    scale_to_dbm: float
    offset_to_dbm: float
    first_receipt_monotonic_ns: int | None = None
    last_receipt_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        if self.generation < 0 or self.sequence < 0:
            raise ValueError("generation and sequence must be non-negative")
        if self.point_count < 2 or self.traces_integrated < 1:
            raise ValueError("point_count and traces_integrated must be positive")
        if self.interval_end_monotonic_ns < self.interval_start_monotonic_ns:
            raise ValueError("interval end precedes interval start")
        if (
            self.first_receipt_monotonic_ns is not None
            and self.last_receipt_monotonic_ns is not None
            and self.last_receipt_monotonic_ns < self.first_receipt_monotonic_ns
        ):
            raise ValueError("last temporal receipt precedes first receipt")
        for name, trace in (
            ("latest_trace_float32", self.latest_trace_float32),
            ("interval_max_trace_float32", self.interval_max_trace_float32),
        ):
            if trace.dtype != np.float32 or trace.ndim != 1 or trace.size != self.point_count:
                raise ValueError(f"{name} must be a point_count-length float32 array")
            if not trace.flags.c_contiguous or not trace.flags.owndata:
                raise ValueError(f"{name} must be contiguous and own its storage")
            if not np.all(np.isfinite(trace)):
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class WaterfallBatch:
    """Contiguous row-major native waterfall rows for one configuration."""

    configuration_generation: int
    batch_sequence: int
    first_row_sequence: int
    row_count: int
    point_count: int
    first_host_timestamp_ns: int
    first_device_timestamp_ns: int
    nominal_row_period_ns: int
    start_frequency_hz: float
    center_frequency_hz: float
    stop_frequency_hz: float
    span_hz: float
    rbw_hz: float
    reference_level_dbm: float
    values: Uint8Array

    def __post_init__(self) -> None:
        if self.configuration_generation < 0 or self.batch_sequence < 0 or self.first_row_sequence < 0:
            raise ValueError("waterfall batch sequences and generation must not be negative")
        if self.row_count <= 0 or self.point_count <= 0 or self.nominal_row_period_ns <= 0:
            raise ValueError("waterfall batch dimensions and row period must be positive")
        if self.values.dtype != np.uint8:
            raise TypeError("WaterfallBatch.values must have dtype uint8")
        if self.values.ndim != 2 or self.values.shape != (self.row_count, self.point_count):
            raise ValueError("WaterfallBatch values must have shape row_count × point_count")
        if not self.values.flags.c_contiguous or not self.values.flags.owndata:
            raise ValueError("WaterfallBatch values must be contiguous and own their storage")


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    source: str
    connected: bool = False
    acquisition_running: bool = False
    sdk_frames_received: int = 0
    display_frames_published: int = 0
    frames_replaced: int = 0
    acquisition_errors: int = 0
    reconnect_count: int = 0
    last_frame_timestamp_ns: int | None = None
    last_error: str | None = None
    sdk_frames_per_second: float = 0.0
    point_count: int | None = None
    device_temperature_c: float | None = None
    reconfiguring: bool = False
    configuration_generation: int = 0


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    source: str
    model: str
    serial: str
    model_code: int | None = None
    sdk_version: str | None = None
    firmware_version: str | None = None
    fpga_version: str | None = None
    hardware_version: str | None = None
    bus_speed: int | None = None
