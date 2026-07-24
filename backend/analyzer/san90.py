"""HAROGIC SAN-90 RTA source owned by a dedicated SDK thread."""

from __future__ import annotations

import ctypes as ct
import math
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.ai_stream import AiStreamConfig, AiStreamPipeline

from .base import AnalyzerSource
from .buffers import IntervalMaxHoldBuffer, LatestFrameBuffer
from .errors import (
    AnalyzerConfigurationError,
    AnalyzerConnectionError,
    AnalyzerStateError,
    AnalyzerTimeoutError,
    SdkError,
    UnsupportedSettingError,
    ControlError,
    ControlErrorCode,
)
from .control_mapping import (
    DETECTOR_VALUES,
    GAIN_STRATEGY_VALUES,
    PREAMPLIFIER_VALUES,
    RBW_MODE_VALUES,
    WINDOW_VALUES,
    enum_name,
)
from .htra import (
    ADAPTIVE,
    API_LAST_PACKET,
    API_LAST_PACKET_WITH_TRIGGER_MISSED,
    API_NO_ERROR,
    API_TRIGGER_MISSED,
    API_WARNING_BUS_TIMEOUT,
    API_WARNING_DATA_NOT_READY,
    API_WARNING_IF_OVERFLOW,
    BUS_TRIGGER,
    DEVICE_N90_R0,
    FORCED_OFF,
    RBW_AUTO,
    RBW_MANUAL,
    BootInfo,
    HtraApi,
    MeasAuxInfo,
    RtaFrameInfo,
    RtaPlotInfo,
    RtaProfile,
    TriggerInfo,
    version_string,
)
from .models import (
    AnalyzerCapabilities,
    AnalyzerActualSettings,
    AnalyzerSettings,
    AnalyzerSettingsState,
    DeviceInfo,
    FrameType,
    RuntimeStatus,
    SpectrumFrame,
    SpectrumTemporalFrame,
    WaterfallBatch,
)
from .spectrum_temporal import LatestSpectrumTemporalExchange, NativeSpectrumTemporalAccumulator
from .tradeoff import SAN90_RESOLUTION_TRADEOFF_STEPS, match_actual_tradeoff_step
from .raw_buffers import (
    DisplaySnapshot,
    DisplaySnapshotExchange,
    RawAmplitudeMapping,
    RawRtaAccumulator,
    RawTraceMetadata,
)
from .waterfall import (
    TimedWaterfallBatchProducer,
    WaterfallProducerMetrics,
    waterfall_rate_for_profile,
    waterfall_rate_override_from_environment,
)

logger = logging.getLogger("san90.controls")


def _positive_environment_rate(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(slots=True)
class _Command:
    identifier: str
    command_type: str
    payload: object
    action: Callable[[], Any]
    future: Future[Any]
    deadline_ns: int


@dataclass(frozen=True, slots=True)
class ReconfigurationMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rollback_attempts: int = 0
    rollback_failures: int = 0
    total_duration_s: float = 0.0
    maximum_duration_s: float = 0.0
    frames_skipped: int = 0
    last_error: str | None = None

    @property
    def mean_duration_s(self) -> float:
        completed = self.successful_requests + self.failed_requests
        return self.total_duration_s / completed if completed else 0.0


@dataclass(frozen=True, slots=True)
class San90Diagnostics:
    """Cumulative measurements for one source instance's RTA run."""

    sdk_read_calls: int = 0
    packets_received: int = 0
    trace_frames_received: int = 0
    timeouts: int = 0
    data_not_ready: int = 0
    invalid_packets: int = 0
    sample_values: int = 0
    minimum_dbm: float | None = None
    maximum_dbm: float | None = None
    sum_dbm: float = 0.0
    sum_squares_dbm: float = 0.0
    sdk_read_total_s: float = 0.0
    sdk_read_max_s: float = 0.0
    conversion_total_s: float = 0.0
    conversion_max_s: float = 0.0
    validation_total_s: float = 0.0
    validation_max_s: float = 0.0
    statistics_total_s: float = 0.0
    statistics_max_s: float = 0.0
    native_copy_total_s: float = 0.0
    native_copy_max_s: float = 0.0
    display_conversion_total_s: float = 0.0
    display_conversion_max_s: float = 0.0
    snapshot_total_s: float = 0.0
    snapshot_max_s: float = 0.0
    display_snapshots_created: int = 0

    @property
    def mean_dbm(self) -> float | None:
        return self.sum_dbm / self.sample_values if self.sample_values else None

    @property
    def standard_deviation_db(self) -> float | None:
        if not self.sample_values:
            return None
        mean = self.sum_dbm / self.sample_values
        variance = max(0.0, self.sum_squares_dbm / self.sample_values - mean * mean)
        return math.sqrt(variance)


class San90Source(AnalyzerSource):
    """Application-facing SAN-90 source.

    All calls that receive a device handle execute on ``san90-sdk-owner``.
    The public methods synchronously submit bounded commands to that thread;
    RTA reads never execute on the caller or an asyncio event loop.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        library_path: str | Path | None = None,
        command_timeout_s: float = 10.0,
        bus_timeout_ms: int = 250,
    ) -> None:
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if command_timeout_s <= 0 or bus_timeout_ms <= 0:
            raise ValueError("timeouts must be positive")
        self._api = HtraApi(library_path)
        self._device_index = device_index
        self._command_timeout_s = command_timeout_s
        self._bus_timeout_ms = bus_timeout_ms
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=32)
        self._latest = LatestFrameBuffer()
        self._max_hold = IntervalMaxHoldBuffer()
        self._state_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._shutdown = False
        self._connected = False
        self._running = False
        self._trigger_active = False
        self._configured = False
        self._device = ct.c_void_p()
        self._boot_profile = self._api.usb_boot_profile()
        self._boot_info: BootInfo | None = None
        self._device_info: DeviceInfo | None = None
        self._settings = AnalyzerSettings(
            mode="rta",
            center_frequency_hz=1.0e9,
            span_hz=None,
            rbw_hz=None,
            vbw_hz=None,
            reference_level_dbm=0.0,
            attenuation_db=None,
            preamplifier="off",
            gain_strategy=None,
            if_agc_enabled=None,
            sweep_time_s=None,
            window=None,
            detector=None,
        )
        self._requested_settings = replace(self._settings)
        self._configuration_generation = 0
        self._reconfiguring = False
        self._reconfiguration_metrics = ReconfigurationMetrics()
        self._profile_out: RtaProfile | None = None
        self._frame_info: RtaFrameInfo | None = None
        self._native_trace: Any = None
        self._sequence = 0
        self._received = 0
        self._published = 0
        self._errors = 0
        self._last_error: str | None = None
        self._last_frame_ns: int | None = None
        self._started_ns: int | None = None
        self._rate_started_ns: int | None = None
        self._rate_received_baseline = 0
        self._temperature_c: float | None = None
        self._diagnostics = San90Diagnostics()
        spectrum_override = os.getenv("SAN90_SPECTRUM_FPS")
        self._spectrum_fps_override = (
            None if spectrum_override is None else _positive_environment_rate("SAN90_SPECTRUM_FPS", 60.0)
        )
        self._spectrum_fps = self._spectrum_fps_override or 60.0
        self._waterfall_override = waterfall_rate_override_from_environment()
        self._raw_accumulator: RawRtaAccumulator | None = None
        self._snapshot_exchange: DisplaySnapshotExchange | None = None
        self._waterfall_producer: TimedWaterfallBatchProducer | None = None
        self._spectrum_temporal_accumulator: NativeSpectrumTemporalAccumulator | None = None
        self._spectrum_temporal_exchange = LatestSpectrumTemporalExchange()
        self._next_spectrum_ns = 0
        self._ai_stream = AiStreamPipeline(AiStreamConfig.from_environment())

    def connect(self) -> None:
        self._ensure_owner_thread()
        self._submit(self._connect_on_owner)
        self._ai_stream.start()

    def disconnect(self) -> None:
        thread = self._thread
        if thread is None:
            return
        try:
            self._submit(self._disconnect_on_owner)
        finally:
            self._ai_stream.stop()
            with self._state_lock:
                self._shutdown = True
            try:
                self._commands.put_nowait(_Command(uuid.uuid4().hex, "shutdown", None, lambda: None, Future(), time.monotonic_ns()))
            except queue.Full:
                pass
            thread.join(timeout=self._command_timeout_s)
            if thread.is_alive():
                raise AnalyzerTimeoutError("SAN-90 SDK owner thread did not shut down")
            with self._state_lock:
                self._thread = None
            self._cancel_pending_commands()

    def start(self) -> None:
        self._require_owner()
        self._submit(self._start_on_owner)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._submit(self._stop_on_owner)

    def get_capabilities(self) -> AnalyzerCapabilities:
        points = tuple(step.point_count for step in SAN90_RESOLUTION_TRADEOFF_STEPS)
        return AnalyzerCapabilities(
            source="san90",
            measurement_modes=("rta",),
            supported_controls=frozenset({
                "center_frequency_hz", "reference_level_dbm", "attenuation_db",
                "preamplifier", "gain_strategy", "rbw_hz", "rbw_mode", "window", "detector",
                "resolution_tradeoff_index",
            }),
            enum_values={
                "preamplifier": tuple(PREAMPLIFIER_VALUES),
                "gain_strategy": tuple(GAIN_STRATEGY_VALUES),
                "rbw_mode": tuple(RBW_MODE_VALUES),
                "window": tuple(WINDOW_VALUES),
                "detector": tuple(DETECTOR_VALUES),
            },
            native_point_counts=points,
            supports_density=True,
            supported_attenuation_values_db=None,
            supports_automatic_attenuation=True,
            preamplifier_modes=tuple(PREAMPLIFIER_VALUES),
            gain_strategy_modes=tuple(GAIN_STRATEGY_VALUES),
            supports_live_frequency_change=False,
            supports_live_amplitude_change=False,
            requires_restart_for_frequency=True,
            requires_restart_for_amplitude=True,
            supports_rbw_control=True,
            rbw_control_mode="auto-or-manual-numeric",
            supported_rbw_values_hz=tuple(step.actual_rbw_hz for step in SAN90_RESOLUTION_TRADEOFF_STEPS),
            rbw_min_hz=None,
            rbw_max_hz=None,
            rbw_is_discrete=False,
            rbw_is_profile_based=False,
            rbw_changes_point_count=True,
            rbw_changes_span=False,
            rbw_requires_restart=True,
            window_modes=tuple(WINDOW_VALUES),
            detector_modes=tuple(DETECTOR_VALUES),
            window_requires_restart=True,
            detector_requires_restart=True,
            supports_resolution_tradeoff=True,
            resolution_tradeoff_steps=SAN90_RESOLUTION_TRADEOFF_STEPS,
            resolution_tradeoff_min_index=0,
            resolution_tradeoff_max_index=len(SAN90_RESOLUTION_TRADEOFF_STEPS) - 1,
            resolution_tradeoff_direction={"left": "time", "right": "frequency"},
            default_resolution_tradeoff_index=5,
            supports_auto_rbw=True,
        )

    def get_settings(self) -> AnalyzerSettings:
        with self._state_lock:
            return replace(self._settings)

    def apply_settings(self, settings: AnalyzerSettings) -> AnalyzerSettings:
        self._validate_settings(settings)
        self._require_owner()
        return self._submit(lambda: self._reconfigure_on_owner(settings), command_type="apply_settings", payload=settings)

    def get_settings_state(self) -> AnalyzerSettingsState:
        with self._state_lock:
            settings = replace(self._settings)
            requested = replace(self._requested_settings)
            frame_info = self._frame_info
            generation = self._configuration_generation
        if frame_info is None:
            raise AnalyzerStateError("SAN-90 has not been configured")
        mapping = self.latest_raw_amplitude_mapping()
        attenuation = settings.attenuation_db
        matched = match_actual_tradeoff_step(
            SAN90_RESOLUTION_TRADEOFF_STEPS,
            actual_rbw_hz=float(settings.rbw_hz or 0.0),
            point_count=int(frame_info.FrameWidth),
            fft_size=int(frame_info.FFTSize),
        )
        return AnalyzerSettingsState(
            requested=requested,
            actual=AnalyzerActualSettings(
                center_frequency_hz=settings.center_frequency_hz,
                start_frequency_hz=float(frame_info.StartFrequency_Hz),
                stop_frequency_hz=float(frame_info.StopFrequency_Hz),
                span_hz=float(frame_info.StopFrequency_Hz-frame_info.StartFrequency_Hz),
                reference_level_dbm=settings.reference_level_dbm,
                attenuation_db=None if attenuation == -1 else attenuation,
                attenuation_automatic=attenuation == -1,
                preamplifier=settings.preamplifier,
                gain_strategy=settings.gain_strategy,
                rbw_hz=float(settings.rbw_hz or 0.0),
                rbw_mode=settings.rbw_mode,
                window=settings.window,
                detector=settings.detector,
                fft_size=int(frame_info.FFTSize),
                scale_to_dbm=None if mapping is None else mapping.scale_db_per_code,
                offset_to_dbm=None if mapping is None else mapping.offset_dbm,
                point_count=int(frame_info.FrameWidth),
                resolution_tradeoff_index=matched.index if matched is not None and settings.rbw_mode == "manual" else None,
                resolution_tradeoff_state=(
                    "auto" if settings.rbw_mode == "auto" else "matched" if matched is not None else "custom"
                ),
                resolution_tradeoff_step_id=matched.id if matched is not None and settings.rbw_mode == "manual" else None,
                frequency_bin_spacing_hz=float(frame_info.StopFrequency_Hz-frame_info.StartFrequency_Hz) / int(frame_info.FrameWidth),
            ),
            configuration_generation=generation,
        )

    def read_frame(self) -> SpectrumFrame | None:
        frame = self._latest.read()
        if frame is not None:
            with self._state_lock:
                self._published += 1
        return frame

    def read_interval_max_hold(self) -> SpectrumFrame | None:
        return self._max_hold.take()

    def read_display_snapshot(self) -> DisplaySnapshot | None:
        exchange = self._snapshot_exchange
        snapshot = exchange.take_latest() if exchange is not None else None
        if snapshot is not None:
            with self._state_lock:
                self._published += 1
        return snapshot

    def read_waterfall_batch(self) -> WaterfallBatch | None:
        producer = self._waterfall_producer
        return producer.exchange.take_latest() if producer is not None else None

    def read_spectrum_temporal(self) -> SpectrumTemporalFrame | None:
        frame = self._spectrum_temporal_exchange.take()
        if frame is not None:
            with self._state_lock:
                self._published += 1
        return frame

    def get_spectrum_temporal_metrics(self) -> dict[str, float | int]:
        accumulator = self._spectrum_temporal_accumulator
        if accumulator is None:
            return {}
        completed = accumulator.completed_intervals
        return {
            "completed_intervals": completed,
            "frames_published_to_exchange": self._spectrum_temporal_exchange.frames_published,
            "frames_displaced": self._spectrum_temporal_exchange.frames_displaced,
            "compatible_maximum_merges": self._spectrum_temporal_exchange.compatible_maximum_merges,
            "incompatible_merge_rejections": self._spectrum_temporal_exchange.incompatible_merge_rejections,
            "traces_preserved_by_merges": self._spectrum_temporal_exchange.traces_preserved_by_merges,
            "total_traces_integrated": accumulator.total_traces_integrated,
            "minimum_traces_integrated": accumulator.minimum_traces_integrated,
            "maximum_traces_integrated": accumulator.maximum_traces_integrated,
            "mean_traces_integrated": accumulator.total_traces_integrated / completed if completed else 0.0,
            "max_hold_update_total_ns": accumulator.max_hold_update_total_ns,
            "mean_max_hold_update_ns_per_interval": accumulator.max_hold_update_total_ns / completed if completed else 0.0,
            "conversion_total_ns": accumulator.conversion_total_ns,
            "mean_conversion_ns": accumulator.conversion_total_ns / completed if completed else 0.0,
            "maximum_conversion_ns": accumulator.conversion_max_ns,
            "finalization_total_ns": accumulator.finalization_total_ns,
            "mean_finalization_ns": accumulator.finalization_total_ns / completed if completed else 0.0,
            "maximum_finalization_ns": accumulator.finalization_max_ns,
            "mean_receipt_span_ns": accumulator.receipt_span_total_ns / completed if completed else 0.0,
            "minimum_receipt_span_ns": accumulator.receipt_span_min_ns,
            "maximum_receipt_span_ns": accumulator.receipt_span_max_ns,
            "missed_interval_deadlines": accumulator.missed_interval_deadlines,
            "frames_replaced": self._spectrum_temporal_exchange.frames_replaced,
            "discarded_incomplete_intervals": accumulator.discarded_incomplete_intervals,
        }

    def get_waterfall_metrics(self) -> WaterfallProducerMetrics | None:
        producer = self._waterfall_producer
        return producer.metrics() if producer is not None else None

    def latest_receipt_monotonic_ns(self) -> int | None:
        """Return host monotonic receipt time without exposing mutable buffers."""
        accumulator = self._raw_accumulator
        metadata = accumulator.metadata if accumulator is not None else None
        return metadata.receipt_monotonic_ns if metadata is not None else None

    def latest_raw_amplitude_mapping(self) -> RawAmplitudeMapping | None:
        """Return the immutable mapping for the current native waterfall codes."""
        accumulator = self._raw_accumulator
        metadata = accumulator.metadata if accumulator is not None else None
        return metadata.mapping if metadata is not None else None

    def get_ai_stream_status(self) -> dict[str, object]:
        return self._ai_stream.status()

    def set_ai_stream_enabled(self, enabled: bool) -> dict[str, object]:
        self._require_owner()
        if enabled:
            self._ai_stream.start_publisher()
            self._submit(lambda: self._ai_stream.set_accepting(True), command_type="ai_stream_enabled", payload=True)
        else:
            self._submit(lambda: self._ai_stream.set_accepting(False), command_type="ai_stream_enabled", payload=False)
            self._ai_stream.stop_publisher()
        return self._ai_stream.status()

    def set_ai_power_profile(self, name: str) -> dict[str, object]:
        self._ai_stream.set_power_profile(name)
        return self._ai_stream.status()

    def latest_ai_preview_png(self) -> bytes | None:
        return self._ai_stream.latest_preview_png()

    def get_status(self) -> RuntimeStatus:
        with self._state_lock:
            elapsed = 0.0 if self._rate_started_ns is None else (time.monotonic_ns() - self._rate_started_ns) / 1e9
            point_count = int(self._frame_info.FrameWidth) if self._frame_info is not None else None
            return RuntimeStatus(
                source="san90",
                connected=self._connected,
                acquisition_running=self._running,
                sdk_frames_received=self._received,
                display_frames_published=self._published,
                frames_replaced=self._spectrum_temporal_exchange.frames_replaced,
                acquisition_errors=self._errors,
                last_frame_timestamp_ns=self._last_frame_ns,
                last_error=self._last_error,
                sdk_frames_per_second=(self._received - self._rate_received_baseline) / elapsed if elapsed > 0 else 0.0,
                point_count=point_count,
                device_temperature_c=self._temperature_c,
                reconfiguring=self._reconfiguring,
                configuration_generation=self._configuration_generation,
            )

    def get_reconfiguration_metrics(self) -> ReconfigurationMetrics:
        with self._state_lock:
            return replace(self._reconfiguration_metrics)

    def get_device_info(self) -> DeviceInfo | None:
        with self._state_lock:
            return self._device_info

    def get_diagnostics(self) -> San90Diagnostics:
        """Return a read-only snapshot without making an SDK call."""
        with self._state_lock:
            return replace(self._diagnostics)

    def get_spectrum_publish_fps(self) -> float:
        return self._spectrum_fps

    def _ensure_owner_thread(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown = False
            self._thread = threading.Thread(target=self._owner_loop, name="san90-sdk-owner", daemon=True)
            self._thread.start()

    def _require_owner(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            raise AnalyzerStateError("SAN-90 must be connected before this operation")

    def _submit(self, action: Callable[[], Any], *, command_type: str = "lifecycle", payload: object = None) -> Any:
        thread = self._thread
        if thread is None or not thread.is_alive():
            raise AnalyzerStateError("SAN-90 SDK owner thread is not running")
        if threading.current_thread() is thread:
            return action()
        future: Future[Any] = Future()
        command = _Command(
            identifier=uuid.uuid4().hex,
            command_type=command_type,
            payload=payload,
            action=action,
            future=future,
            deadline_ns=time.monotonic_ns()+int(self._command_timeout_s*1e9),
        )
        try:
            self._commands.put(command, timeout=self._command_timeout_s)
            return future.result(timeout=self._command_timeout_s)
        except queue.Full as error:
            raise ControlError(ControlErrorCode.DEVICE_BUSY, "SAN-90 command queue is full", recoverable=True) from error
        except FutureTimeoutError as error:
            future.cancel()
            raise ControlError(ControlErrorCode.RECONFIGURATION_TIMEOUT, "Timed out waiting for SAN-90 owner thread", recoverable=True) from error

    def _owner_loop(self) -> None:
        while True:
            with self._state_lock:
                if self._shutdown:
                    return
                running = self._running
            try:
                command = self._commands.get(timeout=0.0 if running else 0.1)
            except queue.Empty:
                if running:
                    try:
                        self._acquire_packet_on_owner()
                    except BaseException as error:
                        with self._state_lock:
                            self._errors += 1
                            self._last_error = f"Unexpected acquisition failure: {error}"
                            self._running = False
                continue
            if command.future.cancelled():
                continue
            if time.monotonic_ns() > command.deadline_ns:
                command.future.set_exception(ControlError(ControlErrorCode.RECONFIGURATION_TIMEOUT, f"Command {command.identifier} expired", recoverable=True))
                continue
            try:
                result = command.action()
                if not command.future.cancelled():
                    command.future.set_result(result)
            except BaseException as error:
                if not command.future.done():
                    command.future.set_exception(error)

    def _cancel_pending_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if not command.future.done():
                command.future.set_exception(ControlError(ControlErrorCode.SHUTTING_DOWN, "SAN-90 owner thread is shutting down", recoverable=False))

    def _connect_on_owner(self) -> None:
        with self._state_lock:
            if self._connected:
                return
        devices = self._api.list_devices(self._boot_profile)
        san90_devices = [device for device in devices if int(device.info.Model) == DEVICE_N90_R0]
        if self._device_index >= len(san90_devices):
            models = ", ".join(str(int(device.info.Model)) for device in devices) or "none"
            raise AnalyzerConnectionError(
                f"SAN-90 index {self._device_index} not found; discovered {len(devices)} device(s), models: {models}"
            )
        selected = san90_devices[self._device_index]
        boot_info = BootInfo()
        status = int(self._api.lib.Device_Open(
            ct.byref(self._device), selected.device_number, ct.byref(self._boot_profile), ct.byref(boot_info)
        ))
        if status != API_NO_ERROR:
            supported = self._api.supported_firmware_versions()
            supported_text = ", ".join(
                f"MCU {version_string(int(item.MFWVersion))}/FPGA {version_string(int(item.FFWVersion))}"
                for item in supported
            ) or "not reported"
            details = (
                f"model={int(boot_info.DeviceInfo.Model)}, "
                f"MCU={version_string(int(boot_info.DeviceInfo.MFWVersion))}, "
                f"FPGA={version_string(int(boot_info.DeviceInfo.FFWVersion))}, "
                f"reported API={version_string(int(boot_info.APIVersion))}; "
                f"library-supported firmware: {supported_text}"
            )
            if self._device.value:
                self._api.lib.Device_Close(ct.byref(self._device))
                self._device = ct.c_void_p()
            raise AnalyzerConnectionError(f"Device_Open failed with HTRA API status {status} ({details})")
        if int(boot_info.DeviceInfo.Model) != DEVICE_N90_R0:
            self._api.lib.Device_Close(ct.byref(self._device))
            self._device = ct.c_void_p()
            raise AnalyzerConnectionError(
                f"Opened model {int(boot_info.DeviceInfo.Model)}, expected SAN-90 model {DEVICE_N90_R0}"
            )
        native = boot_info.DeviceInfo
        info = DeviceInfo(
            source="san90",
            model="HAROGIC SAN-90",
            serial=str(int(native.DeviceUID)),
            model_code=int(native.Model),
            sdk_version=version_string(int(boot_info.APIVersion)),
            firmware_version=version_string(int(native.MFWVersion)),
            fpga_version=version_string(int(native.FFWVersion)),
            hardware_version=f"0x{int(native.HardwareVersion):04x}",
            bus_speed=int(boot_info.BusSpeed),
        )
        with self._state_lock:
            self._boot_info = boot_info
            self._device_info = info
            self._connected = True
            self._last_error = None

    def _disconnect_on_owner(self) -> None:
        if self._trigger_active:
            self._stop_on_owner()
        if self._device.value:
            status = int(self._api.lib.Device_Close(ct.byref(self._device)))
            self._api.require_ok("Device_Close", status)
        self._device = ct.c_void_p()
        with self._state_lock:
            self._connected = False
            self._configured = False
            self._boot_info = None
            self._device_info = None
            self._profile_out = None
            self._frame_info = None
            self._native_trace = None
            self._raw_accumulator = None
            self._snapshot_exchange = None
            self._waterfall_producer = None
            self._spectrum_temporal_accumulator = None
            self._spectrum_temporal_exchange.clear()

    def _configure_on_owner(self, requested: AnalyzerSettings) -> AnalyzerSettings:
        if not self._connected or not self._device.value:
            raise AnalyzerStateError("SAN-90 is not connected")
        if self._trigger_active:
            raise AnalyzerStateError("Stop SAN-90 acquisition before changing configuration")
        profile_in = RtaProfile()
        status = int(self._api.lib.RTA_ProfileDeInit(ct.byref(self._device), ct.byref(profile_in)))
        self._api.require_ok("RTA_ProfileDeInit", status)

        profile_in.CenterFreq_Hz = requested.center_frequency_hz
        profile_in.RefLevel_dBm = requested.reference_level_dbm
        profile_in.BusTimeout_ms = self._bus_timeout_ms
        profile_in.TriggerMode = ADAPTIVE
        profile_in.TriggerSource = BUS_TRIGGER
        profile_in.Preamplifier = PREAMPLIFIER_VALUES.get(requested.preamplifier or "off", FORCED_OFF)
        profile_in.RBWMode = RBW_MODE_VALUES[requested.rbw_mode]
        if requested.rbw_mode == "manual":
            assert requested.rbw_hz is not None
            profile_in.RBWMode = RBW_MANUAL
            profile_in.RBW_Hz = requested.rbw_hz
        else:
            profile_in.RBWMode = RBW_AUTO
        if requested.vbw_hz is not None:
            profile_in.VBWMode = 0  # VBW_Manual in htra_api.h.
            profile_in.VBW_Hz = requested.vbw_hz
        if requested.attenuation_db is not None:
            profile_in.Atten = requested.attenuation_db
        if requested.gain_strategy is not None:
            profile_in.GainStrategy = GAIN_STRATEGY_VALUES[requested.gain_strategy]
        if requested.if_agc_enabled is not None:
            profile_in.EnableIFAGC = int(requested.if_agc_enabled)
        if requested.window is not None:
            profile_in.Window = WINDOW_VALUES[requested.window]
        if requested.detector is not None:
            profile_in.Detector = DETECTOR_VALUES[requested.detector]
        if requested.sweep_time_s is not None:
            profile_in.SweepTimeMode = 7  # SWTMode_Manual in htra_api.h.
            profile_in.SweepTime = requested.sweep_time_s

        profile_out = RtaProfile()
        frame_info = RtaFrameInfo()
        status = int(self._api.lib.RTA_Configuration(
            ct.byref(self._device), ct.byref(profile_in), ct.byref(profile_out), ct.byref(frame_info)
        ))
        self._api.require_ok("RTA_Configuration", status)
        if frame_info.FrameWidth <= 0 or frame_info.PacketFrame <= 0:
            raise AnalyzerConfigurationError(
                f"RTA returned invalid frame dimensions: {frame_info.PacketFrame}x{frame_info.FrameWidth}"
            )
        expected = int(frame_info.PacketFrame) * int(frame_info.FrameWidth)
        if int(frame_info.PacketValidPoints) < expected:
            raise AnalyzerConfigurationError(
                f"RTA PacketValidPoints={frame_info.PacketValidPoints} is smaller than {expected}"
            )
        try:
            native_trace = (ct.c_uint8 * int(frame_info.PacketValidPoints))()
            raw_accumulator = RawRtaAccumulator(int(frame_info.FrameWidth))
            snapshot_exchange = DisplaySnapshotExchange(int(frame_info.FrameWidth))
            spectrum_temporal_accumulator = NativeSpectrumTemporalAccumulator(int(frame_info.FrameWidth))
            waterfall_config = waterfall_rate_for_profile(
                float(profile_out.RBW_Hz), int(frame_info.FrameWidth), self._waterfall_override
            )
            waterfall_producer = self._waterfall_producer
            if waterfall_producer is None:
                waterfall_producer = TimedWaterfallBatchProducer(
                    int(frame_info.FrameWidth), self._configuration_generation, waterfall_config
                )
            else:
                # This is still the SDK owner thread. The reset both discards
                # partial old-generation data and resizes buffers atomically
                # before acquisition can restart.
                waterfall_producer.reconfigure(
                    int(frame_info.FrameWidth), self._configuration_generation, waterfall_config
                )
        except (MemoryError, OverflowError, ValueError) as error:
            raise ControlError(
                ControlErrorCode.BUFFER_RESIZE_FAILED,
                f"Could not allocate RTA buffers for {int(frame_info.FrameWidth)} points: {error}",
                recoverable=True,
            ) from error
        actual = AnalyzerSettings(
            mode="rta",
            center_frequency_hz=float(profile_out.CenterFreq_Hz),
            span_hz=float(frame_info.StopFrequency_Hz - frame_info.StartFrequency_Hz),
            rbw_hz=float(profile_out.RBW_Hz),
            rbw_mode=enum_name(RBW_MODE_VALUES, int(profile_out.RBWMode)) or requested.rbw_mode,
            vbw_hz=float(profile_out.VBW_Hz),
            reference_level_dbm=float(profile_out.RefLevel_dBm),
            attenuation_db=int(profile_out.Atten),
            preamplifier=enum_name(PREAMPLIFIER_VALUES, int(profile_out.Preamplifier)),
            gain_strategy=enum_name(GAIN_STRATEGY_VALUES, int(profile_out.GainStrategy)),
            if_agc_enabled=bool(profile_out.EnableIFAGC),
            sweep_time_s=float(profile_out.SweepTime),
            window=enum_name(WINDOW_VALUES, int(profile_out.Window)),
            detector=enum_name(DETECTOR_VALUES, int(profile_out.Detector)),
        )
        matched_step = match_actual_tradeoff_step(
            SAN90_RESOLUTION_TRADEOFF_STEPS,
            actual_rbw_hz=float(profile_out.RBW_Hz),
            point_count=int(frame_info.FrameWidth),
            fft_size=int(frame_info.FFTSize),
        )
        spectrum_fps = self._spectrum_fps_override or (
            matched_step.spectrum_publish_fps if matched_step is not None else 60.0
        )
        with self._state_lock:
            self._profile_out = profile_out
            self._frame_info = frame_info
            self._native_trace = native_trace
            self._raw_accumulator = raw_accumulator
            self._snapshot_exchange = snapshot_exchange
            self._waterfall_producer = waterfall_producer
            self._spectrum_temporal_accumulator = spectrum_temporal_accumulator
            self._spectrum_temporal_exchange.clear()
            self._spectrum_fps = spectrum_fps
            self._settings = actual
            self._configured = True
        return replace(actual)

    def _reconfigure_on_owner(self, requested: AnalyzerSettings) -> AnalyzerSettings:
        if not self._connected:
            raise ControlError(ControlErrorCode.DEVICE_NOT_CONNECTED, "SAN-90 is not connected", recoverable=True)
        started = time.perf_counter()
        was_running = self._running
        previous_settings = replace(self._settings)
        previous_requested = replace(self._requested_settings)
        previous_generation = self._configuration_generation
        previous_sequence = self._sequence
        previous_rate = self.get_status().sdk_frames_per_second
        with self._state_lock:
            self._reconfiguring = True
            metrics = self._reconfiguration_metrics
            self._reconfiguration_metrics = replace(metrics, total_requests=metrics.total_requests+1)
        try:
            if was_running:
                self._stop_on_owner()
            actual = self._configure_on_owner(requested)
            with self._state_lock:
                self._requested_settings = replace(requested)
                self._configuration_generation = previous_generation+1
                self._sequence = 0
            self._reset_waterfall_producer_on_owner()
            if was_running:
                self._start_on_owner()
                self._confirm_valid_frame_on_owner()
            elapsed = time.perf_counter()-started
            with self._state_lock:
                metrics = self._reconfiguration_metrics
                self._reconfiguration_metrics = replace(
                    metrics,
                    successful_requests=metrics.successful_requests+1,
                    total_duration_s=metrics.total_duration_s+elapsed,
                    maximum_duration_s=max(metrics.maximum_duration_s, elapsed),
                    frames_skipped=metrics.frames_skipped+int(previous_rate*elapsed),
                    last_error=None,
                )
            logger.info("configuration_transaction status=success generation=%s duration_ms=%.3f center_hz=%.3f reference_dbm=%.3f attenuation=%s preamplifier=%s gain_strategy=%s", self._configuration_generation, elapsed*1000, actual.center_frequency_hz, actual.reference_level_dbm, actual.attenuation_db, actual.preamplifier, actual.gain_strategy)
            return actual
        except BaseException as original_error:
            with self._state_lock:
                metrics = self._reconfiguration_metrics
                self._reconfiguration_metrics = replace(metrics, rollback_attempts=metrics.rollback_attempts+1)
            try:
                if self._trigger_active:
                    self._stop_on_owner()
                self._configure_on_owner(previous_settings)
                with self._state_lock:
                    self._requested_settings = previous_requested
                    # Rollback restores the same immutable measurement generation.
                    # Preserve the sequence too, so consumers do not reject the
                    # restarted stream as older data within that generation.
                    self._configuration_generation = previous_generation
                    self._sequence = previous_sequence
                self._reset_waterfall_producer_on_owner()
                if was_running:
                    self._start_on_owner()
                    self._confirm_valid_frame_on_owner()
            except BaseException as rollback_error:
                elapsed = time.perf_counter()-started
                with self._state_lock:
                    metrics = self._reconfiguration_metrics
                    self._reconfiguration_metrics = replace(metrics, failed_requests=metrics.failed_requests+1, rollback_failures=metrics.rollback_failures+1, total_duration_s=metrics.total_duration_s+elapsed, maximum_duration_s=max(metrics.maximum_duration_s,elapsed), last_error=str(rollback_error))
                raise ControlError(ControlErrorCode.ROLLBACK_FAILED, f"Configuration failed and rollback failed: {rollback_error}", sdk_status=getattr(rollback_error,"status",None), recoverable=False) from original_error
            elapsed = time.perf_counter()-started
            with self._state_lock:
                metrics = self._reconfiguration_metrics
                self._reconfiguration_metrics = replace(metrics, failed_requests=metrics.failed_requests+1, total_duration_s=metrics.total_duration_s+elapsed, maximum_duration_s=max(metrics.maximum_duration_s,elapsed), frames_skipped=metrics.frames_skipped+int(previous_rate*elapsed), last_error=str(original_error))
            logger.info("configuration_transaction status=rolled_back generation=%s duration_ms=%.3f error=%s", self._configuration_generation, elapsed*1000, original_error)
            if isinstance(original_error, ControlError):
                raise ControlError(
                    original_error.code,
                    str(original_error),
                    sdk_status=original_error.sdk_status,
                    requested_value=original_error.requested_value,
                    previous_actual_value=original_error.previous_actual_value,
                    recoverable=original_error.recoverable,
                ) from original_error
            raise ControlError(ControlErrorCode.SDK_CONFIGURATION_FAILED, str(original_error), sdk_status=getattr(original_error,"status",None), recoverable=True) from original_error
        finally:
            with self._state_lock:
                self._reconfiguring = False

    def _confirm_valid_frame_on_owner(self) -> None:
        before = self._received
        deadline = time.monotonic()+min(2.0,self._command_timeout_s/2)
        while self._running and self._received == before and time.monotonic()<deadline:
            self._acquire_packet_on_owner()
        if self._received == before:
            raise ControlError(ControlErrorCode.FIRST_FRAME_TIMEOUT, "No valid SAN-90 frame arrived after restart", recoverable=True)

    def _start_on_owner(self) -> None:
        if not self._connected:
            raise AnalyzerStateError("SAN-90 is not connected")
        if self._running:
            return
        if self._trigger_active:
            raise AnalyzerStateError("SAN-90 trigger remains active after an acquisition error; stop before restarting")
        if not self._configured:
            self._configure_on_owner(self._settings)
        status = int(self._api.lib.RTA_BusTriggerStart(ct.byref(self._device)))
        self._api.require_ok("RTA_BusTriggerStart", status)
        with self._state_lock:
            if self._configuration_generation == 0:
                self._configuration_generation = 1
            self._reset_waterfall_producer_on_owner()
            now_ns = time.monotonic_ns()
            self._next_spectrum_ns = now_ns
            self._rate_started_ns = now_ns
            self._rate_received_baseline = self._received
            self._running = True
            self._trigger_active = True
            if self._started_ns is None:
                self._started_ns = time.monotonic_ns()
            self._last_error = None

    def _reset_waterfall_producer_on_owner(self) -> None:
        producer = self._waterfall_producer
        frame_info = self._frame_info
        profile = self._profile_out
        if producer is None and frame_info is None and profile is None:
            # Permits isolated transaction tests that replace configuration;
            # the real configuration path always installs all three together.
            return
        if producer is None or frame_info is None or profile is None:
            raise AnalyzerStateError("SAN-90 waterfall buffers are not configured")
        config = waterfall_rate_for_profile(
            float(profile.RBW_Hz), int(frame_info.FrameWidth), self._waterfall_override
        )
        producer.reconfigure(int(frame_info.FrameWidth), self._configuration_generation, config)
        try:
            self._ai_stream.configure(
                int(frame_info.PacketFrame),
                int(frame_info.FrameWidth),
                self._configuration_generation,
            )
        except (MemoryError, ValueError) as error:
            logger.error("AI stream buffer configuration failed; disabling AI output: %s", error)
            self._ai_stream.set_accepting(False)

    def _stop_on_owner(self) -> None:
        if not self._trigger_active:
            return
        status = int(self._api.lib.RTA_BusTriggerStop(ct.byref(self._device)))
        with self._state_lock:
            self._running = False
            self._trigger_active = False
        self._api.require_ok("RTA_BusTriggerStop", status)

    def _acquire_packet_on_owner(self) -> None:
        frame_info = self._frame_info
        profile = self._profile_out
        native_trace = self._native_trace
        if frame_info is None or profile is None or native_trace is None:
            raise AnalyzerStateError("SAN-90 acquisition started without RTA buffers")
        plot = RtaPlotInfo()
        trigger = TriggerInfo()
        auxiliary = MeasAuxInfo()
        read_started = time.perf_counter()
        status = int(self._api.lib.RTA_GetRealTimeSpectrum_Raw(
            ct.byref(self._device), native_trace, ct.byref(plot), ct.byref(trigger), ct.byref(auxiliary)
        ))
        read_elapsed = time.perf_counter() - read_started
        with self._state_lock:
            diagnostics = self._diagnostics
            self._diagnostics = replace(
                diagnostics,
                sdk_read_calls=diagnostics.sdk_read_calls + 1,
                sdk_read_total_s=diagnostics.sdk_read_total_s + read_elapsed,
                sdk_read_max_s=max(diagnostics.sdk_read_max_s, read_elapsed),
                timeouts=diagnostics.timeouts + int(status == API_WARNING_BUS_TIMEOUT),
                data_not_ready=diagnostics.data_not_ready + int(status == API_WARNING_DATA_NOT_READY),
            )
        if status in {API_WARNING_BUS_TIMEOUT, API_WARNING_DATA_NOT_READY}:
            return
        if status in {API_LAST_PACKET, API_TRIGGER_MISSED, API_LAST_PACKET_WITH_TRIGGER_MISSED}:
            return
        overload = status == API_WARNING_IF_OVERFLOW
        if status != API_NO_ERROR and not overload:
            error = SdkError("RTA_GetRealTimeSpectrum_Raw", status)
            with self._state_lock:
                self._errors += 1
                self._last_error = str(error)
                self._running = False
            return

        width = int(frame_info.FrameWidth)
        count = int(frame_info.PacketFrame)
        raw = np.ctypeslib.as_array(native_trace)[: width * count].reshape(count, width)
        validation_started = time.perf_counter()
        valid = raw.shape == (count, width) and raw.dtype == np.uint8 and raw.flags.c_contiguous
        validation_elapsed = time.perf_counter() - validation_started
        if not valid:
            with self._state_lock:
                diagnostics = self._diagnostics
                self._diagnostics = replace(
                    diagnostics,
                    invalid_packets=diagnostics.invalid_packets + 1,
                    validation_total_s=diagnostics.validation_total_s + validation_elapsed,
                    validation_max_s=max(diagnostics.validation_max_s, validation_elapsed),
                )
                self._errors += 1
                self._last_error = "RTA returned malformed native trace data"
                self._running = False
            return

        mapping = RawAmplitudeMapping(float(plot.ScaleTodBm), float(plot.OffsetTodBm))
        statistics_started = time.perf_counter()
        raw_minimum = float(np.min(raw))
        raw_maximum = float(np.max(raw))
        raw_sum = float(np.sum(raw, dtype=np.float64))
        raw_sum_squares = float(np.einsum("ij,ij->", raw, raw, dtype=np.float64, casting="unsafe"))
        sample_values = raw.size
        packet_minimum = raw_minimum * mapping.scale_db_per_code + mapping.offset_dbm
        packet_maximum = raw_maximum * mapping.scale_db_per_code + mapping.offset_dbm
        packet_sum = mapping.scale_db_per_code * raw_sum + mapping.offset_dbm * sample_values
        packet_sum_squares = (
            mapping.scale_db_per_code * mapping.scale_db_per_code * raw_sum_squares
            + 2.0 * mapping.scale_db_per_code * mapping.offset_dbm * raw_sum
            + mapping.offset_dbm * mapping.offset_dbm * sample_values
        )
        statistics_elapsed = time.perf_counter() - statistics_started
        packet_timestamp_ns = int(auxiliary.nsSinceEpoch) or time.time_ns()
        # TraceTimestampStep is expressed in the SDK system-timer domain (for
        # this SAN-90 it is 16384 counts), not seconds. PacketAcqTime is
        # explicitly seconds and its PacketFrame ratio matches the measured
        # native trace rate, so use that for host-monotonic row segmentation.
        timestamp_step_ns = max(1, round(float(frame_info.PacketAcqTime) * 1e9 / count))
        start = float(frame_info.StartFrequency_Hz)
        stop = float(frame_info.StopFrequency_Hz)
        center = (start + stop) / 2.0
        span = stop - start
        host_timestamp_ns = time.time_ns()
        receipt_monotonic_ns = time.monotonic_ns()
        latest_sequence = self._sequence + count - 1
        latest_device_timestamp_ns = packet_timestamp_ns + (count - 1) * timestamp_step_ns
        metadata = RawTraceMetadata(
            sequence=latest_sequence,
            device_timestamp_ns=latest_device_timestamp_ns,
            host_timestamp_ns=host_timestamp_ns,
            receipt_monotonic_ns=receipt_monotonic_ns,
            start_frequency_hz=start,
            center_frequency_hz=center,
            stop_frequency_hz=stop,
            span_hz=span,
            rbw_hz=float(profile.RBW_Hz),
            reference_level_dbm=float(profile.RefLevel_dBm),
            mapping=mapping,
            configuration_generation=self._configuration_generation,
        )
        accumulator = self._raw_accumulator
        exchange = self._snapshot_exchange
        waterfall_producer = self._waterfall_producer
        temporal_accumulator = self._spectrum_temporal_accumulator
        if accumulator is None or exchange is None or waterfall_producer is None or temporal_accumulator is None:
            raise AnalyzerStateError("RTA raw display buffers are not configured")
        native_copy_started = time.perf_counter()
        accumulator.update(raw, metadata, accumulate_interval=False)
        completed_temporal = temporal_accumulator.add_packet(raw, metadata)
        if completed_temporal is not None:
            self._spectrum_temporal_exchange.publish(completed_temporal)
        waterfall_producer.add_packet(raw, metadata, trace_timestamp_step_ns=timestamp_step_ns)
        self._ai_stream.offer_packet(raw, metadata, trace_timestamp_step_ns=timestamp_step_ns)
        native_copy_elapsed = time.perf_counter() - native_copy_started
        self._sequence += count

        spectrum_due = receipt_monotonic_ns >= self._next_spectrum_ns
        if spectrum_due:
            period_ns = max(1, int(1e9 / self._spectrum_fps))
            while self._next_spectrum_ns <= receipt_monotonic_ns:
                self._next_spectrum_ns += period_ns
        snapshot_created = exchange.publish(accumulator, spectrum=spectrum_due, waterfall=False)
        if spectrum_due:
            values = accumulator.copy_latest_dbm()
            frame = SpectrumFrame(
                sequence=latest_sequence,
                timestamp_ns=latest_device_timestamp_ns,
                values=values,
                point_count=width,
                start_frequency_hz=start,
                center_frequency_hz=center,
                stop_frequency_hz=stop,
                span_hz=span,
                rbw_hz=float(profile.RBW_Hz),
                vbw_hz=float(profile.VBW_Hz),
                reference_level_dbm=float(profile.RefLevel_dBm),
                sweep_time_s=float(profile.SweepTime) if math.isfinite(profile.SweepTime) else None,
                frame_type=FrameType.CURRENT,
                overload=overload,
                dropped_frames=exchange.replaced,
                configuration_generation=self._configuration_generation,
            )
            self._latest.publish(frame)
        with self._state_lock:
            diagnostics = self._diagnostics
            self._diagnostics = replace(
                diagnostics,
                packets_received=diagnostics.packets_received + 1,
                trace_frames_received=diagnostics.trace_frames_received + count,
                sample_values=diagnostics.sample_values + sample_values,
                minimum_dbm=packet_minimum if diagnostics.minimum_dbm is None else min(diagnostics.minimum_dbm, packet_minimum),
                maximum_dbm=packet_maximum if diagnostics.maximum_dbm is None else max(diagnostics.maximum_dbm, packet_maximum),
                sum_dbm=diagnostics.sum_dbm + packet_sum,
                sum_squares_dbm=diagnostics.sum_squares_dbm + packet_sum_squares,
                validation_total_s=diagnostics.validation_total_s + validation_elapsed,
                validation_max_s=max(diagnostics.validation_max_s, validation_elapsed),
                statistics_total_s=diagnostics.statistics_total_s + statistics_elapsed,
                statistics_max_s=max(diagnostics.statistics_max_s, statistics_elapsed),
                native_copy_total_s=diagnostics.native_copy_total_s + native_copy_elapsed,
                native_copy_max_s=max(diagnostics.native_copy_max_s, native_copy_elapsed),
                display_conversion_total_s=diagnostics.display_conversion_total_s + (exchange.last_conversion_s if snapshot_created else 0.0),
                display_conversion_max_s=max(diagnostics.display_conversion_max_s, exchange.last_conversion_s if snapshot_created else 0.0),
                snapshot_total_s=diagnostics.snapshot_total_s + (exchange.last_publish_s if snapshot_created else 0.0),
                snapshot_max_s=max(diagnostics.snapshot_max_s, exchange.last_publish_s if snapshot_created else 0.0),
                display_snapshots_created=diagnostics.display_snapshots_created + int(snapshot_created),
            )
            self._received += count
            self._last_frame_ns = latest_device_timestamp_ns
            self._temperature_c = float(auxiliary.Temperature) * 0.01

    @staticmethod
    def _validate_settings(settings: AnalyzerSettings) -> None:
        if settings.mode != "rta":
            raise UnsupportedSettingError("Phase 5 SAN-90 integration supports only SDK RTA mode")
        if settings.span_hz is not None:
            raise UnsupportedSettingError(
                "RTA_Profile_TypeDef has no direct span field; leave span unset and use returned FrameInfo start/stop"
            )
        if not math.isfinite(settings.center_frequency_hz) or settings.center_frequency_hz <= 0:
            raise AnalyzerConfigurationError("center_frequency_hz must be finite and positive")
        if not math.isfinite(settings.reference_level_dbm):
            raise AnalyzerConfigurationError("reference_level_dbm must be finite")
        if settings.rbw_mode not in RBW_MODE_VALUES:
            raise AnalyzerConfigurationError(f"Unsupported rbw_mode {settings.rbw_mode!r}; expected one of {tuple(RBW_MODE_VALUES)}")
        if settings.rbw_mode == "manual" and settings.rbw_hz is None:
            raise AnalyzerConfigurationError("manual RBW mode requires rbw_hz")
        for name in ("rbw_hz", "vbw_hz", "sweep_time_s"):
            value = getattr(settings, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise AnalyzerConfigurationError(f"{name} must be finite and positive")
        if settings.attenuation_db is not None and not -128 <= settings.attenuation_db <= 127:
            raise AnalyzerConfigurationError("attenuation_db does not fit the SDK int8 field")
        for name, values in (
            ("preamplifier", PREAMPLIFIER_VALUES),
            ("gain_strategy", GAIN_STRATEGY_VALUES),
            ("window", WINDOW_VALUES),
            ("detector", DETECTOR_VALUES),
        ):
            value = getattr(settings, name)
            if value is not None and value not in values:
                raise AnalyzerConfigurationError(f"Unsupported {name} value {value!r}; expected one of {tuple(values)}")
