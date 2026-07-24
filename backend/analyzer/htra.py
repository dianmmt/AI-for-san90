"""Minimal ctypes binding for the verified HTRA API 0.55.88 RTA subset.

Definitions in this module mirror the canonical 0.55.88 ``htra_api.h``.  Do
not extend them from the bundled Python wrapper; that wrapper has known ABI
drift.  ``tests/test_htra_abi.py`` compares these layouts with a C compiler.
"""

from __future__ import annotations

import ctypes as ct
import platform
from dataclasses import dataclass
from pathlib import Path

from .errors import AnalyzerConnectionError, SdkError

SDK_VERSION_INT = (0 << 16) | (55 << 8) | 88
SDK_VERSION = "0.55.88"
MAX_DEVICE = 256
DEVICE_N90_R0 = 67

# Values copied from the named enums in htra_api.h 0.55.88.
USB = 0x00
USB_PORT_AND_POWER_PORT = 0x00
RBW_MANUAL = 0x00
RBW_AUTO = 0x01
ADAPTIVE = 0x01
BUS_TRIGGER = 0x02
FORCED_OFF = 0x01
LOW_NOISE_PREFERRED = 0x00
HIGH_LINEARITY_PREFERRED = 0x01

API_NO_ERROR = 0
API_WARNING_BUS_TIMEOUT = -10
API_WARNING_IF_OVERFLOW = -12
API_LAST_PACKET = -301
API_TRIGGER_MISSED = -302
API_LAST_PACKET_WITH_TRIGGER_MISSED = -303
API_WARNING_DATA_NOT_READY = -304


class BootProfile(ct.Structure):
    _fields_ = [
        ("PhysicalInterface", ct.c_int),
        ("DevicePowerSupply", ct.c_int),
        ("ETH_IPVersion", ct.c_int),
        ("ETH_IPAddress", ct.c_uint8 * 16),
        ("ETH_RemotePort", ct.c_uint16),
        ("ETH_ErrorCode", ct.c_int32),
        ("ETH_ReadTimeOut", ct.c_int32),
    ]


class NativeDeviceInfo(ct.Structure):
    _fields_ = [
        ("DeviceUID", ct.c_uint64),
        ("Model", ct.c_uint16),
        ("HardwareVersion", ct.c_uint16),
        ("MFWVersion", ct.c_uint32),
        ("FFWVersion", ct.c_uint32),
        ("PMUVersion", ct.c_uint16),
        ("AGUVersion", ct.c_uint16),
    ]


class NativeFirmwareVersion(ct.Structure):
    _fields_ = [
        ("FFWVersion", ct.c_uint32),
        ("MFWVersion", ct.c_uint32),
        ("BusVersion", ct.c_uint32),
        ("PMUVersion", ct.c_uint16),
        ("AGUVersion", ct.c_uint16),
    ]


class BootInfo(ct.Structure):
    _fields_ = [
        ("DeviceInfo", NativeDeviceInfo),
        ("BusSpeed", ct.c_uint32),
        ("BusVersion", ct.c_uint32),
        ("APIVersion", ct.c_uint32),
        ("ErrorCodes", ct.c_int * 7),
        ("Errors", ct.c_int),
        ("WarningCodes", ct.c_int * 7),
        ("Warnings", ct.c_int),
    ]


class MeasAuxInfo(ct.Structure):
    _fields_ = [
        ("MaxIndex", ct.c_uint32),
        ("MaxPower_dBm", ct.c_float),
        ("Temperature", ct.c_int16),
        ("RFState", ct.c_uint16),
        ("BBState", ct.c_uint16),
        ("GainPattern", ct.c_uint16),
        ("ConvertPattern", ct.c_uint32),
        ("SysTimeStamp", ct.c_double),
        ("AbsoluteTimeStamp", ct.c_double),
        ("Latitude", ct.c_float),
        ("Longitude", ct.c_float),
        ("Altitude", ct.c_float),
        ("SATHealth", ct.c_float),
        ("IFAGCGain", ct.c_double),
        ("RefClkFreqOffset", ct.c_double),
        ("nsSinceEpoch", ct.c_uint64),
    ]


class TriggerInfo(ct.Structure):
    _fields_ = [
        ("SysTimerCountOfFirstDataPoint", ct.c_uint64),
        ("InPacketTriggeredDataSize", ct.c_uint16),
        ("InPacketTriggerEdges", ct.c_uint16),
        ("StartDataIndexOfTriggerEdges", ct.c_uint32 * 25),
        ("SysTimerCountOfEdges", ct.c_uint64 * 25),
        ("EdgeType", ct.c_int8 * 25),
    ]


class RtaProfile(ct.Structure):
    _fields_ = [
        ("CenterFreq_Hz", ct.c_double),
        ("RefLevel_dBm", ct.c_double),
        ("RBW_Hz", ct.c_double),
        ("VBW_Hz", ct.c_double),
        ("RBWMode", ct.c_int),
        ("VBWMode", ct.c_int),
        ("DecimateFactor", ct.c_uint32),
        ("Window", ct.c_int),
        ("SweepTimeMode", ct.c_int),
        ("SweepTime", ct.c_double),
        ("Detector", ct.c_int),
        ("TraceDetectMode", ct.c_int),
        ("TraceDetectRatio", ct.c_uint32),
        ("TraceDetector", ct.c_int),
        ("RxPort", ct.c_int),
        ("BusTimeout_ms", ct.c_uint32),
        ("TriggerSource", ct.c_int),
        ("TriggerEdge", ct.c_int),
        ("TriggerMode", ct.c_int),
        ("TriggerAcqTime", ct.c_double),
        ("TriggerOutMode", ct.c_int),
        ("TriggerOutPulsePolarity", ct.c_int),
        ("TriggerLevel_dBm", ct.c_double),
        ("TriggerLevel_SafeTime", ct.c_double),
        ("TriggerDelay", ct.c_double),
        ("PreTriggerTime", ct.c_double),
        ("TriggerTimerSync", ct.c_int),
        ("TriggerTimer_Period", ct.c_double),
        ("EnableReTrigger", ct.c_uint8),
        ("ReTrigger_Period", ct.c_double),
        ("ReTrigger_Count", ct.c_uint16),
        ("GainStrategy", ct.c_int),
        ("Preamplifier", ct.c_int),
        ("AnalogIFBWGrade", ct.c_uint8),
        ("IFGainGrade", ct.c_uint8),
        ("EnableDebugMode", ct.c_uint8),
        ("ReferenceClockSource", ct.c_int),
        ("ReferenceClockFrequency", ct.c_double),
        ("EnableReferenceClockOut", ct.c_uint8),
        ("SystemClockSource", ct.c_int),
        ("ExternalSystemClockFrequency", ct.c_double),
        ("Atten", ct.c_int8),
        ("EnableIFAGC", ct.c_uint8),
        ("DCCancelerMode", ct.c_int),
        ("QDCMode", ct.c_int),
        ("QDCIGain", ct.c_float),
        ("QDCQGain", ct.c_float),
        ("QDCPhaseComp", ct.c_float),
        ("DCCIOffset", ct.c_int8),
        ("DCCQOffset", ct.c_int8),
        ("LOOptimization", ct.c_int),
    ]


class RtaFrameInfo(ct.Structure):
    _fields_ = [
        ("StartFrequency_Hz", ct.c_double),
        ("StopFrequency_Hz", ct.c_double),
        ("POI", ct.c_double),
        ("TraceTimestampStep", ct.c_double),
        ("TimeResolution", ct.c_double),
        ("PacketAcqTime", ct.c_double),
        ("PacketCount", ct.c_uint32),
        ("PacketFrame", ct.c_uint32),
        ("FFTSize", ct.c_uint32),
        ("FrameWidth", ct.c_uint32),
        ("FrameHeight", ct.c_uint32),
        ("PacketSamplePoints", ct.c_uint32),
        ("PacketValidPoints", ct.c_uint32),
        ("MaxDensityValue", ct.c_uint32),
        ("GainParameter", ct.c_uint32),
    ]


class RtaPlotInfo(ct.Structure):
    _fields_ = [
        ("ScaleTodBm", ct.c_float),
        ("OffsetTodBm", ct.c_float),
        ("SpectrumBitmapIndex", ct.c_uint64),
    ]


@dataclass(frozen=True, slots=True)
class DiscoveredDevice:
    device_number: int
    info: NativeDeviceInfo


def default_library_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "armv7l": "armv7",
    }.get(machine)
    if architecture is None:
        raise AnalyzerConnectionError(f"Unsupported SDK host architecture: {machine}")
    return root / "harogic" / "Linux_API" / "htraapi" / "lib" / architecture / "libhtraapi.so.0.55.88"


def version_string(value: int) -> str:
    return f"{(value >> 16) & 0xff}.{(value >> 8) & 0xff}.{value & 0xff}"


class HtraApi:
    """Loaded native library with signatures for the Phase 5 RTA subset."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        self.path = Path(library_path).expanduser().resolve() if library_path else default_library_path()
        if not self.path.is_file():
            raise AnalyzerConnectionError(f"HTRA API shared library not found: {self.path}")
        try:
            self.lib = ct.CDLL(str(self.path))
        except OSError as error:
            raise AnalyzerConnectionError(f"Could not load HTRA API library {self.path}: {error}") from error
        self._bind()
        actual = int(self.lib.Get_APIVersion())
        if actual != SDK_VERSION_INT:
            raise AnalyzerConnectionError(
                f"HTRA API version mismatch: expected {SDK_VERSION}, loaded {version_string(actual)} from {self.path}"
            )

    def _bind(self) -> None:
        device_ref = ct.POINTER(ct.c_void_p)
        self.lib.Get_APIVersion.argtypes = []
        self.lib.Get_APIVersion.restype = ct.c_int
        self.lib.APISupportFirmwareVersions.argtypes = [
            ct.POINTER(ct.POINTER(NativeFirmwareVersion)), ct.POINTER(ct.c_uint32),
        ]
        self.lib.APISupportFirmwareVersions.restype = ct.c_int
        self.lib.Device_List.argtypes = [
            ct.POINTER(BootProfile), ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_uint8), ct.POINTER(NativeDeviceInfo),
        ]
        self.lib.Device_List.restype = ct.c_int
        self.lib.Device_Open.argtypes = [device_ref, ct.c_int, ct.POINTER(BootProfile), ct.POINTER(BootInfo)]
        self.lib.Device_Open.restype = ct.c_int
        self.lib.Device_Close.argtypes = [device_ref]
        self.lib.Device_Close.restype = ct.c_int
        self.lib.RTA_ProfileDeInit.argtypes = [device_ref, ct.POINTER(RtaProfile)]
        self.lib.RTA_ProfileDeInit.restype = ct.c_int
        self.lib.RTA_Configuration.argtypes = [
            device_ref, ct.POINTER(RtaProfile), ct.POINTER(RtaProfile), ct.POINTER(RtaFrameInfo),
        ]
        self.lib.RTA_Configuration.restype = ct.c_int
        self.lib.RTA_BusTriggerStart.argtypes = [device_ref]
        self.lib.RTA_BusTriggerStart.restype = ct.c_int
        self.lib.RTA_BusTriggerStop.argtypes = [device_ref]
        self.lib.RTA_BusTriggerStop.restype = ct.c_int
        self.lib.RTA_GetRealTimeSpectrum_Raw.argtypes = [
            device_ref, ct.POINTER(ct.c_uint8), ct.POINTER(RtaPlotInfo),
            ct.POINTER(TriggerInfo), ct.POINTER(MeasAuxInfo),
        ]
        self.lib.RTA_GetRealTimeSpectrum_Raw.restype = ct.c_int

    @staticmethod
    def usb_boot_profile() -> BootProfile:
        profile = BootProfile()
        profile.PhysicalInterface = USB
        profile.DevicePowerSupply = USB_PORT_AND_POWER_PORT
        return profile

    def list_devices(self, boot_profile: BootProfile) -> list[DiscoveredDevice]:
        count = ct.c_uint8()
        numbers = (ct.c_uint8 * MAX_DEVICE)()
        infos = (NativeDeviceInfo * MAX_DEVICE)()
        status = int(self.lib.Device_List(ct.byref(boot_profile), ct.byref(count), numbers, infos))
        if status != API_NO_ERROR:
            raise SdkError("Device_List", status)
        return [
            DiscoveredDevice(
                int(numbers[i]),
                NativeDeviceInfo.from_buffer_copy(bytes(infos[i])),
            )
            for i in range(int(count.value))
        ]

    def supported_firmware_versions(self) -> list[NativeFirmwareVersion]:
        versions = ct.POINTER(NativeFirmwareVersion)()
        count = ct.c_uint32()
        status = int(self.lib.APISupportFirmwareVersions(ct.byref(versions), ct.byref(count)))
        self.require_ok("APISupportFirmwareVersions", status)
        return [NativeFirmwareVersion.from_buffer_copy(bytes(versions[index])) for index in range(count.value)]

    @staticmethod
    def require_ok(operation: str, status: int) -> None:
        if status != API_NO_ERROR:
            raise SdkError(operation, status)
