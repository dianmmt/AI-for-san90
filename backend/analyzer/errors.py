"""Application-facing analyzer errors."""

from __future__ import annotations

from enum import Enum
from typing import Any


class AnalyzerError(RuntimeError):
    """Base class for analyzer failures."""


class AnalyzerConfigurationError(AnalyzerError):
    """A requested analyzer configuration is invalid."""


class AnalyzerConnectionError(AnalyzerError):
    """The analyzer could not be discovered, opened, or reached."""


class AnalyzerStateError(AnalyzerError):
    """An operation is not valid in the analyzer's current state."""


class AnalyzerTimeoutError(AnalyzerError):
    """An analyzer operation did not complete before its deadline."""


class UnsupportedSettingError(AnalyzerConfigurationError):
    """The selected source or measurement mode cannot apply a setting."""


class SdkError(AnalyzerError):
    """The native HTRA API returned a failure or warning status."""

    def __init__(self, operation: str, status: int) -> None:
        self.operation = operation
        self.status = status
        super().__init__(f"{operation} failed with HTRA API status {status}")


class ControlErrorCode(str, Enum):
    DEVICE_NOT_CONNECTED = "DEVICE_NOT_CONNECTED"
    DEVICE_BUSY = "DEVICE_BUSY"
    RECONFIGURATION_TIMEOUT = "RECONFIGURATION_TIMEOUT"
    UNSUPPORTED_SETTING = "UNSUPPORTED_SETTING"
    VALUE_OUT_OF_RANGE = "VALUE_OUT_OF_RANGE"
    SDK_CONFIGURATION_FAILED = "SDK_CONFIGURATION_FAILED"
    SDK_RESTART_FAILED = "SDK_RESTART_FAILED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    STALE_CONFIGURATION = "STALE_CONFIGURATION"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    UNSUPPORTED_RBW = "UNSUPPORTED_RBW"
    UNSUPPORTED_WINDOW = "UNSUPPORTED_WINDOW"
    UNSUPPORTED_DETECTOR = "UNSUPPORTED_DETECTOR"
    INVALID_PROFILE = "INVALID_PROFILE"
    PROFILE_CONFIGURATION_FAILED = "PROFILE_CONFIGURATION_FAILED"
    BUFFER_RESIZE_FAILED = "BUFFER_RESIZE_FAILED"
    FIRST_FRAME_TIMEOUT = "FIRST_FRAME_TIMEOUT"


class ControlError(AnalyzerError):
    def __init__(
        self,
        code: ControlErrorCode,
        message: str,
        *,
        sdk_status: int | None = None,
        requested_value: Any = None,
        previous_actual_value: Any = None,
        recoverable: bool = False,
    ) -> None:
        self.code = code
        self.sdk_status = sdk_status
        self.requested_value = requested_value
        self.previous_actual_value = previous_actual_value
        self.recoverable = recoverable
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "sdk_status": self.sdk_status,
            "requested_value": self.requested_value,
            "previous_actual_value": self.previous_actual_value,
            "recoverable": self.recoverable,
        }
