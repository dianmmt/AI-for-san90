"""Common contract implemented by simulated and physical analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AnalyzerCapabilities, AnalyzerSettings, AnalyzerSettingsState, DeviceInfo, RuntimeStatus, SpectrumFrame


class AnalyzerSource(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Discover and open the selected source."""

    @abstractmethod
    def disconnect(self) -> None:
        """Stop acquisition and release source resources."""

    @abstractmethod
    def start(self) -> None:
        """Begin acquisition without blocking the caller."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition cleanly."""

    @abstractmethod
    def get_capabilities(self) -> AnalyzerCapabilities:
        """Return source capabilities known to the application."""

    @abstractmethod
    def get_settings(self) -> AnalyzerSettings:
        """Return actual accepted settings."""

    @abstractmethod
    def apply_settings(self, settings: AnalyzerSettings) -> AnalyzerSettings:
        """Validate and apply settings, returning actual accepted values."""

    @abstractmethod
    def get_settings_state(self) -> AnalyzerSettingsState:
        """Return requested settings, actual metadata, and configuration generation."""

    @abstractmethod
    def read_frame(self) -> SpectrumFrame | None:
        """Return the latest frame, or None if no newer frame is available."""

    @abstractmethod
    def get_status(self) -> RuntimeStatus:
        """Return an immutable runtime snapshot."""

    @abstractmethod
    def get_device_info(self) -> DeviceInfo | None:
        """Return device identity after connection."""

    @abstractmethod
    def get_spectrum_publish_fps(self) -> float:
        """Return the active capability-driven spectrum publication target."""
