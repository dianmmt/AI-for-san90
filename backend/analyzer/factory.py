"""Environment-selectable analyzer source factory."""

from __future__ import annotations

import os

from .base import AnalyzerSource
from .errors import AnalyzerConfigurationError
from .simulator import SimulatorSource


def create_analyzer_source(source: str | None = None, **kwargs: object) -> AnalyzerSource:
    selected = (source or os.getenv("ANALYZER_SOURCE", "simulator")).strip().lower()
    if selected == "simulator":
        return SimulatorSource(**kwargs)
    if selected == "san90":
        from .san90 import San90Source

        return San90Source(**kwargs)
    raise AnalyzerConfigurationError(
        f"Unknown ANALYZER_SOURCE={selected!r}; expected 'simulator' or 'san90'"
    )
