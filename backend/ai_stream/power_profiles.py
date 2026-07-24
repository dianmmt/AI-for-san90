"""Fixed absolute-power mappings used by external AI consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PowerProfile:
    name: str
    min_dbm: float
    max_dbm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_dbm) or not np.isfinite(self.max_dbm) or self.max_dbm <= self.min_dbm:
            raise ValueError("power profile limits must be finite and increasing")

    @property
    def db_per_gray_level(self) -> float:
        return (self.max_dbm - self.min_dbm) / 255.0


POWER_PROFILES: dict[str, PowerProfile] = {
    "normal": PowerProfile("normal", -130.0, -50.0),
    "external_lna": PowerProfile("external_lna", -120.0, -20.0),
    "strong_signal": PowerProfile("strong_signal", -100.0, 0.0),
}
DEFAULT_POWER_PROFILE = "external_lna"


def require_power_profile(name: str) -> PowerProfile:
    try:
        return POWER_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown AI power profile {name!r}; expected one of {tuple(POWER_PROFILES)}"
        ) from error


def dbm_to_gray8(
    waterfall_dbm: np.ndarray,
    min_dbm: float,
    max_dbm: float,
    *,
    output: np.ndarray | None = None,
    workspace: np.ndarray | None = None,
) -> np.ndarray:
    """Map fixed absolute dBm limits to C-contiguous GRAY8 using nearest rounding."""
    if waterfall_dbm.shape != (640, 640) or waterfall_dbm.dtype != np.float32:
        raise ValueError("waterfall_dbm must be a 640x640 float32 array")
    if not np.isfinite(min_dbm) or not np.isfinite(max_dbm) or max_dbm <= min_dbm:
        raise ValueError("GRAY8 dBm limits must be finite and increasing")
    if output is None:
        output = np.empty((640, 640), dtype=np.uint8)
    if output.shape != (640, 640) or output.dtype != np.uint8 or not output.flags.c_contiguous:
        raise ValueError("GRAY8 output must be a contiguous 640x640 uint8 array")
    # A reusable float workspace avoids changing the fixed formula while
    # allowing the publisher to retain buffer ownership without allocations.
    if workspace is None:
        workspace = np.empty_like(waterfall_dbm)
    if workspace.shape != (640, 640) or workspace.dtype != np.float32 or not workspace.flags.c_contiguous:
        raise ValueError("GRAY8 workspace must be a contiguous 640x640 float32 array")
    np.subtract(waterfall_dbm, np.float32(min_dbm), out=workspace)
    np.multiply(workspace, np.float32(255.0 / (max_dbm - min_dbm)), out=workspace)
    np.rint(workspace, out=workspace)
    np.clip(workspace, 0.0, 255.0, out=workspace)
    np.copyto(output, workspace, casting="unsafe")
    return output
