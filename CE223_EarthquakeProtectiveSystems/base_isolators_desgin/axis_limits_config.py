"""
Per-image axis limits for digitization (displacement and force as on the plot).

U0 and P0 are obtained from the digitized cloud using these limits (no rescaling).
For new images, add a sidecar <basename>_limits.json next to the image with:
  {"x_min": -4, "x_max": 4, "y_min": -12, "y_max": 11}
"""
from __future__ import annotations

import json
from pathlib import Path

from isolator_curve_digitizer import AxisLimits

# Default limits (x = displacement [in], y = force [kips]) matching plot axis labels.
DEFAULT_AXIS_LIMITS: dict[str, AxisLimits] = {
    "strain10.png": AxisLimits(x_min=-0.4, x_max=0.4, y_min=-2.2, y_max=2.2),
    "strain74.png": AxisLimits(x_min=-2.5, x_max=2.5, y_min=-7.5, y_max=7.5),
    "strain124.png": AxisLimits(x_min=-4.0, x_max=4.0, y_min=-12.0, y_max=11.0),
    "strain180.png": AxisLimits(x_min=-6.0, x_max=6.0, y_min=-15.0, y_max=15.0),
}


def get_axis_limits(image_name: str, image_dir: Path) -> AxisLimits | None:
    """
    Return axis limits for an image: from sidecar <basename>_limits.json if present,
    else from DEFAULT_AXIS_LIMITS. Returns None if no limits are defined.
    """
    base = Path(image_name).stem
    sidecar = image_dir / f"{base}_limits.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return AxisLimits(
                x_min=float(data["x_min"]),
                x_max=float(data["x_max"]),
                y_min=float(data["y_min"]),
                y_max=float(data["y_max"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
    return DEFAULT_AXIS_LIMITS.get(image_name)
