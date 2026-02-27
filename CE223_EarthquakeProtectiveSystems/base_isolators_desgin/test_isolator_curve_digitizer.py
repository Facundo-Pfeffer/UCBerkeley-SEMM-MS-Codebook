from __future__ import annotations

from pathlib import Path

import numpy as np

from axis_limits_config import get_axis_limits
from isolator_curve_digitizer import IsolatorCurveDigitizer


BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "input_diagrams"


def test_u0_from_axis_limits():
    """U0 (max |displacement|) is obtained from the image via per-image axis limits (no rescaling)."""
    for name in ("strain10.png", "strain74.png", "strain124.png", "strain180.png"):
        limits = get_axis_limits(name, IMG_DIR)
        assert limits is not None, f"{name}: no axis limits defined"
        img = IMG_DIR / name
        if not img.exists():
            continue
        digitizer = IsolatorCurveDigitizer(limits, min_bin_density=3)
        pts = digitizer.digitize(
            img,
            max_points=8000,
            shuffle=True,
            known_max_displacement=None,
        )
        max_abs_u = float(np.max(np.abs(pts[:, 0])))
        axis_extent = max(abs(limits.x_min), abs(limits.x_max))
        # Loop should reach a reasonable fraction of the axis and not exceed it.
        assert max_abs_u >= 0.3 * axis_extent, (
            f"{name}: U0={max_abs_u} too small for axis extent {axis_extent}"
        )
        assert max_abs_u <= axis_extent * 1.02, (
            f"{name}: U0={max_abs_u} exceeds axis extent {axis_extent}"
        )

