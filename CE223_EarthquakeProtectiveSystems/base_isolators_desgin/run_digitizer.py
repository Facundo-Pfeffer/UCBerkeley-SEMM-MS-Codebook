from __future__ import annotations

"""
Batch driver script for digitizing isolator force–displacement plots.

Axis limits (and thus U0, P0) come from the plot scale: per-image limits are read from
axis_limits_config (sidecar <basename>_limits.json or defaults). No rescaling.

Usage (from this directory):
    python run_digitizer.py
"""

from pathlib import Path

import numpy as np

from axis_limits_config import get_axis_limits
from isolator_curve_digitizer import IsolatorCurveDigitizer


def main() -> None:
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input_diagrams"

    imgs = sorted(input_dir.glob("strain*.png"))
    if not imgs:
        raise SystemExit(f"No strain*.png files found in {input_dir}")

    for img in imgs:
        limits = get_axis_limits(img.name, input_dir)
        if limits is None:
            print(f"Skipping {img.name} (no axis limits defined).")
            continue
        print(f"Digitizing {img.name} ...", flush=True)
        digitizer = IsolatorCurveDigitizer(limits, min_bin_density=3)
        pts_cal = digitizer.digitize(
            img,
            max_points=16000,
            shuffle=True,
            known_max_displacement=None,
        )
        if img.name == "strain74.png":
            noise1 = (pts_cal[:, 0] < -2.3) & (pts_cal[:, 1] > -2.46)
            noise2 = (pts_cal[:, 0] < -1.9) & (pts_cal[:, 0] > -2.0) & (pts_cal[:, 1] > -1.5)
            pts_cal = pts_cal[~(noise1 | noise2)]
        csv_path = img.with_suffix(".csv")
        np.savetxt(
            csv_path,
            pts_cal,
            delimiter=",",
            header="disp_in,force_kip",
            comments="",
        )
        u0 = float(np.max(np.abs(pts_cal[:, 0])))
        print(
            f"  -> {pts_cal.shape[0]} points, U0={u0:.4f} in, "
            f"force in [{pts_cal[:,1].min():.3f}, {pts_cal[:,1].max():.3f}] kips"
        )


if __name__ == "__main__":
    main()

