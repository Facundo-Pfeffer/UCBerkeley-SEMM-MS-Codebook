# Base Isolator Curve Digitizer

This module provides a robust, scriptable way to digitize scanned
force–displacement plots for base isolators (e.g. the `strain*.png`
files in `input_diagrams/`).

## Core ideas

- Automatically detect the **plotting rectangle** using dark-pixel density
  (works even with noisy scans and axis frames).
- Map every “ink” pixel to a **(displacement, force)** pair in physical units.
- Optionally:
  - Shuffle and subsample to cap the total number of points.
  - **Re-calibrate** the displacement axis using a known maximum displacement.
- Save clouds to CSV for reproducible workflows, while always returning
  the cloud as a NumPy array for in-memory use.

## Files

- `isolator_curve_digitizer.py`
  - `AxisLimits`: container for `(x_min, x_max, y_min, y_max)`.
  - `IsolatorCurveDigitizer`:
    - `digitize(image_path, max_points=None, shuffle=True, known_max_displacement=None)`
      → `np.ndarray (N, 2)` with columns `[disp, force]`.
    - `digitize_to_csv(...)` → same as `digitize`, plus writes a CSV.
  - `digitize_many(...)`: helper for multiple images.

- `run_digitizer.py`
  - Batch script which:
    - Finds all `strain*.png` in `input_diagrams/`.
    - Uses a common `AxisLimits`.
    - Applies **known max displacement** calibration per image
      (see table below).
    - Saves `strainXX.csv` with header `disp_in,force_kip`.

- `isolator_plotly.py`
  - `PlotStyle`: styling configuration for Plotly figures.
  - `create_force_displacement_figure(points, title, meta=None)`: returns
    a styled Plotly figure showing the digitized cloud.

- `build_isolator_dashboard.py`
  - Uses the digitizer and Plotly helpers to build a single
    `isolator_dashboard.html` report page with four interactive figures and
    short textual summaries.

- `test_isolator_curve_digitizer.py`
  - Pytest that verifies the calibrated cloud’s max \(|x|\) matches the
    known max displacement for each `strain*.png` image.

## Known maximum displacements (benchmark)

For the four provided plots, the maximum displacements are:

| Image         | Max disp [in] |
|--------------|---------------|
| `strain10`   | 0.3185        |
| `strain124`  | 2.405         |
| `strain180`  | 4.0365        |
| `strain74`   | 5.8435        |

These values are used in:

- `run_digitizer.py` (to rescale the horizontal axis of each cloud).
- `test_isolator_curve_digitizer.py` (to check that max \(|x|\) matches).

## Usage

From `CE223_EarthquakeProtectiveSystems/base_isolators_desgin/`:

```bash
python run_digitizer.py
```

This will produce:

- `input_diagrams/strain10.csv`
- `input_diagrams/strain124.csv`
- `input_diagrams/strain180.csv`
- `input_diagrams/strain74.csv`

Each CSV has two columns:

```text
disp_in,force_kip
...many rows...
```

To use the class directly:

```python
from pathlib import Path
from isolator_curve_digitizer import AxisLimits, IsolatorCurveDigitizer

limits = AxisLimits(x_min=-0.25, x_max=0.25, y_min=-2.5, y_max=1.5)
dig = IsolatorCurveDigitizer(limits)

pts = dig.digitize(
    Path("input_diagrams/strain10.png"),
    max_points=8000,
    known_max_displacement=0.3185,  # optional calibration
)
# pts is a NumPy array with shape (N, 2)
```

### Building the Plotly dashboard

From the same directory:

```bash
python build_isolator_dashboard.py
```

This produces `isolator_dashboard.html` containing:

- A short project description.
- One interactive force–displacement plot per `strain*.png` image.
- A small summary (calibrated max displacement and max |force|) for each case.

## Running tests

From the project root (or this directory):

```bash
pytest CE223_EarthquakeProtectiveSystems/base_isolators_desgin/test_isolator_curve_digitizer.py
```

This confirms that the calibrated \(|x|\) max for each `strain*.png`
agrees with the table above to within a small tolerance.

