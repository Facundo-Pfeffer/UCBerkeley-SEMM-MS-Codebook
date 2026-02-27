# Axis limits for digitization

Displacement (u) and force (F) are read from the image by mapping the plot rectangle to **physical axis limits**. Those limits must match the axis labels on the plot so that U0 and P0 come from the digitized curve (no rescaling).

## Default limits

The dashboard and digitizer use default limits for `strain10.png`, `strain74.png`, `strain124.png`, `strain180.png` (see `axis_limits_config.py`).

## New images / overriding limits

For any image, you can provide a **sidecar JSON file** with the same base name and suffix `_limits.json`. Example for `mytest.png`:

**mytest_limits.json**
```json
{
  "x_min": -4.0,
  "x_max": 4.0,
  "y_min": -12.0,
  "y_max": 11.0
}
```

- `x_min`, `x_max`: displacement range (e.g. inches) as on the plot’s horizontal axis.
- `y_min`, `y_max`: force range (e.g. kips) as on the plot’s vertical axis.

Place the file next to the image (e.g. in `input_diagrams/`). The digitizer will use it and U0 = max|u|, P0 = max|F| from the digitized cloud.
