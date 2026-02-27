# CE223 – Earthquake Protective Systems

This directory collects the numerical experiments and interactive dashboards for the CE223 course.
It currently contains **two main projects**:

- **Base isolator design** (`base_isolators_desgin/`)
- **SDOF damping model comparisons** (`sdof_hysteresis/`)

A shared FFT-based SDOF response utility is provided in `fft_sdof_response.py`, and the
Kobe University ground motion is stored under `input_ground_motion/`.

## Directory structure

- `base_isolators_desgin/`  
  Digitization and analysis of base-isolator hysteresis loops, plus an equivalent linear
  SDOF iteration against a real earthquake record. Produces:
  - `isolator_dashboard.html`: hysteresis curves, metrics, and trends.
  - `isolator_iteration_dashboard.html`: equivalent-SDOF iteration and Newmark/FFT comparison.

- `sdof_hysteresis/`  
  Harmonic hysteresis loops and energy-dissipation analysis for three SDOF damping models
  (Kelvin–Voigt, hysteretic, fractional Kelvin–Voigt), plus their earthquake response via FFT.
  Produces:
  - `sdof_hysteresis_dashboard.html`: harmonic loops, \(K_1\)/\(K_2\), EDC vs frequency, and
    earthquake response for Models A/B/C.
  - `sdof_hysteresis_loops.html`: focused view of the loops only.

- `input_ground_motion/`  
  Ground motion records in PEER AT2 format. Currently includes:
  - `RSN1108_KOBE_KBU090.AT2` – 090 component of the 1995 Kobe, Japan earthquake.

- `highlighted_htmls/`  
  Canonical, GitHub-Pages-ready copies of the CE223 dashboards. These are the files linked
  from the main website.

- `fft_sdof_response.py`  
  Shared FFT-based SDOF solver for base-isolator and SDOF hysteresis projects.

## Usage

From the repository root, ensure dependencies (NumPy, Plotly, etc.) are installed in your
Python environment. Then:

```bash
# Base isolator dashboards
python CE223_EarthquakeProtectiveSystems/base_isolators_desgin/build_isolator_dashboard.py
python CE223_EarthquakeProtectiveSystems/base_isolators_desgin/build_isolator_iteration_dashboard.py

# SDOF hysteresis dashboard
python CE223_EarthquakeProtectiveSystems/sdof_hysteresis/build_sdof_hysteresis_dashboard.py
```

All scripts write their **canonical outputs** directly to
`CE223_EarthquakeProtectiveSystems/highlighted_htmls/`, which are then served by GitHub Pages.

## Deployment notes

To preview locally, from the repository root run:

```bash
python -m http.server 8000
```

and visit:

- `http://localhost:8000/webpage/index.html` – main landing page (with CE223 tile).
- `http://localhost:8000/CE223_EarthquakeProtectiveSystems/highlighted_htmls/...` – CE223 dashboards.

On GitHub Pages, the same dashboards are available through the `/dist/` mirror, so relative
paths in the HTML assume `highlighted_htmls/` is at the site root.

