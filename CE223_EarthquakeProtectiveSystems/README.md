# CE223 – Earthquake Protective Systems

This directory collects numerical experiments, response-history analyses, and interactive dashboards developed for **CE223 – Earthquake Protective Systems**.

The work is organized around base isolation, hysteretic single-degree-of-freedom models, two-degree-of-freedom isolation theory, nonlinear friction pendulum systems, lead-rubber bearings, equivalent linearization, frequency-domain response analysis, and ground-motion response visualization.

## Directory structure

```text
CE223_EarthquakeProtectiveSystems/
├── base_isolators_desgin/
├── frequency_domain/
│   └── fft_sdof_response.py
├── highlighted_htmls/
├── highlighted_pdfs/
├── input_ground_motion/
├── newmark_mdof_demo/
├── nonlinear_dynamic_analyses/
├── sdof_hysteresis/
├── two_dof_isolation/
└── README.md
```

## Project folders

### `base_isolators_desgin/`

Digitization and analysis of base-isolator hysteresis loops, including equivalent-linear SDOF iteration against earthquake input.

Main outputs:

- `isolator_dashboard.html`  
  Interactive dashboard with digitized hysteresis loops, equivalent parameters, loop metrics, and trends.

- `isolator_iteration_dashboard.html`  
  Equivalent-SDOF iteration dashboard comparing Newmark and FFT-based response estimates.

### `frequency_domain/`

Frequency-domain utilities used by CE223 projects.

Main file:

- `fft_sdof_response.py`  
  Shared FFT-based SDOF response utility used for frequency-domain response calculations and comparisons against time-domain Newmark solutions.

### `sdof_hysteresis/`

Harmonic hysteresis loops and energy-dissipation analysis for several SDOF damping models.

The models include:

- Kelvin–Voigt damping
- Hysteretic damping
- Fractional Kelvin–Voigt damping

Main outputs:

- `sdof_hysteresis_dashboard.html`  
  Full dashboard with harmonic loops, stiffness interpretation, equivalent damping trends, frequency dependence, and earthquake response.

- `sdof_hysteresis_loops.html`  
  Focused dashboard containing only the hysteresis-loop comparison.

### `two_dof_isolation/`

Material related to two-degree-of-freedom isolation theory and response calculations.

This folder is intended for models where the isolated superstructure is not reduced to a single rigid mass, and where both isolation-level and superstructure deformation effects may be represented.

### `newmark_mdof_demo/`

Demonstration scripts for Newmark integration in multi-degree-of-freedom systems.

This folder is useful as a reference for time-integration implementation details before moving to the nonlinear dashboard models.

### `nonlinear_dynamic_analyses/`

Main folder for nonlinear response-history dashboards used in the FPS and LRB problems.

Current dashboard builders:

```text
build_fps_bilinear_kinematic_hardening_dashboard.py
build_fps_bouc_wen_dashboard.py
build_lrb_bouc_wen_dashboard_.py
```

#### FPS bilinear dashboard with kinematic hardening

Script:

```bash
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_fps_bilinear_kinematic_hardening_dashboard.py
```

Generated dashboard:

```text
highlighted_htmls/CE223_FPS_Bilinear_Kobe_Sylmar.html
```

This dashboard analyzes a rigid-superstructure friction pendulum system using a bilinear regularization with kinematic hardening.

Main features:

- nonlinear bilinear FPS force law;
- Newmark time integration;
- return mapping for the nonlinear restoring force;
- kinematic hardening regularization;
- equivalent viscously damped linear comparison;
- floor-response spectra for nonstructural components with \(\zeta_p=2\%\);
- Kobe and Sylmar ground-motion response histories.

The ideal FPS quantities are:

\[
K_p=\frac{Mg}{R},
\qquad
Q=\mu Mg.
\]

The bilinear regularization uses:

\[
F_y=Q,
\qquad
k=\frac{F_y}{u_y},
\qquad
H=\frac{kK_p}{k-K_p}.
\]

#### FPS Bouc--Wen dashboard

Script:

```bash
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_fps_bouc_wen_dashboard.py
```

Generated dashboard:

```text
highlighted_htmls/CE223_FPS_BoucWen_Kobe_Sylmar.html
```

This dashboard approximates the FPS hysteresis loop using a Bouc--Wen model.

Main features:

- ideal FPS loop used as the calibration target;
- Bouc--Wen approximation of the prescribed cyclic test;
- nonlinear time-history analysis under Kobe and Sylmar motions;
- normalized force histories \(F/W\);
- force-displacement loops;
- comparison with the bilinear/plasticity FPS model.

The Bouc--Wen restoring force is:

\[
F(t)
=
\alpha K_1 u(t)
+
(1-\alpha)K_1u_yz(t).
\]

The hysteretic variable satisfies:

\[
u_y\dot z
+
\gamma z|\dot u||z|^{n-1}
+
\beta\dot u|z|^n
-
\dot u
=
0.
\]

#### LRB Bouc--Wen dashboard

Script:

```bash
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_lrb_bouc_wen_dashboard_.py
```

Generated dashboard:

```text
highlighted_htmls/CE223_LRB_BoucWen_Kobe_Sylmar.html
```

This dashboard analyzes a rigid mass supported on lead-rubber bearings using a Bouc--Wen hysteretic model calibrated from a recorded force-displacement loop.

Main features:

- cyclic calibration of one lead-rubber bearing;
- full cyclic loading path shown from \(t=0\);
- Plotly sliders for \(\beta\), \(\gamma\), and \(n\);
- \(z(t)\) plots for the Bouc--Wen hysteretic parameter;
- nonlinear time-history analysis under Kobe and Sylmar motions;
- equivalent viscously damped linear oscillator comparison;
- detailed hover information for response histories and hysteresis loops.

The stiffness interpretation used in the calibration is:

\[
K_2=S=0.92\,\mathrm{kN/mm},
\qquad
K_1=10S=9.20\,\mathrm{kN/mm},
\qquad
\alpha=\frac{K_2}{K_1}=0.10.
\]

The selected Bouc--Wen parameters are:

\[
\beta=0.50,
\qquad
\gamma=0.50,
\qquad
n=1.00.
\]

The restoring force is decomposed as:

\[
F(t)
=
\alpha K_1u(t)
+
(1-\alpha)K_1u_yz(t).
\]

The dashboard reports the two force components:

\[
\alpha K_1u=K_2u,
\qquad
(1-\alpha)K_1u_yz.
\]

## Shared data and outputs

### `input_ground_motion/`

Ground-motion records used across CE223 analyses.

Currently includes:

```text
RSN1108_KOBE_KBU090.AT2
```

This is the 090 component of the 1995 Kobe, Japan earthquake.

Additional project-specific records may be stored inside the corresponding analysis folder. For example:

```text
nonlinear_dynamic_analyses/SYLMAR360.txt
```

### `highlighted_htmls/`

Canonical GitHub-Pages-ready dashboard outputs.

The nonlinear dashboard scripts write their final HTML files directly to this folder. These are the files linked from the main course/project webpage.

Main generated dashboards include:

```text
CE223_FPS_Bilinear_Kobe_Sylmar.html
CE223_FPS_BoucWen_Kobe_Sylmar.html
CE223_LRB_BoucWen_Kobe_Sylmar.html
```

### `highlighted_pdfs/`

Folder for exported PDF versions of selected reports, figures, or dashboards.

## Usage

From the repository root, run the desired dashboard builder.

```bash
# Base-isolator dashboards
python CE223_EarthquakeProtectiveSystems/base_isolators_desgin/build_isolator_dashboard.py
python CE223_EarthquakeProtectiveSystems/base_isolators_desgin/build_isolator_iteration_dashboard.py

# SDOF hysteresis dashboard
python CE223_EarthquakeProtectiveSystems/sdof_hysteresis/build_sdof_hysteresis_dashboard.py

# FPS bilinear dashboard with kinematic hardening
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_fps_bilinear_kinematic_hardening_dashboard.py

# FPS Bouc--Wen dashboard
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_fps_bouc_wen_dashboard.py

# LRB Bouc--Wen dashboard
python CE223_EarthquakeProtectiveSystems/nonlinear_dynamic_analyses/build_lrb_bouc_wen_dashboard_.py
```

All main dashboards are written to:

```text
CE223_EarthquakeProtectiveSystems/highlighted_htmls/
```

## Local preview

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/webpage/index.html
```

or open a dashboard directly:

```text
http://localhost:8000/CE223_EarthquakeProtectiveSystems/highlighted_htmls/CE223_FPS_Bilinear_Kobe_Sylmar.html
http://localhost:8000/CE223_EarthquakeProtectiveSystems/highlighted_htmls/CE223_FPS_BoucWen_Kobe_Sylmar.html
http://localhost:8000/CE223_EarthquakeProtectiveSystems/highlighted_htmls/CE223_LRB_BoucWen_Kobe_Sylmar.html
```

## Notes

- The HTML dashboards are generated artifacts. After changing model parameters, plotting options, or ground-motion paths, rerun the corresponding builder script.
- The nonlinear dashboards are intended to act as reproducible numerical reports, not only plotting scripts.
- When a parameter is estimated from a figure, the assumption should be documented directly in the code and in the dashboard text.
- The `highlighted_htmls/` folder contains the canonical website-ready versions of the dashboards.