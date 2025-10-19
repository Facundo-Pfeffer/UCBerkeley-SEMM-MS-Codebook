# CEE231 – Solid Mechanics

This folder supports the development of homework assignments and exploratory analyses for UC Berkeley’s CEE231 (Solid Mechanics).

## What’s here

- `ce231_specific_homework_runs/` – runnable scripts for the assignments (`01HW.py` … `06HW.py`).
- `highlighted_htmls/` – self‑contained interactive results you can open in a browser (and they are also linked from the portfolio website).

## Unique capabilities

- **Directional Young’s Modulus visualizer**
  - Computes the directional modulus \(E(\mathbf{d})\) over the unit sphere using the Voigt form of the compliance matrix.
  - Generates dense point clouds with hoverable tooltips (direction cosines, \(E\), angles).
  - Useful for diagnosing anisotropy in cubic and general materials.

- **Material models by construction**
  - Quick creators for cubic materials (\(c_{11}, c_{12}, c_{44}\)).
  - Direct handling of general symmetric 6×6 stiffness matrices for alloys.

- **Assignment‑ready utilities**
  - Clean function boundaries for: direction vectors, stress/strain (Voigt↔tensor), and energy calculations.
  - Reproducible defaults (sampling resolution, colorscales, seeds) to match write‑ups and figures.
  - Export helpers to produce self‑contained HTML plots for submissions.

- **Numerical robustness**
  - Guards against singular matrices and ill‑conditioning when inverting \(C\) to \(S\).
  - Optional resolution/precision knobs for fast previews vs. publication‑quality output.

## How to run

1. Open any script under `ce231_specific_homework_runs/` (e.g., `06HW.py`).
2. Run the file to generate interactive figures (and optional HTML exports under `highlighted_htmls/`).
3. Open the generated HTML files directly in your browser for review or inclusion in a report.

## Quick links to interactive examples

- `highlighted_htmls/Directional_Youngs_Modulus_A.html` – Fe (cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_B.html` – Nb (cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_C.html` – NiTi alloy (general anisotropy)

> Note: these HTML files are also linked from the portfolio website for easy viewing.
