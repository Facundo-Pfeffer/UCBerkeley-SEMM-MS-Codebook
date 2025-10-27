# CEE231 – Solid Mechanics

This is the repository portfolio folder for CE231 Solid Mechanics at UC Berkeley under the instruction of Sanjay Govindjee.

## Contents

- `ce231_specific_homework_runs/` – Python scripts for homework assignments (`01HW.py` through `07HW.py`).
- `highlighted_htmls/` – Interactive visualizations that can be opened directly in a browser.

## Key features

- **Directional Young's Modulus visualizer** – Computes and plots \(E(\mathbf{d})\) over the unit sphere from the compliance matrix, showing material anisotropy for cubic and general materials (Fe, Nb, NiTi alloy).

- **Material property tools** – Functions to construct stiffness/compliance matrices for cubic materials and general anisotropic materials, compute stress/strain in Voigt notation, and calculate strain energy.

## Usage

1. Run any script in `ce231_specific_homework_runs/` (e.g., `python 06HW.py`).
2. Interactive Plotly figures are generated and can be exported as HTML.
3. View the HTML files in `highlighted_htmls/` or directly linked from the portfolio website.

## Example outputs

- `highlighted_htmls/Directional_Youngs_Modulus_A.html` – Iron (Fe, cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_B.html` – Niobium (Nb, cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_C.html` – NiTi alloy (general anisotropy)
