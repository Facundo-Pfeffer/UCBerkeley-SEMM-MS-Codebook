# CEE231 – Solid Mechanics

This is the repository portfolio folder for CEE231 Solid Mechanics at UC Berkeley under the instruction of Sanjay Govindjee.

## Contents

- `ce231_specific_homework_runs/` – Python scripts for homework assignments and dashboard generators
  - `homework_scripts/` – Individual homework Python scripts (`hw01.py` through `hw11.py`)
  - `dashboard_generators/` – Scripts that generate interactive dashboards
  - `plotly_templates.py` – Shared UC Berkeley styling and templates
- `highlighted_htmls/` – Web-ready interactive visualizations (deployed by GitHub Actions)
- `highlighted_pdfs/` – PDF homework solutions

## Key Features

### Anisotropic Elasticity Visualizations
- **Directional Young's Modulus** – Computes and plots E(d) over the unit sphere from the compliance matrix, showing material anisotropy for cubic and general materials (Fe, Nb, NiTi alloy).
- **3D Point Cloud Representation** – Interactive rotation and inspection of crystallographic symmetries.

### Viscoelastic Response Dashboards
- **Step Response** – Two-parameter viscoelastic model with relaxation modulus E_r(t) = E₁ + E₂·exp[-(t/β)²]
- **Sinusoidal Response** – Standard Linear Solid (SLS) subjected to harmonic strain; hysteresis loop visualization
- **Square Wave Response** – SLS with Backward Euler time integration showing viscous strain evolution

### Material Property Tools
- Functions to construct stiffness/compliance matrices for cubic and general anisotropic materials
- Compute stress/strain in Voigt notation
- Calculate strain energy and characteristic values

## Usage

### Running Homework Scripts
```bash
cd ce231_specific_homework_runs/homework_scripts
python hw09.py  # Generates both step and sinusoidal response dashboards
```

### Regenerating Dashboards
```bash
cd ce231_specific_homework_runs/dashboard_generators
python step_response_dashboard.py
python sls_sinusoidal_dashboard.py
python sls_square_wave_dashboard.py
```

All dashboards are automatically saved to `highlighted_htmls/` for web deployment.

## Interactive Outputs

### Anisotropic Materials (HW7)
- `highlighted_htmls/Directional_Youngs_Modulus_A.html` – Iron (Fe, cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_B.html` – Niobium (Nb, cubic)
- `highlighted_htmls/Directional_Youngs_Modulus_C.html` – NiTi alloy (general anisotropy)
- `highlighted_htmls/CEE231_HW7_Convergence_Analysis.html` – Convergence analysis

### Viscoelastic Materials (HW8-HW9)
- `highlighted_htmls/step_response_clean.html` – Viscoelastic step response
- `highlighted_htmls/sls_sinusoidal_clean.html` – SLS sinusoidal response
- `highlighted_htmls/sls_square_wave_clean.html` – SLS square wave with time integration

## Documentation

See `ce231_specific_homework_runs/README.md` for detailed documentation on dashboard generation, styling templates, and development workflow.

---

**Author**: Facundo L. Pfeffer  
**Institution**: University of California, Berkeley  
**Course**: CEE231 - Solid Mechanics
