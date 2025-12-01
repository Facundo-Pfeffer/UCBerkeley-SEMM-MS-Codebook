# CEE 231 - Solid Mechanics: Homework & Dashboards

Organized repository for CEE 231 Solid Mechanics course homework assignments and interactive dashboards.

## 📁 Folder Structure

```
ce231_specific_homework_runs/
│
├── homework_scripts/          # Individual homework Python scripts (hw01.py - hw11.py)
├── dashboard_generators/      # Python scripts that generate dashboards → ../highlighted_htmls/
├── hw7_outputs/              # HW7 specific outputs (LaTeX, README)
│
├── plotly_templates.py       # Shared Plotly styling and templates
└── README.md                 # This file

../highlighted_htmls/          # Web-ready HTML files (deployed by GitHub Actions)
```

## 🎨 Plotly Templates

The `plotly_templates.py` file provides centralized styling for all dashboards:

- **UC Berkeley Colors**: Official brand colors (Berkeley Blue, California Gold, etc.)
- **Axis Styling**: Consistent grid, line, and tick configurations
- **Layout Presets**: Standard layouts for different dashboard types
- **Reusable Components**: Sliders, tables, titles, markers, etc.

### Usage Example

```python
from plotly_templates import (
    UCBerkeleyColors as Colors,
    get_axis_style,
    get_slider_style,
    save_figure
)

# Use Berkeley Blue for a line
line_style = dict(color=Colors.BERKELEY_BLUE, width=3)

# Apply standard axis styling
fig.update_xaxes(**get_axis_style())

# Save with standard configuration
save_figure(fig, "output.html")
```

## 📊 Available Dashboards

### 1. **Viscoelastic Step Response Dashboard**
**File**: `../highlighted_htmls/step_response_clean.html`  
**Generator**: `dashboard_generators/step_response_dashboard.py`

Interactive visualization of viscoelastic material subjected to step strain input.

- **Material Model**: Two-parameter viscoelastic
- **Relaxation Modulus**: E_r(t) = E₁ + E₂ · exp[-(t/β)²]
- **Applied Strain**: ε(t) = ε₀ · h(t)
- **Interactive Control**: 12 strain amplitudes (5e-7 to 2e-3)
- **Plots**: Stress response, Relaxation modulus, Strain input, Characteristic values table

**To regenerate**:
```bash
cd dashboard_generators
python step_response_dashboard.py
```

### 2. **SLS Sinusoidal Response Dashboard**
**File**: `../highlighted_htmls/sls_sinusoidal_clean.html`  
**Generator**: `dashboard_generators/sls_sinusoidal_dashboard.py`

Interactive visualization of Standard Linear Solid subjected to sinusoidal strain.

- **Material Model**: Standard Linear Solid (Zener model)
- **Material Properties**: E_re = 1 MPa, E_rg = 2 MPa, τ_R = 1 s
- **Applied Strain**: ε(t) = ε₀ sin(ωt) h(t), ω = 2π rad/s
- **Interactive Control**: 4 strain amplitudes (0.005, 0.01, 0.02, 0.05)
- **Plots**: Stress response, Strain input, Hysteresis loop, Characteristic values table

**To regenerate**:
```bash
cd dashboard_generators
python sls_sinusoidal_dashboard.py
```

### 3. **SLS Square Wave Response Dashboard (Backward Euler)**
**File**: `../highlighted_htmls/sls_square_wave_clean.html`  
**Generator**: `dashboard_generators/sls_square_wave_dashboard.py`

Interactive visualization of Standard Linear Solid with square wave strain using time integration.

- **Material Model**: Standard Linear Solid with internal variable
- **Material Properties**: E₀ = 2.0 MPa, E₁ = 1.5 MPa, η = 1.0 MPa·s, τ_R = 0.667 s
- **Applied Strain**: Square wave, period = 1 s, amplitude = 0.01, duty cycle = 50%
- **Time Integration**: Backward Euler scheme (γ = τ_R/(τ_R + Δt))
- **Interactive Control**: 6 time step values (0.001 to 0.1 s) to show convergence
- **Plots**: Stress & strain (dual axis), Viscous strain evolution, Stress-strain path, Characteristic values
- **Key Feature**: Explains why stress becomes negative when strain is zero (viscous strain memory effect)

**To regenerate**:
```bash
cd dashboard_generators
python sls_square_wave_dashboard.py
```

## 🔧 Development Workflow

### Creating a New Dashboard

1. **Import shared templates**:
   ```python
   from plotly_templates import UCBerkeleyColors, get_axis_style, save_figure
   ```

2. **Use consistent styling**:
   - Colors from `UCBerkeleyColors`
   - Axis styling from `get_axis_style()`
   - Slider configuration from `get_slider_style()`

3. **Save to correct location** (highlighted_htmls for web deployment):
   ```python
   output_path = os.path.join('..', '..', 'highlighted_htmls', 'your_dashboard.html')
   save_figure(fig, output_path)
   ```

4. **Add generator script** to `dashboard_generators/`

### Modifying Existing Dashboards

1. Edit the generator script in `dashboard_generators/`
2. Run the script to regenerate the HTML in `../highlighted_htmls/`
3. Test the output HTML file
4. The updated dashboard will be automatically deployed via GitHub Actions

## 📚 Homework Scripts

Individual homework Python scripts (`hw01.py` through `hw11.py`) are located in `homework_scripts/`.

Each script contains problem-specific implementations:
- **hw01 - hw07**: Various solid mechanics problems
- **hw08**: Viscoelastic step response (basis for dashboard)
- **hw09**: SLS sinusoidal response (basis for dashboard)
- **hw11**: Power-law plasticity Dash app backend

## 🎓 HW7 Documentation

Documentation files for Homework 7 (Anisotropic Materials Analysis) are in `hw7_outputs/`:
- `HW7_README.md`: Detailed explanation of HW7 methodology
- `HW7_plot_explanation.tex`: Full LaTeX documentation  
- `HW7_plot_explanation_short.tex`: Short LaTeX version

**Note**: HW7 interactive HTML visualizations are deployed from:
- `CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_*.html` (Materials A, B, C)
- `CEE231_SolidMechanics/highlighted_htmls/CEE231_HW7_Convergence_Analysis.html`

These are automatically copied to the deployed site by the GitHub Actions workflow.

## 🌐 Web Integration

All dashboards in `../highlighted_htmls/` are web-ready and automatically deployed via GitHub Actions. They can be:
- Linked directly from course pages
- Embedded in portfolio websites via `<iframe>`
- Shared as standalone interactive visualizations

**Example integration** (on deployed site):
```html
<a href="CEE231_SolidMechanics/highlighted_htmls/step_response_clean.html" target="_blank">
    Step Response Dashboard
</a>
```

**Note**: The GitHub Actions workflow copies `highlighted_htmls/` folders to the deployment root, preserving their relative paths.

## 📝 Notes

- All dashboards use **UC Berkeley official colors**
- Interactive controls are **single synchronized sliders**
- Plots include **hover tooltips** for detailed information
- Generated HTML files use **CDN for Plotly.js** (lightweight)

---

**Author**: Facundo Pfeffer  
**Course**: CEE 231 - Solid Mechanics  
**Institution**: University of California, Berkeley

