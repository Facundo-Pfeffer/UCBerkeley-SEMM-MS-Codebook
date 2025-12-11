
# CE225 Dynamics - Structural Dynamics Analysis Package

A Python package for structural dynamics analysis, featuring numerical methods for Single Degree of Freedom (SDOF) systems, Multi-Degree-of-Freedom (MDOF) systems, earthquake response spectrum generation, and modal analysis. Developed for UC Berkeley's CE225 Structural Dynamics course.

## Features

### Core Functionality
- **Numerical Methods**: Implementation of various numerical integration methods for solving equations of motion (EOM) of SDOF systems
- **Response Spectrum Analysis**: Tools for generating and analyzing elastic response spectra for earthquake engineering applications
- **MDOF Modal Analysis**: Complete workflow for extracting mode shapes, damping ratios, and modal properties from experimental acceleration data
- **Modal Response Analysis**: Time-history analysis of MDOF systems using modal superposition and Newmark's method
- **Response Spectrum Analysis (RSA)**: Spectral analysis using design spectra with SRSS modal combination
- **Interactive Visualization**: Plotly-based plotting capabilities with logarithmic scaling, hover information, and customizable styling

### Key Components
- `SDOFHarmonicVibration`: Class for analyzing SDOF systems under harmonic loading
- `ElasticResponseSpectrumBuilder`: Response spectrum generation with customizable parameters
- `ModeShapeAnalyzer`: Extracts mode shapes from acceleration data using RMS ratios and correlation analysis
- `DampingAnalyzer`: Computes damping ratios using logarithmic decrement method
- `ModalResponseAnalyzer`: Performs modal response analysis for MDOF systems
- Interactive plotting with hover text displaying system parameters (time, displacement, velocity, acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

### Basic SDOF Analysis

```python
import math
from CEE225_Dynamics.response_spectrum_builder.response_spectrum_builder import ElasticResponseSpectrumBuilder
from CEE225_Dynamics.sdof_numerical_methods import SDOFHarmonicVibration

# Define system parameters
wn = math.pi * 2  # Natural frequency [rad/s]
zeta = 0.05       # Damping ratio
k = 5             # Elastic constant [kips/in]
m = k/(wn**2)     # Mass [kips s²/in]

# Create SDOF system
sdof_system = SDOFHarmonicVibration(
    forcing_function=your_forcing_function,
    elastic_constant=k,
    damping_ratio=zeta,
    natural_frequency=wn,
    initial_displacement=0,
    initial_velocity=0,
)

# Generate response spectrum
builder = ElasticResponseSpectrumBuilder(
    sdof_system,
    normalization_value=8*12,
    period_range=(0.02, 20)
)

# Create interactive plot
builder.plot(use_log_x=True, use_log_y=True)
```

### HW11: Complete MDOF Analysis Pipeline

```bash
cd CEE225_Dynamics/ce225_specific_homework_runs/HW11
python run_hw11.py
```

This orchestrates the complete analysis workflow:
1. Mode shape extraction from experimental data
2. Damping ratio computation
3. Modal response analysis (Problem 3)
4. Response spectrum analysis (Problem 4)

## Course-Specific Homework

The `ce225_specific_homework_runs` folder contains implementations and solutions for specific homework assignments from UC Berkeley's CE225 course taught by Prof. Matthew DeJong.

### HW11: 3-Story MDOF Building Vibration Analysis

**Final Project** - Comprehensive analysis of a 3-story MDOF building structure through experimental acceleration data and earthquake response.

#### Overview

HW11 implements a complete structural dynamics analysis workflow for a 3-story building structure:

1. **Mode Shape Extraction**: Computes mode shapes from acceleration data using RMS ratios and correlation analysis
2. **Damping Analysis**: Determines damping ratios using logarithmic decrement method
3. **Problem 3 - Modal Response Analysis**: Time-history analysis of building response to ground motion using modal superposition
4. **Problem 4 - Response Spectrum Analysis**: Spectral analysis using design response spectrum with SRSS modal combination

#### Project Structure

```
HW11/
├── input_files/
│   ├── mode_1_excitation.csv          # Acceleration data for Mode 1
│   ├── mode_2_excitation.csv          # Acceleration data for Mode 2
│   ├── mode_3_excitation.csv          # Acceleration data for Mode 3
│   ├── ground_motion_excitation.csv    # 100% Loma Prieta at Palo Alto ground motion
│   └── spectrum.csv                    # Design response spectrum (PSA vs period)
├── run_hw11.py                        # Main orchestration script
├── mode_shape_analyzer.py             # ModeShapeAnalyzer class
├── damping_analyzer.py                # DampingAnalyzer class
├── modal_response_analyzer.py         # ModalResponseAnalyzer class
├── problem3_modal_response.py         # Problem 3 implementation
├── problem3_summary.py                # Problem 3 HTML summary generator
├── problem4_response_spectrum.py     # Problem 4 implementation
├── newmark_sdof.py                    # Newmark's method for SDOF systems
├── data_loader.py                     # DataLoader class
├── mode_shape_plotter.py              # ModeShapePlotter class
├── damping_plotter.py                 # DampingPlotter class
├── plotting_utils.py                  # Color palettes and styling
└── html_updater.py                    # HTML matrix updater utility
```

#### Architecture

**Core Classes:**

- **`ModeShapeAnalyzer`**: 
  - Computes mode shapes using RMS ratios and cross-correlation
  - Applies amplitude and shape-consistency filters
  - Computes statistical metrics (mean, std, CV, confidence intervals)
  - Supports mass normalization and orthogonalization

- **`DampingAnalyzer`**:
  - Detects decay portion of free vibration response
  - Extracts peaks using prominence-based detection
  - Computes damping ratios via logarithmic decrement
  - Provides statistical analysis (mean, std) across multiple cycles

- **`ModalResponseAnalyzer`**:
  - Computes participation factors (Γ_n)
  - Computes effective modal mass (M_n*) and height (h_n*)
  - Computes modal static story shears and base moments
  - Solves modal equations using Newmark's method
  - Performs modal superposition for floor responses

- **`DataLoader`**:
  - Loads acceleration data from CSV files
  - Auto-detects number of floors
  - Handles unit conversions (g to m/s²)

#### Methodology

##### Mode Shape Extraction

1. **RMS Computation**: Compute root-mean-square (RMS) amplitudes for each floor over the entire time history
2. **Sign Determination**: Use cross-correlation with reference floor (top floor) to determine phase relationships
3. **Normalization**: Normalize the mode shape so the maximum absolute value is 1.0
4. **Instantaneous Shapes**: Compute normalized mode shapes at each time step
5. **Data Filtering**: Apply amplitude and shape-consistency filters:
   - Amplitude filter: Keep only time steps where RMS amplitude exceeds threshold
   - Shape consistency filter: Keep only shapes similar to reference (correlation > threshold)
6. **Statistical Metrics**: Compute mean, standard deviation, coefficient of variation, and 95% confidence intervals

##### Damping Analysis

1. **Decay Detection**: Identify decay start using moving average envelope and derivative analysis
2. **Peak Extraction**: Extract positive and negative peaks using prominence-based detection
3. **Logarithmic Decrement**: Compute δ = (1/n) ln(A_1/A_{n+1}) for each cycle pair
4. **Damping Ratio**: Compute ζ = δ / (2π) for small damping
5. **Statistical Analysis**: Compute mean and standard deviation across multiple cycles

##### Problem 3: Modal Response Analysis

**Objective**: Determine and plot the response of the 3-story building to the 100% Loma Prieta at Palo Alto ground motion.

**Analysis Steps**:

1. **Load Ground Motion**: Load table acceleration from `ground_motion_excitation.csv`
2. **Solve Modal Equations**: For each mode n, solve:
   ```
   D̈_n + 2ζ_n ω_n Ḋ_n + ω_n² D_n = -ü_g(t)
   ```
   using Newmark's constant-average acceleration method

3. **Compute Modal Coordinates**: 
   ```
   q_n(t) = Γ_n D_n(t)
   ```

4. **Compute Floor Responses**:
   - Displacement: `u_j(t) = Σ_n φ_jn q_n(t)`
   - Velocity: `u̇_j(t) = Σ_n φ_jn Γ_n Ḋ_n(t)`
   - Acceleration: `ü_j(t) = Σ_n φ_jn Γ_n D̈_n(t) + ü_g(t)`

5. **Compute Base Forces**:
   - Base Shear: `V_b(t) = Σ_n V_{b,n}^st A_n(t)` where `A_n = ω_n² D_n`
   - Base Moment: `M_b(t) = Σ_n M_{b,n}^st A_n(t)`

**Outputs**:
- Modal displacement responses q_n(t) for each mode
- Floor displacement responses u_j(t) [inches]
- Floor acceleration responses ü_j(t) [in/s²] with measured data overlay
- Base shear V_b(t) [kips]
- Base overturning moment M_b(t) [kip-ft]
- Ground motion plot
- Comprehensive HTML summary with LaTeX report section

##### Problem 4: Response Spectrum Analysis

**Objective**: Compute spectral ordinates for the 3-DOF building using the provided design spectrum and apply SRSS to estimate peak responses.

**Analysis Steps**:

1. **Load Design Spectrum**: Load PSA (Pseudo-Spectral Acceleration) vs period for multiple damping ratios (0%, 1%, 2%, 3%, 5%)

2. **Interpolate Spectral Ordinates**:
   - Period interpolation: Linear interpolation within each damping column
   - Damping interpolation: Linear interpolation between nearest damping columns
   - For each mode n: `A_{n,0} = PSA(T_n, ζ_n)` [g]

3. **Compute Spectral Displacement**:
   ```
   D_{n,0} = (T_n / (2π))² A_{n,0} g
   ```

4. **Compute Modal Static Responses**:
   - Floor displacements: `u_{j,n}^st = Γ_n φ_jn D_{n,0}`
   - Base shear: `V_{b,n}^st = Γ_n Σ_j m_j φ_jn`

5. **SRSS Combination**:
   - Floor displacement: `u_j = √(Σ_n (u_{j,n}^st)²)`
   - Base shear: `V_b = √(Σ_n (V_{b,n}^st A_{n,0} g)²)`

**Outputs**:
- Response spectrum plot with modal points marked
- Table of modal spectral ordinates (T_n, ζ_n, A_{n,0}, D_{n,0})
- Table of SRSS floor displacements
- Table of SRSS base shear
- Comprehensive HTML summary with computation steps

#### Running the Analysis

**Prerequisites**:
```bash
pip install numpy pandas plotly scipy
```

**Execution**:
```bash
cd CEE225_Dynamics/ce225_specific_homework_runs/HW11
python run_hw11.py
```

The script orchestrates the complete workflow:
1. Loads acceleration data from all three mode excitation CSV files
2. Computes mode shapes and statistics for each mode
3. Computes damping ratios for each mode
4. Runs Problem 3: Modal response analysis
5. Runs Problem 4: Response spectrum analysis
6. Generates interactive Plotly visualizations
7. Creates comprehensive HTML summary pages

**Output Location**: `CEE225_Dynamics/highlighted_htmls/`

#### Output Files

**Mode Shape Analysis**:
- `mode_1_analysis.html` - Detailed analysis for Mode 1 (filtered)
- `mode_2_analysis.html` - Detailed analysis for Mode 2 (filtered)
- `mode_3_analysis.html` - Detailed analysis for Mode 3 (filtered)
- `mode_*_analysis_raw.html` - Raw (unfiltered) analyses for sensitivity
- `all_mode_shapes.html` - Combined comparison of all modes
- `mode_*_damping.html` - Damping analysis visualizations

**Problem 3 Outputs**:
- `problem3_modal_displacements.html` - Modal displacement responses q_n(t)
- `problem3_floor_displacements.html` - Floor displacement responses u_j(t)
- `problem3_floor_accelerations.html` - Floor acceleration responses ü_j(t)
- `problem3_base_shear.html` - Base shear V_b(t)
- `problem3_base_moment.html` - Base overturning moment M_b(t)
- `problem3_ground_motion.html` - Ground motion plot
- `problem3_summary.html` - Comprehensive summary page with LaTeX report

**Problem 4 Outputs**:
- `problem4_spectrum.html` - Response spectrum with modal points
- `problem4_summary.html` - Comprehensive summary with tables and computation steps

#### Key Features

- **Mass-Normalized Mode Shapes**: Mode shapes are mass-normalized so that `Φ^T M Φ = I` for use in modal analysis
- **Unit Conversions**: Automatic handling of unit conversions (g ↔ m/s², m ↔ inches, N ↔ kips, N·m ↔ kip-ft)
- **Measured Data Comparison**: Problem 3 overlays computed responses with measured floor accelerations and displacements
- **Interactive Plots**: All plots include hover information showing exact values, mode contributions, and computation details
- **Consistent Color Scheme**: Professional color palette (Berkeley Blue, Crimson Red, Dark Green, Black) with thinner lines for modal contributions
- **Comprehensive Documentation**: LaTeX report sections explaining methodology and computations

#### System Properties

- **Mass Matrix** (diagonal): `M = diag([1180, 1180, 910])` [kg]
- **Natural Frequencies**: `f_1 = 2.00 Hz`, `f_2 = 7.20 Hz`, `f_3 = 13.75 Hz`
- **Floor Heights**: `[2.0828, 4.1656, 6.2484]` [m] (consistent with overturning moment lever arms)
- **Damping Ratios**: Extracted from experimental data using logarithmic decrement (typically 1-2%)

#### Web Integration

The generated HTML files are automatically deployed via GitHub Actions to the portfolio website. Access from:
- Main CEE225 page: `webpage/cee225-dynamics.html`
- HW11 outputs: `highlighted_htmls/problem3_summary.html`, `highlighted_htmls/problem4_summary.html`

## Dependencies

- **NumPy**: Array programming and numerical computations
- **Pandas**: Data loading and manipulation
- **Plotly**: Interactive data visualization
- **SciPy**: Signal processing (peak detection, correlation)
- **Python 3.x** with standard libraries (math, functools, pathlib)

## References

[1] M. DeJong, "Lectures for cee225: Structural dynamics." Course lectures, University of California, Berkeley, 2025. Fall 2025 semester SEMM MS Program.

[2] A. K. Chopra, *Dynamics of Structures: Theory and Applications to Earthquake Engineering*. Boston: Pearson Education, 4th, global edition ed., 2014.

[3] M. DeJong, F. Filippou, S. Govindjee, A. D. Kiureghian, K. Mosalam, J. Moehle, and E. Opabola, "Semm graduate program primer: 2025," SEMM Reports Series UCB/SEMM-2025/04, University of California, Berkeley, July 2025.

[4] C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[5] P. T. Inc., "Plotly: Collaborative data science." https://plotly.com/python/, 2015. Accessed: 2025-10-04

## License

This project is developed for educational purposes as part of the UC Berkeley CE225 Structural Dynamics course.
