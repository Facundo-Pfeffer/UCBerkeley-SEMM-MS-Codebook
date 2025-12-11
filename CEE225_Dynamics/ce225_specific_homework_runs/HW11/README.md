# HW11: 3-Story MDOF Building Vibration Analysis

Final Project for CEE225 - Structural Dynamics

## Overview

This project analyzes the dynamic behavior of a 3-story MDOF (Multi-Degree-of-Freedom) building structure through experimental acceleration data. The analysis extracts mode shapes, computes statistical confidence metrics, and provides interactive visualizations.

## Project Structure

```
HW11/
├── input_files/
│   ├── mode_1_excitation.csv    # Acceleration data for Mode 1
│   ├── mode_2_excitation.csv    # Acceleration data for Mode 2
│   └── mode_3_excitation.csv    # Acceleration data for Mode 3
├── run_hw11.py                  # Main analysis script (homework runner)
├── building_frame.py             # BuildingFrame class for visualization
├── data_loader.py                # DataLoader class for CSV data loading
├── mode_shape_analyzer.py        # ModeShapeAnalyzer class for computations
├── mode_shape_plotter.py          # ModeShapePlotter class for plotting
├── plotting_utils.py             # Plotting utilities (colors, styles)
├── html_updater.py               # HTML matrix updater utility
└── README.md                     # This file
```

## Data Format

The CSV files contain acceleration measurements with the following columns:
- `time`: Time in seconds
- `L1AccX_filtered`: Filtered acceleration at Floor 1 (Level 1)
- `L2AccX_filtered`: Filtered acceleration at Floor 2 (Level 2)
- `L3AccX_filtered`: Filtered acceleration at Floor 3 (Level 3)

## Architecture

The project uses an object-oriented design for scalability and maintainability:

- **`BuildingFrame`**: Handles building visualization for any number of stories
- **`DataLoader`**: Loads acceleration data from CSV files (auto-detects number of floors)
- **`ModeShapeAnalyzer`**: Computes mode shapes using RMS ratios and correlation analysis
- **`ModeShapePlotter`**: Creates interactive Plotly visualizations
- **`run_hw11.py`**: Main script that orchestrates the analysis workflow

## Methodology

### Mode Shape Extraction

1. **RMS Computation**: Compute root-mean-square (RMS) amplitudes for each floor over the entire time history
2. **Sign Determination**: Use cross-correlation with reference floor to determine phase relationships
3. **Normalization**: Normalize the mode shape so the maximum absolute value is 1.0
4. **Instantaneous Shapes**: Compute normalized mode shapes at each time step
5. **Data Filtering**: Apply amplitude and shape-consistency filters to improve statistical reliability
6. **Statistical Metrics**: Compute mean, standard deviation, coefficient of variation, and 95% confidence intervals

### Statistical Validation

The analysis computes statistics over instantaneous mode shapes:
- **Mean mode shape**: Average across all filtered instantaneous shapes
- **Standard deviation**: Spread of mode shape values
- **Coefficient of variation**: Relative uncertainty (CV = σ/μ)
- **95% Confidence intervals**: Statistical bounds assuming normal distribution

## Running the Analysis

### Prerequisites

Install required Python packages:
```bash
pip install numpy pandas plotly
```

### Execution

Run the analysis script:
```bash
python run_hw11.py
```

The script will:
1. Load acceleration data from all three CSV files
2. Compute mode shapes and statistics for each mode
3. Generate interactive Plotly visualizations
4. Save HTML files to `../../highlighted_htmls/`:
   - `mode_1_analysis.html` - Detailed analysis for Mode 1
   - `mode_2_analysis.html` - Detailed analysis for Mode 2
   - `mode_3_analysis.html` - Detailed analysis for Mode 3
   - `all_mode_shapes.html` - Combined comparison of all modes

## Output Visualizations

Each mode analysis includes three main plots:

1. **Building Frame**: Visual representation of the building structure with mode shape displacement (fixed-fixed column interpolation)
2. **Gaussian Distributions**: Probability density functions for each floor's mode shape component
3. **Time Variation**: Grouped time-averaged mode shape values compared to the reference mode shape

The analysis generates both filtered (cleaned) and raw (unfiltered) visualizations to demonstrate the impact of data cleaning.

## Web Integration

The generated HTML files are automatically deployed via GitHub Actions to:
- `CEE225_Dynamics/highlighted_htmls/` in the deployed site

Navigation pages:
- **Menu**: `final_project_menu.html` - Main navigation for the project
- **Step 1**: `step1_mode_shapes.html` - Individual mode shape analyses
- **Step 3**: `step3_combined.html` - Combined mode shape comparison

Access from the main CEE225 page: [CEE225 Dynamics](../webpage/cee225-dynamics.html)

## Interpretation

### Mode Shapes

- **Mode 1 (Fundamental)**: Typically all floors move in the same direction, lowest natural frequency
- **Mode 2 (Second Mode)**: Often shows one floor moving opposite to others, creating a node
- **Mode 3 (Third Mode)**: Higher mode with more complex deformation pattern, multiple nodes

### Statistical Metrics

- **Low CV (< 10%)**: High confidence in the mode shape value
- **Medium CV (10-20%)**: Moderate confidence, some uncertainty
- **High CV (> 20%)**: Lower confidence, significant uncertainty

### Confidence Intervals

The 95% confidence intervals indicate the range where the true mode shape value is expected to lie with 95% probability. Narrower intervals indicate higher confidence.

## Author

**Facundo L. Pfeffer**  
MS Student, Structural Engineering, Mechanics and Materials  
University of California, Berkeley

## Course

CEE225 - Structural Dynamics  
Fall 2025

