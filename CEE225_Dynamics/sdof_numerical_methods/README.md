# Single Degree of Freedom (SDOF) Dynamic Response Analysis

## Overview

This project provides a comprehensive numerical analysis toolkit for single degree of freedom systems under harmonic excitation followed by free vibration. The implementation compares analytical solutions with numerical methods to evaluate accuracy and stability.

## Problem Description

The system analyzes a damped single degree of freedom oscillator with:
- **Natural frequency**: ωₙ = 2π rad/s
- **Damping ratio**: ζ = 0.05 (baseline)
- **Elastic constant**: k = 5 kips/in
- **Forcing function**: p(t) = 8sin(πt/0.4) for t ≤ 1.2s, then 0

## Architecture

### Core Classes

**`SolutionPoint`**
- Container for solution data at each time step
- Stores displacement, velocity, acceleration, and metadata

**`SDOFHarmonicVibration`**
- Analytical solution implementation
- Provides exact response for validation

**`AbsSDOFNumericMethod`**
- Abstract base class for numerical methods
- Handles common functionality and error tracking

**`CentralDifferenceMethod`**
- Explicit numerical integration scheme
- Conditionally stable (requires small time steps)

**`AverageAccelerationMethod`**
- Implicit numerical integration scheme
- Unconditionally stable

## Analysis Results

### Problem A: Method Comparison (ζ = 0.05, Δt = 0.1s)
Compares numerical methods against exact solution showing:
- Central Difference: Excellent accuracy for stable time steps
- Average Acceleration: Good accuracy with unconditional stability
- Static Solution: Baseline comparison

### Problem B: Damping Ratio Study
Analyzes system behavior for varying damping ratios:
- ζ = 0.01: Lightly damped response
- ζ = 0.10: Moderately damped response  
- ζ = 0.25: Heavily damped response

### Problem C: Time Step Sensitivity
Investigates numerical stability:
- **Δt = 0.35s**: Central Difference becomes unstable
- **Δt = 0.20s**: Marginally stable behavior
- **Δt = 0.05s**: Stable and accurate results

## Features

- **Interactive Plotting**: Professional Plotly visualizations with hover details
- **Error Tracking**: Accumulated absolute error calculations
- **Modular Design**: Easy to extend with new numerical methods
- **Comprehensive Analysis**: Multiple damping ratios and time step studies
- **Professional Output**: Timestamped HTML reports

## Key Insights

1. **Stability**: Central Difference Method requires Δt ≤ Δt_critical for stability
2. **Accuracy**: Both methods show excellent agreement with analytical solution for appropriate time steps
3. **Damping Effects**: Higher damping ratios reduce oscillation amplitude and improve numerical stability
4. **Method Selection**: Average Acceleration preferred for larger time steps due to unconditional stability

## Usage

```python
# Define system parameters
params = dict(
    time_step=0.1,
    time_stop=4,
    elastic_constant=5,
    damping_ratio=0.05,
    natural_frequency=2*math.pi,
    initial_displacement=0,
    initial_velocity=0,
)

# Create analytical solution
exact_solution = SDOFHarmonicVibration(**params)

# Compare numerical methods
central_diff = CentralDifferenceMethod(**params, exact_solution=exact_solution.eom)
avg_accel = AverageAccelerationMethod(**params, exact_solution=exact_solution.eom)

# Generate interactive plots
plot_displacement_vs_time(solutions={
    "Central Difference": central_diff.solution_set,
    "Average Acceleration": avg_accel.solution_set,
    "Exact Solution": exact_solution.get_cloud_points(0.1, 4)
})
```

## Mathematical Foundation

### Equation of Motion
The governing differential equation for the SDOF system:

```
mü + cu̇ + ku = p(t)
```

Where:
- m: mass
- c: damping coefficient
- k: stiffness
- u: displacement
- p(t): external forcing function

### Numerical Methods

**Central Difference Method (Explicit)**
- Time stepping: uᵢ₊₁ = (p̂ᵢ₊₁)/k̂
- Stability condition: Δt ≤ Δt_critical
- Best for: Small time steps, linear systems

**Average Acceleration Method (Implicit)**
- Assumes constant average acceleration over time step
- Unconditionally stable for linear systems
- Best for: Larger time steps, nonlinear systems

## File Structure

```
HW5 CE 220 - Facundo Pfeffer/
├── HW5.py                    # Main analysis script
├── plotly_generator.py       # Visualization utilities
├── README.md                 # This file
└── [output_files]/          # Generated HTML plots
    ├── Dynamic_Response_zeta_*.html
    ├── *_Comparison_Delta_t.html
    └── u_vs_t.html
```

## Requirements

- `numpy`: Numerical computations
- `plotly`: Interactive visualization
- `functools`: LRU caching for performance

## Educational Value

This implementation serves as an excellent educational tool for:
- Understanding numerical integration methods
- Analyzing stability and accuracy trade-offs
- Comparing explicit vs implicit schemes
- Studying dynamic response of engineering systems
- Visualizing the effects of damping on system response

## Results Summary

The analysis demonstrates that:
1. Both numerical methods converge to the analytical solution when properly implemented
2. Time step selection is critical for the Central Difference Method stability
3. The Average Acceleration Method provides robust performance across different time steps
4. Damping ratio significantly affects both the physical response and numerical stability

## Author

Facundo L. Pfeffer  
Course: CEE 225 - Dynamics  
Date: Fall 2025
