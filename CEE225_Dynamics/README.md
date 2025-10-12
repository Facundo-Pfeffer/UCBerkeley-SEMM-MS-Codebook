
# CE225 Dynamics - Structural Dynamics Analysis Package

A Python package for structural dynamics analysis, featuring numerical methods for Single Degree of Freedom (SDOF) systems and earthquake response spectrum generation. Developed for UC Berkeley's CE225 Structural Dynamics course.

## Features

### Core Functionality
- **Numerical Methods**: Implementation of various numerical integration methods for solving equations of motion (EOM) of SDOF systems
- **Response Spectrum Analysis**: Tools for generating and analyzing elastic response spectra for earthquake engineering applications
- **Interactive Visualization**: Plotly-based plotting capabilities with logarithmic scaling and customizable grid options

### Key Components
- `SDOFHarmonicVibration`: Class for analyzing SDOF systems under harmonic loading
- `ElasticResponseSpectrumBuilder`: Response spectrum generation with customizable parameters
- Interactive plotting with hover text displaying system parameters (time, displacement, velocity, acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

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

## Course-Specific Homework

The `ce225_specific_homework_runs` folder contains implementations and solutions for specific homework assignments from UC Berkeley's CE225 course taught by Prof. Matthew DeJong.

## Dependencies

- NumPy: Array programming and numerical computations
- Plotly: Interactive data visualization
- Python 3.x with standard libraries (math, functools)

## References

[1] M. DeJong, "Lectures for cee225: Structural dynamics." Course lectures, University of California, Berkeley, 2025. Fall 2025 semester SEMM MS Program.

[2] A. K. Chopra, *Dynamics of Structures: Theory and Applications to Earthquake Engineering*. Boston: Pearson Education, 4th, global edition ed., 2014.

[3] M. DeJong, F. Filippou, S. Govindjee, A. D. Kiureghian, K. Mosalam, J. Moehle, and E. Opabola, "Semm graduate program primer: 2025," SEMM Reports Series UCB/SEMM-2025/04, University of California, Berkeley, July 2025.

[4] C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[5] P. T. Inc., "Plotly: Collaborative data science." https://plotly.com/python/, 2015. Accessed: 2025-10-04

## License

This project is developed for educational purposes as part of the UC Berkeley CE225 Structural Dynamics course.
