# Facundo L. Pfeffer - SEMM Graduate Portfolio

A professional portfolio website showcasing graduate work in Structural Engineering, Mechanics and Materials (SEMM) at UC Berkeley.

## Overview

This portfolio demonstrates computational analysis work in structural dynamics, solid mechanics, and structural analysis, featuring interactive visualizations and numerical methods implementations.

## Features

### 🎓 Academic Focus
- **CEE225 - Structural Dynamics**: SDOF systems, numerical integration, response spectrum analysis
- **CEE220 - Structural Analysis**: Truss analysis, equilibrium matrix methods
- **CEE231 - Solid Mechanics**: Material properties, directional Young's modulus analysis

### 💻 Technical Implementation
- **Interactive Visualizations**: Plotly-based dynamic response plots
- **Numerical Methods**: Central Difference and Average Acceleration methods
- **Error Analysis**: Stability studies and convergence analysis
- **Modular Architecture**: Object-oriented Python implementations

### 🎨 Design
- **UC Berkeley Branding**: Official colors (#003262 Berkeley Blue, #FDB515 California Gold)
- **Responsive Layout**: Mobile-friendly design using HTML5UP Phantom template
- **Professional Navigation**: Easy access to all projects and information

## Website Structure

```
html5up-phantom/
├── index.html                 # Main portfolio homepage
├── cee231-mechanics.html      # CE231 course page with interactive tiles
├── cee225-dynamics.html      # Structural Dynamics overview
├── sdof-numerical-methods.html # SDOF methods detailed page
├── about.html                # Personal and academic information
├── contact.html              # Contact information and collaboration
├── assets/                   # CSS, JS, and image resources
│   └── images/
│       ├── portraits/        # Headshots & people images
│       └── projects/         # Thumbnails and project visuals
└── README.md                 # This file
```

## Key Projects Showcased

### 1. SDOF Numerical Methods
- **Implementation**: Central Difference and Average Acceleration methods
- **Features**: Error tracking, stability analysis, interactive visualization
- **Applications**: Harmonic excitation, free vibration analysis

### 2. Response Spectrum Builder
- **Purpose**: Earthquake engineering applications
- **Capabilities**: Elastic response spectrum generation
- **Visualization**: Interactive plotting with logarithmic scaling

### 3. Structural Analysis Tools
- **Truss Solver**: Statically determinate truss analysis
- **Matrix Methods**: Equilibrium matrix construction
- **Verification**: Global equilibrium checks

### 4. Solid Mechanics Analysis
- **Material Properties**: Directional Young's modulus visualization
- **Compliance Matrix**: 3D material behavior analysis
- **Interactive Plots**: 3D point cloud visualizations

## Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python (NumPy, Plotly, SciPy)
- **Visualization**: Plotly interactive charts
- **Version Control**: Git/GitHub
- **Template**: HTML5UP Phantom (customized)

## Getting Started

### Viewing the Portfolio
1. Open `index.html` in a web browser
2. Navigate through different sections using the menu
3. Click on project tiles to view detailed information

### Running the Code
1. Navigate to the respective course folders in the main repository
2. Install required dependencies from `requirements.txt`
3. Run Python scripts to generate interactive visualizations

### Dependencies
```bash
# For CEE225 Dynamics
pip install numpy plotly

# For CEE220 Analysis  
pip install numpy plotly

# For CEE231 Mechanics
pip install numpy plotly
```

## Customization

### Adding New Projects
1. Create new HTML page following the template structure
2. Add navigation links to main menu
3. Update project tiles on homepage
4. Include technical details and code snippets

### Modifying Styling
- Edit `assets/css/main.css` for color scheme changes
- UC Berkeley colors are already implemented
- Maintain responsive design principles

## Contact Information

- **Email**: facundo.pfeffer@berkeley.edu
- **Institution**: University of California, Berkeley
- **Program**: MS in Structural Engineering, Mechanics and Materials
- **GitHub**: github.com/facundopfeffer

## License

This portfolio is developed for educational purposes as part of the UC Berkeley SEMM program. The HTML5UP Phantom template is used under the CCA 3.0 license.

## Acknowledgments

- **UC Berkeley SEMM Program**: Faculty and staff support
- **HTML5UP**: Phantom template design
- **Open Source Community**: Python scientific computing libraries

---

*Last updated: Fall 2025*
