# Facundo L. Pfeffer - SEMM Graduate Portfolio

A professional portfolio website showcasing graduate work in Structural Engineering, Mechanics and Materials (SEMM) at UC Berkeley.

## Overview

This portfolio demonstrates computational analysis work in structural dynamics, solid mechanics, and structural analysis, featuring interactive visualizations and numerical methods implementations.

## Features

### Academic Focus
- **CEE225 - Structural Dynamics**: SDOF systems, numerical integration, response spectrum analysis
- **CEE220 - Structural Analysis**: Truss analysis, equilibrium matrix methods
- **CEE231 - Solid Mechanics**: Material properties, directional Young's modulus analysis

### Technical Implementation
- **Interactive Visualizations**: Plotly-based dynamic response plots
- **Numerical Methods**: Central Difference and Average Acceleration methods
- **Error Analysis**: Stability studies and convergence analysis
- **Modular Architecture**: Object-oriented Python implementations

### Design
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

## Sections and Content

### CE231 – Solid Mechanics
- Landing page: `cee231-mechanics.html`
- Interactive results (Directional Young's Modulus) are published from the repository folder `CEE231_SolidMechanics/highlighted_htmls/` and are accessible on the live site at:
  - `/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_A.html`
  - `/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_B.html`
  - `/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_C.html`

### CE225 – Structural Dynamics
- Landing page: `cee225-dynamics.html`
- Homework viewer embeds the PDF published from `CEE225_Dynamics/highlighted_pdfs/` and is accessible at:
  - `/CEE225_Dynamics/highlighted_pdfs/CE%20225%20HW2%20-%20Facundo%20L.%20Pfeffer.pdf`

## How deployment works (important)

This repository uses GitHub Actions to build and deploy the website to GitHub Pages.

- The website source lives under `webpage/`.
- During deployment, the workflow also scans the repository for folders named `highlighted_htmls` and `highlighted_pdfs` anywhere in the repo and mirrors them into the published site at the same relative paths.
- This lets you keep artifacts (HTML exports, PDFs) close to the course code while still serving them on the live site.

Add new artifacts:
1. Create a folder named `highlighted_htmls` or `highlighted_pdfs` under any course directory.
2. Drop your HTML/PDF files there.
3. Push to the default branch (or manually run the Pages workflow).
4. Link to them from pages in `webpage/` using the repo-relative path, e.g. `/MyCourse/highlighted_pdfs/file.pdf`.

## Local preview

You can open `webpage/index.html` directly in a browser for a quick preview. For a closer production match, use any static server:

```bash
python -m http.server --directory webpage 5500
# then open http://localhost:5500/
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
