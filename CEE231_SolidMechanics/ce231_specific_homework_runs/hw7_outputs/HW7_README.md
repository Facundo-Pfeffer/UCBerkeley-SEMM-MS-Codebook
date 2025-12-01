# CEE 231 - Homework 7: Convergence Analysis

## Overview
This assignment analyzes the convergence behavior of the torsional stiffness series as a function of the number of terms.

## Important Distinction: N vs n

**This is crucial to understand:**

- **N** = Number of terms in the series (1st term, 2nd term, 3rd term, ...)
- **n** = The actual index values used in the series (only odd values: 1, 3, 5, 7, ...)

**Relationship:** `n_max = 2N - 1`

### Examples:
- **N = 50 terms** → evaluates n = 1, 3, 5, 7, ..., 97, **99**
- **N = 100 terms** → evaluates n = 1, 3, 5, 7, ..., 197, **199**
- **N = 500 terms** → evaluates n = 1, 3, 5, 7, ..., 997, **999**

## Files

### Python Script
- **`hw07.py`** - Main convergence analysis script
  - Uses clear variable names: `num_terms`, `n_odd`, `term_numbers`
  - Hover tooltips show both term number and corresponding n value
  - Console output clearly states maximum n value reached
  - Generates interactive HTML plot

### LaTeX Explanations
- **`HW7_plot_explanation.tex`** - Comprehensive academic explanation
  - Full section structure with detailed analysis
  - Mathematical justification
  - Explicitly explains N vs n distinction
  
- **`HW7_plot_explanation_short.tex`** - Concise homework-friendly version
  - Compact paragraph format
  - Key observations highlighted
  - Suitable for assignment inclusion

### Output
- **`CEE231_HW7_Convergence_Analysis.html`** - Interactive Plotly visualization
  - Upper panel: Torsional stiffness convergence
  - Lower panel: Relative change (log scale)
  - Hover to see term number N and corresponding odd index n

## Key Results (N = 50 terms, n_max = 99)

- **k_T ≈ 374.95 kN·mm²**
- **Relative error < 0.01%** compared to fully converged value
- **Relative change < 10⁻⁴%** at 50th term

## Running the Code

```bash
# Make sure numpy and plotly are installed in your virtual environment
pip install numpy plotly

# Run the analysis
python hw07.py
```

## Plot Features

The generated plot includes:
1. **Convergence curve** showing k_T(N) approaching asymptotic value
2. **Dashed line** indicating fully converged value
3. **Interactive hover** displaying:
   - Term number (N)
   - Odd index value (n)
   - Torsional stiffness value
4. **Log-scale panel** showing exponential decay of relative change

## Axis Labels

- **X-axis:** "Number of Terms (N)" - counts 1, 2, 3, ..., 500
- **Hover data:** Shows corresponding odd n values (1, 3, 5, ..., 999)
- This makes it clear that N=50 means the 50th term with index n=99


