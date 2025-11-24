#!/usr/bin/env python3
"""
Static HTML Dashboard Generator for 1D Power-law Plasticity
Converts the interactive Dash app to a static HTML file for GitHub Pages deployment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_templates import (
    UCBerkeleyColors as Colors,
    get_axis_style,
    save_figure,
)

# Import functions from the homework script
homework_scripts_path = os.path.join(os.path.dirname(__file__), '..', 'homework_scripts')
sys.path.insert(0, homework_scripts_path)

# Import the functions we need
import importlib.util
spec = importlib.util.spec_from_file_location("hw11", os.path.join(homework_scripts_path, "11HW.py"))
hw11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw11)

strain_history_vector = hw11.strain_history_vector
integrate_power_law_backward_euler = hw11.integrate_power_law_backward_euler

# Default parameters (matching the Dash app defaults)
E_GPa = 200.0
Y0_MPa = 200.0
K_MPa = 300.0
n_hard = 0.3
a = 0.05
a_prime = 0.01
tau = 10.0
n_steps_per_tau = 200

# Convert to SI units
E = E_GPa * 1e9
Y0 = Y0_MPa * 1e6
K = K_MPa * 1e6

# Compute the response
print("Computing plasticity response...")
t, eps_n, sigma_n, eps_p_n, epsp_bar_n = integrate_power_law_backward_euler(
    E=E,
    Y0=Y0,
    K=K,
    n_hard=n_hard,
    a=a,
    a_prime=a_prime,
    tau=tau,
    n_steps_per_tau=int(n_steps_per_tau),
)

# Create subplot figure with 3 plots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        'Stress–Strain Response: σ(ε)',
        'Equivalent Plastic Strain vs Time: ε̄ᵖ(t)',
        'Plastic Strain vs Time: εᵖ(t)'
    ),
    vertical_spacing=0.12,
    row_heights=[0.4, 0.3, 0.3]
)

# Plot 1: Stress-Strain
fig.add_trace(
    go.Scatter(
        x=eps_n,
        y=sigma_n / 1e6,  # MPa
        mode="lines",
        line=dict(width=3, color=Colors.BERKELEY_BLUE),
        name="σ(ε)",
        hovertemplate="ε: %{x:.4f}<br>σ: %{y:.2f} MPa<extra></extra>",
    ),
    row=1, col=1
)

# Plot 2: Equivalent plastic strain
fig.add_trace(
    go.Scatter(
        x=t,
        y=epsp_bar_n,
        mode="lines",
        line=dict(width=3, color=Colors.FOUNDERS_ROCK),
        name="ε̄ᵖ(t)",
        hovertemplate="t: %{x:.2f} s<br>ε̄ᵖ: %{y:.4f}<extra></extra>",
    ),
    row=2, col=1
)

# Plot 3: Plastic strain
fig.add_trace(
    go.Scatter(
        x=t,
        y=eps_p_n,
        mode="lines",
        line=dict(width=3, color=Colors.CALIFORNIA_GOLD),
        name="εᵖ(t)",
        hovertemplate="t: %{x:.2f} s<br>εᵖ: %{y:.4f}<extra></extra>",
    ),
    row=3, col=1
)

# Apply axis styling
axis_style = get_axis_style()

# Update axes
fig.update_xaxes(title_text="<b>Strain ε [–]</b>", row=1, col=1, **axis_style)
fig.update_yaxes(title_text="<b>Stress σ [MPa]</b>", row=1, col=1, **axis_style)

fig.update_xaxes(title_text="<b>Time t [s]</b>", row=2, col=1, **axis_style)
fig.update_yaxes(title_text="<b>Equivalent plastic strain ε̄ᵖ [–]</b>", row=2, col=1, **axis_style)

fig.update_xaxes(title_text="<b>Time t [s]</b>", row=3, col=1, **axis_style)
fig.update_yaxes(title_text="<b>Plastic strain εᵖ [–]</b>", row=3, col=1, **axis_style)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>1D Power-law Plasticity – Backward Euler Integration</b><br>'
             '<sub>Rate-independent uniaxial plasticity with power-law isotropic hardening</sub>',
        font=dict(size=22, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center',
        y=0.98,
    ),
    height=1200,
    plot_bgcolor=Colors.BG_LIGHT,
    paper_bgcolor=Colors.BG_WHITE,
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(t=120, b=60, l=80, r=40),
    showlegend=False,
    hovermode='closest',
)

# Format subplot titles
for i, annotation in enumerate(fig['layout']['annotations']):
    if i < 3:  # Only format the subplot titles
        annotation['font'] = dict(size=16, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif')
        annotation['x'] = 0.5
        annotation['xanchor'] = 'center'

# Add parameter information as annotation
param_text = (
    f"<b>Material Parameters:</b> E = {E_GPa:.0f} GPa, Y₀ = {Y0_MPa:.0f} MPa, "
    f"K = {K_MPa:.0f} MPa, n = {n_hard:.2f}<br>"
    f"<b>Loading Parameters:</b> a = {a:.3f}, a′ = {a_prime:.4f}, "
    f"τ = {tau:.0f} s, Resolution = {n_steps_per_tau} steps/τ"
)

fig.add_annotation(
    text=param_text,
    xref="paper", yref="paper",
    x=0.5, y=-0.02,
    xanchor="center", yanchor="top",
    showarrow=False,
    font=dict(size=11, color=Colors.TEXT_LIGHT, family='Arial, sans-serif'),
    align="center",
)

# Save to highlighted_htmls (web deployment folder)
output_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    '..', 
    'highlighted_htmls', 
    'plasticity_backward_euler.html'
)
save_figure(fig, output_path)
print(f"\n[SUCCESS] Static dashboard generated: {output_path}")
print(f"  Parameters: E={E_GPa:.0f} GPa, Y₀={Y0_MPa:.0f} MPa, K={K_MPa:.0f} MPa, n={n_hard:.2f}")
print(f"  Loading: a={a:.3f}, a′={a_prime:.4f}, τ={tau:.0f} s")

