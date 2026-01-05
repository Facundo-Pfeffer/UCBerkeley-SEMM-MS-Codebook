#!/usr/bin/env python3
"""
SLS Sinusoidal Response Dashboard Generator
Uses shared plotly_templates for consistent styling
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
    get_slider_style,
    get_table_header_style,
    get_table_cells_style,
    format_hover_template,
    save_figure_with_template
)

# Material parameters (fixed)
E_re = 1.0      # MPa - equilibrium modulus
E_rg = 2.0      # MPa - glassy component
tau_R = 1.0     # s - relaxation time
omega = 2*np.pi # rad/s - frequency

# Strain amplitude values - linearly spaced for very smooth slider (50 values)
eps0_vals = np.linspace(0.002, 0.05, 50)
eps0_default = 0.01

# Time range
t_max = 4.0
npts = 1000

def compute_sls_response(E_re, E_rg, tau_R, eps0, omega, t_max, npts):
    """Compute SLS response to sinusoidal strain."""
    t = np.linspace(0, t_max, npts)
    eps = eps0 * np.sin(omega * t)
    
    # Stress coefficients
    denom = (1/tau_R**2 + omega**2)
    A = eps0 * (E_re + E_rg * omega**2 / denom)
    B = eps0 * E_rg * (omega/tau_R) / denom
    C = -B
    
    sigma = A * np.sin(omega * t) + B * np.cos(omega * t) + C * np.exp(-t/tau_R)
    return t, eps, sigma, A, B, C

# Create subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Stress Response σ(t)',
        'Strain Input ε(t)',
        'Stress-Strain Hysteresis',
        'Characteristic Values'
    ),
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "table"}]],
    horizontal_spacing=0.12,
    vertical_spacing=0.15
)

# Find index closest to default
default_idx = int(np.argmin(np.abs(eps0_vals - eps0_default)))

# Generate all data
for i, eps0 in enumerate(eps0_vals):
    t, eps, sigma, A, B, C = compute_sls_response(E_re, E_rg, tau_R, eps0, omega, t_max, npts)
    visible = (i == default_idx)
    
    # Stress vs time
    fig.add_trace(go.Scatter(
        x=t, y=sigma,
        mode='lines',
        line=dict(color=Colors.BERKELEY_BLUE, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('t', 'σ', '.3f', '.4e') + ' MPa'
    ), row=1, col=1)
    
    # Strain vs time
    fig.add_trace(go.Scatter(
        x=t, y=eps,
        mode='lines',
        line=dict(color=Colors.CALIFORNIA_GOLD, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('t', 'ε', '.3f', '.4f')
    ), row=1, col=2)
    
    # Hysteresis loop
    fig.add_trace(go.Scatter(
        x=eps, y=sigma,
        mode='lines',
        line=dict(color=Colors.MEDALIST, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('ε', 'σ', '.4f', '.4e') + ' MPa'
    ), row=2, col=1)
    
    # Table
    sigma_max = np.max(np.abs(sigma))
    period = 2*np.pi/omega
    freq = omega/(2*np.pi)
    loss_angle = np.arctan(B/A) * 180/np.pi
    
    fig.add_trace(go.Table(
        header=get_table_header_style(),
        cells=get_table_cells_style(),
        visible=visible
    ), row=2, col=2)
    
    fig.data[-1].header.values = ['<b>Parameter</b>', '<b>Value</b>']
    fig.data[-1].cells.values = [
        ['ε₀', 'ω', 'Frequency', 'Period', 'E<sub>re</sub>', 'E<sub>rg</sub>', 
         'τ<sub>R</sub>', 'σ<sub>max</sub>', 'A', 'B', 'C', 'Loss Angle'],
        [f'{eps0:.3f}', f'{omega:.3f} rad/s', f'{freq:.3f} Hz', f'{period:.3f} s',
         f'{E_re:.1f} MPa', f'{E_rg:.1f} MPa', f'{tau_R:.1f} s',
         f'{sigma_max:.4e} MPa', f'{A:.4e} MPa', f'{B:.4e} MPa', 
         f'{C:.4e} MPa', f'{loss_angle:.2f}°']
    ]

# Apply axis styling
axis_style = get_axis_style()
fig.update_xaxes(title_text='<b>t [s]</b>', row=1, col=1, **axis_style)
fig.update_yaxes(title_text='<b>σ(t) [MPa]</b>', row=1, col=1, **axis_style)
fig.update_xaxes(title_text='<b>t [s]</b>', row=1, col=2, **axis_style)
fig.update_yaxes(title_text='<b>ε(t) [-]</b>', row=1, col=2, **axis_style)
fig.update_xaxes(title_text='<b>ε(t) [-]</b>', row=2, col=1, **axis_style)
fig.update_yaxes(title_text='<b>σ(t) [MPa]</b>', row=2, col=1, **axis_style)

# Create slider
steps = []
for i, eps0 in enumerate(eps0_vals):
    visibility = []
    for j in range(len(eps0_vals)):
        visibility.extend([True] * 4 if j == i else [False] * 4)
    
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}],
        label=f"{eps0:.3f}"
    ))

slider = get_slider_style(
    steps=steps,
    active_index=int(np.argmin(np.abs(eps0_vals - eps0_default))),
    prefix="<b>Strain Amplitude ε₀ = </b>"
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Standard Linear Solid (SLS) - Sinusoidal Strain Dashboard</b><br>' +
             '<sub>E<sub>re</sub> = 1 MPa  |  E<sub>rg</sub> = 2 MPa  |  τ<sub>R</sub> = 1 s  |  ω = 2π rad/s</sub>',
        font=dict(size=20, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    sliders=[slider],
    height=900,
    width=1500,
    plot_bgcolor=Colors.BG_LIGHT,
    paper_bgcolor=Colors.BG_WHITE,
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(t=120, b=120, l=80, r=80)
)

# Format subplot titles
for annotation in fig['layout']['annotations'][:4]:
    annotation['text'] = f'<b>{annotation["text"]}</b>'
    annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif')

# Explanation text
explanation = """
					<p><strong>What is Sinusoidal Loading?</strong> This dashboard shows how a viscoelastic material responds when it is cyclically stretched and compressed in a smooth, wave-like pattern (like a sine wave). This type of loading is common in engineering applications like vibration analysis and fatigue testing.</p>
					
					<p><strong>How to Read the Plots:</strong></p>
					<ul style="margin-left: 1.5rem;">
						<li><strong>Top Left - Stress Response:</strong> Shows how the stress oscillates in response to the cyclic strain. Notice that the stress wave is slightly "out of phase" with the strain—it peaks at a slightly different time. This phase difference indicates energy dissipation (damping).</li>
						<li><strong>Top Right - Strain Input:</strong> The applied sinusoidal strain (deformation) oscillates smoothly between positive and negative values, representing cyclic stretching and compression.</li>
						<li><strong>Bottom Left - Stress-Strain Hysteresis Loop:</strong> This is perhaps the most important plot! It shows stress vs. strain in a closed loop. The area inside the loop represents energy lost per cycle (hysteresis). A larger loop means more energy dissipation. In purely elastic materials, this would be a straight line with no loop.</li>
						<li><strong>Bottom Right - Characteristic Values:</strong> Key parameters including the loss angle (how much the stress lags behind strain), maximum stress, and material constants.</li>
					</ul>
					
					<p><strong>Using the Slider:</strong> Adjust the strain amplitude (ε₀) to see how different amounts of cyclic deformation affect the response. Larger amplitudes produce larger stress responses and wider hysteresis loops, indicating more energy dissipation.</p>
					
					<p><strong>Real-World Example:</strong> Think of a car's shock absorber. As the car goes over bumps, the shock absorber experiences cyclic loading. The hysteresis loop shows how much energy is absorbed and dissipated as heat—this is what makes the ride smoother. A material with a larger loop area would provide better damping.</p>
"""

# Save to highlighted_htmls (web deployment folder)
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'highlighted_htmls', 'sls_sinusoidal_clean.html')
save_figure_with_template(fig, output_path, title="Standard Linear Solid (SLS) - Sinusoidal Response", div_id='sls-sinusoidal-plot', explanation=explanation)
print(f"  {len(eps0_vals)} strain amplitudes | Single slider | 4 plots")
