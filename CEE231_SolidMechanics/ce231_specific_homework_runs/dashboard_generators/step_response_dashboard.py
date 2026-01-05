#!/usr/bin/env python3
"""
Viscoelastic Step Response Dashboard Generator
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

# Material parameters
E1 = 10.0  # MPa
E2 = 10.0  # MPa
beta = 5.0  # s

# Strain values - logarithmically spaced for very smooth slider (60 values)
eps0_vals = np.logspace(np.log10(5e-7), np.log10(2e-3), 60)
eps0_default = 1e-4

# Time range
t_min = -1.0
t_max = 4.0
npts = 1200

def compute_response(beta, eps0):
    """Compute viscoelastic response."""
    t = np.linspace(t_min * beta, t_max * beta, npts)
    t_norm = t / beta
    step = np.where(t >= 0, 1.0, 0.0)
    Er = (E1 + E2 * np.exp(-(np.maximum(t, 0) / beta) ** 2)) * step
    sigma = eps0 * Er
    eps = eps0 * step
    return t_norm, Er, sigma, eps

# Create subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Stress Response σ(t)',
        'Relaxation Modulus E<sub>r</sub>(t)',
        'Step Strain Input ε(t)',
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
    t_norm, Er, sigma, eps = compute_response(beta, eps0)
    visible = (i == default_idx)
    
    # Stress plot
    sigma0 = eps0 * (E1 + E2)
    sigma_inf = eps0 * E1
    
    fig.add_trace(go.Scatter(
        x=t_norm, y=sigma,
        mode='lines',
        line=dict(color=Colors.BERKELEY_BLUE, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('t/β', 'σ', '.2f', '.3e') + ' MPa'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0.1, t_max-0.2], y=[sigma0, sigma_inf],
        mode='markers+text',
        marker=dict(size=10, color=Colors.BERKELEY_BLUE, line=dict(width=2, color='white')),
        text=[f'σ(0⁺)={sigma0:.2e}', f'σ(∞)={sigma_inf:.2e}'],
        textposition=['top center', 'bottom center'],
        textfont=dict(size=11, color=Colors.TEXT_DARK, family='Arial, sans-serif'),
        visible=visible,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Modulus plot
    Er0 = E1 + E2
    Er_inf = E1
    
    fig.add_trace(go.Scatter(
        x=t_norm, y=Er,
        mode='lines',
        line=dict(color=Colors.FOUNDERS_ROCK, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('t/β', 'E<sub>r</sub>', '.2f', '.1f') + ' MPa'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0.1, t_max-0.2], y=[Er0, Er_inf],
        mode='markers+text',
        marker=dict(size=10, color=Colors.FOUNDERS_ROCK, line=dict(width=2, color='white')),
        text=[f'E<sub>r</sub>(0⁺)={Er0:.1f}', f'E<sub>r</sub>(∞)={Er_inf:.1f}'],
        textposition=['top center', 'bottom center'],
        textfont=dict(size=11, color=Colors.TEXT_DARK, family='Arial, sans-serif'),
        visible=visible,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=2)
    
    # Strain plot
    fig.add_trace(go.Scatter(
        x=t_norm, y=eps,
        mode='lines',
        line=dict(color=Colors.CALIFORNIA_GOLD, width=4),
        visible=visible,
        showlegend=False,
        hovertemplate=format_hover_template('t/β', 'ε', '.2f', '.3e')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0.1], y=[eps0],
        mode='markers+text',
        marker=dict(size=12, color=Colors.CALIFORNIA_GOLD, symbol='square', line=dict(width=2, color='white')),
        text=[f'ε₀={eps0:.2e}'],
        textposition='top center',
        textfont=dict(size=11, color=Colors.TEXT_DARK, family='Arial, sans-serif'),
        visible=visible,
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)
    
    # Table
    t_90 = beta * np.sqrt(-np.log(0.1))
    fig.add_trace(go.Table(
        header=get_table_header_style(),
        cells=get_table_cells_style(),
        visible=visible
    ), row=2, col=2)
    
    fig.data[-1].header.values = ['<b>Parameter</b>', '<b>Value</b>']
    fig.data[-1].cells.values = [
        ['ε₀', 'β', 'σ(0⁺)', 'σ(∞)', 'E<sub>r</sub>(0⁺)', 'E<sub>r</sub>(∞)', 't<sub>90%</sub>'],
        [f'{eps0:.2e}', f'{beta:.1f} s', f'{sigma0:.3e} MPa', f'{sigma_inf:.3e} MPa',
         f'{Er0:.1f} MPa', f'{Er_inf:.1f} MPa', f'{t_90:.2f} s']
    ]

# Apply axis styling
axis_style = get_axis_style()
fig.update_xaxes(title_text='<b>t/β</b>', row=1, col=1, range=[t_min, t_max], **axis_style)
fig.update_yaxes(title_text='<b>σ(t) [MPa]</b>', row=1, col=1, **axis_style)
fig.update_xaxes(title_text='<b>t/β</b>', row=1, col=2, range=[t_min, t_max], **axis_style)
fig.update_yaxes(title_text='<b>E<sub>r</sub>(t) [MPa]</b>', row=1, col=2, **axis_style)
fig.update_xaxes(title_text='<b>t/β</b>', row=2, col=1, range=[t_min, t_max], **axis_style)
fig.update_yaxes(title_text='<b>ε(t) [-]</b>', row=2, col=1, **axis_style)

# Create slider
steps = []
for i, eps0 in enumerate(eps0_vals):
    eps_margin = eps0 * 0.3
    visibility = []
    for j in range(len(eps0_vals)):
        visibility.extend([True] * 7 if j == i else [False] * 7)
    
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}, {"yaxis3.range": [-eps_margin, eps0 + eps_margin]}],
        label=f"{eps0:.0e}"
    ))

slider = get_slider_style(
    steps=steps,
    active_index=int(np.argmin(np.abs(eps0_vals - eps0_default))),
    prefix="<b>Applied Strain ε₀ = </b>"
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Viscoelastic Step Response Dashboard</b><br><sub>E₁ = E₂ = 10 MPa  |  β = 5 s</sub>',
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
    margin=dict(t=100, b=120, l=80, r=80)
)

# Format subplot titles
for annotation in fig['layout']['annotations'][:4]:
    annotation['text'] = f'<b>{annotation["text"]}</b>'
    annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif')

# Explanation text
explanation = """
					<p><strong>What is a Step Response?</strong> This visualization shows how a viscoelastic material responds when it is suddenly stretched to a constant length and held there. Unlike purely elastic materials (like a spring), viscoelastic materials exhibit time-dependent behavior.</p>
					
					<p><strong>How to Read the Plots:</strong></p>
					<ul style="margin-left: 1.5rem;">
						<li><strong>Top Left - Stress Response:</strong> Shows how the stress (force per area) in the material changes over time. Notice how it starts high and gradually decreases—this is called "stress relaxation." The material "remembers" its initial state but gradually relaxes.</li>
						<li><strong>Top Right - Relaxation Modulus:</strong> This represents the material's stiffness over time. It starts at a higher value (glassy modulus) and decreases toward an equilibrium value as the material relaxes.</li>
						<li><strong>Bottom Left - Step Strain Input:</strong> The applied strain (deformation) jumps from 0 to a constant value at time t=0. This is the "step" that triggers the response.</li>
						<li><strong>Bottom Right - Characteristic Values:</strong> Key parameters describing the material's response, including maximum stress, relaxation time, and material constants.</li>
					</ul>
					
					<p><strong>Using the Slider:</strong> Adjust the strain amplitude (ε₀) to see how different amounts of stretching affect the material's response. Larger strains produce proportionally larger stresses, but the relaxation behavior remains similar.</p>
					
					<p><strong>Real-World Example:</strong> Imagine stretching a rubber band and holding it at a fixed length. Initially, it feels very tight (high stress), but over time, the force required to maintain that length decreases as the material relaxes. This is exactly what these plots show!</p>
"""

# Save to highlighted_htmls (web deployment folder)
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'highlighted_htmls', 'step_response_clean.html')
save_figure_with_template(fig, output_path, title="Viscoelastic Step Response", div_id='step-response-plot', explanation=explanation)
print(f"  {len(eps0_vals)} strain values | Single slider | 4 plots")
