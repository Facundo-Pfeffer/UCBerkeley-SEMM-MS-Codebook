#!/usr/bin/env python3
"""
SLS Square Wave Response Dashboard (Backward Euler)
Standard Linear Solid subjected to square wave strain with time integration
"""

import sys
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.append(str(DASHBOARD_ROOT))

PROJECT_ROOT = DASHBOARD_ROOT.parent

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
    save_figure
)

# Material parameters
E0 = 2.0      # MPa - equilibrium modulus
E1 = 1.5      # MPa - spring component
eta = 1.0     # MPa·s - viscosity
tau_R = eta / E1  # s - relaxation time (2/3 s)

# Time parameters
t_max = 5.0   # s
# Create smooth range of time steps (50 values for very smooth slider)
dt_vals = np.concatenate([
    np.linspace(0.001, 0.01, 20),   # Fine steps in small dt range
    np.linspace(0.011, 0.05, 20),   # Medium steps
    np.linspace(0.055, 0.1, 11)     # Coarser steps for large dt
])
dt_default = 0.01

def square_wave_strain(t):
    """
    Square wave strain function.
    ε(t) = 0.01 if (t mod 1) <= 0.5, else 0.0
    """
    if t <= 0:
        return 0.0
    t_mod = t % 1.0
    return 0.01 if t_mod <= 0.5 else 0.0

def compute_sls_backward_euler(E0, E1, tau_R, dt, t_max):
    """
    Compute SLS response using Backward Euler scheme.
    
    Governing equations:
    σ = E₀·ε + E₁·(ε - ε^v)
    dε^v/dt = (1/τ_R)·(ε - ε^v)
    
    Backward Euler update:
    γ = τ_R/(τ_R + Δt)
    ε^v_{n+1} = γ·ε^v_n + (1-γ)·ε_{n+1}
    σ_{n+1} = (E₀ + γE₁)·ε_{n+1} - γE₁·ε^v_n
    """
    # Time array
    n_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n_steps)
    
    # Initialize arrays
    eps = np.zeros(n_steps)
    eps_v = np.zeros(n_steps)
    sigma = np.zeros(n_steps)
    
    # Relaxation factor
    gamma = tau_R / (tau_R + dt)
    
    # Initial conditions (quiescent)
    eps[0] = square_wave_strain(0)
    eps_v[0] = 0.0
    sigma[0] = E0 * eps[0] + E1 * (eps[0] - eps_v[0])
    
    # Time stepping
    for i in range(1, n_steps):
        # Get current strain from square wave
        eps[i] = square_wave_strain(t[i])
        
        # Update viscous strain (Backward Euler)
        eps_v[i] = gamma * eps_v[i-1] + (1 - gamma) * eps[i]
        
        # Update stress
        sigma[i] = (E0 + gamma * E1) * eps[i] - gamma * E1 * eps_v[i-1]
    
    return t, eps, eps_v, sigma, gamma

# Create subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Stress & Strain Response',
        'Viscous Strain Evolution ε<sup>v</sup>(t)',
        'Stress-Strain Path',
        'Characteristic Values'
    ),
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "table"}]],
    horizontal_spacing=0.12,
    vertical_spacing=0.15
)

# Find index closest to default
default_idx = int(np.argmin(np.abs(dt_vals - dt_default)))

# Generate all data for different time steps
for i, dt in enumerate(dt_vals):
    t, eps, eps_v, sigma, gamma = compute_sls_backward_euler(E0, E1, tau_R, dt, t_max)
    visible = (i == default_idx)
    
    # Plot 1: Stress and Strain on same axes
    # Strain (square wave)
    fig.add_trace(go.Scatter(
        x=t, y=eps * 100,  # Convert to percentage
        mode='lines',
        name='Strain ε(t)',
        line=dict(color=Colors.CALIFORNIA_GOLD, width=3, dash='dot'),
        visible=visible,
        showlegend=True,
        legendgroup='1',
        yaxis='y1',
        hovertemplate='t=%{x:.3f} s<br>ε=%{y:.2f}%<extra></extra>'
    ), row=1, col=1)
    
    # Stress - multiply by 100 to match strain scale
    fig.add_trace(go.Scatter(
        x=t, y=sigma * 100,  # Multiply by 100 for similar scale
        mode='lines',
        name='Stress σ(t)×100',
        line=dict(color=Colors.BERKELEY_BLUE, width=3),
        visible=visible,
        showlegend=True,
        legendgroup='1',
        yaxis='y1',  # Use same y-axis as strain
        hovertemplate='t=%{x:.3f} s<br>σ=%{y:.2f} (×100 MPa)<extra></extra>'
    ), row=1, col=1)
    
    # Plot 2: Viscous strain evolution
    fig.add_trace(go.Scatter(
        x=t, y=eps_v * 100,  # Convert to percentage
        mode='lines',
        line=dict(color=Colors.FOUNDERS_ROCK, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate='t=%{x:.3f} s<br>ε<sup>v</sup>=%{y:.3f}%<extra></extra>'
    ), row=1, col=2)
    
    # Plot 3: Stress-Strain path
    fig.add_trace(go.Scatter(
        x=eps * 100, y=sigma,
        mode='lines',
        line=dict(color=Colors.MEDALIST, width=3),
        visible=visible,
        showlegend=False,
        hovertemplate='ε=%{x:.2f}%<br>σ=%{y:.4f} MPa<extra></extra>'
    ), row=2, col=1)
    
    # Table with characteristic values
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    eps_v_max = np.max(eps_v)
    E_tangent = E0 + gamma * E1
    E_unrelaxed = E0 + E1
    
    fig.add_trace(go.Table(
        header=get_table_header_style(),
        cells=get_table_cells_style(),
        visible=visible
    ), row=2, col=2)
    
    fig.data[-1].header.values = ['<b>Parameter</b>', '<b>Value</b>']
    fig.data[-1].cells.values = [
        ['E₀', 'E₁', 'η', 'τ<sub>R</sub>', 'Δt', 'γ', 
         'E<sub>tangent</sub>', 'E<sub>unrelaxed</sub>',
         'σ<sub>max</sub>', 'σ<sub>min</sub>', 'ε<sup>v</sup><sub>max</sub>', 'Steps'],
        [f'{E0:.2f} MPa', f'{E1:.2f} MPa', f'{eta:.2f} MPa·s', f'{tau_R:.3f} s',
         f'{dt:.4f} s', f'{gamma:.4f}',
         f'{E_tangent:.3f} MPa', f'{E_unrelaxed:.1f} MPa',
         f'{sigma_max:.4f} MPa', f'{sigma_min:.4f} MPa', 
         f'{eps_v_max*100:.3f}%', f'{len(t)}']
    ]

# Apply axis styling
axis_style = get_axis_style()

# Plot 1: Single y-axis for both strain and stress (stress×100)
fig.update_xaxes(title_text='<b>Time [s]</b>', row=1, col=1, **axis_style)
fig.update_yaxes(
    title_text='<b>Strain ε(t) [%] & Stress σ(t)×100 [MPa]</b>',
    row=1, col=1, 
    range=[-2, 6],  # Range to show both strain and stress×100
    showgrid=True,
    gridcolor=axis_style['gridcolor'],
    gridwidth=1,
    zeroline=True,
    zerolinecolor='rgba(0, 0, 0, 0.3)',
    zerolinewidth=2,
    showline=True,
    linewidth=2,
    linecolor='black',
    mirror=True,
    ticks='outside',
    tickwidth=1.5,
    tickcolor='black'
)

# Plot 2: Viscous strain - auto scale
fig.update_xaxes(title_text='<b>Time [s]</b>', row=1, col=2, **axis_style)
fig.update_yaxes(
    title_text='<b>Viscous Strain ε<sup>v</sup>(t) [%]</b>', 
    row=1, col=2,
    # Let Plotly auto-scale based on data
    **axis_style
)

# Plot 3: Stress-Strain path
fig.update_xaxes(title_text='<b>Strain [%]</b>', row=2, col=1, **axis_style)
fig.update_yaxes(title_text='<b>Stress [MPa]</b>', row=2, col=1, **axis_style)

# Create slider
steps = []
for i, dt in enumerate(dt_vals):
    # Each dt has 5 traces (strain, stress, eps_v, stress-strain path, table)
    visibility = []
    for j in range(len(dt_vals)):
        visibility.extend([True] * 5 if j == i else [False] * 5)
    # Add the invisible trace for secondary y-axis
    visibility.append(False)
    
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}],
        label=f"{dt:.4f}s"
    ))

slider = get_slider_style(
    steps=steps,
    active_index=int(np.argmin(np.abs(dt_vals - dt_default))),
    prefix="<b>Time Step Δt = </b>",
    suffix=" s"
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Standard Linear Solid - Square Wave Input (Backward Euler)</b><br>' +
             '<sub>E₀ = 2.0 MPa  |  E₁ = 1.5 MPa  |  η = 1.0 MPa·s  |  τ<sub>R</sub> = 0.667 s</sub>',
        font=dict(size=20, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    sliders=[slider],
    height=900,
    width=1600,
    plot_bgcolor=Colors.BG_LIGHT,
    paper_bgcolor=Colors.BG_WHITE,
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(t=120, b=120, l=80, r=100),
    legend=dict(
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=Colors.BERKELEY_BLUE,
        borderwidth=1
    )
)

# Format subplot titles
for annotation in fig['layout']['annotations'][:4]:
    annotation['text'] = f'<b>{annotation["text"]}</b>'
    annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif')


# Save to highlighted_htmls (web deployment folder)
output_path = PROJECT_ROOT / 'highlighted_htmls' / 'sls_square_wave_clean.html'
save_figure(fig, output_path)
print(f"  {len(dt_vals)} time step values | Single slider | 4 plots")
print(f"  Time range: 0 to {t_max} s")
print(f"  Square wave: period = 1 s, amplitude = 0.01, duty cycle = 50%")

