#!/usr/bin/env python3
"""
Exponential Saturation Hardening Plasticity Summary Page Generator
==================================================================
Creates a comprehensive summary HTML page explaining the analysis steps
and displaying the four required plots.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_templates import (
    UCBerkeleyColors as Colors,
    get_axis_style,
    format_hover_template,
)

# Copy the integration functions directly to avoid import issues
# (This avoids potential circular dependencies or module-level execution)

def solve_delta_epsp_bar_exponential(
    sigma_trial: float,
    epsp_bar_n: float,
    E: float,
    Y0: float,
    H: float,
    alpha: float,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Solve for Δε̄^p_n during a plastic step."""
    abs_sigma_trial = abs(sigma_trial)
    
    # Quick elastic check
    Y_n = Y0 + H * (1.0 - np.exp(-alpha * epsp_bar_n))
    f_trial = abs_sigma_trial - Y_n
    if f_trial <= 0.0:
        return 0.0
    
    # Initial guess
    eps_shift = max(epsp_bar_n, 1e-12)
    H_bar_n = H * alpha * np.exp(-alpha * eps_shift)
    denom_guess = E + H_bar_n
    delta = max(f_trial / denom_guess, 0.0)
    
    for iter_count in range(max_iter):
        eps_bar_curr = epsp_bar_n + delta
        
        Y_curr = Y0 + H * (1.0 - np.exp(-alpha * eps_bar_curr))
        g_val = abs_sigma_trial - E * delta - Y_curr
        
        if abs(g_val) < tol * max(Y0, 1.0):
            break
        
        if eps_bar_curr <= 0.0:
            g_prime = -E
        else:
            H_bar_curr = H * alpha * np.exp(-alpha * eps_bar_curr)
            g_prime = -E - H_bar_curr
        
        if g_prime >= 0.0:
            g_prime = -E
        
        delta_new = delta - g_val / g_prime
        
        if delta_new < 0.0:
            delta_new = 0.0
        upper_bound = 10.0 * f_trial / E
        if delta_new > upper_bound:
            delta_new = upper_bound
        
        # Check for convergence stagnation
        if abs(delta_new - delta) < tol * max(delta, 1e-12):
            break
        
        delta = delta_new
    
    return max(delta, 0.0)


def integrate_exponential_saturation_backward_euler(
    E: float,
    Y0: float,
    H: float,
    alpha: float,
    eps0: float,
    beta: float,
    t_max: float,
    n_steps: int,
):
    """Integrate using backward Euler with return mapping."""
    t = np.linspace(0.0, t_max, n_steps + 1)
    dt = t_max / n_steps
    
    eps = eps0 * np.sin(beta * t)
    
    sigma = np.zeros_like(t)
    eps_p = np.zeros_like(t)
    epsp_bar = np.zeros_like(t)
    epsp_bar_dot = np.zeros_like(t)
    
    eps_p[0] = 0.0
    epsp_bar[0] = 0.0
    sigma[0] = E * (eps[0] - eps_p[0])
    
    Y_n = Y0
    
    for n in range(n_steps):
        delta_eps_n = eps[n + 1] - eps[n]
        
        sigma_trial = sigma[n] + E * delta_eps_n
        eps_p_trial = eps_p[n]
        Y_trial = Y_n
        f_trial = abs(sigma_trial) - Y_trial
        
        if f_trial <= 0.0:
            sigma[n + 1] = sigma_trial
            eps_p[n + 1] = eps_p_trial
            epsp_bar[n + 1] = epsp_bar[n]
            epsp_bar_dot[n + 1] = 0.0
            Y_n = Y_trial
        else:
            delta_epsp_bar_n = solve_delta_epsp_bar_exponential(
                sigma_trial, epsp_bar[n], E, Y0, H, alpha
            )
            
            epsp_bar[n + 1] = epsp_bar[n] + delta_epsp_bar_n
            epsp_bar_dot[n + 1] = delta_epsp_bar_n / dt
            
            sign_trial = np.sign(sigma_trial) if sigma_trial != 0.0 else 1.0
            eps_p[n + 1] = eps_p[n] + delta_epsp_bar_n * sign_trial
            
            Y_n = Y0 + H * (1.0 - np.exp(-alpha * epsp_bar[n + 1]))
            sigma[n + 1] = sign_trial * Y_n
        
        # Progress indicator every 10% of steps
        if (n + 1) % max(1, n_steps // 10) == 0:
            progress = 100 * (n + 1) / n_steps
            print(f"  Progress: {progress:.0f}%", end='\r')
    
    print()  # New line after progress
    return t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot

# Default parameters from problem statement
E = 210000.0  # MPa (210 GPa)
Y0 = 410.0    # MPa
H = 275.0     # MPa
alpha = 100.0  # dimensionless
eps0 = 0.01   # strain amplitude
beta = 1.0    # rad/s
t_max = 30.0  # s
n_steps = 3000  # number of time steps (finer resolution)


def create_plots(t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot, E, Y0, H, alpha, beta, eps0):
    """Create the six plots with detailed hover information."""
    # Compute additional quantities for hover info
    # Yield function: f = |σ| - Y(ε̄^p)
    Y = Y0 + H * (1.0 - np.exp(-alpha * epsp_bar))
    f_yield = np.abs(sigma) - Y
    
    # Determine elastic/plastic state
    is_plastic = f_yield > -1e-10  # Small tolerance for numerical precision
    
    # Total strain (for context)
    eps_total = eps
    
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            'Total Strain ε(t) vs Time',
            'Stress σ(t) vs Time',
            'Plastic Strain ε<sup>p</sup>(t) vs Time',
            'Accumulated Plastic Strain ε̄<sup>p</sup>(t) vs Time',
            'Plastic Strain Rate ε̇̄<sup>p</sup>(t) vs Time',
            'Stress-Strain Path σ vs ε'
        ),
        specs=[[{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}]],
        vertical_spacing=0.10
    )
    
    axis_style = get_axis_style()
    
    # Plot 1: Total Strain vs Time
    hover_text_strain = [
        f't = {ti:.6f} s<br>' +
        f'ε = {epsi:.6f}<br>' +
        f'ε̇ = {epsi_dot:.6e} s⁻¹<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}<br>' +
        f'ε<sup>p</sup> = {epspi:.6f}<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}'
        for ti, epsi, epsi_dot, si, Yi, fi, isp, epspi, epspbi in 
        zip(t, eps_total, beta * eps0 * np.cos(beta * t), sigma, Y, f_yield, is_plastic, eps_p, epsp_bar)
    ]
    fig.add_trace(go.Scatter(
        x=t, y=eps_total,
        mode='lines',
        line=dict(color=Colors.LAWRENCE, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_strain
    ), row=1, col=1)
    
    # Plot 2: Stress vs Time
    hover_text_stress = [
        f't = {ti:.6f} s<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}<br>' +
        f'ε = {epsi:.6f}<br>' +
        f'ε<sup>p</sup> = {epspi:.6f}<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}'
        for ti, si, Yi, fi, isp, epsi, epspi, epspbi in 
        zip(t, sigma, Y, f_yield, is_plastic, eps_total, eps_p, epsp_bar)
    ]
    fig.add_trace(go.Scatter(
        x=t, y=sigma,
        mode='lines',
        line=dict(color=Colors.BERKELEY_BLUE, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_stress
    ), row=2, col=1)
    
    # Plot 3: Plastic Strain vs Time
    hover_text_eps_p = [
        f't = {ti:.6f} s<br>' +
        f'ε<sup>p</sup> = {epspi:.6f}<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}<br>' +
        f'ε̇̄<sup>p</sup> = {epspdi:.6e} s⁻¹'
        for ti, epspi, si, Yi, fi, isp, epspbi, epspdi in 
        zip(t, eps_p, sigma, Y, f_yield, is_plastic, epsp_bar, epsp_bar_dot)
    ]
    fig.add_trace(go.Scatter(
        x=t, y=eps_p,
        mode='lines',
        line=dict(color=Colors.CALIFORNIA_GOLD, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_eps_p
    ), row=3, col=1)
    
    # Plot 4: Accumulated Plastic Strain vs Time
    hover_text_epsp_bar = [
        f't = {ti:.6f} s<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'H(ε̄<sup>p</sup>) = {Hi:.2f} MPa<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}<br>' +
        f'ε̇̄<sup>p</sup> = {epspdi:.6e} s⁻¹'
        for ti, epspbi, Yi, Hi, si, fi, isp, epspdi in 
        zip(t, epsp_bar, Y, H * alpha * np.exp(-alpha * epsp_bar), sigma, f_yield, is_plastic, epsp_bar_dot)
    ]
    fig.add_trace(go.Scatter(
        x=t, y=epsp_bar,
        mode='lines',
        line=dict(color=Colors.FOUNDERS_ROCK, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_epsp_bar
    ), row=4, col=1)
    
    # Plot 5: Plastic Strain Rate vs Time
    dt_array = np.concatenate(([t[1] - t[0]], np.diff(t)))
    hover_text_epsp_dot = [
        f't = {ti:.6f} s<br>' +
        f'ε̇̄<sup>p</sup> = {epspdi:.6e} s⁻¹<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}<br>' +
        f'Δε̄<sup>p</sup> ≈ {epspdi * dti:.8f}<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}'
        for ti, epspdi, epspbi, si, Yi, fi, isp, dti in 
        zip(t, epsp_bar_dot, epsp_bar, sigma, Y, f_yield, is_plastic, dt_array)
    ]
    fig.add_trace(go.Scatter(
        x=t, y=epsp_bar_dot,
        mode='lines',
        line=dict(color=Colors.MEDALIST, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_epsp_dot
    ), row=5, col=1)
    
    # Plot 6: Stress-Strain Path
    hover_text_stress_strain = [
        f'ε = {epsi:.6f}<br>' +
        f'σ = {si:.2f} MPa<br>' +
        f'|σ| = {abs(si):.2f} MPa<br>' +
        f't = {ti:.6f} s<br>' +
        f'Y(ε̄<sup>p</sup>) = {Yi:.2f} MPa<br>' +
        f'f = {fi:.6f} MPa<br>' +
        f'State: {"Plastic" if isp else "Elastic"}<br>' +
        f'ε<sup>p</sup> = {epspi:.6f}<br>' +
        f'ε̄<sup>p</sup> = {epspbi:.6f}<br>' +
        f'ε̇̄<sup>p</sup> = {epspdi:.6e} s⁻¹'
        for epsi, si, ti, Yi, fi, isp, epspi, epspbi, epspdi in 
        zip(eps_total, sigma, t, Y, f_yield, is_plastic, eps_p, epsp_bar, epsp_bar_dot)
    ]
    fig.add_trace(go.Scatter(
        x=eps_total, y=sigma,
        mode='lines',
        line=dict(color=Colors.WELLMAN_TILE, width=3),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_stress_strain
    ), row=6, col=1)
    
    # Apply axis styling (all vertical)
    fig.update_xaxes(title_text='<b>Time t [s]</b>', row=1, col=1, range=[0, t_max], **axis_style)
    fig.update_yaxes(title_text='<b>Total Strain ε</b>', row=1, col=1, **axis_style)
    fig.update_xaxes(title_text='<b>Time t [s]</b>', row=2, col=1, range=[0, t_max], **axis_style)
    fig.update_yaxes(title_text='<b>Stress σ [MPa]</b>', row=2, col=1, **axis_style)
    fig.update_xaxes(title_text='<b>Time t [s]</b>', row=3, col=1, range=[0, t_max], **axis_style)
    fig.update_yaxes(title_text='<b>Plastic Strain ε<sup>p</sup></b>', row=3, col=1, **axis_style)
    fig.update_xaxes(title_text='<b>Time t [s]</b>', row=4, col=1, range=[0, t_max], **axis_style)
    fig.update_yaxes(title_text='<b>Accumulated Plastic Strain ε̄<sup>p</sup></b>', row=4, col=1, **axis_style)
    fig.update_xaxes(title_text='<b>Time t [s]</b>', row=5, col=1, range=[0, t_max], **axis_style)
    fig.update_yaxes(title_text='<b>Plastic Strain Rate ε̇̄<sup>p</sup> [s⁻¹]</b>', row=5, col=1, **axis_style)
    fig.update_xaxes(title_text='<b>Total Strain ε</b>', row=6, col=1, **axis_style)
    fig.update_yaxes(title_text='<b>Stress σ [MPa]</b>', row=6, col=1, **axis_style)
    
    # Update layout (all vertical, 6 rows x 1 column)
    fig.update_layout(
        height=2400,  # Increased height for 6 vertical plots
        width=1200,
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(t=100, b=80, l=80, r=80)
    )
    
    # Format subplot titles
    for annotation in fig['layout']['annotations'][:6]:
        annotation['text'] = f'<b>{annotation["text"]}</b>'
        annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif')
    
    return fig


def generate_summary_html(t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot, plot_html):
    """Generate the complete summary HTML page."""
    
    # Compute key statistics
    sigma_max = np.max(np.abs(sigma))
    epsp_bar_max = np.max(epsp_bar)
    epsp_bar_dot_max = np.max(epsp_bar_dot)
    
    # Find when plasticity first occurs (more precise detection)
    plastic_indices = np.where(epsp_bar > 1e-10)[0]
    t_yield = t[plastic_indices[0]] if len(plastic_indices) > 0 else None
    
    # Also compute the exact yield time by interpolation if possible
    if t_yield is not None and plastic_indices[0] > 0:
        # Interpolate between last elastic and first plastic point
        t_elastic = t[plastic_indices[0] - 1]
        t_plastic = t[plastic_indices[0]]
        epsp_elastic = epsp_bar[plastic_indices[0] - 1]
        epsp_plastic = epsp_bar[plastic_indices[0]]
        if epsp_plastic > epsp_elastic:
            # Linear interpolation to find exact yield point
            t_yield = t_elastic + (t_plastic - t_elastic) * (1e-10 - epsp_elastic) / (epsp_plastic - epsp_elastic)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Exponential Saturation Hardening Plasticity - Summary</title>
		<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
		<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
		<script>
			window.MathJax = {{
				tex: {{
					inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
					displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
					processEscapes: true,
					processEnvironments: true
				}}
			}};
		</script>
		<style>
			body {{
				font-family: 'Arial', sans-serif;
				line-height: 1.6;
				color: #2c3e50;
				max-width: 1200px;
				margin: 0 auto;
				padding: 2rem;
				background-color: #f9fafb;
			}}
			h1 {{
				color: #003262;
				border-bottom: 3px solid #FDB515;
				padding-bottom: 0.5rem;
				margin-bottom: 2rem;
			}}
			h2 {{
				color: #003262;
				margin-top: 2rem;
				margin-bottom: 1rem;
				border-left: 4px solid #FDB515;
				padding-left: 1rem;
			}}
			h3 {{
				color: #003262;
				margin-top: 1.5rem;
				margin-bottom: 0.75rem;
			}}
			.section {{
				background: white;
				border: 1px solid #e5e7eb;
				border-radius: 8px;
				padding: 1.5rem;
				margin: 2rem 0;
			}}
			.equation-block {{
				background: #f9fafb;
				border-left: 4px solid #003262;
				padding: 1rem;
				margin: 1rem 0;
				overflow-x: auto;
			}}
			table {{
				width: 100%;
				border-collapse: collapse;
				margin: 1rem 0;
			}}
			th, td {{
				border: 1px solid #e5e7eb;
				padding: 0.75rem;
				text-align: left;
			}}
			th {{
				background-color: #003262;
				color: white;
			}}
			tr:nth-child(even) {{
				background-color: #f9fafb;
			}}
			.plot-container {{
				margin: 1.5rem 0;
				border: 1px solid #e5e7eb;
				border-radius: 8px;
				overflow: hidden;
			}}
			ul {{
				margin: 1rem 0;
				padding-left: 2rem;
			}}
			li {{
				margin: 0.5rem 0;
			}}
		</style>
	</head>
	<body>
		<div class="container">
			<h1>Exponential Saturation Hardening Elastoplasticity Analysis</h1>
			<p style="font-size: 1.1em; color: #6b7280; margin-bottom: 2rem;">
				This page summarizes the analysis of a one-dimensional nonlinear exponential saturation 
				isotropic hardening elastoplasticity model subjected to sinusoidal strain loading. 
				The analysis uses backward Euler integration with return mapping to solve the 
				consistency equation at each time step.
			</p>

			<section class="section">
				<h2>Problem Statement</h2>
				<p>The model is subjected to a sinusoidal strain history:</p>
				<div class="equation-block">
					$$\\varepsilon(t) = \\varepsilon_0 \\sin(\\beta t), \\quad 0 \\leq t \\leq 30~\\text{{s}}$$
				</div>
				<p>where $\\varepsilon_0 = 0.01$ and $\\beta = 1~\\text{{rad/s}}$.</p>
				
				<h3>Material Parameters</h3>
				<table>
					<tr>
						<th>Parameter</th>
						<th>Symbol</th>
						<th>Value</th>
					</tr>
					<tr>
						<td>Young's Modulus</td>
						<td>$E$</td>
						<td>210 GPa = 210,000 MPa</td>
					</tr>
					<tr>
						<td>Initial Yield Stress</td>
						<td>$Y_0$</td>
						<td>410 MPa</td>
					</tr>
					<tr>
						<td>Saturation Hardening</td>
						<td>$H$</td>
						<td>275 MPa</td>
					</tr>
					<tr>
						<td>Saturation Parameter</td>
						<td>$\\alpha$</td>
						<td>100 (dimensionless)</td>
					</tr>
				</table>
			</section>

			<section class="section">
				<h2>Constitutive Model</h2>
				
				<h3>Stress–Strain Relation</h3>
				<div class="equation-block">
					$$\\sigma = E(\\varepsilon - \\varepsilon^p)$$
				</div>
				<p>where $\\varepsilon^p$ is the signed plastic strain.</p>

				<h3>Flow Rule</h3>
				<div class="equation-block">
					$$\\dot{{\\varepsilon}}^p = \\dot{{\\bar{{\\varepsilon}}}}^p \\, \\text{{sign}}(\\sigma), \\quad \\dot{{\\bar{{\\varepsilon}}}}^p = |\\dot{{\\varepsilon}}^p|$$
				</div>
				<p>where $\\bar{{\\varepsilon}}^p$ is the accumulated (equivalent) plastic strain.</p>

				<h3>Yield Condition</h3>
				<div class="equation-block">
					$$f(\\sigma, \\bar{{\\varepsilon}}^p) = |\\sigma| - \\left(Y_0 + H\\left[1 - \\exp(-\\alpha \\bar{{\\varepsilon}}^p)\\right]\\right) \\leq 0$$
				</div>
				<p>The flow strength $Y(\\bar{{\\varepsilon}}^p)$ follows an exponential saturation law:</p>
				<div class="equation-block">
					$$Y(\\bar{{\\varepsilon}}^p) = Y_0 + H\\left[1 - \\exp(-\\alpha \\bar{{\\varepsilon}}^p)\\right]$$
				</div>
				<p>As $\\bar{{\\varepsilon}}^p \\to \\infty$, the flow strength approaches $Y_0 + H = 685~\\text{{MPa}}$.</p>

				<h3>Hardening Rate</h3>
				<div class="equation-block">
					$$H(\\bar{{\\varepsilon}}^p) = \\frac{{dY}}{{d\\bar{{\\varepsilon}}^p}} = H \\alpha \\exp(-\\alpha \\bar{{\\varepsilon}}^p) > 0$$
				</div>
				<p>The hardening rate decreases exponentially with accumulated plastic strain, starting at $H \\alpha = 27,500~\\text{{MPa}}$ when $\\bar{{\\varepsilon}}^p = 0$.</p>
			</section>

			<section class="section">
				<h2>Computational Method</h2>
				
				<h3>Backward Euler Integration with Return Mapping</h3>
				<p>The time integration uses a backward Euler scheme with the following steps:</p>
				
				<ol>
					<li><strong>Elastic Trial State:</strong> For each time step $n+1$, compute the trial stress assuming no plastic flow:
						<div class="equation-block">
							$$\\sigma_{{n+1}}^{{\\text{{tr}}}} = E(\\varepsilon_{{n+1}} - \\varepsilon_n^p)$$
						</div>
					</li>
					
					<li><strong>Yield Function Check:</strong> Evaluate the trial yield function:
						<div class="equation-block">
							$$f_{{n+1}}^{{\\text{{tr}}}} = |\\sigma_{{n+1}}^{{\\text{{tr}}}}| - Y(\\bar{{\\varepsilon}}_n^p)$$
						</div>
					</li>
					
					<li><strong>Elastic Step:</strong> If $f_{{n+1}}^{{\\text{{tr}}}} \\leq 0$, accept the trial state:
						<div class="equation-block">
							$$\\sigma_{{n+1}} = \\sigma_{{n+1}}^{{\\text{{tr}}}}, \\quad \\varepsilon_{{n+1}}^p = \\varepsilon_n^p, \\quad \\bar{{\\varepsilon}}_{{n+1}}^p = \\bar{{\\varepsilon}}_n^p$$
						</div>
					</li>
					
					<li><strong>Plastic Step:</strong> If $f_{{n+1}}^{{\\text{{tr}}}} > 0$, solve for the plastic strain increment $\\Delta\\bar{{\\varepsilon}}_{{n+1}}^p$ using Newton's method to satisfy consistency:
						<div class="equation-block">
							$$|\\sigma_{{n+1}}^{{\\text{{tr}}}}| - E \\Delta\\bar{{\\varepsilon}}_{{n+1}}^p - Y(\\bar{{\\varepsilon}}_n^p + \\Delta\\bar{{\\varepsilon}}_{{n+1}}^p) = 0$$
						</div>
						<p>The updated state is then:</p>
						<div class="equation-block">
							$$\\bar{{\\varepsilon}}_{{n+1}}^p = \\bar{{\\varepsilon}}_n^p + \\Delta\\bar{{\\varepsilon}}_{{n+1}}^p$$
						</div>
						<div class="equation-block">
							$$\\varepsilon_{{n+1}}^p = \\varepsilon_n^p + \\Delta\\bar{{\\varepsilon}}_{{n+1}}^p \\, \\text{{sign}}(\\sigma_{{n+1}}^{{\\text{{tr}}}})$$
						</div>
						<div class="equation-block">
							$$\\sigma_{{n+1}} = \\text{{sign}}(\\sigma_{{n+1}}^{{\\text{{tr}}}}) \\, Y(\\bar{{\\varepsilon}}_{{n+1}}^p)$$
						</div>
					</li>
					
					<li><strong>Plastic Strain Rate:</strong> Approximate the equivalent plastic strain rate:
						<div class="equation-block">
							$$\\dot{{\\bar{{\\varepsilon}}}}_{{n+1}}^p \\approx \\frac{{\\Delta\\bar{{\\varepsilon}}_{{n+1}}^p}}{{\\Delta t}}$$
						</div>
					</li>
				</ol>

				<h3>Newton Solver for Consistency Equation</h3>
				<p>For plastic steps, the scalar nonlinear equation for $\\Delta\\bar{{\\varepsilon}}^p$ is solved using Newton's method with:</p>
				<ul>
					<li>Initial guess based on effective tangent modulus: $\\Delta\\bar{{\\varepsilon}}^p \\approx f^{{\\text{{tr}}}} / (E + H(\\bar{{\\varepsilon}}_n^p))$</li>
					<li>Derivative: $dg/d(\\Delta\\bar{{\\varepsilon}}^p) = -E - H(\\bar{{\\varepsilon}}_n^p + \\Delta\\bar{{\\varepsilon}}^p)$</li>
					<li>Convergence tolerance: $10^{{-8}}$ relative to $Y_0$</li>
					<li>Maximum iterations: 50</li>
				</ul>
			</section>

			<section class="section">
				<h2>Results</h2>
				<p>The four plots below show the time evolution of stress, plastic strain, accumulated plastic strain, and plastic strain rate over the 30-second loading period.</p>
				
				<div class="plot-container">
					{plot_html}
				</div>

				<h3>Key Statistics</h3>
				<table>
					<tr>
						<th>Quantity</th>
						<th>Maximum Value</th>
					</tr>
					<tr>
						<td>Stress $|\\sigma|$</td>
						<td>{sigma_max:.2f} MPa</td>
					</tr>
					<tr>
						<td>Accumulated Plastic Strain $\\bar{{\\varepsilon}}^p$</td>
						<td>{epsp_bar_max:.6f}</td>
					</tr>
					<tr>
						<td>Plastic Strain Rate $\\dot{{\\bar{{\\varepsilon}}}}^p$</td>
						<td>{epsp_bar_dot_max:.6e} s⁻¹</td>
					</tr>
					{"<tr><td>First Yield Time</td><td>{:.6f} s</td></tr>".format(t_yield) if t_yield is not None else ""}
				</table>
			</section>

			<section class="section">
				<h2>Observations</h2>
				<ul>
					<li>The sinusoidal strain input causes alternating plastic loading and unloading cycles.</li>
					<li>During each loading half-cycle, the material yields and accumulates plastic strain.</li>
					<li>The exponential saturation hardening causes the flow strength to increase rapidly at first, then approach the saturation value $Y_0 + H = 685~\\text{{MPa}}$.</li>
					<li>The plastic strain rate $\\dot{{\\bar{{\\varepsilon}}}}^p$ is non-zero only during active plastic loading phases.</li>
					<li>The accumulated plastic strain $\\bar{{\\varepsilon}}^p$ increases monotonically, as required by the model.</li>
				</ul>
			</section>

		</div>
	</body>
</html>
"""
    return html


def main():
    """Generate the summary HTML page."""
    print("\n" + "="*70)
    print("CEE231 - Exponential Saturation Hardening Summary Generator")
    print("="*70 + "\n")
    
    print("Computing response with default parameters...")
    print(f"  Parameters: E={E} MPa, Y0={Y0} MPa, H={H} MPa, α={alpha}")
    print(f"  Strain: ε₀={eps0}, β={beta} rad/s, t_max={t_max} s, n_steps={n_steps}")
    print("  This may take a moment...")
    
    try:
        t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot = integrate_exponential_saturation_backward_euler(
            E=E, Y0=Y0, H=H, alpha=alpha,
            eps0=eps0, beta=beta, t_max=t_max,
            n_steps=n_steps
        )
        print(f"  ✓ Integration complete: {len(t)} time steps")
    except Exception as e:
        print(f"  ✗ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Creating plots...")
    fig = create_plots(t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot, E, Y0, H, alpha, beta, eps0)
    
    # Convert plot to HTML (embed directly, no external file)
    # Get the HTML and extract the plot div and script
    # Include config to enable home/reset button (home button is enabled by default)
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [],  # Keep all default buttons including home
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'exponential_saturation_plasticity',
            'height': 2400,
            'width': 1200,
            'scale': 1
        }
    }
    plot_html_full = fig.to_html(include_plotlyjs='cdn', div_id='plasticity-plots', 
                                  full_html=False, config=config)
    plot_html = plot_html_full
    
    print("Generating summary HTML...")
    html_content = generate_summary_html(t, eps, sigma, eps_p, epsp_bar, epsp_bar_dot, plot_html)
    
    # Save to highlighted_htmls
    output_path = Path(__file__).parent.parent.parent / 'highlighted_htmls' / 'exponential_saturation_plasticity_summary.html'
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"\n[SUCCESS] Summary generated: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

