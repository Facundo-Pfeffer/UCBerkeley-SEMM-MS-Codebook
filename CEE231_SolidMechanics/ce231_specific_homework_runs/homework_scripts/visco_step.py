#!/usr/bin/env python3
"""
Step Response Dashboard (from HW8) - extracted as reusable function.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from plot_config import (
    COLORS,
    AXIS_STYLE,
    TABLE_HEADER_STYLE,
    TABLE_CELL_STYLE,
    LAYOUT_DEFAULTS,
    SLIDER_STYLE,
    ANNOTATION_FONTS,
)

HW_DIR = Path(__file__).resolve().parent


def build_step_response_dashboard():
    """Build viscoelastic step response dashboard."""
    
    # Material parameters
    E1 = 10.0
    E2 = 10.0
    eps0_default = 1e-4
    beta_default = 5.0
    
    eps0_vals = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    
    t_over_beta_max = 4.0
    t_over_beta_min = -1.0
    npts = 1200
    
    def compute_all(beta, eps0):
        t = np.linspace(t_over_beta_min * beta, t_over_beta_max * beta, npts, endpoint=False)
        x = t / beta
        step = np.where(t >= 0, 1.0, 0.0)
        Er = (E1 + E2 * np.exp(-(np.maximum(t, 0) / beta) ** 2)) * step
        sigma = eps0 * Er
        eps = eps0 * step
        return x, Er, sigma, eps
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Stress Response</b> σ(t)",
            "<b>Relaxation Modulus</b> E<sub>r</sub>(t)",
            "<b>Step Strain</b> ε(t)",
            "<b>Characteristic Values</b>"
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.10, vertical_spacing=0.14
    )
    
    combo_keys = []
    sigma_ids, Er_ids, eps_ids, table_ids = [], [], [], []
    sigma_annot_ids, Er_annot_ids, eps_annot_ids = [], [], []
    
    for eps0 in eps0_vals:
        beta = beta_default
        x, Er, sigma, eps = compute_all(beta, eps0)
        combo_keys.append(eps0)
        
        # Stress response
        fig.add_trace(go.Scatter(
            x=x, y=sigma, mode="lines",
            name=f"σ(t) | ε₀={eps0:.0e}",
            visible=False,
            line=dict(color=COLORS['stress'], width=2.5)
        ), row=1, col=1)
        sigma_ids.append(len(fig.data)-1)
        
        # Stress markers
        sigma0 = eps0 * (E1 + E2)
        siginf = eps0 * E1
        fig.add_trace(go.Scatter(
            x=[0.05, t_over_beta_max - 0.1], 
            y=[sigma0, siginf],
            mode="markers+text",
            marker=dict(size=9, color=COLORS['stress'], symbol='circle', 
                       line=dict(width=1.5, color='white')),
            text=[f"σ(0⁺) = {sigma0:.2e} MPa", f"σ(∞) = {siginf:.2e} MPa"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=10, color=COLORS['annotation'], family='Arial, sans-serif'),
            visible=False,
            showlegend=False,
            hoverinfo='text',
            hovertext=[f"Initial stress: {sigma0:.4e} MPa", f"Final stress: {siginf:.4e} MPa"]
        ), row=1, col=1)
        sigma_annot_ids.append(len(fig.data)-1)
        
        # Relaxation modulus
        fig.add_trace(go.Scatter(
            x=x, y=Er, mode="lines",
            name=f"E_r(t) | ε₀={eps0:.0e}",
            visible=False,
            showlegend=False,
            line=dict(color=COLORS['modulus'], width=2.5)
        ), row=1, col=2)
        Er_ids.append(len(fig.data)-1)
        
        # E_r markers
        Er0 = E1 + E2
        Erinf = E1
        fig.add_trace(go.Scatter(
            x=[0.05, t_over_beta_max - 0.1], 
            y=[Er0, Erinf],
            mode="markers+text",
            marker=dict(size=9, color=COLORS['modulus'], symbol='circle',
                       line=dict(width=1.5, color='white')),
            text=[f"E<sub>r</sub>(0⁺) = {Er0:.1f} MPa", f"E<sub>r</sub>(∞) = {Erinf:.1f} MPa"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=10, color=COLORS['annotation'], family='Arial, sans-serif'),
            visible=False,
            showlegend=False,
            hoverinfo='text',
            hovertext=[f"Initial modulus: {Er0:.1f} MPa", f"Final modulus: {Erinf:.1f} MPa"]
        ), row=1, col=2)
        Er_annot_ids.append(len(fig.data)-1)
        
        # Step strain
        fig.add_trace(go.Scatter(
            x=x, y=eps, mode="lines",
            name=f"ε(t) | ε₀={eps0:.0e}",
            visible=False,
            showlegend=False,
            line=dict(color=COLORS['strain'], width=3.5)
        ), row=2, col=1)
        eps_ids.append(len(fig.data)-1)
        
        # Strain marker
        fig.add_trace(go.Scatter(
            x=[0.05],
            y=[eps0],
            mode="markers+text",
            marker=dict(size=10, color=COLORS['strain'], symbol='square',
                       line=dict(width=1.5, color='white')),
            text=[f"ε₀ = {eps0:.2e}"],
            textposition="top center",
            textfont=dict(size=11, color=COLORS['annotation'], family='Arial, sans-serif'),
            visible=False,
            showlegend=False,
            hoverinfo='text',
            hovertext=[f"Applied strain: {eps0:.4e}"]
        ), row=2, col=1)
        eps_annot_ids.append(len(fig.data)-1)
        
        # Table
        t_90 = 1.52 * beta
        fig.add_trace(go.Table(
            header=dict(values=["<b>Parameter</b>", "<b>Value</b>"], **TABLE_HEADER_STYLE),
            cells=dict(
                values=[
                    ["<b>ε₀</b>", "<b>β</b>", "<b>σ(0⁺)</b>", "<b>σ(∞)</b>", 
                     "<b>E<sub>r</sub>(0⁺)</b>", "<b>E<sub>r</sub>(∞)</b>", 
                     "<b>t<sub>90%</sub></b>"],
                    [f"{eps0:.2e}", f"{beta:g} s", f"{sigma0:.3e} MPa", f"{siginf:.3e} MPa",
                     f"{Er0:.1f} MPa", f"{Erinf:.1f} MPa", f"{t_90:.2f} s"]
                ],
                height=28,
                **TABLE_CELL_STYLE
            ),
            visible=False
        ), row=2, col=2)
        table_ids.append(len(fig.data)-1)
    
    # Set default visibility
    default_idx = combo_keys.index(eps0_default)
    fig.data[sigma_ids[default_idx]].visible = True
    fig.data[sigma_annot_ids[default_idx]].visible = True
    fig.data[Er_ids[default_idx]].visible = True
    fig.data[Er_annot_ids[default_idx]].visible = True
    fig.data[eps_ids[default_idx]].visible = True
    fig.data[eps_annot_ids[default_idx]].visible = True
    fig.data[table_ids[default_idx]].visible = True
    
    # Apply axis styling
    fig.update_xaxes(title_text="<b>t/β</b>", row=1, col=1, range=[t_over_beta_min, t_over_beta_max], **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>σ(t) [MPa]</b>", row=1, col=1, **AXIS_STYLE)
    
    fig.update_xaxes(title_text="<b>t/β</b>", row=1, col=2, range=[t_over_beta_min, t_over_beta_max], **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>E<sub>r</sub>(t) [MPa]</b>", row=1, col=2, **AXIS_STYLE)
    
    eps_margin = eps0_default * 0.3
    fig.update_xaxes(title_text="<b>t/β</b>", row=2, col=1, range=[t_over_beta_min, t_over_beta_max], **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>ε(t) [-]</b>", row=2, col=1, 
                     range=[-eps_margin, eps0_default + eps_margin], **AXIS_STYLE)
    
    fig.update_layout(
        title_text="<b>Viscoelastic Step-Response Dashboard</b> (E₁ = E₂ = 10 MPa)",
        height=950, 
        width=1300,
        margin=dict(t=200, b=160, l=80, r=80),
        **LAYOUT_DEFAULTS
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations'][:4]:
        annotation['font'] = dict(size=13, family='Arial, sans-serif', color=COLORS['stress'])
    
    def mask_for(eps0_sel):
        vis = [False]*len(fig.data)
        idx = combo_keys.index(eps0_sel)
        vis[sigma_ids[idx]] = True
        vis[sigma_annot_ids[idx]] = True
        vis[Er_ids[idx]] = True
        vis[Er_annot_ids[idx]] = True
        vis[eps_ids[idx]] = True
        vis[eps_annot_ids[idx]] = True
        vis[table_ids[idx]] = True
        return vis
    
    # Create slider steps
    steps_eps = []
    for e0 in eps0_vals:
        eps_margin = e0 * 0.3
        yaxis_range = [-eps_margin, e0 + eps_margin]
        vis = mask_for(e0)
        steps_eps.append(dict(
            method="update",
            args=[{"visible": vis}, {"yaxis3.range": yaxis_range}],
            label=f"{e0:.0e}"
        ))
    
    # Mathematical formulation annotations
    formulation_annotations = [
        dict(text="<b>Mathematical Formulation</b>",
             x=0.5, y=1.165, xref="paper", yref="paper",
             xanchor="center", showarrow=False, font=ANNOTATION_FONTS['title']),
        dict(text="<b>Step Input:</b> ε(t) = ε₀·h(t)  →  dε/dt = ε₀·δ(t)",
             x=0.02, y=1.135, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
        dict(text="<b>Boltzmann Superposition:</b> σ(t) = ∫ E<sub>r</sub>(t−s)·(dε/ds)·ds = ε₀·E<sub>r</sub>(t)·h(t)",
             x=0.02, y=1.11, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
        dict(text="<b>Relaxation Modulus:</b> E<sub>r</sub>(t) = E₁ + E₂·exp[−(t/β)²]",
             x=0.02, y=1.085, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
        dict(text="<b>Result:</b> σ(t) = ε₀·E<sub>r</sub>(t)   |   σ(0⁺) = ε₀(E₁+E₂)   |   σ(∞) = ε₀·E₁",
             x=0.02, y=1.06, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
    ]
    
    # Add slider
    sliders = [dict(
        active=eps0_vals.index(eps0_default),
        yanchor="bottom", y=-0.08, xanchor="right", x=0.98,
        currentvalue=dict(prefix="<b>Initial Strain ε₀ = </b>", visible=True, xanchor="right",
                         font=dict(size=11, family='Arial, sans-serif', color=COLORS['stress'])),
        pad=dict(b=3, t=3), len=0.45, steps=steps_eps,
        **SLIDER_STYLE
    )]
    
    fig.update_layout(
        sliders=sliders,
        annotations=list(fig['layout']['annotations']) + formulation_annotations
    )
    return fig


if __name__ == "__main__":
    fig = build_step_response_dashboard()
    output_path = HW_DIR / "visco_dashboard_step.html"
    fig.write_html(output_path, include_plotlyjs="cdn", auto_open=False)
    print(f"[SUCCESS] Generated: {output_path}")

