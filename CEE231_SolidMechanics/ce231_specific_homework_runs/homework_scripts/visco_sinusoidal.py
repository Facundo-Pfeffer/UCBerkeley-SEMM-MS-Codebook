#!/usr/bin/env python3
"""
Sinusoidal Response Dashboard (from original HW9) - extracted as reusable function.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plot_config import COLORS, AXIS_STYLE, TABLE_HEADER_STYLE, TABLE_CELL_STYLE, LAYOUT_DEFAULTS, SLIDER_STYLE, ANNOTATION_FONTS


def build_sinusoidal_dashboard():
    """Build Standard Linear Solid (SLS) sinusoidal strain dashboard."""
    
    # Material parameters
    E_re_default = 1.0
    E_rg_default = 2.0
    tau_R_default = 1.0
    eps0_default = 0.01
    omega_default = 2*np.pi
    
    eps0_vals = [0.005, 0.01, 0.02, 0.05]
    omega_vals = [np.pi, 2*np.pi, 4*np.pi, 6*np.pi]
    E_re_vals = [0.5, 1.0, 2.0, 3.0]
    E_rg_vals = [1.0, 2.0, 3.0, 4.0]
    tau_R_vals = [0.5, 1.0, 2.0, 3.0]
    
    t_max = 4.0
    npts = 1000
    
    def compute_response(E_re, E_rg, tau_R, eps0, omega):
        t = np.linspace(0, t_max, npts)
        eps = eps0 * np.sin(omega * t)
        
        denom = (1/tau_R**2 + omega**2)
        A = eps0 * (E_re + E_rg * omega**2 / denom)
        B = eps0 * E_rg * (omega/tau_R) / denom
        C = -eps0 * E_rg * (omega/tau_R) / denom
        
        sigma = A * np.sin(omega * t) + B * np.cos(omega * t) + C * np.exp(-t/tau_R)
        E_r = E_re + E_rg * np.exp(-t/tau_R)
        
        return t, eps, sigma, E_r, A, B, C
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Stress Response</b> σ(t)",
            "<b>Strain Input</b> ε(t)",
            "<b>Stress-Strain (Hysteresis)</b>",
            "<b>Characteristic Values</b>"
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.10, vertical_spacing=0.14
    )
    
    combo_keys = []
    sigma_ids, eps_ids, hyst_ids, table_ids = [], [], [], []
    
    for eps0 in eps0_vals:
        for omega in omega_vals:
            for E_re in E_re_vals:
                for E_rg in E_rg_vals:
                    for tau_R in tau_R_vals:
                        combo_keys.append((eps0, omega, E_re, E_rg, tau_R))
                        
                        t, eps, sigma, E_r, A, B, C = compute_response(E_re, E_rg, tau_R, eps0, omega)
                        
                        # Stress vs time
                        fig.add_trace(go.Scatter(
                            x=t, y=sigma, mode="lines", name=f"σ(t)", visible=False,
                            line=dict(color=COLORS['stress'], width=2.5),
                            hovertemplate='t = %{x:.3f} s<br>σ = %{y:.4e} MPa<extra></extra>'
                        ), row=1, col=1)
                        sigma_ids.append(len(fig.data)-1)
                        
                        # Strain vs time
                        fig.add_trace(go.Scatter(
                            x=t, y=eps, mode="lines", name=f"ε(t)", visible=False,
                            line=dict(color=COLORS['strain'], width=2.5),
                            hovertemplate='t = %{x:.3f} s<br>ε = %{y:.4f}<extra></extra>'
                        ), row=1, col=2)
                        eps_ids.append(len(fig.data)-1)
                        
                        # Hysteresis loop
                        fig.add_trace(go.Scatter(
                            x=eps, y=sigma, mode="lines", name=f"Hysteresis", visible=False,
                            line=dict(color=COLORS['hysteresis'], width=2.5),
                            hovertemplate='ε = %{x:.4f}<br>σ = %{y:.4e} MPa<extra></extra>'
                        ), row=2, col=1)
                        hyst_ids.append(len(fig.data)-1)
                        
                        # Table
                        sigma_max = np.max(np.abs(sigma))
                        period = 2*np.pi/omega
                        freq = omega/(2*np.pi)
                        loss_angle = np.arctan(B/A) * 180/np.pi
                        
                        fig.add_trace(go.Table(
                            header=dict(values=["<b>Parameter</b>", "<b>Value</b>"], **TABLE_HEADER_STYLE),
                            cells=dict(
                                values=[
                                    ["<b>ε₀</b>", "<b>ω</b>", "<b>f</b>", "<b>Period</b>",
                                     "<b>E<sub>re</sub></b>", "<b>E<sub>rg</sub></b>", "<b>τ<sub>R</sub></b>",
                                     "<b>σ<sub>max</sub></b>", "<b>A</b>", "<b>B</b>", "<b>C</b>", 
                                     "<b>Loss Angle</b>"],
                                    [f"{eps0:.3f}", f"{omega:.3f} rad/s", f"{freq:.3f} Hz", f"{period:.3f} s",
                                     f"{E_re:.2f} MPa", f"{E_rg:.2f} MPa", f"{tau_R:.2f} s",
                                     f"{sigma_max:.4e} MPa", f"{A:.4e} MPa", f"{B:.4e} MPa", f"{C:.4e} MPa",
                                     f"{loss_angle:.2f}°"]
                                ],
                                height=26,
                                **TABLE_CELL_STYLE
                            ),
                            visible=False
                        ), row=2, col=2)
                        table_ids.append(len(fig.data)-1)
    
    # Set default visibility
    default_key = (eps0_default, omega_default, E_re_default, E_rg_default, tau_R_default)
    default_idx = combo_keys.index(default_key)
    fig.data[sigma_ids[default_idx]].visible = True
    fig.data[eps_ids[default_idx]].visible = True
    fig.data[hyst_ids[default_idx]].visible = True
    fig.data[table_ids[default_idx]].visible = True
    
    # Apply axis styling
    fig.update_xaxes(title_text="<b>t [s]</b>", row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>σ(t) [MPa]</b>", row=1, col=1, **AXIS_STYLE)
    
    fig.update_xaxes(title_text="<b>t [s]</b>", row=1, col=2, **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>ε(t) [-]</b>", row=1, col=2, **AXIS_STYLE)
    
    fig.update_xaxes(title_text="<b>ε(t) [-]</b>", row=2, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text="<b>σ(t) [MPa]</b>", row=2, col=1, **AXIS_STYLE)
    
    fig.update_layout(
        title_text="<b>Standard Linear Solid (SLS) - Sinusoidal Strain Dashboard</b>",
        height=950, 
        width=1400,
        margin=dict(t=250, b=180, l=80, r=80),
        showlegend=False,
        **LAYOUT_DEFAULTS
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations'][:4]:
        annotation['font'] = dict(size=13, family='Arial, sans-serif', color=COLORS['stress'])
    
    def mask_for(eps0_sel, omega_sel, E_re_sel, E_rg_sel, tau_R_sel):
        vis = [False]*len(fig.data)
        idx = combo_keys.index((eps0_sel, omega_sel, E_re_sel, E_rg_sel, tau_R_sel))
        vis[sigma_ids[idx]] = True
        vis[eps_ids[idx]] = True
        vis[hyst_ids[idx]] = True
        vis[table_ids[idx]] = True
        return vis
    
    # Create slider steps
    steps_eps0 = [dict(method="update", args=[{"visible": mask_for(e0, omega_default, E_re_default, E_rg_default, tau_R_default)}], label=f"{e0:.3f}") for e0 in eps0_vals]
    steps_omega = [dict(method="update", args=[{"visible": mask_for(eps0_default, w, E_re_default, E_rg_default, tau_R_default)}], label=f"{w:.2f}") for w in omega_vals]
    steps_E_re = [dict(method="update", args=[{"visible": mask_for(eps0_default, omega_default, E, E_rg_default, tau_R_default)}], label=f"{E:.1f}") for E in E_re_vals]
    steps_E_rg = [dict(method="update", args=[{"visible": mask_for(eps0_default, omega_default, E_re_default, E, tau_R_default)}], label=f"{E:.1f}") for E in E_rg_vals]
    steps_tau_R = [dict(method="update", args=[{"visible": mask_for(eps0_default, omega_default, E_re_default, E_rg_default, tau)}], label=f"{tau:.1f}") for tau in tau_R_vals]
    
    # Mathematical formulation annotations
    formulation_annotations = [
        dict(text="<b>Mathematical Formulation</b>",
             x=0.5, y=1.17, xref="paper", yref="paper",
             xanchor="center", showarrow=False, font=ANNOTATION_FONTS['title']),
        dict(text="<b>Strain Input:</b> ε(t) = ε₀·sin(ωt)·h(t)",
             x=0.02, y=1.145, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
        dict(text="<b>Relaxation Modulus:</b> E<sub>r</sub>(t) = E<sub>re</sub> + E<sub>rg</sub>·exp(−t/τ<sub>R</sub>)",
             x=0.02, y=1.12, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
        dict(text="<b>Stress Response:</b> σ(t) = A·sin(ωt) + B·cos(ωt) + C·exp(−t/τ<sub>R</sub>)   |   A = ε₀(E<sub>re</sub> + E<sub>rg</sub>ω²τ<sub>R</sub>²/(1+ω²τ<sub>R</sub>²))   |   B = C = ε₀E<sub>rg</sub>ωτ<sub>R</sub>/(1+ω²τ<sub>R</sub>²) · (±1)",
             x=0.02, y=1.095, xref="paper", yref="paper",
             xanchor="left", showarrow=False, font=ANNOTATION_FONTS['text']),
    ]
    
    # Add sliders
    sliders = [
        dict(active=eps0_vals.index(eps0_default), yanchor="bottom", y=-0.02, xanchor="right", x=0.98,
             currentvalue=dict(prefix="<b>ε₀ = </b>", visible=True, xanchor="right",
                              font=dict(size=10, family='Arial, sans-serif', color=COLORS['stress'])),
             pad=dict(b=2, t=2), len=0.35, steps=steps_eps0, **SLIDER_STYLE),
        dict(active=omega_vals.index(omega_default), yanchor="bottom", y=-0.06, xanchor="right", x=0.98,
             currentvalue=dict(prefix="<b>ω = </b>", suffix=" rad/s", visible=True, xanchor="right",
                              font=dict(size=10, family='Arial, sans-serif', color=COLORS['stress'])),
             pad=dict(b=2, t=2), len=0.35, steps=steps_omega, **SLIDER_STYLE),
        dict(active=E_re_vals.index(E_re_default), yanchor="bottom", y=-0.10, xanchor="right", x=0.98,
             currentvalue=dict(prefix="<b>E<sub>re</sub> = </b>", suffix=" MPa", visible=True, xanchor="right",
                              font=dict(size=10, family='Arial, sans-serif', color=COLORS['stress'])),
             pad=dict(b=2, t=2), len=0.35, steps=steps_E_re, **SLIDER_STYLE),
        dict(active=E_rg_vals.index(E_rg_default), yanchor="bottom", y=-0.14, xanchor="right", x=0.98,
             currentvalue=dict(prefix="<b>E<sub>rg</sub> = </b>", suffix=" MPa", visible=True, xanchor="right",
                              font=dict(size=10, family='Arial, sans-serif', color=COLORS['stress'])),
             pad=dict(b=2, t=2), len=0.35, steps=steps_E_rg, **SLIDER_STYLE),
        dict(active=tau_R_vals.index(tau_R_default), yanchor="bottom", y=-0.18, xanchor="right", x=0.98,
             currentvalue=dict(prefix="<b>τ<sub>R</sub> = </b>", suffix=" s", visible=True, xanchor="right",
                              font=dict(size=10, family='Arial, sans-serif', color=COLORS['stress'])),
             pad=dict(b=2, t=2), len=0.35, steps=steps_tau_R, **SLIDER_STYLE),
    ]
    
    fig.update_layout(
        sliders=sliders,
        annotations=list(fig['layout']['annotations']) + formulation_annotations
    )
    return fig


if __name__ == "__main__":
    fig = build_sinusoidal_dashboard()
    fig.write_html("sls_sinusoidal_dashboard.html", include_plotlyjs="cdn", auto_open=False)
    print("[SUCCESS] Generated: sls_sinusoidal_dashboard.html")

