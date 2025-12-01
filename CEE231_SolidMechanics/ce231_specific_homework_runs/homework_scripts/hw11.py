import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import sys
import os

# Add parent directory to path to import plotly_templates
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from plotly_templates import (
    UCBerkeleyColors as Colors,
    get_axis_style,
    get_plot_layout_style,
)


# ---------------------------------------------------------------------------
# strain history ε(t)
# ---------------------------------------------------------------------------

def strain_history(t_val: float, a: float, a_prime: float, tau: float) -> float:
    """
    Prescribed strain history:
        ε(t) = a * t / τ,                             t < τ
        ε(t) = [a + a' (t - τ)/τ] * cos(2π (t-τ)/τ), t ≥ τ
    """
    if t_val < tau:
        return a * t_val / tau
    factor = a + a_prime * (t_val - tau) / tau
    return factor * np.cos(2.0 * np.pi * (t_val - tau) / tau)


def strain_history_vector(t: np.ndarray, a: float, a_prime: float, tau: float) -> np.ndarray:
    """Vectorized version of the strain history."""
    t = np.asarray(t)
    eps = np.empty_like(t, dtype=float)

    mask_early = t < tau
    mask_late = ~mask_early

    eps[mask_early] = a * t[mask_early] / tau
    if np.any(mask_late):
        factor = a + a_prime * (t[mask_late] - tau) / tau
        eps[mask_late] = factor * np.cos(2.0 * np.pi * (t[mask_late] - tau) / tau)

    return eps


# ---------------------------------------------------------------------------
# Newton solver for Δε̄^p_n from g(Δε̄^p_n) = 0
# ---------------------------------------------------------------------------

def solve_delta_epsp_bar(
    sigma_trial: float,
    epsp_bar_n: float,
    E: float,
    Y0: float,
    K: float,
    n_hard: float,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """
    Solve for Δε̄^p_n during a plastic step using the scalar equation:

        g(Δε̄^p_n) =
            |σ_trial| - E Δε̄^p_n
            - [Y0 + K (ε̄^p_n + Δε̄^p_n)^n_hard] = 0.

    Returns Δε̄^p_n ≥ 0.
    """
    abs_sigma_trial = abs(sigma_trial)

    # quick elastic check
    Y_n = Y0 + K * (epsp_bar_n ** n_hard)
    f_trial = abs_sigma_trial - Y_n
    if f_trial <= 0.0:
        return 0.0

    # initial guess based on an effective tangent (robust for small ε̄^p_n)
    eps_shift = max(epsp_bar_n, 1e-12)
    denom_guess = E + K * n_hard * (eps_shift ** (n_hard - 1.0))
    delta = max(f_trial / denom_guess, 0.0)

    for _ in range(max_iter):
        eps_bar_curr = epsp_bar_n + delta

        g_val = (
            abs_sigma_trial
            - E * delta
            - (Y0 + K * (eps_bar_curr ** n_hard))
        )

        if abs(g_val) < tol * max(Y0, 1.0):
            break

        if eps_bar_curr <= 0.0:
            g_prime = -E
        else:
            g_prime = -E - K * n_hard * (eps_bar_curr ** (n_hard - 1.0))

        # safeguard against non-negative derivative
        if g_prime >= 0.0:
            g_prime = -E

        delta_new = delta - g_val / g_prime

        # simple clipping to keep iteration stable
        if delta_new < 0.0:
            delta_new = 0.0
        upper_bound = 10.0 * f_trial / E
        if delta_new > upper_bound:
            delta_new = upper_bound

        delta = delta_new

    return max(delta, 0.0)


# ---------------------------------------------------------------------------
# backward-euler integration for 1D power-law hardening
# ---------------------------------------------------------------------------

def integrate_power_law_backward_euler(
    E: float,
    Y0: float,
    K: float,
    n_hard: float,
    a: float,
    a_prime: float,
    tau: float,
    n_steps_per_tau: int,
):
    """
    Integrate σ_n, ε_n, ε^p_n, ε̄^p_n over t ∈ (0, 6τ) using a backward Euler step.
    Naming follows A&G: index n, trial state, and Δε̄^p_n.
    """
    t_end = 6.0 * tau
    n_steps = 6 * n_steps_per_tau
    t = np.linspace(0.0, t_end, n_steps + 1)

    # state variables
    eps_n = strain_history_vector(t, a, a_prime, tau)
    sigma_n = np.zeros_like(t)
    eps_p_n = np.zeros_like(t)
    epsp_bar_n = np.zeros_like(t)

    # initial flow stress: Y_0 at ε̄^p_0 = 0
    Y_n = Y0

    for n in range(n_steps):
        # strain increment Δε_n
        delta_eps_n = eps_n[n + 1] - eps_n[n]

        # trial state: plastic flow frozen
        sigma_trial = sigma_n[n] + E * delta_eps_n
        eps_p_trial = eps_p_n[n]
        Y_trial = Y_n
        f_trial = abs(sigma_trial) - Y_trial

        if f_trial <= 0.0:
            # elastic step: accept trial state
            sigma_n[n + 1] = sigma_trial
            eps_p_n[n + 1] = eps_p_trial
            epsp_bar_n[n + 1] = epsp_bar_n[n]
            Y_n = Y_trial
        else:
            # plastic step: solve for Δε̄^p_n
            delta_epsp_bar_n = solve_delta_epsp_bar(
                sigma_trial, epsp_bar_n[n], E, Y0, K, n_hard
            )

            # update equivalent plastic strain
            epsp_bar_n[n + 1] = epsp_bar_n[n] + delta_epsp_bar_n

            # plastic strain increment Δε^p_n = Δε̄^p_n sign(σ_trial)
            sign_trial = np.sign(sigma_trial) if sigma_trial != 0.0 else 1.0
            eps_p_n[n + 1] = eps_p_n[n] + delta_epsp_bar_n * sign_trial

            # updated flow stress Y_{n+1}
            Y_n = Y0 + K * (epsp_bar_n[n + 1] ** n_hard)

            # updated stress σ_{n+1} using yield condition
            sigma_n[n + 1] = sign_trial * Y_n

    return t, eps_n, sigma_n, eps_p_n, epsp_bar_n


# ---------------------------------------------------------------------------
# plotly figure generators
# ---------------------------------------------------------------------------

def create_stress_strain_figure(eps_n, sigma_n):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps_n,
            y=sigma_n / 1e6,  # MPa
            mode="lines",
            line=dict(width=3, color=Colors.BERKELEY_BLUE),
            name="σ(ε)",
        )
    )
    axis_style = get_axis_style()
    fig.update_layout(
        title="Stress–Strain Response: σ(ε)",
        xaxis_title="Strain ε [–]",
        yaxis_title="Stress σ [MPa]",
        template="plotly_white",
        margin=dict(l=60, r=10, t=60, b=50),
        font=dict(family="Arial", size=14),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def create_epspbar_time_figure(t, epsp_bar_n):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=epsp_bar_n,
            mode="lines",
            line=dict(width=3, color=Colors.FOUNDERS_ROCK),
            name="ε̄ᵖ(t)",
        )
    )
    axis_style = get_axis_style()
    fig.update_layout(
        title="Equivalent Plastic Strain vs Time: ε̄ᵖ(t)",
        xaxis_title="Time t [s]",
        yaxis_title="Equivalent plastic strain ε̄ᵖ [–]",
        template="plotly_white",
        margin=dict(l=60, r=10, t=60, b=50),
        font=dict(family="Arial", size=14),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def create_epsp_time_figure(t, eps_p_n):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=eps_p_n,
            mode="lines",
            line=dict(width=3, color=Colors.CALIFORNIA_GOLD),
            name="εᵖ(t)",
        )
    )
    axis_style = get_axis_style()
    fig.update_layout(
        title="Plastic Strain vs Time: εᵖ(t)",
        xaxis_title="Time t [s]",
        yaxis_title="Plastic strain εᵖ [–]",
        template="plotly_white",
        margin=dict(l=60, r=10, t=60, b=50),
        font=dict(family="Arial", size=14),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "1D Power-law Plasticity – Backward Euler"

# Add custom CSS for Berkeley-themed slider styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom slider styling with Berkeley colors */
            .rc-slider-track {
                background-color: #003262 !important;
                height: 6px !important;
            }
            .rc-slider-handle {
                border-color: #003262 !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 8px rgba(0, 50, 98, 0.4) !important;
            }
            .rc-slider-handle:hover {
                border-color: #FDB515 !important;
                box-shadow: 0 3px 12px rgba(253, 181, 21, 0.6) !important;
            }
            .rc-slider-handle:active {
                border-color: #FDB515 !important;
                box-shadow: 0 4px 16px rgba(253, 181, 21, 0.8) !important;
            }
            .rc-slider-rail {
                background-color: rgba(200, 200, 200, 0.3) !important;
                height: 6px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def slider_with_label(id_, label, min_, max_, step, value, unit=""):
    return html.Div(
        style={"marginBottom": "24px"},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "8px",
                },
                children=[
                    html.Span(
                        f"{label}",
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "color": Colors.TEXT_DARK,
                            "letterSpacing": "0.3px",
                            "fontFamily": "Arial, sans-serif",
                        },
                    ),
                    html.Span(
                        id=f"{id_}-value",
                        style={
                            "fontWeight": "700",
                            "fontSize": "14px",
                            "color": Colors.BERKELEY_BLUE,
                            "backgroundColor": Colors.BG_LIGHT,
                            "padding": "4px 10px",
                            "borderRadius": "12px",
                            "minWidth": "60px",
                            "textAlign": "center",
                            "fontFamily": "Arial, sans-serif",
                            "border": f"1px solid {Colors.BERKELEY_BLUE}",
                        },
                    ),
                ],
            ),
            dcc.Slider(
                id=id_,
                min=min_,
                max=max_,
                step=step,
                value=value,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "marginTop": "4px",
                    "fontSize": "10px",
                    "color": Colors.TEXT_LIGHT,
                    "fontFamily": "Arial, sans-serif",
                },
                children=[
                    html.Span(f"{min_:.2f}" if min_ < 1 else f"{int(min_)}"),
                    html.Span(f"{unit}", style={"fontStyle": "italic"}),
                    html.Span(f"{max_:.2f}" if max_ < 1 else f"{int(max_)}"),
                ],
            ),
        ],
    )


app.layout = html.Div(
    style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "30px 20px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": Colors.BG_LIGHT,
        "minHeight": "100vh",
    },
    children=[
        html.Div(
            style={
                "backgroundColor": Colors.BG_WHITE,
                "borderRadius": "12px",
                "padding": "40px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
                "border": f"2px solid {Colors.BERKELEY_BLUE}",
            },
            children=[
                html.Div(
                    style={
                        "textAlign": "center",
                        "marginBottom": "40px",
                        "paddingBottom": "30px",
                        "borderBottom": f"3px solid {Colors.CALIFORNIA_GOLD}",
                    },
                    children=[
                        html.H1(
                            "1D Power-law Plasticity",
                            style={
                                "textAlign": "center",
                                "marginBottom": "8px",
                                "fontSize": "32px",
                                "fontWeight": "700",
                                "color": Colors.BERKELEY_BLUE,
                                "fontFamily": "Arial, sans-serif",
                            },
                        ),
                        html.P(
                            "Backward Euler Integration",
                            style={
                                "textAlign": "center",
                                "marginBottom": "15px",
                                "fontSize": "18px",
                                "color": Colors.TEXT_LIGHT,
                                "fontWeight": "500",
                                "fontFamily": "Arial, sans-serif",
                            },
                        ),
                        html.P(
                            "Rate-independent uniaxial plasticity with power-law isotropic hardening. "
                            "Material and loading parameters can be adjusted with the sliders.",
                            style={
                                "textAlign": "center",
                                "marginBottom": "0",
                                "fontSize": "14px",
                                "color": Colors.TEXT_LIGHT,
                                "maxWidth": "700px",
                                "margin": "0 auto",
                                "fontFamily": "Arial, sans-serif",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
                    children=[
                        # left column – controls
                        html.Div(
                            style={
                                "flex": "0 0 360px",
                                "backgroundColor": Colors.BG_LIGHT,
                                "padding": "28px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                                "border": f"1px solid {Colors.BERKELEY_BLUE}",
                            },
                            children=[
                                html.H3(
                                    "Material Parameters",
                                    style={
                                        "marginTop": 0,
                                        "marginBottom": "20px",
                                        "fontSize": "20px",
                                        "fontWeight": "700",
                                        "color": Colors.BERKELEY_BLUE,
                                        "paddingBottom": "12px",
                                        "borderBottom": f"2px solid {Colors.BERKELEY_BLUE}",
                                        "fontFamily": "Arial, sans-serif",
                                    },
                        ),
                        slider_with_label(
                            "E-slider",
                            "Young's modulus E",
                            min_=100.0,
                            max_=300.0,
                            step=10.0,
                            value=200.0,
                            unit="GPa",
                        ),
                        slider_with_label(
                            "Y0-slider",
                            "Initial yield strength Y₀",
                            min_=100.0,
                            max_=400.0,
                            step=10.0,
                            value=200.0,
                            unit="MPa",
                        ),
                        slider_with_label(
                            "K-slider",
                            "Hardening modulus K",
                            min_=0.0,
                            max_=600.0,
                            step=20.0,
                            value=300.0,
                            unit="MPa",
                        ),
                        slider_with_label(
                            "n-slider",
                            "Hardening exponent n",
                            min_=0.1,
                            max_=0.7,
                            step=0.05,
                            value=0.3,
                            unit="–",
                        ),
                        html.H3(
                                    "Loading Parameters",
                                    style={
                                        "marginTop": "28px",
                                        "marginBottom": "20px",
                                        "fontSize": "20px",
                                        "fontWeight": "700",
                                        "color": Colors.BERKELEY_BLUE,
                                        "paddingTop": "20px",
                                        "paddingBottom": "12px",
                                        "borderTop": f"2px solid {Colors.GRID}",
                                        "borderBottom": f"2px solid {Colors.CALIFORNIA_GOLD}",
                                        "fontFamily": "Arial, sans-serif",
                                    },
                        ),
                        slider_with_label(
                            "a-slider",
                            "Ramp amplitude a",
                            min_=0.01,
                            max_=0.10,
                            step=0.005,
                            value=0.05,
                            unit="–",
                        ),
                        slider_with_label(
                            "aprime-slider",
                            "Oscillation growth a′",
                            min_=0.0,
                            max_=0.05,
                            step=0.0025,
                            value=0.01,
                            unit="–",
                        ),
                        slider_with_label(
                            "tau-slider",
                            "Characteristic time τ",
                            min_=5.0,
                            max_=20.0,
                            step=1.0,
                            value=10.0,
                            unit="s",
                        ),
                        slider_with_label(
                            "nsteps-slider",
                                    "Time resolution",
                            min_=50,
                            max_=400,
                            step=10,
                            value=200,
                            unit="steps/τ",
                        ),
                    ],
                ),
                # right column – figures
                html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "600px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "30px",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "backgroundColor": Colors.BG_WHITE,
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                                        "border": f"1px solid {Colors.GRID}",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="stress-strain-graph",
                                            config={"displayModeBar": False},
                                            style={"height": "100%"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": Colors.BG_WHITE,
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                                        "border": f"1px solid {Colors.GRID}",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="epspbar-time-graph",
                                            config={"displayModeBar": False},
                                            style={"height": "100%"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": Colors.BG_WHITE,
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                                        "border": f"1px solid {Colors.GRID}",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="epsp-time-graph",
                                            config={"displayModeBar": False},
                                            style={"height": "100%"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "marginTop": "30px",
                        "paddingTop": "20px",
                        "borderTop": f"2px solid {Colors.GRID}",
                        "textAlign": "center",
                    },
                    children=[
                        html.P(
                            "Note: Stresses are plotted in MPa; internal computations use SI units.",
                            style={
                                "fontSize": "13px",
                                "color": Colors.TEXT_LIGHT,
                                "fontStyle": "italic",
                                "margin": 0,
                                "fontFamily": "Arial, sans-serif",
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Callbacks to update slider value displays
@app.callback(Output("E-slider-value", "children"), Input("E-slider", "value"))
def update_E_value(value):
    return f"{value:.0f}"

@app.callback(Output("Y0-slider-value", "children"), Input("Y0-slider", "value"))
def update_Y0_value(value):
    return f"{value:.0f}"

@app.callback(Output("K-slider-value", "children"), Input("K-slider", "value"))
def update_K_value(value):
    return f"{value:.0f}"

@app.callback(Output("n-slider-value", "children"), Input("n-slider", "value"))
def update_n_value(value):
    return f"{value:.2f}"

@app.callback(Output("a-slider-value", "children"), Input("a-slider", "value"))
def update_a_value(value):
    return f"{value:.3f}"

@app.callback(Output("aprime-slider-value", "children"), Input("aprime-slider", "value"))
def update_aprime_value(value):
    return f"{value:.4f}"

@app.callback(Output("tau-slider-value", "children"), Input("tau-slider", "value"))
def update_tau_value(value):
    return f"{value:.0f}"

@app.callback(Output("nsteps-slider-value", "children"), Input("nsteps-slider", "value"))
def update_nsteps_value(value):
    return f"{value:.0f}"

@app.callback(
    Output("stress-strain-graph", "figure"),
    Output("epspbar-time-graph", "figure"),
    Output("epsp-time-graph", "figure"),
    Input("E-slider", "value"),
    Input("Y0-slider", "value"),
    Input("K-slider", "value"),
    Input("n-slider", "value"),
    Input("a-slider", "value"),
    Input("aprime-slider", "value"),
    Input("tau-slider", "value"),
    Input("nsteps-slider", "value"),
    prevent_initial_call=False,
)
def update_plots(E_GPa, Y0_MPa, K_MPa, n_hard, a, a_prime, tau, n_steps_per_tau):
    # convert to SI units
    E = E_GPa * 1e9
    Y0 = Y0_MPa * 1e6
    K = K_MPa * 1e6

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

    # Ensure data is valid (not empty)
    if len(t) == 0 or len(eps_n) == 0:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(**get_plot_layout_style(height=400))
        return empty_fig, empty_fig, empty_fig

    fig_ss = create_stress_strain_figure(eps_n, sigma_n)
    fig_epspbar = create_epspbar_time_figure(t, epsp_bar_n)
    fig_epsp = create_epsp_time_figure(t, eps_p_n)

    return fig_ss, fig_epspbar, fig_epsp


# ---------------------------------------------------------------------------

# Expose server for deployment
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
