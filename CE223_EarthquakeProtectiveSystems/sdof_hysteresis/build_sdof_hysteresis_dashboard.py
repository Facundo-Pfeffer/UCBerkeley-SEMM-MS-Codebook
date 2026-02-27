from __future__ import annotations

"""
CE223 – SDOF hysteresis comparison (Models A, B, C) for prescribed
harmonic displacement u(t) = u0 sin(omega t).

Run from this directory:
    python build_sdof_hysteresis_dashboard.py

Outputs (under harmonic_hysteresis_dashboard/):
    - sdof_hysteresis_loops.html      : f–u loops for the three models
    - sdof_hysteresis_dashboard.html  : mini-report with theory summary and plots
    - config.json                     : parameters used in the run
"""

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from sdof_frequency_response import (
    load_peer_at2_to_mps2,
    sdof_frequency_response_for_models,
)
from sdof_hysteresis_plotly import create_sdof_hysteresis_figure, save_figure_html


BASE_DIR = Path(__file__).resolve().parent

# For CE223, match the CE225/CEE231 pattern:
# course_root/highlighted_htmls/*.html so that ../../assets/... resolves
# correctly after deployment (see DEVELOPMENT_NOTES.md).
OUTPUT_DIR = BASE_DIR.parent / "highlighted_htmls"


def sdof_parameters() -> dict:
    """Return baseline physical and model parameters."""
    m = 1.0  # kg
    omega_n = math.pi  # rad/s
    k = m * omega_n**2  # N/m
    u0 = 1.0

    # Model-specific parameters at resonance (from CE223 notes)
    c = 0.30 * math.pi  # viscous dashpot (Model A)
    delta = 0.30  # hysteretic loss factor (Model B)
    alpha = 0.30
    c_alpha = 1.473 * math.pi

    return dict(
        m=m,
        omega_n=omega_n,
        k=k,
        u0=u0,
        c=c,
        delta=delta,
        alpha=alpha,
        c_alpha=c_alpha,
    )


def harmonic_time_history(omega: float, u0: float, n_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return time array and prescribed displacement u(t) = u0 sin(omega t) over one cycle."""
    T = 2.0 * math.pi / omega
    t = np.linspace(0.0, T, n_points, endpoint=False)
    u = u0 * np.sin(omega * t)
    return t, u


def loop_area_poly(u: np.ndarray, f: np.ndarray) -> float:
    """
    Polygonal approximation of loop area in the (u, f) plane, mirroring
    MATLAB's polyarea for one closed cycle.

    This is the numerical EDC definition requested in the assignment:
    sample one full cycle, close the loop, and apply the shoelace formula.
    """
    u = np.asarray(u, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    n = u.size
    if n < 3:
        return 0.0

    # Ensure the loop is closed before applying the shoelace formula.
    if not (math.isclose(u[0], u[-1]) and math.isclose(f[0], f[-1])):
        u = np.concatenate([u, u[:1]])
        f = np.concatenate([f, f[:1]])

    x = u
    y = f
    area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return abs(area)


def model_forces(
    t: np.ndarray,
    u: np.ndarray,
    *,
    omega: float,
    k: float,
    c: float,
    delta: float,
    alpha: float,
    c_alpha: float,
) -> dict[str, np.ndarray]:
    """Compute f(t) for Models A, B, C at a given frequency."""
    u0 = float(np.max(np.abs(u)))
    du_dt = u0 * omega * np.cos(omega * t)

    # Model A: Kelvin–Voigt
    fA = k * u + c * du_dt

    # Model B: hysteretic idealization
    fB = k * u0 * np.sin(omega * t) + k * delta * u0 * np.cos(omega * t)

    # Model C: fractional Kelvin–Voigt with baseline stiffness k
    amp = c_alpha * omega**alpha
    shift = math.pi * alpha / 2.0
    frac_term = amp * np.sin(omega * t + shift)
    fC = k * u + frac_term

    return {"Model A (viscous)": fA, "Model B (hysteretic)": fB, "Model C (fractional)": fC}


def analytic_K1_K2_EDC(omega: float, params: dict) -> dict[str, dict[str, float]]:
    """Analytical K1, K2, and EDC for Models A, B, C at frequency omega."""
    k = params["k"]
    c = params["c"]
    delta = params["delta"]
    alpha = params["alpha"]
    c_alpha = params["c_alpha"]
    u0 = params["u0"]

    # Model A
    K1_A = k
    K2_A = c * omega
    EDC_A = math.pi * K2_A * u0**2

    # Model B
    K1_B = k
    K2_B = k * delta
    EDC_B = math.pi * K2_B * u0**2

    # Model C
    K1_C = k + c_alpha * omega**alpha * math.cos(math.pi * alpha / 2.0)
    K2_C = c_alpha * omega**alpha * math.sin(math.pi * alpha / 2.0)
    EDC_C = math.pi * K2_C * u0**2

    return {
        "Model A (viscous)": {"K1": K1_A, "K2": K2_A, "EDC": EDC_A},
        "Model B (hysteretic)": {"K1": K1_B, "K2": K2_B, "EDC": EDC_B},
        "Model C (fractional)": {"K1": K1_C, "K2": K2_C, "EDC": EDC_C},
    }


def edc_vs_frequency_numeric(
    params: dict,
    n_freq: int = 80,
    n_time: int = 2000,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Numerical EDC(ω) curves for Models A, B, C using a polyarea-style
    loop integration over one full cycle, as in the assignment:

      - 0.5 ω_n ≤ ω ≤ 2 ω_n,
      - at least 1000 time points per cycle (we use n_time ≥ 2000),
      - loop explicitly closed before applying the shoelace formula.
    """
    omega_n = params["omega_n"]
    k = params["k"]
    c = params["c"]
    delta = params["delta"]
    alpha = params["alpha"]
    c_alpha = params["c_alpha"]
    u0 = params["u0"]

    if n_time < 1000:
        n_time = 1000

    omega = np.linspace(0.5 * omega_n, 2.0 * omega_n, n_freq)

    edc_A = np.zeros_like(omega)
    edc_B = np.zeros_like(omega)
    edc_C = np.zeros_like(omega)

    for i, w in enumerate(omega):
        t, u = harmonic_time_history(w, u0, n_points=n_time)
        forces = model_forces(
            t,
            u,
            omega=w,
            k=k,
            c=c,
            delta=delta,
            alpha=alpha,
            c_alpha=c_alpha,
        )
        edc_A[i] = loop_area_poly(u, forces["Model A (viscous)"])
        edc_B[i] = loop_area_poly(u, forces["Model B (hysteretic)"])
        edc_C[i] = loop_area_poly(u, forces["Model C (fractional)"])

    return omega, {
        "Model A (viscous)": edc_A,
        "Model B (hysteretic)": edc_B,
        "Model C (fractional)": edc_C,
    }


def build_hysteresis_loops_html() -> None:
    """Create a simple f–u comparison plot and save it as standalone HTML."""
    params = sdof_parameters()
    omega = params["omega_n"]
    t, u = harmonic_time_history(omega, params["u0"])
    forces = model_forces(
        t,
        u,
        omega=omega,
        k=params["k"],
        c=params["c"],
        delta=params["delta"],
        alpha=params["alpha"],
        c_alpha=params["c_alpha"],
    )

    fig = create_sdof_hysteresis_figure(
        u={
            "Model A (viscous)": u,
            "Model B (hysteretic)": u,
            "Model C (fractional)": u,
        },
        f=forces,
        title="SDOF hysteresis comparison at ω = ωₙ",
    )
    out_path = OUTPUT_DIR / "sdof_hysteresis_loops.html"
    save_figure_html(fig, out_path)
    print(f"Wrote {out_path}")


def build_dashboard_html() -> None:
    """Build a compact dashboard with theory summary and hysteresis plot."""
    params = sdof_parameters()
    omega = params["omega_n"]
    t, u = harmonic_time_history(omega, params["u0"])
    forces = model_forces(
        t,
        u,
        omega=omega,
        k=params["k"],
        c=params["c"],
        delta=params["delta"],
        alpha=params["alpha"],
        c_alpha=params["c_alpha"],
    )
    metrics = analytic_K1_K2_EDC(omega, params)
    omega_grid, edc_curves = edc_vs_frequency_numeric(params)

    fig = create_sdof_hysteresis_figure(
        u={
            "Model A (viscous)": u,
            "Model B (hysteretic)": u,
            "Model C (fractional)": u,
        },
        f=forces,
        title="SDOF hysteresis comparison at ω = ωₙ",
    )
    fig_div = to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )

    # EDC(ω) figure using analytic expressions (no polyarea).
    fig_edc = go.Figure()
    omega_ratio = omega_grid / params["omega_n"]

    def _add_edc_trace(label: str, y_vals: np.ndarray, color: str, legendrank: int) -> None:
        fig_edc.add_trace(
            go.Scatter(
                x=omega_ratio,
                y=y_vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.8),
                legendrank=legendrank,
            )
        )

    _add_edc_trace("Model A (viscous)", edc_curves["Model A (viscous)"], "rgb(0, 50, 98)", 1)
    _add_edc_trace("Model B (hysteretic)", edc_curves["Model B (hysteretic)"], "rgb(22, 163, 74)", 2)
    _add_edc_trace("Model C (fractional)", edc_curves["Model C (fractional)"], "rgb(197, 48, 48)", 3)

    fig_edc.update_layout(
        template="plotly_white",
        autosize=True,
        height=480,
        title=dict(
            text="EDC(ω) vs frequency ratio ω / ωₙ",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1e293b"),
        ),
        xaxis=dict(
            title="Frequency ratio ω / ωₙ",
            gridcolor="rgba(0, 0, 0, 0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0, 0, 0)",
        ),
        yaxis=dict(
            title="EDC(ω) [N·m]",
            gridcolor="rgba(0, 0, 0, 0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0, 0, 0)",
        ),
        # Extra top space plus legend moved to bottom to avoid overlapping
        margin=dict(l=70, r=40, t=80, b=90),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.18,
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )
    fig_edc_div = to_html(
        fig_edc,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )

    # Frequency-domain earthquake response (Models A, B, C under Kobe KBU090)
    gm_path = (
        BASE_DIR.parent / "input_ground_motion" / "RSN1108_KOBE_KBU090.AT2"
    )
    ug_ddot, dt = load_peer_at2_to_mps2(gm_path)
    t_gm, responses = sdof_frequency_response_for_models(ug_ddot, dt, params)

    # Build 4-panel figure: u(t), u̇(t), ü_abs(t), ü_g(t), with all models
    fig_resp = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Relative displacement u(t)",
            "Relative velocity u̇(t)",
            "Absolute acceleration ü_abs(t)",
            "Ground acceleration ü_g(t)",
        ),
    )

    model_colors = {
        "Model A (viscous)": "rgb(0, 50, 98)",
        "Model B (hysteretic)": "rgb(22, 163, 74)",
        "Model C (fractional)": "rgb(197, 48, 48)",
    }

    for label, resp in responses.items():
        u_t = resp[:, 0]
        v_t = resp[:, 1]
        a_abs_t = resp[:, 2]
        color = model_colors.get(label, "rgb(0,0,0)")

        fig_resp.add_trace(
            go.Scatter(
                x=t_gm,
                y=u_t,
                mode="lines",
                name=f"{label} – u(t)",
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )
        fig_resp.add_trace(
            go.Scatter(
                x=t_gm,
                y=v_t,
                mode="lines",
                name=f"{label} – u̇(t)",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig_resp.add_trace(
            go.Scatter(
                x=t_gm,
                y=a_abs_t,
                mode="lines",
                name=f"{label} – ü_abs(t)",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig_resp.add_trace(
        go.Scatter(
            x=t_gm,
            y=ug_ddot,
            mode="lines",
            name="Ground – ü_g(t) [m/s²]",
            line=dict(color="rgb(0,0,0)", width=2.2, dash="dot"),
            showlegend=True,
        ),
        row=4,
        col=1,
    )

    fig_resp.update_layout(
        template="plotly_white",
        autosize=True,
        height=720,
        title=dict(
            text="Frequency-domain earthquake response (Kobe KBU090)",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1e293b"),
        ),
        legend=dict(
            orientation="h",
            y=-0.12,
            yanchor="top",
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=13, family="Arial", color="#111827"),
        ),
        margin=dict(l=70, r=25, t=80, b=90),
    )
    fig_resp.update_xaxes(title_text="Time [s]", row=4, col=1)
    fig_resp.update_yaxes(title_text="u(t) [m]", row=1, col=1)
    fig_resp.update_yaxes(title_text="u̇(t) [m/s]", row=2, col=1)
    fig_resp.update_yaxes(title_text="ü_abs(t) [m/s²]", row=3, col=1)
    fig_resp.update_yaxes(title_text="ü_g(t) [m/s²]", row=4, col=1)

    fig_resp_div = to_html(
        fig_resp,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )

    rows_html = []
    for label, vals in metrics.items():
        rows_html.append(
            f"<tr><td>{label}</td>"
            f"<td>{vals['K1']:.4f}</td>"
            f"<td>{vals['K2']:.4f}</td>"
            f"<td>{vals['EDC']:.4f}</td></tr>"
        )
    summary_rows = "\n".join(rows_html)

    template = r"""<!DOCTYPE HTML>
<!--
  Phantom by HTML5 UP (site-wide template)
-->
<html>
  <head>
    <title>SDOF hysteresis – CE223</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
    <meta name="description" content="SDOF hysteresis comparison for three damping models in CE223." />
    <link rel="stylesheet" href="../../assets/css/main.css" />
    <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>

    <!-- MathJax for LaTeX rendering (match CEE225 highlighted pages) -->
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>

    <!-- Plotly + small style tweak: remove borders around modebar buttons -->
    <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
    <style>
      .modebar-btn {
        border: none !important;
        box-shadow: none !important;
      }
      .modebar {
        box-shadow: none !important;
      }
      .plot-embed {
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.5rem;
        background: #ffffff;
        margin-bottom: 1.5rem;
      }
      .plot-embed .plotly-graph-div {
        width: 100% !important;
      }
      /* SDOF explainer styling */
      .sdof-box {
        margin-bottom: 1.75rem;
      }
      .sdof-box h3,
      .sdof-box h4 {
        margin-bottom: 0.75rem;
      }
      .sdof-box p {
        margin-bottom: 0.85rem;
      }
      .sdof-box ol,
      .sdof-box ul {
        margin-left: 1.5rem;
        margin-bottom: 0.75rem;
      }
      .sdof-box li {
        margin-bottom: 0.4rem;
      }
    </style>
  </head>
  <body class="is-preload">
    <!-- Wrapper -->
    <div id="wrapper">

      <!-- Header -->
      <header id="header">
        <div class="inner">
          <!-- Logo (match CE225 header style, link back to portfolio home) -->
          <a href="../../index.html" class="logo">
            <span class="symbol"><img src="../../images/CE223/CE223_Thumbnail.png" alt="CE223 thumbnail" /></span>
            <span class="title">CE223 – Earthquake Protective Systems</span>
          </a>
          <!-- Navigation will be generated by navigation.js -->
        </div>
      </header>

      <!-- Main -->
      <div id="main">
        <div class="inner">
          <h1>SDOF hysteresis – Models A, B, and C</h1>
          <p style="font-size: 1.1em; color: #6b7280; margin-bottom: 2rem;">
            Single-degree-of-freedom oscillator with prescribed harmonic displacement
            \(u(t) = u_0 \sin(\omega t)\). This page explains three common damping
            idealizations and shows how they affect the force–displacement loop shape,
            the energy dissipated per cycle, and the effective stiffness felt by the
            structure.
          </p>

          <section class="box sdof-box">
            <h3 style="letter-spacing: 0.15em; font-size: 0.9em; text-transform: uppercase; color: #6b7280;">
              Key questions this page answers
            </h3>
            <p>
              The visualizations and formulas below are organized to answer four practical questions:
            </p>
            <ol>
              <li><strong>How do different damping models change the hysteresis loop?</strong>
                We compare the force–displacement loops of three models for the same imposed motion.</li>
              <li><strong>How much energy is dissipated per cycle?</strong>
                We compute the loop area (EDC) numerically and track how it varies with loading frequency.</li>
              <li><strong>What do “storage” and “loss” stiffness mean?</strong>
                We introduce \(K_1\) and \(K_2\) as in‑phase and out‑of‑phase stiffness, linked to loop tilt and area.</li>
              <li><strong>When is a fractional model useful?</strong>
                We show how the fractional Kelvin–Voigt model can interpolate between purely viscous and purely hysteretic behavior.</li>
            </ol>
          </section>

          <section class="box sdof-box">
            <h3>Problem setting and notation</h3>
            <p>
              The analysis is performed in relative coordinates \(u(t)\) (m), with internal
              resisting force \(f(t)\) (N). The internal force is decomposed as
              \(f(t) = f_r(t) + f_d(t)\), where:
            </p>
            <ul>
              <li>\(f_r(t) = k\,u(t)\) is the elastic restoring force, with stiffness
                \(k\) (N/m).</li>
              <li>\(f_d(t)\) is the dissipative force, specified by the damping model.</li>
            </ul>
            <p>
              The baseline parameters are:
              \[
              m = 1~\mathrm{kg},\qquad
              \omega_n = \pi~\mathrm{rad/s},\qquad
              k = m\omega_n^2 = \pi^2~\mathrm{N/m}.
              \]
            </p>
            <p>
              In Part (a), the displacement is prescribed in steady state as
              \[
              u(t) = u_0 \sin(\omega t),\qquad
              u_0 = 1~\mathrm{m},
              \]
              with circular frequency \(\omega\) (rad/s). One cycle has duration
              \(T = 2\pi/\omega\). The hysteresis loop is the parametric curve
              \(\bigl(u(t), f(t)\bigr)\) for \(t \in [0, T]\). Its enclosed area
              equals the closed-loop work
              \(\mathrm{EDC} = \oint f\,du\) (J), representing the energy dissipated
              per cycle, while the overall tilt of the loop reflects the in-phase
              stiffness that stores recoverable strain energy.
            </p>
          </section>

          <section class="box sdof-box">
            <h3>Harmonic decomposition and dynamic stiffness</h3>
            <p>
              For the imposed motion \(u(t) = u_0 \sin(\omega t)\) with
              \(\dot{u}(t) = u_0 \omega \cos(\omega t)\), any linear model produces
              a steady-state force at the same frequency that can be written uniquely as
              \[
              f(t) = K_1(\omega)\,u_0 \sin(\omega t) + K_2(\omega)\,u_0 \cos(\omega t),
              \]
              which defines:
            </p>
            <ul>
              <li>\(K_1(\omega)\) (N/m): storage stiffness, multiplying the in-phase term
                  \(\sin(\omega t)\),</li>
              <li>\(K_2(\omega)\) (N/m): loss stiffness, multiplying the quadrature term
                  \(\cos(\omega t)\), in phase with \(\dot{u}(t)\).</li>
            </ul>
            <p>
              Introducing the complex dynamic stiffness
              \[
              \hat{R}(\omega) = K_1(\omega) + \mathrm{i}K_2(\omega),\qquad
              F(\omega) = \hat{R}(\omega)\,U(\omega),
              \]
              aligns this time-domain decomposition with the Fourier-domain representation,
              where \(U(\omega)\) and \(F(\omega)\) are the Fourier transforms of
              \(u(t)\) and \(f(t)\). For real-valued time histories the spectra satisfy
              \[
              U(-\omega) = \overline{U(\omega)},\qquad
              F(-\omega) = \overline{F(\omega)},
              \]
              which implies the conjugate-symmetry requirement
              \(\hat{R}(-\omega) = \overline{\hat{R}(\omega)}\). This condition is
              enforced explicitly in the frequency-domain forms of Models B and C in
              the earthquake-response scripts by using \(\operatorname{sgn}(\omega)\).
            </p>
          </section>

          <section class="box sdof-box">
            <h3>Damping models under prescribed motion</h3>
            <ol>
              <li>
                <strong>Model A – Kelvin–Voigt (viscous)</strong><br />
                Constitutive law:
                \[
                f(t) = k\,u(t) + c\,\dot{u}(t),
                \]
                with viscous coefficient \(c\) (N·s/m). For the imposed motion,
                \[
                f(t) = k\,u_0 \sin(\omega t) + (c\omega)\,u_0 \cos(\omega t),
                \]
                so
                \[
                K_1^{(A)}(\omega) = k,\qquad
                K_2^{(A)}(\omega) = c\omega.
                \]
                In the frequency domain,
                \[
                \hat{R}^{(A)}(\omega) = k + \mathrm{i}c\omega,
                \]
                which automatically satisfies
                \(\hat{R}^{(A)}(-\omega) = \overline{\hat{R}^{(A)}(\omega)}\).
              </li>
              <li>
                <strong>Model B – hysteretic (structural, harmonic idealization)</strong><br />
                Harmonic idealization:
                \[
                f(t) = k\,u_0 \sin(\omega t) + k\delta\,u_0 \cos(\omega t),
                \]
                with loss factor \(\delta\) (dimensionless). Comparison with the
                decomposition gives
                \[
                K_1^{(B)}(\omega) = k,\qquad
                K_2^{(B)}(\omega) = k\delta,
                \]
                so the loop area and energy dissipated per cycle are independent
                of \(\omega\). For two-sided spectra, a frequency-domain
                implementation uses
                \[
                \hat{R}^{(B)}(\omega)
                  = k\Bigl(1 + \mathrm{i}\delta\,\operatorname{sgn}(\omega)\Bigr),
                \]
                which enforces
                \(\hat{R}^{(B)}(-\omega) = \overline{\hat{R}^{(B)}(\omega)}\).
              </li>
              <li>
                <strong>Model C – fractional Kelvin–Voigt</strong><br />
                Constitutive law:
                \[
                f(t) = k\,u(t) + c_\alpha D^\alpha u(t),\qquad 0 < \alpha < 1,
                \]
                where \(c_\alpha\) (N·s\(^{-\alpha}\)/m) is a material constant and
                \(D^\alpha\) denotes a linear fractional-derivative operator of order
                \(\alpha\). For harmonic motion the identity
                \[
                D^\alpha[\sin(\omega t)]
                  = \omega^\alpha \sin\!\left(\omega t + \frac{\pi\alpha}{2}\right)
                \]
                leads to
                \[
                K_1^{(C)}(\omega)
                  = k + c_\alpha \omega^\alpha
                    \cos\!\left(\frac{\pi\alpha}{2}\right),\qquad
                K_2^{(C)}(\omega)
                  = c_\alpha \omega^\alpha
                    \sin\!\left(\frac{\pi\alpha}{2}\right).
                \]
                The same fractional element therefore adds both a storage
                contribution \(\Delta K_1\) and a loss contribution \(\Delta K_2\),
                which is why matching dissipation (loop area) at a target frequency
                generally changes the loop tilt unless the elastic term in Model C
                is adjusted.
              </li>
            </ol>
          </section>

          <section class="box sdof-box">
            <h3>Hysteresis loops at resonance (ω = ωₙ)</h3>
            <div class="plot-embed">
              __FIG_DIV__
            </div>
            <p class="figure-caption">
              Force–displacement hysteresis loops for the three models at \(\omega = \omega_n\).
              All loops share the same elastic stiffness \(k\), but differ in how dissipation is
              introduced and how the loss stiffness \(K_2(\omega)\) depends on frequency.
              At \(\omega = \omega_n\), the parameters are chosen so that Models A and B have
              identical \((K_1, K_2)\), and Model C is calibrated to match the same
              \(\mathrm{EDC}(\omega_n)\). The overlay highlights how the fractional model can
              reproduce the loop area while still exhibiting a different tilt if the elastic
              term is not further adjusted.
            </p>
          </section>

          <section class="box sdof-box">
            <h3>Energy dissipated per cycle vs frequency</h3>
            <p>
              The energy dissipated per cycle is defined geometrically as the loop area
              \(\mathrm{EDC} = \oint f\,du\) in the \((u,f)\) plane. In this dashboard it is
              evaluated <em>numerically</em> by integrating the simulated loop for each model.
            </p>
            <p>
              For each frequency \(\omega\) in the range
              \(0.5\,\omega_n \le \omega \le 2\,\omega_n\), one full cycle
              \(T = 2\pi / \omega\) is sampled finely in time. The prescribed displacement
              \(u(t)\) and the corresponding force \(f(t)\) are computed for each model,
              the loop is explicitly closed by appending the starting point, and the area is
              obtained from a polygonal (polyarea‑style) formula in the \((u,f)\) plane.
            </p>
            <p>
              The resulting \(\mathrm{EDC}(\omega)\) curves match the analytic identity
              \(\mathrm{EDC}(\omega) = \pi K_2(\omega) u_0^2\) and make the frequency
              dependence of dissipation clear: viscous damping grows approximately
              linearly with \(\omega\), the hysteretic idealization is nearly constant,
              and the fractional model exhibits intermediate \(\omega^\alpha\) behavior.
            </p>
            <div class="plot-embed">
              __EDC_FIG_DIV__
            </div>
            <p class="figure-caption">
              Model A (viscous) grows approximately linearly with \(\omega\), Model B (hysteretic
              idealization) is frequency independent, and Model C (fractional) exhibits intermediate
              \(\omega^\alpha\) dependence. At \(\omega = \omega_n\), all three curves intersect by
              construction, confirming equal \(\mathrm{EDC}\) at resonance.
            </p>
          </section>

          <section class="box sdof-box">
            <h3>Storage and loss stiffness, and energy per cycle at ω = ωₙ</h3>
            <p>
              The table below lists the analytical storage stiffness \(K_1(\omega_n)\),
              loss stiffness \(K_2(\omega_n)\), and energy dissipated per cycle
              \(\mathrm{EDC}(\omega_n)\) for \(u_0 = 1\). Stiffnesses are given in
              N/m and \(\mathrm{EDC}\) in N·m (J).
            </p>
            <div class="summary-table-wrap">
              <table class="summary-table" aria-label="K₁, K₂, and EDC at resonance">
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">\(K_1(\omega_n)\)</th>
                    <th scope="col">\(K_2(\omega_n)\)</th>
                    <th scope="col">\(\mathrm{EDC}(\omega_n)\)</th>
                  </tr>
                </thead>
                <tbody>
__SUMMARY_ROWS__
                </tbody>
              </table>
            </div>
          </section>
          
          <section class="box sdof-box">
            <h3>Earthquake response via frequency-domain analysis</h3>
            <p>
              In relative coordinates \(u(t)\), the SDOF balance under base excitation is
              \[
                m u''(t) + f_d(t) + k\,u(t) = -m\,\ddot{u}_g(t),
              \]
              where \(f_d(t)\) is the model-dependent damping force and \(\ddot{u}_g(t)\)
              is the ground acceleration. Introducing the dynamic stiffness \(\hat{R}(\omega)\)
              such that \(F(\omega) = \hat{R}(\omega) U(\omega)\), Fourier transforming gives
              \[
                \bigl(\hat{R}(\omega) - m\omega^2\bigr)U(\omega) = -m\,\ddot{U}_g(\omega),
                \qquad
                U(\omega) = H(\omega)\,\ddot{U}_g(\omega),\quad
                H(\omega) = -\frac{m}{\hat{R}(\omega) - m\omega^2}.
              \]
            </p>
            <p>
              For each damping model, the appropriate \(\hat{R}(\omega)\) is used:
              Model A: \(\hat{R}^{(A)}(\omega) = k + i c \omega\);
              Model B: \(\hat{R}^{(B)}(\omega) = k\bigl(1 + i\delta\,\operatorname{sgn}(\omega)\bigr)\);
              Model C: \(\hat{R}^{(C)}(\omega) = k_C + c_\alpha (i\omega)^\alpha\) with the
              two-sided fractional power ensuring conjugate symmetry. The spectra
              \(U(\omega)\), \(V(\omega) = i\omega U(\omega)\), and
              \(\ddot{U}_{\mathrm{abs}}(\omega)\) are computed using NumPy's FFT routines
              and transformed back to the time domain via inverse FFT to obtain the
              displacement, velocity, and absolute acceleration responses under the
              Kobe KBU090 motion.
            </p>
            <p>
              The figure below compares the three models on a common time axis. Because
              the FFT method treats long records efficiently and makes the frequency
              dependence of \(\hat{R}(\omega)\) explicit, it is well suited to this
              comparison; however, it also relies on linearization and periodic extension
              of the record, so care is needed when interpreting results for strongly
              nonlinear behavior or for motions with significant trends.
            </p>
            <div class="plot-embed">
              __RESP_FIG_DIV__
            </div>
          </section>
        </div>
      </div>

      <!-- Footer -->
      <footer id="footer"></footer>
    </div>

    <!-- Scripts -->
    <script src="../../assets/js/jquery.min.js"></script>
    <script src="../../assets/js/browser.min.js"></script>
    <script src="../../assets/js/breakpoints.min.js"></script>
    <script src="../../assets/js/util.js"></script>
    <script src="../../assets/js/navigation.js"></script>
    <script src="../../assets/js/main.js"></script>
  </body>
</html>
"""

    html = (
        template.replace("__FIG_DIV__", fig_div)
        .replace("__EDC_FIG_DIV__", fig_edc_div)
        .replace("__SUMMARY_ROWS__", summary_rows)
        .replace("__RESP_FIG_DIV__", fig_resp_div)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sdof_hysteresis_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def write_config_json(params: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = OUTPUT_DIR / "config.json"
    cfg_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote {cfg_path}")


def main() -> None:
    params = sdof_parameters()
    write_config_json(params)
    build_hysteresis_loops_html()
    build_dashboard_html()


if __name__ == "__main__":
    main()

