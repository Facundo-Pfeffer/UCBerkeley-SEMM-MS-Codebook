from __future__ import annotations

"""
CE223 – Base Isolators: Iterative Equivalent-SDOF Response Dashboard

This script visualizes the fixed-point iteration used to match:
- an assumed peak isolator displacement U_max,
- the resulting shear strain γ,
- and the equivalent linear SDOF properties (K1, ζ_eff, c)
for a given ground motion.

It builds a small HTML report showing:
- Key iteration metrics (final γ, K1, ζ_eff, U_max),
- Convergence history across iterations,
- Ground-motion preview.

Run from this directory:

    python build_isolator_iteration_dashboard.py

The dashboard is written to:

    CE223_EarthquakeProtectiveSystems/base_isolators_desgin/isolator_iteration_dashboard.html
"""

import math
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from fft_sdof_response import sdof_response_fft_ground_motion
from isolator_sdof_iteration import NUM_BEARINGS, IterationRecord, iterate_isolator_response
from newmark_sdof import newmark_sdof


BASE_DIR = Path(__file__).resolve().parent
# Legacy local path kept for backward compatibility; canonical output is
# written under CE223_EarthquakeProtectiveSystems/highlighted_htmls.
OUTPUT_HTML = BASE_DIR / "isolator_iteration_dashboard.html"
# Highlighted HTMLs for CE223 live one level up
HIGHLIGHTED_HTML_DIR = BASE_DIR.parent / "highlighted_htmls"
# Ground motions for CE223 live one level up, under input_ground_motion/
INPUT_GM_DIR = BASE_DIR.parent / "input_ground_motion"


def load_ground_motion() -> tuple[np.ndarray, float, str]:
    """
    Load ground motion as (ug_ddot, dt, label).

    A PEER-format .AT2 file must exist in input_ground_motion/.
    The first one found is used. The file is expected to have:
    - Header lines starting with '%' including a line of the form
      '%NPTS= XXXX, DT=  .0100 SEC, ...'
    - Acceleration values in units of g on subsequent lines.
    """
    if not INPUT_GM_DIR.exists():
        raise FileNotFoundError(
            f"Ground motion directory {INPUT_GM_DIR} not found. "
            "Expected RSN1108_KOBE_KBU090.AT2 (or similar) there."
        )

    at2_files = sorted(
        list(INPUT_GM_DIR.glob("*.AT2")) + list(INPUT_GM_DIR.glob("*.at2"))
    )
    if not at2_files:
        raise FileNotFoundError(
            f"No .AT2 ground motion file found in {INPUT_GM_DIR}. "
            "Place RSN1108_KOBE_KBU090.AT2 (or compatible) in that folder."
        )

    path = at2_files[0]

    # Parse header to get DT
    dt = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("%NPTS"):
                # Example: %NPTS=   3200, DT=   .0100 SEC,
                text = line.upper()
                if "DT=" in text:
                    dt_part = text.split("DT=")[1]
                    # take up to "SEC"
                    if "SEC" in dt_part:
                        dt_str = dt_part.split("SEC")[0].strip().replace(",", "")
                    else:
                        dt_str = dt_part.strip().split()[0]
                    dt = float(dt_str)
                break

    if dt is None:
        raise ValueError(
            f"Could not parse DT from header of ground motion file {path}."
        )

    # Load accelerations in g, skipping header lines starting with '%'
    acc_data = np.loadtxt(path, comments="%")
    # acc_data may be 1D or 2D depending on how many values per line
    acc_g = np.asarray(acc_data, dtype=float).ravel()
    if acc_g.size < 2:
        raise ValueError(f"Ground motion file {path} has too few samples.")

    # Convert from g to in/s^2 assuming 1 g ≈ 386.09 in/s^2
    ug_ddot = acc_g * 386.09
    label = path.name
    return ug_ddot, dt, label


def build_iteration_fig(records: list[IterationRecord]) -> str:
    iters = np.array([r.iteration for r in records], dtype=float)
    gamma = np.array([r.gamma_percent for r in records], dtype=float)
    zeta = np.array([r.zeta_eff for r in records], dtype=float)
    U_max = np.array([r.U_max_in for r in records], dtype=float)
    K1 = np.array([r.K1 for r in records], dtype=float)

    # customdata columns: [U_max, K1, zeta]
    custom = np.stack((U_max, K1, zeta), axis=-1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=gamma,
            mode="lines+markers",
            name="Shear strain γ [%]",
            line=dict(color="rgb(0, 55, 95)", width=2.8),
            marker=dict(size=8, color="rgb(0, 55, 95)"),
            yaxis="y1",
            customdata=custom,
            hovertemplate=(
                "Iteration %{x}<br>"
                "γ = %{y:.2f} %<br>"
                "U_max = %{customdata[0]:.3f} in<br>"
                "K₁ = %{customdata[1]:.3f}<br>"
                "ζ_eff = %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=zeta,
            mode="lines+markers",
            name="Effective damping ζ_eff",
            line=dict(color="rgb(197, 48, 48)", width=2.4, dash="dash"),
            marker=dict(size=8, color="rgb(197, 48, 48)"),
            yaxis="y2",
            customdata=custom,
            hovertemplate=(
                "Iteration %{x}<br>"
                "ζ_eff = %{y:.3f}<br>"
                "γ = %{customdata[0]*100/3.25:.2f} %<br>"
                "U_max = %{customdata[0]:.3f} in<br>"
                "K₁ = %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=420,
        title=dict(
            text="Iteration convergence: shear strain γ and damping ζ_eff",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1e293b"),
        ),
        xaxis=dict(
            title="Iteration",
            dtick=1,
            gridcolor="rgba(0,0,0,0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0,0,0)",
        ),
        yaxis=dict(
            title="γ [%]",
            gridcolor="rgba(0,0,0,0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0,0,0)",
        ),
        yaxis2=dict(
            title="ζ_eff",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            showline=True,
            mirror=False,
            linecolor="rgb(197, 48, 48)",
        ),
        legend=dict(
            orientation="h",
            y=0.97,
            yanchor="bottom",
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=70, r=70, t=90, b=60),
    )

    return to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )


def build_ground_motion_preview(ug_ddot: np.ndarray, dt: float, label: str) -> str:
    t = np.arange(0.0, ug_ddot.size * dt, dt)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ug_ddot,
            mode="lines",
            name="Ground acceleration",
            line=dict(color="rgb(0, 55, 95)", width=2.2),
        )
    )
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=320,
        title=dict(
            text=f"Input ground motion (preview) – {label}",
            x=0.5,
            xanchor="center",
            font=dict(size=18, family="Arial", color="#1e293b"),
        ),
        xaxis=dict(
            title="Time [s]",
            gridcolor="rgba(0,0,0,0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0,0,0)",
        ),
        yaxis=dict(
            title="ü_g(t) [in/s²]",
            gridcolor="rgba(0,0,0,0.12)",
            showline=True,
            mirror=True,
            linecolor="rgb(0,0,0)",
        ),
        margin=dict(l=70, r=25, t=60, b=55),
    )

    return to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )


def build_response_fig(
    response_td: np.ndarray, response_fft: np.ndarray, ug_ddot: np.ndarray, dt: float
) -> str:
    """
    Build a 4-panel time-history figure comparing:
    - relative displacement u(t),
    - relative velocity u̇(t),
    - absolute acceleration ü_abs(t),
    - ground acceleration ü_g(t),
    for both the time-domain Newmark solution and the FFT-based solution.
    """
    n = response_td.shape[0]
    t = np.arange(0.0, n * dt, dt)
    u_td = response_td[:, 0]
    v_td = response_td[:, 1]
    a_rel_td = response_td[:, 2]
    ug = ug_ddot[:n]
    a_abs_td = a_rel_td + ug

    # FFT-based solutions
    u_fft = response_fft[:, 0]
    v_fft = response_fft[:, 1]
    a_abs_fft = response_fft[:, 2]

    fig = make_subplots(
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

    fig.add_trace(
        go.Scatter(
            x=t,
            y=u_td,
            mode="lines",
            name="Newmark – u(t) [in]",
            line=dict(color="rgb(0,55,95)", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=u_fft,
            mode="lines",
            name="FFT – u(t) [in]",
            line=dict(color="rgb(220,38,38)", width=1.8, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=v_td,
            mode="lines",
            name="Newmark – u̇(t) [in/s]",
            line=dict(color="rgb(0,55,95)", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=v_fft,
            mode="lines",
            name="FFT – u̇(t) [in/s]",
            line=dict(color="rgb(220,38,38)", width=1.8, dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=a_abs_td,
            mode="lines",
            name="Newmark – ü_abs(t) [in/s²]",
            line=dict(color="rgb(0,55,95)", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=a_abs_fft,
            mode="lines",
            name="FFT – ü_abs(t) [in/s²]",
            line=dict(color="rgb(220,38,38)", width=1.8, dash="dash"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ug,
            mode="lines",
            name="ü_g(t) [in/s²]",
            line=dict(color="rgb(0,0,0)", width=2.3, dash="dot"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=700,
        title=dict(
            text="Equivalent-SDOF response histories (Newmark vs FFT)",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1e293b"),
        ),
        showlegend=True,
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
    fig.update_xaxes(title_text="Time [s]", row=4, col=1)
    fig.update_yaxes(title_text="u(t) [in]", row=1, col=1)
    fig.update_yaxes(title_text="u̇(t) [in/s]", row=2, col=1)
    fig.update_yaxes(title_text="ü_abs(t) [in/s²]", row=3, col=1)
    fig.update_yaxes(title_text="ü_g(t) [in/s²]", row=4, col=1)

    return to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )


def compute_fixed_base_response(
    ug_ddot: np.ndarray,
    dt: float,
    *,
    T_fixed: float = 0.3,
    zeta_fixed: float = 0.05,
    m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the response of a fixed-base SDOF used as a comparison system.

    The system has natural period T_fixed and damping ratio zeta_fixed. It is
    excited by the same ground acceleration record as the isolated system.

    Returns
    -------
    t : np.ndarray
        Time vector [s].
    a_abs : np.ndarray
        Absolute acceleration history of the mass [in/s²].
    k_fixed : float
        Stiffness of the fixed-base SDOF.
    c_fixed : float
        Viscous damping coefficient of the fixed-base SDOF.
    """
    ug = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug.size
    if n == 0:
        raise ValueError("ug_ddot must contain at least one time point for fixed-base SDOF.")

    omega_n = 2.0 * math.pi / T_fixed
    k_fixed = m * omega_n**2
    c_fixed = 2.0 * zeta_fixed * omega_n * m

    # Forcing in relative coordinates: p(t) = -m * ü_g(t)
    p = -m * ug
    resp = newmark_sdof(m=m, k=k_fixed, c=c_fixed, p=p, dt=dt, u0=0.0, v0=0.0)
    a_rel = resp[:, 2]
    a_abs = a_rel + ug[: a_rel.size]
    t = np.arange(0.0, a_rel.size * dt, dt)
    return t, a_abs, k_fixed, c_fixed


def build_fixed_base_comparison_fig(
    t_iso: np.ndarray,
    a_abs_iso: np.ndarray,
    t_fixed: np.ndarray,
    a_abs_fixed: np.ndarray,
    ug_ddot: np.ndarray,
    dt: float,
) -> str:
    """
    Build a comparison figure: isolated vs fixed-base absolute acceleration.
    """
    n = min(a_abs_iso.size, a_abs_fixed.size, ug_ddot.size)
    t = t_iso[:n]
    iso = a_abs_iso[:n]
    fixed = a_abs_fixed[:n]
    ug = ug_ddot[:n]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Absolute acceleration – isolated vs fixed-base SDOF",
            "Ground acceleration ü_g(t)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=iso,
            mode="lines",
            name="Isolated mass – ü_abs(t)",
            line=dict(color="rgb(0,55,95)", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_fixed[:n],
            y=fixed,
            mode="lines",
            name="Fixed base T = 0.30 s, ζ = 0.05 – ü_abs(t)",
            line=dict(color="rgb(220,38,38)", width=1.8, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ug,
            mode="lines",
            name="ü_g(t) [in/s²]",
            line=dict(color="rgb(0,0,0)", width=2.0, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=520,
        title=dict(
            text="Absolute acceleration: isolated vs fixed-base structure",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1e293b"),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.14,
            yanchor="top",
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(size=13, family="Arial", color="#111827"),
        ),
        margin=dict(l=70, r=25, t=80, b=90),
    )
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="ü_abs(t) [in/s²]", row=1, col=1)
    fig.update_yaxes(title_text="ü_g(t) [in/s²]", row=2, col=1)

    return to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config=dict(displayModeBar=True, responsive=True),
    )


def build_html() -> None:
    # 1. Load or synthesize ground motion
    ug_ddot, dt, gm_label = load_ground_motion()

    # 2. Run the equivalent-SDOF iteration
    # For now, take m = 1.0 in consistent pseudo-units; scaling of m does not
    # affect the functional form of the iteration.
    m_eff = 1.0
    # Initial guess for peak displacement [in]; take a low-strain value so that
    # the convergence from small to large deformation is clearly visible.
    U_guess_in = 0.32

    records, response = iterate_isolator_response(
        ug_ddot=ug_ddot,
        dt=dt,
        m=m_eff,
        U_guess_in=U_guess_in,
    )

    if not records:
        raise RuntimeError("No iteration records produced.")

    final = records[-1]

    fig_iter_div = build_iteration_fig(records)
    fig_gm_div = build_ground_motion_preview(ug_ddot, dt, gm_label)

    # 3. Frequency-domain (FFT-based) response using converged equivalent SDOF
    # System stiffness = 4×K1; ζ_eff unchanged, c = 2 ζ √(k_system m).
    k_system = NUM_BEARINGS * final.K1
    omega_n_final = math.sqrt(k_system / m_eff)
    c_final = 2.0 * final.zeta_eff * omega_n_final * m_eff
    response_fft = sdof_response_fft_ground_motion(
        ug_ddot=ug_ddot,
        dt=dt,
        m=m_eff,
        k=k_system,
        c=c_final,
        zero_pad_factor=2,
        return_components="absolute",
    )

    fig_resp_div = build_response_fig(response, response_fft, ug_ddot, dt)

    # 4. Fixed-base SDOF with T = 0.30 s, ζ = 0.05 for acceleration comparison
    n_resp = response.shape[0]
    t_iso = np.arange(0.0, n_resp * dt, dt)
    a_rel_iso = response[:, 2]
    ug_trim = ug_ddot[:n_resp]
    a_abs_iso = a_rel_iso + ug_trim

    t_fixed, a_abs_fixed, k_fixed, c_fixed = compute_fixed_base_response(
        ug_trim, dt, T_fixed=0.3, zeta_fixed=0.05, m=m_eff
    )
    fig_fixed_div = build_fixed_base_comparison_fig(
        t_iso=t_iso,
        a_abs_iso=a_abs_iso,
        t_fixed=t_fixed,
        a_abs_fixed=a_abs_fixed,
        ug_ddot=ug_trim,
        dt=dt,
    )

    # Peak absolute accelerations (for commentary), expressed in g.
    peak_iso = float(np.max(np.abs(a_abs_iso)))
    peak_fixed = float(np.max(np.abs(a_abs_fixed)))
    peak_iso_g = peak_iso / 386.09
    peak_fixed_g = peak_fixed / 386.09
    ratio_iso_fixed = peak_iso / peak_fixed if peak_fixed > 0.0 else float("nan")
    reduction_percent = (1.0 - ratio_iso_fixed) * 100.0 if peak_fixed > 0.0 else float("nan")

    # Build iteration summary rows
    iter_rows = []
    for r in records:
        iter_rows.append(
            f"<tr>"
            f"<td>{r.iteration:d}</td>"
            f"<td>{r.gamma_percent:6.2f}</td>"
            f"<td>{r.U_max_in:6.3f}</td>"
            f"<td>{r.K1:8.3f}</td>"
            f"<td>{r.zeta_eff:6.3f}</td>"
            f"</tr>"
        )
    iter_table_body = "\n".join(iter_rows)

    template = r"""<!DOCTYPE HTML>
<html>
  <head>
    <title>Base isolator – SDOF iteration (CE223)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
    <meta name="description" content="Iterative equivalent-SDOF response for CE223 base isolator design." />
    <link rel="stylesheet" href="../../assets/css/main.css" />
    <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>

    <!-- MathJax for LaTeX rendering (match CEE225/CE223 dashboards) -->
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>

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
      .sdof-box {
        margin-bottom: 1.75rem;
      }
      .sdof-box h3 {
        margin-bottom: 0.75rem;
      }
      .sdof-box p {
        margin-bottom: 0.85rem;
      }
      .key-metrics-title {
        letter-spacing: 0.15em;
        font-size: 0.9em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.75rem;
      }
      .key-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.25rem;
      }
      .key-metric-item h4 {
        margin: 0 0 0.25rem;
        font-size: 1.0em;
        font-weight: 600;
        color: #111827;
      }
      .key-metric-item p {
        margin: 0;
        font-size: 0.95em;
        color: #4b5563;
      }
      .iter-table-wrap {
        overflow-x: auto;
      }
      table.iter-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95em;
      }
      table.iter-table th,
      table.iter-table td {
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid #e5e7eb;
        text-align: right;
        white-space: nowrap;
      }
      table.iter-table th:first-child,
      table.iter-table td:first-child {
        text-align: left;
      }
      table.iter-table thead th {
        background-color: #f9fafb;
        font-weight: 600;
        color: #374151;
      }
    </style>
  </head>
  <body class="is-preload">
    <div id="wrapper">
      <header id="header">
        <div class="inner">
          <a href="../../index.html" class="logo">
            <span class="symbol"><img src="../../images/CE223/CE223_Thumbnail.png" alt="CE223 thumbnail" /></span>
            <span class="title">CE223 – Earthquake Protective Systems</span>
          </a>
        </div>
      </header>

      <div id="main">
        <div class="inner">
          <h1>Equivalent-SDOF iteration for base isolator response</h1>
          <p style="font-size: 1.05em; color: #6b7280; margin-bottom: 2rem;">
            This page tracks how an equivalent linear SDOF model is calibrated to a nonlinear
            base isolator by iterating between assumed peak displacement, shear strain, and
            response under a specified ground motion. Two complementary solution strategies
            are used throughout: a time-domain Newmark method and a frequency-domain
            FFT-based method, both applied to the same linearized SDOF equation.
          </p>

          <section class="box sdof-box">
            <h3>Four-bearing isolation system</h3>
            <p>
              The problem considers a rigid superstructure of mass
              \(m = 200\times 10^3\,\mathrm{kg}\) supported by
              <strong>4 high-damping rubber (HDR) bearings</strong>. The hysteresis
              curves (Figure 1 in the problem) describe the cyclic behavior of
              <em>one</em> bearing; shear strain is the horizontal displacement divided
              by the total rubber thickness \(H = 3.25\,\mathrm{in}\) for that bearing.
            </p>
            <p>
              For the equivalent SDOF, the four bearings act in parallel, so the
              <strong>system storage stiffness</strong> is
              \(k = 4\times K_1\), where \(K_1\) is the per-bearing storage stiffness
              from the hysteresis curves.
            </p>
            <p>
              The effective damping ratio \(\zeta_{\mathrm{eff}}\) from the curves is
              preserved at the system level for the following reason. Each hysteresis
              test is performed with <strong>axial load \(M/4\)</strong> on that bearing
              (one quarter of the total isolated mass). The per-bearing equivalent
              viscous coefficient \(c_1\) inferred from \(\zeta_{\mathrm{eff}}\) therefore
              incorporates this factor (it is the \(c\) that gives \(\zeta_{\mathrm{eff}}\)
              for a SDOF with stiffness \(K_1\) and mass \(M/4\)). When the four bearings
              are in parallel, \(c_{\mathrm{system}} = 4 c_1\) and
              \(k_{\mathrm{system}} = 4 K_1\). In the ratio
              \(\zeta = c\big/(2\sqrt{km}\big)\), the factor 4 in \(c\) and the factor 4
              in \(k\) cancel, so \(\zeta_{\mathrm{system}} = \zeta_{\mathrm{eff}}\).
              We therefore use \(c = 2\zeta_{\mathrm{eff}}\sqrt{km}\) with the
              system stiffness \(k = 4 K_1\) and full mass \(m\). All iteration and
              response calculations use this system-level \(k\) and \(c\).
            </p>
          </section>

          <section class="box sdof-box">
            <h3>Solution methods overview</h3>
            <p>
              In relative coordinates \(u(t)\), the isolated mass is modeled as a single
              degree of freedom (SDOF) with equation of motion
              \[
                m\,u''(t) + c\,u'(t) + k\,u(t) = -m\,\ddot{u}_g(t),
              \]
              where \(m\) is the effective mass, \(k = 4 K_1\) is the
              <strong>system</strong> storage stiffness (four times the per-bearing
              \(K_1\) from the hysteresis curves), \(c\) is a viscous coefficient
              chosen to reproduce \(\zeta_{\mathrm{eff}}\), and
              \(\ddot{u}_g(t)\) is the ground acceleration record.
            </p>
            <p>
              The dashboard compares two ways of solving this equation:
            </p>
            <ol>
              <li>
                <strong>Time-domain Newmark method.</strong>
                The continuous equation is discretized on a uniform time grid with step
                \(\Delta t\). Newmark's constant-average-acceleration scheme
                (\(\beta = 1/4\), \(\gamma = 1/2\)) is used to advance displacement,
                velocity, and acceleration from time step \(i\) to \(i+1\), using
                the standard effective-stiffness and effective-load formulas. This is
                implemented in NumPy in the function <code>newmark_sdof</code>, which
                stores \(u_i\), \(u'_i\), and \(u''_i\) in arrays and loops once over
                all time steps.
              </li>
              <li>
                <strong>Frequency-domain FFT method.</strong>
                The same SDOF equation is solved by taking the discrete Fourier transform
                of the ground motion with <code>numpy.fft.fft</code>, forming the transfer
                function
                \[
                  H(\omega) = -\frac{m}{k + i c \omega - m\omega^2},
                \]
                and computing
                \(U(\omega) = H(\omega)\,\ddot{U}_g(\omega)\). Velocity and absolute
                acceleration follow from spectral derivatives:
                \(V(\omega) = i\omega U(\omega)\),
                \(\ddot{U}_{\mathrm{rel}}(\omega) = -\omega^2 U(\omega)\),
                \(\ddot{U}_{\mathrm{abs}}(\omega) = \ddot{U}_{\mathrm{rel}}(\omega) +
                \ddot{U}_g(\omega)\). An inverse FFT
                (<code>numpy.fft.ifft</code>) with zero-padding is used to return
                \(u(t)\), \(u'(t)\), and \(\ddot{u}_{\mathrm{abs}}(t)\) in the time domain.
              </li>
            </ol>
            <p>
              Both methods use identical SDOF parameters \((m, k, c)\), so differences
              between the blue (Newmark) and red (FFT) traces in the plots quantify only
              numerical effects, not changes in the underlying physical model.
            </p>
          </section>

          <section class="box sdof-box">
            <h3 class="key-metrics-title">Key metrics after convergence</h3>
            <div class="key-metrics-grid">
              <div class="key-metric-item">
                <h4>Peak relative displacement</h4>
                <p>U_max ≈ {U_max_final:.3f} in over the analyzed ground motion.</p>
              </div>
              <div class="key-metric-item">
                <h4>Shear strain level</h4>
                <p>γ ≈ {gamma_final:.1f}% corresponding to the converged U_max.</p>
              </div>
              <div class="key-metric-item">
                <h4>Per-bearing storage stiffness</h4>
                <p>K₁ ≈ {K1_final:.3f} kip/in (from hysteresis). System stiffness k = 4×K₁ ≈ {K1_system:.3f} kip/in.</p>
              </div>
              <div class="key-metric-item">
                <h4>Effective damping ratio</h4>
                <p>ζ_eff ≈ {zeta_final:.3f}, mapped from the hysteresis-based metrics.</p>
              </div>
            </div>
          </section>

          <section class="box sdof-box">
            <h3>Iteration history</h3>
            <p>
              Each iteration proceeds as a fixed-point update:
            </p>
            <ol>
              <li>Start from the current peak displacement estimate \(U_{\max}\) and compute
                  the corresponding shear strain using the known rubber thickness
                  \(H = 3.25~\mathrm{in}\): \(\gamma = 100\,U_{\max}/H\).</li>
              <li>Interpolate the per-bearing storage stiffness \(K_1(\gamma)\) and
                  effective damping ratio \(\zeta_{\mathrm{eff}}(\gamma)\) from the
                  isolator hysteresis library.</li>
              <li>Set system stiffness \(k = 4 K_1\) (four bearings in parallel), form
                  \(c = 2\,\zeta_{\mathrm{eff}}\,\omega_n m\) with
                  \(\omega_n = \sqrt{k/m}\), and solve
                  \(m u'' + c u' + k u = -m \ddot{u}_g(t)\) using Newmark's method
                  in relative coordinates.</li>
              <li>Extract a new peak displacement from the response,
                  \(U_{\max}^{\text{new}} = \max_t |u(t)|\), and repeat until the
                  relative change \(|U_{\max}^{\text{new}} - U_{\max}|/U_{\max}\)
                  falls below the convergence tolerance.</li>
            </ol>
            <p>
              The plot below shows how the assumed shear strain \(\gamma\) and the
              associated damping ratio \(\zeta_{\mathrm{eff}}\) evolve and settle
              as the iteration converges.
            </p>
            <div class="plot-embed">
              __ITER_FIG__
            </div>
            <div class="iter-table-wrap">
              <table class="iter-table" aria-label="Iteration history for equivalent-SDOF calibration">
                <thead>
                  <tr>
                    <th>Iteration</th>
                    <th>γ [%]</th>
                    <th>U_max [in]</th>
                    <th>K₁ (per bearing)</th>
                    <th>ζ_eff</th>
                  </tr>
                </thead>
                <tbody>
__ITER_ROWS__
                </tbody>
              </table>
            </div>
          </section>

          <section class="box sdof-box">
            <h3>Input ground motion</h3>
            <p>
              The equivalent-SDOF system is driven by the ground acceleration time history
              shown below. When a CSV file is present in <code>input_ground_motion/</code>,
              it is used directly; otherwise a synthetic pulse-like motion is generated for
              demonstration purposes.
            </p>
            <div class="plot-embed">
              __GM_FIG__
            </div>
          </section>

          <section class="box sdof-box">
            <h3>Equivalent-SDOF response histories</h3>
            <p>
              The calibrated equivalent-SDOF model is then solved using both methods
              described above. Newmark's scheme provides the reference time histories
              in blue, while the FFT-based convolution produces the overlaid red curves.
              For each quantity the two traces share the same axes, allowing direct
              visual comparison of displacement, velocity, and absolute acceleration
              of the isolated mass under the Kobe ground motion.
            </p>
            <div class="plot-embed">
              __RESP_FIG__
            </div>
          </section>

          <section class="box sdof-box">
            <h3>Isolated vs fixed-base absolute acceleration</h3>
            <p>
              To quantify the benefit of isolation, we compare the calibrated isolated system with a
              <strong>fixed-base SDOF</strong> having period \(T = 0.30~\mathrm{s}\) and damping
              ratio \(\zeta = 0.05\). Both are driven by the same ground acceleration record.
            </p>
            <div class="plot-embed">
              __FIXED_FIG__
            </div>
            <p>
              For this motion, the isolated mass reaches a peak absolute acceleration of approximately
              <strong>{peak_iso_g:.2f} g</strong>, while the fixed-base system reaches about
              <strong>{peak_fixed_g:.2f} g</strong>. In other words, the isolated system experiences
              roughly <strong>{ratio_iso_fixed:.2f}</strong> times the peak acceleration of the
              fixed-base structure (a reduction of about <strong>{reduction_percent:.0f}%</strong>).
            </p>
            <p>
              The price paid for this reduction in acceleration is larger relative displacement in the
              isolator levels (visible in the response histories above), which is precisely what base
              isolation is designed to trade: <em>less force and acceleration in the superstructure,
              more controlled motion in the isolation layer</em>.
            </p>
          </section>
        </div>
      </div>

      <footer id="footer"></footer>
    </div>

    <script src="../../assets/js/jquery.min.js"></script>
    <script src="../../assets/js/browser.min.js"></script>
    <script src="../../assets/js/breakpoints.min.js"></script>
    <script src="../../assets/js/util.js"></script>
    <script src="../../assets/js/navigation.js"></script>
    <script src="../../assets/js/main.js"></script>
  </body>
</html>
"""

    k_system_final = NUM_BEARINGS * final.K1
    html = (
        template.replace("__ITER_FIG__", fig_iter_div)
        .replace("__GM_FIG__", fig_gm_div)
        .replace("__RESP_FIG__", fig_resp_div)
        .replace("__FIXED_FIG__", fig_fixed_div)
        .replace("__ITER_ROWS__", iter_table_body)
        .replace("{U_max_final:.3f}", f"{final.U_max_in:.3f}")
        .replace("{gamma_final:.1f}", f"{final.gamma_percent:.1f}")
        .replace("{K1_final:.3f}", f"{final.K1:.3f}")
        .replace("{K1_system:.3f}", f"{k_system_final:.3f}")
        .replace("{zeta_final:.3f}", f"{final.zeta_eff:.3f}")
        .replace("{peak_iso_g:.2f}", f"{peak_iso_g:.2f}")
        .replace("{peak_fixed_g:.2f}", f"{peak_fixed_g:.2f}")
        .replace("{ratio_iso_fixed:.2f}", f"{ratio_iso_fixed:.2f}")
        .replace("{reduction_percent:.0f}", f"{reduction_percent:.0f}")
    )

    # Canonical output: CE223_EarthquakeProtectiveSystems/highlighted_htmls
    HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)
    highlighted_path = HIGHLIGHTED_HTML_DIR / "isolator_iteration_dashboard.html"
    highlighted_path.write_text(dedent(html), encoding="utf-8")

    print(f"Wrote iteration highlight dashboard {highlighted_path}")


def main() -> None:
    build_html()


if __name__ == "__main__":
    main()

