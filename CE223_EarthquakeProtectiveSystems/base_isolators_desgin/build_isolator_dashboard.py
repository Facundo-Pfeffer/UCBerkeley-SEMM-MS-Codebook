from __future__ import annotations

"""
Build an interactive Plotly dashboard summarizing digitized isolator curves.

EVERY run of this script regenerates the summary table and all figures from the
current images and algorithm. The HTML file is overwritten with fresh data.

  python build_isolator_dashboard.py        # Write isolator_dashboard.html (table updated)
  python build_isolator_dashboard.py --serve   # Serve locally; table rebuilt on every page load

Output: isolator_dashboard.html (in this script's directory).
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html

from axis_limits_config import get_axis_limits
from isolator_curve_digitizer import IsolatorCurveDigitizer
from isolator_plotly import create_force_displacement_figure
from isolator_metrics import (
    IsolatorMetrics,
    compute_envelope_point_mask,
    compute_isolator_metrics,
)


BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "input_diagrams"
# Legacy local path kept for backward compatibility, but the canonical
# output for deployment is now written under CE223_EarthquakeProtectiveSystems/highlighted_htmls.
OUTPUT_HTML = BASE_DIR / "isolator_dashboard.html"
HIGHLIGHTED_HTML_DIR = BASE_DIR.parent / "highlighted_htmls"
HIGHLIGHTED_HTML_DIR = BASE_DIR.parent / "highlighted_htmls"

STRAIN_LABELS = {
    "strain10.png": "ε ≈ 9.8%",
    "strain74.png": "ε ≈ 74%",
    "strain124.png": "ε ≈ 124%",
    "strain180.png": "ε ≈ 180%",
}

# Numeric strain (%) for ordering plots and summary table (increasing strain)
STRAIN_PERCENT = {
    "strain10.png": 9.8,
    "strain74.png": 74.0,
    "strain124.png": 124.0,
    "strain180.png": 180.0,
}

# Target peak displacements U0 [in] from test documentation.
# These are used to gently rescale the digitized curves in x so that the
# maximum absolute displacement matches the known value for each strain.
KNOWN_MAX_DISP = {
    "strain10.png": 0.32,
    "strain74.png": 2.40,
    "strain124.png": 4.00,
    "strain180.png": 5.85,
}

# When a scanned plot shows multiple overlaid cycles at the SAME amplitude, the
# envelope (F_max, F_min per bin) equals one cycle's loop—no division needed.
# Only use >1 when the envelope spans nested cycles of different amplitudes.
NUM_CYCLES_OVERRIDE: dict[str, int] = {}


def _metrics_per_cycle(metrics: IsolatorMetrics, n_cycles: int) -> IsolatorMetrics:
    """Return metrics with W_D and derived K2, K1, zeta_eff as per-cycle values."""
    if n_cycles <= 1:
        return metrics
    wd = metrics.WD / n_cycles
    K2 = wd / (math.pi * metrics.U0**2) if metrics.U0 > 0 else float("nan")
    if math.isfinite(metrics.K0) and math.isfinite(K2) and metrics.K0 >= K2:
        K1 = math.sqrt(metrics.K0**2 - K2**2)
    else:
        K1 = float("nan")
    zeta = (
        wd / (2.0 * math.pi * 1.0 * K1 * metrics.U0**2)
        if K1 > 0 and metrics.U0 > 0
        else float("nan")
    )
    return IsolatorMetrics(
        U0=metrics.U0,
        P0=metrics.P0,
        K0=metrics.K0,
        WD=wd,
        K2=K2,
        K1=K1,
        zeta_eff=zeta,
        u0_point=metrics.u0_point,
        p0_point=metrics.p0_point,
    )


def _build_sections_and_summary() -> tuple[str, str]:
    """Build per-strain plot sections and summary table. U0/P0 come from image via axis limits (no rescaling)."""
    # Increasing strain order: 9.8% → 74% → 124% → 180%
    ordered_names = sorted(
        [
            n
            for n in STRAIN_PERCENT
            if (IMG_DIR / n).exists() and get_axis_limits(n, IMG_DIR) is not None
        ],
        key=lambda name: STRAIN_PERCENT.get(name, 0),
    )

    sections: list[str] = []
    summary_rows: list[tuple[float, str, IsolatorMetrics]] = []  # (strain_percent, strain_label, metrics)

    for name in ordered_names:
        limits = get_axis_limits(name, IMG_DIR)
        if limits is None:
            continue
        img = IMG_DIR / name
        if not img.exists():
            continue

        digitizer = IsolatorCurveDigitizer(limits, min_bin_density=3)
        pts = digitizer.digitize(
            img,
            max_points=16000,
            shuffle=True,
            known_max_displacement=KNOWN_MAX_DISP.get(name),
        )

        if name == "strain74.png":
            noise1 = (pts[:, 0] < -2.3) & (pts[:, 1] > -2.46)
            noise2 = (pts[:, 0] < -1.9) & (pts[:, 0] > -2.0) & (pts[:, 1] > -1.5)
            pts = pts[~(noise1 | noise2)]

        strain_text = STRAIN_LABELS.get(name, "")
        title = f"Isolator Hysteresis – {name.replace('.png', '')}"

        metrics = compute_isolator_metrics(pts)
        n_cycles = NUM_CYCLES_OVERRIDE.get(name, 1)
        metrics = _metrics_per_cycle(metrics, n_cycles)
        summary_rows.append((float(STRAIN_PERCENT.get(name, float("nan"))), strain_text or name, metrics))

        # Points that define the loop envelopes used for the area computation
        envelope_mask = compute_envelope_point_mask(pts)
        envelope_pts = pts[envelope_mask]

        # Add margin (8%) between data and plot borders for better visualization
        x_span = limits.x_max - limits.x_min
        y_span = limits.y_max - limits.y_min
        margin = 0.08
        x_range = (limits.x_min - margin * x_span, limits.x_max + margin * x_span)
        y_range = (limits.y_min - margin * y_span, limits.y_max + margin * y_span)

        fig = create_force_displacement_figure(
            pts,
            title=title,
            meta={
                "strain": strain_text,
                "max_disp": f"{metrics.U0:.4f}",
                "notes": "Digitized from scanned force–displacement plot",
            },
            x_range=x_range,
            y_range=y_range,
            u0_point=metrics.u0_point,
            p0_point=metrics.p0_point,
        )

        # Overlay envelope points (used for area) in green
        if envelope_pts.size > 0:
            fig.add_trace(
                go.Scattergl(
                    x=envelope_pts[:, 0],
                    y=envelope_pts[:, 1],
                    mode="markers",
                    name="Area envelope points",
                    marker=dict(size=4, color="rgb(0, 160, 0)", opacity=0.9),
                    hovertemplate="envelope point<br>u = %{x:.4f} in<br>F = %{y:.4f} kips<extra></extra>",
                )
            )

        fig_div = to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            config=dict(displayModeBar=True, responsive=True),
        )

        section_html = f"""
        <section class="report-section">
            <h2>{strain_text or name}</h2>
            <div class="plot-embed">
                {fig_div}
            </div>
            <p class="figure-caption">
                Force–displacement loop for {strain_text or name} from image-based digitization.
                Green marker: point of maximum absolute displacement \\(U_0\\). Red marker: point of
                maximum absolute force \\(P_0\\).
            </p>
        </section>
        """
        sections.append(dedent(section_html))

    # Single summary table: one row per strain, columns = U0, P0, K0, WD, K2, K1, zeta_eff
    summary_table_rows = []
    for _, strain_label, m in summary_rows:
        summary_table_rows.append(
            f"                    <tr>"
            f"<td>{strain_label}</td>"
            f"<td>{m.U0:.4f}</td>"
            f"<td>{m.P0:.3f}</td>"
            f"<td>{m.K0:.3f}</td>"
            f"<td>{m.WD:.3f}</td>"
            f"<td>{m.K2:.3f}</td>"
            f"<td>{m.K1:.3f}</td>"
            f"<td>{m.zeta_eff:.3f}</td>"
            f"</tr>"
        )
    summary_table_body = "\n".join(summary_table_rows)

    # Trend plots: K1, K2, zeta vs shear strain (%)
    trend_x = [s for (s, _, _) in summary_rows]
    trend_labels = [lbl for (_, lbl, _) in summary_rows]
    trend_K1 = [m.K1 for (_, _, m) in summary_rows]
    trend_K2 = [m.K2 for (_, _, m) in summary_rows]
    trend_zeta = [m.zeta_eff for (_, _, m) in summary_rows]

    def _trend_fig(y_vals: list[float], *, title: str, y_title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_x,
                y=y_vals,
                mode="lines+markers",
                showlegend=False,
                line=dict(color="rgb(0, 55, 95)", width=2),
                marker=dict(size=8, color="rgb(0, 55, 95)"),
                customdata=np.array(trend_labels, dtype=object),
                hovertemplate="Strain: %{customdata}<br>%{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, family="Arial", color="#1e293b")),
            xaxis=dict(
                title=dict(text="Shear strain [%]"),
                tickmode="array",
                tickvals=trend_x,
                ticktext=[f"{v:g}" for v in trend_x],
                gridcolor="rgba(0, 0, 0, 0.12)",
                showline=True,
                mirror=True,
                linecolor="rgb(0, 0, 0)",
            ),
            yaxis=dict(
                title=dict(text=y_title),
                gridcolor="rgba(0, 0, 0, 0.12)",
                showline=True,
                mirror=True,
                linecolor="rgb(0, 0, 0)",
            ),
            margin=dict(l=70, r=25, t=60, b=55),
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor="rgb(250, 250, 250)",
            hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial"),
        )
        return fig

    trend_figs = [
        _trend_fig(trend_K1, title="Storage stiffness K1 vs shear strain", y_title="K1 [kips/in]"),
        _trend_fig(trend_K2, title="Loss stiffness K2 vs shear strain", y_title="K2 [kips/in]"),
        _trend_fig(trend_zeta, title="Equivalent damping \u03b6_eff vs shear strain", y_title="\u03b6_eff [-]"),
    ]
    trend_divs = "\n".join(
        f'<div class="plot-embed trend-plot">{to_html(fig, include_plotlyjs=False, full_html=False, config=dict(displayModeBar=True, responsive=True))}</div>'
        for fig in trend_figs
    )

    summary_html = f"""
        <section class="box summary-section">
            <h3>Summary of computed quantities</h3>
            <p>
                The table below collects, for each strain level, the quantities derived from the
                digitized hysteresis loop: maximum displacement \\(U_0\\) and force \\(P_0\\), secant
                stiffness \\(K_0 = P_0/U_0\\), energy dissipated per cycle \\(W_D\\), loss stiffness
                \\(K_2 = W_D/(\\pi U_0^2)\\), storage stiffness \\(K_1\\) from \\(K_0^2 = K_1^2 + K_2^2\\),
                and equivalent viscous damping ratio \\(\\zeta_\\mathrm{{eff}}\\) evaluated at
                \\(\\omega/\\omega_n = 1\\).
            </p>
            <div class="summary-table-wrap">
            <table class="summary-table" aria-label="Summary of isolator parameters by strain level">
                <thead>
                    <tr>
                        <th scope="col">Strain</th>
                        <th scope="col">\\(U_0\\) [in]</th>
                        <th scope="col">\\(P_0\\) [kips]</th>
                        <th scope="col">\\(K_0\\) [kips/in]</th>
                        <th scope="col">\\(W_D\\) [kip·in]</th>
                        <th scope="col">\\(K_2\\) [kips/in]</th>
                        <th scope="col">\\(K_1\\) [kips/in]</th>
                        <th scope="col">\\(\\zeta_\\mathrm{{eff}}\\)</th>
                    </tr>
                </thead>
                <tbody>
{summary_table_body}
                </tbody>
            </table>
            </div>
            <p class="report-generated" style="margin-top:1rem;font-size:0.9rem;color:#6b7280;">
                Report generated on {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC.
                Re-run <code>python build_isolator_dashboard.py</code> to refresh the table and figures.
            </p>
        </section>

        <section class="box">
            <h3>Trends with shear strain</h3>
            <p>
                The following plots summarize how \\(K_1\\), \\(K_2\\), and \\(\\zeta_\\mathrm{{eff}}\\) vary with shear strain,
                based on the values computed from each digitized hysteresis loop.
            </p>
            <div class="trend-grid">
                {trend_divs}
            </div>
        </section>
"""
    return "\n".join(sections), dedent(summary_html)


def build_dashboard() -> None:
    sections_html, summary_html = _build_sections_and_summary()

    # Use a raw string for the HTML template so that LaTeX-style
    # backslashes and CSS braces are treated literally. We then
    # inject the per-strain sections and the summary table with
    # placeholder replacement.
    template = r"""<!DOCTYPE HTML>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <title>CE223 – Base Isolator Hysteresis Dashboard</title>
    <link rel="stylesheet" href="../../assets/css/main.css" />
    <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>
    <style>
        /* Match site header width: same inner width as Phantom (68em) */
        .ce223-dashboard .container {
            max-width: 68em;
            margin-left: auto;
            margin-right: auto;
        }
        #main.ce223-dashboard {
            padding-top: 0.75rem;
        }
        .inner-report {
            font-family: Arial, Helvetica, sans-serif;
            font-size: 1rem;
            line-height: 1.55;
            color: #2c3e50;
            max-width: 100%;
            margin: 0;
            padding: 0 1rem;
        }
        .inner-report header.major {
            text-align: center;
            margin-bottom: 1.25rem;
        }
        .inner-report header.major h2 {
            font-size: 1.85rem;
            font-weight: 700;
            color: #003262;
            margin: 0 0 0.5rem 0;
            letter-spacing: 0.02em;
        }
        .summary-lead {
            font-size: 1.05rem;
            color: #6b7280;
            max-width: 40em;
            margin: 0 auto;
            line-height: 1.6;
        }
        .inner-report .box {
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .inner-report .box h3 {
            font-size: 1.2rem;
            font-weight: 700;
            color: #003262;
            margin: 0 0 0.75rem 0;
            border-left: 4px solid #FDB515;
            padding-left: 0.75rem;
        }
        .inner-report .box p {
            margin: 0 0 0.75rem 0;
        }
        .inner-report .box p:last-child {
            margin-bottom: 0;
        }
        .inner-report .box ol, .inner-report .box ul {
            margin: 0.5rem 0 0.75rem 1.5rem;
            padding: 0;
        }
        .inner-report .box li {
            margin-bottom: 0.5rem;
        }
        .report-section {
            background: #fff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #003262;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .report-section h2 {
            font-size: 1.35rem;
            font-weight: 700;
            color: #003262;
            margin: 0 0 0.5rem 0;
        }
        .summary-section {
            margin-top: 0.5rem;
        }
        .summary-table-wrap {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 1rem 0;
        }
        .summary-table {
            width: 100%;
            min-width: 32rem;
            border-collapse: collapse;
            font-size: 0.95rem;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
        }
        .summary-table thead th {
            background: #003262;
            color: #fff;
            font-weight: 600;
            text-align: center;
            padding: 0.55rem 0.5rem;
        }
        .summary-table thead th:first-child {
            text-align: left;
        }
        .summary-table tbody td {
            padding: 0.5rem 0.5rem;
            border-top: 1px solid #e5e7eb;
            color: #2c3e50;
        }
        .summary-table tbody td:first-child {
            font-weight: 600;
            color: #003262;
        }
        .summary-table tbody td:not(:first-child) {
            text-align: right;
        }
        .summary-table tbody tr:nth-child(even) {
            background: #f9fafb;
        }
        .plot-embed {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.5rem;
            background: #fff;
            margin: 1rem 0;
            overflow-x: auto;
        }
        .plot-embed .plotly {
            max-width: 100%;
        }
        .trend-grid {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            margin-top: 1rem;
        }
        .trend-plot {
            margin: 0;
        }
        .figure-caption {
            font-size: 0.9rem;
            color: #6b7280;
            margin: 0.5rem 0 0 0;
            line-height: 1.5;
        }
        @media (max-width: 736px) {
            #main.ce223-dashboard {
                padding-top: 0.5rem;
            }
            .inner-report {
                padding: 0 0.75rem;
            }
            .inner-report header.major h2 {
                font-size: 1.45rem;
            }
            .summary-lead {
                font-size: 1rem;
            }
            .inner-report .box, .report-section {
                padding: 1rem;
            }
            .inner-report .box h3 {
                font-size: 1.1rem;
            }
            .report-section h2 {
                font-size: 1.2rem;
            }
            .summary-table {
                font-size: 0.85rem;
            }
            .summary-table thead th, .summary-table tbody td {
                padding: 0.4rem 0.35rem;
            }
        }
    </style>
    <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script async src="../../assets/js/navigation.js"></script>
</head>
<body class="is-preload">
    <div id="page-wrapper">
        <header id="header"></header>

        <section id="main" class="wrapper style1 ce223-dashboard">
            <div class="container inner-report">
                <header class="major">
                    <h2>CE223 – Base Isolator Hysteresis</h2>
                    <p class="summary-lead">
                        This report recovers numerical force–displacement data from published scanned
                        hysteresis plots of a base isolator tested at four shear strain levels. Each
                        plot is digitized into a dense point cloud, the axes are calibrated using
                        known test values, and effective stiffness and energy-dissipation parameters
                        are computed. The goal is to characterize the isolator’s hysteretic behavior
                        for use in design or analysis when raw test data are not available.
                    </p>
                </header>

                <section class="box">
                    <h3>What is being done and why</h3>
                    <p>
                        Base isolators are devices that reduce seismic demand on a structure by
                        supporting it on flexible, energy-dissipating elements. Their force–displacement
                        response under cyclic loading is hysteretic: the load path on loading differs
                        from that on unloading, and the area enclosed by the loop represents energy
                        dissipated per cycle. Quantifying this behavior—peak force \(P_0\) and
                        displacement \(U_0\), secant stiffness \(K_0 = P_0/U_0\), and equivalent
                        viscous damping \(\zeta_\mathrm{{eff}}\)—is standard in earthquake engineering
                        for design and assessment.
                    </p>
                    <p>
                        This page starts from <em>scanned images</em> of force–displacement loops
                        (from reports or papers) rather than raw data. Each image is digitized to
                        obtain a cloud of \((u, F)\) points; the same mechanical definitions are
                        applied to compute \(U_0\), \(P_0\), \(K_0\), \(W_D\), \(K_1\), \(K_2\),
                        and \(\zeta_\mathrm{{eff}}\). The result is a consistent set of parameters
                        across four strain levels (approximately 9.8%, 74%, 124%, and 180%) for
                        comparison and potential use in models. All values reported here are
                        <strong>per bearing</strong>; the problem’s isolation system uses
                        <strong>4 HDR bearings</strong> in parallel, so the system storage stiffness
                        for SDOF analysis is \(4\times K_1\).
                    </p>
                </section>

                <section class="box">
                    <h3>Digitization workflow and quantity definitions</h3>
                    <p>The following steps are applied to each scanned PNG:</p>
                    <ol>
                        <li>The image is converted to grayscale and the plotting window is detected from dark-pixel density.</li>
                        <li>The outer frame and axis lines are removed in image space.</li>
                        <li>Remaining dark pixels are mapped to physical coordinates \((u, F)\) using strain-specific axis limits taken from the original plot labels (displacement in inches, force in kips).</li>
                        <li>Small connected components (annotation digits and specks) are filtered out.</li>
                        <li>Physical axis limits (displacement and force) are set from the plot labels (or from a per-image sidecar <code>basename_limits.json</code>), so \(U_0\) and \(P_0\) are obtained directly from the digitized curve.</li>
                    </ol>
                    <p>From the calibrated point cloud the following quantities are computed:</p>
                    <ul>
                        <li><strong>\(U_0\)</strong> and <strong>\(P_0\)</strong>: maximum absolute displacement and force.</li>
                        <li><strong>\(K_0 = P_0/U_0\)</strong>: secant stiffness to the loop tip.</li>
                        <li><strong>\(W_D\)</strong>: area enclosed by the loop (envelope integration over displacement bins).</li>
                        <li><strong>\(K_2 = W_D/(\pi U_0^2)\)</strong>: loss stiffness.</li>
                        <li><strong>\(K_1\)</strong>: storage stiffness from \(K_0^2 = K_1^2 + K_2^2\).</li>
                        <li><strong>\(\zeta_\mathrm{{eff}} = W_D/\bigl(2\pi (\omega/\omega_n) K_1 U_0^2\bigr)\)</strong>, evaluated at \(\omega/\omega_n = 1\).</li>
                    </ul>
                    <p>On each figure, the green marker indicates the point corresponding to \(U_0\) and the red marker the point corresponding to \(P_0\).</p>
                </section>

                __SECTIONS_HTML__

                __SUMMARY_HTML__
            </div>
        </section>

        <footer id="footer"></footer>
    </div>

    <script src="../../assets/js/jquery.min.js"></script>
    <script src="../../assets/js/jquery.scrollex.min.js"></script>
    <script src="../../assets/js/jquery.scrolly.min.js"></script>
    <script src="../../assets/js/browser.min.js"></script>
    <script src="../../assets/js/breakpoints.min.js"></script>
    <script src="../../assets/js/util.js"></script>
    <script src="../../assets/js/main.js"></script>
</body>
</html>
"""
    full_html = (
        template.replace("__SECTIONS_HTML__", sections_html)
        .replace("__SUMMARY_HTML__", summary_html)
    )

    # Canonical output: CE223_EarthquakeProtectiveSystems/highlighted_htmls
    HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)
    highlighted_path = HIGHLIGHTED_HTML_DIR / "isolator_dashboard.html"
    highlighted_path.write_text(full_html, encoding="utf-8")

    print(f"Highlight dashboard written to {highlighted_path}")
    print("Summary table and all figures updated.")


if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv:
        import http.server
        import webbrowser
        import threading

        build_dashboard()
        port = 8766
        out_path = OUTPUT_HTML.resolve()

        class RebuildHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(BASE_DIR), **kwargs)

            def do_GET(self):
                if self.path.strip("/") == "isolator_dashboard.html" or self.path == "/":
                    build_dashboard()
                    self.path = "/isolator_dashboard.html" if self.path == "/" else self.path
                super().do_GET(self)

        with http.server.HTTPServer(("", port), RebuildHandler) as httpd:
            url = f"http://127.0.0.1:{port}/isolator_dashboard.html"
            print(f"Serving at {url} — summary table is rebuilt on every request.")
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            httpd.serve_forever()
    else:
        build_dashboard()

