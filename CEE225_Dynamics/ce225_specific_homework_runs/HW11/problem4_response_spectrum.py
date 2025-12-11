#!/usr/bin/env python3
"""
Problem #4: Response Spectrum Analysis (RSA)
--------------------------------------------
Computes spectral ordinates for the 3-DOF building using the provided design
spectrum, applies SRSS to estimate peak floor displacements and base shear,
and generates plots/HTML summary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modal_response_analyzer import ModalResponseAnalyzer
from plotting_utils import Colors, get_axis_style


def load_spectrum(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df


def map_damping_brackets(zeta_decimal: float):
    """
    Map damping to bracketing spectrum columns for interpolation.
    Damping columns correspond to 0%,1%,2%,3%,5% → A1..A5.
    Returns (lower_idx, upper_idx, lower_pct, upper_pct, weight_upper).
    """
    pct = zeta_decimal * 100.0
    targets = [0, 1, 2, 3, 5]
    # Clamp within available range
    if pct <= targets[0]:
        return 0, 0, targets[0], targets[0], 0.0
    if pct >= targets[-1]:
        return len(targets) - 1, len(targets) - 1, targets[-1], targets[-1], 0.0
    # Find brackets
    for i in range(len(targets) - 1):
        if targets[i] <= pct <= targets[i + 1]:
            low, high = targets[i], targets[i + 1]
            w_up = (pct - low) / (high - low) if high > low else 0.0
            return i, i + 1, low, high, w_up
    # Fallback (should not reach)
    return 0, 0, targets[0], targets[0], 0.0


def map_damping_to_column(zeta_decimal: float):
    """Return the nearest listed damping percent among {0,1,2,3,5} for reporting."""
    pct = zeta_decimal * 100.0
    targets = [0, 1, 2, 3, 5]
    nearest = min(targets, key=lambda x: abs(x - pct))
    return nearest


def get_psa_g(df_spectrum, T_mode, zeta_decimal):
    """
    Interpolate pseudo-acceleration (in g) for a given period and damping.
    Period interpolation: linear within each damping column.
    Damping interpolation: linear between nearest available damping columns.
    """
    damping_targets = [0, 1, 2, 3, 5]
    lower_idx, upper_idx, low_pct, up_pct, w_up = map_damping_brackets(zeta_decimal)
    cols = [c for c in df_spectrum.columns if c != 'Tn']
    # Map target index to column name (A1..A5)
    def col_name(idx):
        return cols[idx]  # assuming order A_1..A_5
    Tn = df_spectrum['Tn'].values
    psa_low = np.interp(T_mode, Tn, df_spectrum[col_name(lower_idx)].values)
    psa_up = np.interp(T_mode, Tn, df_spectrum[col_name(upper_idx)].values)
    psa = (1 - w_up) * psa_low + w_up * psa_up
    return psa, (low_pct, up_pct, w_up)


def plot_spectrum(df_spectrum, T_modes, psa_modes, damping_info, output_path):
    fig = go.Figure()
    # Plot all damping curves
    cols = [c for c in df_spectrum.columns if c != 'Tn']
    color_map = {
        'A_1': Colors.BERKELEY_BLUE,
        'A_2': Colors.CALIFORNIA_GOLD,
        'A_3': Colors.FOUNDERS_ROCK,
        'A_4': Colors.ORANGE,
        'A_5': Colors.PURPLE,
    }
    for c in cols:
        # damping labels per spec: A1->0%, A2->1%, A3->2%, A4->3%, A5->5%
        damping_labels = {'A_1': '0%', 'A_2': '1%', 'A_3': '2%', 'A_4': '3%', 'A_5': '5%'}
        fig.add_trace(
            go.Scatter(
                x=df_spectrum['Tn'],
                y=df_spectrum[c],
                mode='lines',
                name=f'{c} (ζ={damping_labels.get(c, "?")})',
                line=dict(color=color_map.get(c, '#888'), width=2),
                hovertemplate="T = %{x:.2f} s<br>A<sub>n,0</sub> = %{y:.3f} g<extra></extra>"
            )
        )
    # Mark modal points
    for i, (T, psa) in enumerate(zip(T_modes, psa_modes)):
        low_pct, up_pct, w_up = damping_info[i]
        # Determine the effective color based on lower bracket
        targets = [0, 1, 2, 3, 5]
        cols_order = [c for c in df_spectrum.columns if c != 'Tn']
        lower_col = cols_order[[0,1,2,3,4][targets.index(low_pct) if low_pct in targets else 0]]
        fig.add_trace(
            go.Scatter(
                x=[T], y=[psa],
                mode='markers',
                name=f'Mode {i+1}',
                marker=dict(size=10, color='black', symbol='diamond'),
                hovertemplate=(
                    f"Mode {i+1}<br>T = {T:.3f} s<br>A<sub>n,0</sub> = {psa:.3f} g"
                    f"<br>ζ interp: {low_pct:.1f}% → {up_pct:.1f}% (w={w_up:.2f})"
                    "<extra></extra>"
                ),
                showlegend=False
            )
        )
    fig.update_layout(
        title=dict(text='Design Spectrum with Modal Ordinates', x=0.5,
                   font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif', weight='bold')),
        xaxis=get_axis_style() | dict(title='Period T [s]'),
        yaxis=get_axis_style() | dict(title='Pseudo-acceleration [g]'),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=500,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.08,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.15)', borderwidth=1),
        margin=dict(l=80, r=80, t=90, b=70)
    )
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def create_summary_html(output_dir, mode_table, floor_table, base_shear_kips,
                        spectrum_plot_file):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Problem 4: Response Spectrum Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.5; max-width: 1100px; margin: 0 auto; padding: 2rem; color: #2c3e50; }}
h1, h2, h3 {{ color: #003262; }}
section {{ margin-bottom: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #e5e7eb; padding: 0.6rem; text-align: left; }}
th {{ background: #003262; color: white; }}
tr:nth-child(even) {{ background: #f7f7f7; }}
.plot-container {{ border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; margin-top: 1rem; }}
</style>
</head>
<body>
<h1>Problem 4: Peak Response via Response Spectrum Analysis</h1>
<p>This summary presents the RSA of the 3-DOF frame under the 100% Loma Prieta (Palo Alto) motion. It shows (1) the design spectrum and modal ordinates obtained by period and damping interpolation; (2) the resulting spectral displacements per mode; (3) SRSS peak floor displacements; and (4) SRSS base shear. Compare these SRSS results to the direct time-history results from Problem 3.</p>

<p><strong>Computation steps (RSA):</strong></p>
<ul>
  <li><em>Period interpolation:</em> For each damping curve, A<sub>n,0</sub> at the modal period T<sub>n</sub> is obtained by linear interpolation in T.</li>
  <li><em>Damping interpolation:</em> A<sub>n,0</sub> is then interpolated between the bounding damping curves (0, 1, 2, 3, 5%) toward the mode damping.</li>
  <li><em>Spectral displacement:</em> D<sub>n,0</sub> = A<sub>n,0</sub> / ω<sub>n</sub><sup>2</sup> (reported in inches).</li>
  <li><em>SRSS floor displacement:</em> modal contributions Gamma<sub>n</sub> · φ<sub>jn</sub> · D<sub>n,0</sub> are combined as u<sub>j,SRSS</sub> = √( Σ (Gamma<sub>n</sub> · φ<sub>jn</sub> · D<sub>n,0</sub>)² ) (assumes statistical independence).</li>
  <li><em>SRSS base shear:</em> modal pseudo-accelerations multiply modal static base shears V<sub>b,n</sub><sup>st</sup>; combined as V<sub>b,SRSS</sub> = √( Σ ( V<sub>b,n</sub><sup>st</sup> · A<sub>n,0</sub> )² ), then converted to kips.</li>
</ul>

<section>
  <h2>Design Spectrum and Modal Ordinates</h2>
  <p>The provided design spectrum (Item vi) includes 0, 1, 2, 3, and 5% damping curves. Each modal point is obtained by interpolating in period along the appropriate damping curves and then interpolating between damping curves to match the mode’s damping.</p>
  <div class="plot-container">
    <iframe src="{spectrum_plot_file}" width="100%" height="520px" style="border:none;"></iframe>
  </div>
  <h3>Modal Spectral Ordinates</h3>
  {mode_table}
</section>

<section>
  <h2>SRSS Peak Floor Displacements (in)</h2>
  <p>Peak floor displacements are combined by SRSS from modal spectral displacements (Gamma_n * phi_jn * D_n0). Values are given in inches.</p>
  {floor_table}
</section>

<section>
  <h2>SRSS Base Shear</h2>
  <p>The SRSS base shear combines modal pseudo-accelerations with modal static base shears V_{{b,n}}^{{st}}; the result is reported in kips.</p>
  <p><strong>V_b,SRSS = {base_shear_kips:.2f} kips</strong></p>
</section>

</body>
</html>"""
    out_path = output_dir / "problem4_summary.html"
    out_path.write_text(html, encoding='utf-8')
    print(f"  Generated: {out_path}")


def run_problem4(mass_matrix=None, mode_shapes=None, natural_freqs=None, damping_ratios=None,
                 floor_heights=None, output_dir=None):
    script_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = script_dir.parent.parent / 'highlighted_htmls'
    output_dir.mkdir(exist_ok=True)

    # Defaults (if not provided)
    if mass_matrix is None:
        mass_matrix = np.array([[1180, 0, 0],
                                [0, 1180, 0],
                                [0, 0, 910]], dtype=float)
    if mode_shapes is None:
        mode_shapes = np.array([
            [0.771, -1.916, 2.051],
            [1.755, -1.331, -1.903],
            [2.495, 1.982, 0.914]
        ]) * 1e-2
    if natural_freqs is None:
        f_n = np.array([2.00, 7.20, 13.75])   # Hz
        natural_freqs = 2 * np.pi * f_n
    if damping_ratios is None:
        damping_ratios = np.array([0.0113, 0.0157, 0.0093])  # decimal
    if floor_heights is None:
        floor_heights = np.array([3.5, 7.0, 10.5])

    analyzer = ModalResponseAnalyzer(
        mass_matrix=mass_matrix,
        mode_shapes=mode_shapes,
        natural_freqs=natural_freqs,
        damping_ratios=damping_ratios,
        floor_heights=floor_heights
    )

    # Load spectrum
    spectrum_path = script_dir / "input_files" / "spectrum.csv"
    df_spec = load_spectrum(spectrum_path)

    # Modal spectral ordinates
    T_modes = 2 * np.pi / natural_freqs
    psa_results = [get_psa_g(df_spec, T_modes[i], damping_ratios[i]) for i in range(len(natural_freqs))]
    psa_g = np.array([r[0] for r in psa_results])
    damping_info = [r[1] for r in psa_results]  # (low_pct, up_pct, w_up)
    psa_mps2 = psa_g * 9.81
    D_modes_m = psa_mps2 / (natural_freqs ** 2)
    D_modes_in = D_modes_m * 39.3701
    A_modes_inps2 = psa_mps2 * 39.3701

    # SRSS floor displacements
    n_floors = Phi.shape[0]
    u_srss_in = np.zeros(n_floors)
    for j in range(n_floors):
        terms = []
        for n in range(len(omega_n)):
            terms.append((analyzer.Gamma[n] * Phi[j, n] * D_modes_in[n]) ** 2)
        u_srss_in[j] = np.sqrt(np.sum(terms))

    # SRSS base shear
    Vb_static = analyzer.V_static[0, :]  # kg (since s_n are kg)
    Vb_modal_N = Vb_static * psa_mps2
    Vb_srss_N = np.sqrt(np.sum(Vb_modal_N ** 2))
    Vb_srss_kips = Vb_srss_N / 4448.22

    # Tables
    mode_rows = []
    for i in range(len(omega_n)):
        low_pct, up_pct, w_up = damping_info[i]
        zeta_pct = zeta[i] * 100.0
        mode_rows.append(
            f"<tr><td>Mode {i+1}</td>"
            f"<td>{T_modes[i]:.3f}</td>"
            f"<td>{zeta_pct:.2f}% (bracket {low_pct:.0f}%→{up_pct:.0f}%, w={w_up:.2f})</td>"
            f"<td>{psa_g[i]:.3f}</td>"
            f"<td>{A_modes_inps2[i]:.1f}</td>"
            f"<td>{D_modes_in[i]:.3f}</td></tr>"
        )
    mode_table = f"""
    <table>
      <thead><tr><th>Mode</th><th>T [s]</th><th>ζ (interp)</th><th>A<sub>n,0</sub> [g]</th><th>A<sub>n,0</sub> [in/s²]</th><th>D<sub>n,0</sub> [in]</th></tr></thead>
      <tbody>{''.join(mode_rows)}</tbody>
    </table>
    """

    floor_rows = []
    for j in range(n_floors):
        floor_rows.append(f"<tr><td>Floor {j+1}</td><td>{u_srss_in[j]:.3f}</td></tr>")
    floor_table = f"""
    <table>
      <thead><tr><th>Floor</th><th>u<sub>SRSS</sub> [in]</th></tr></thead>
      <tbody>{''.join(floor_rows)}</tbody>
    </table>
    """

    # Plots
    spectrum_plot = output_dir / "problem4_spectrum.html"
    plot_spectrum(df_spec, T_modes, psa_g, damping_info, spectrum_plot)

    # Summary
    create_summary_html(
        output_dir=output_dir,
        mode_table=mode_table,
        floor_table=floor_table,
        base_shear_kips=Vb_srss_kips,
        spectrum_plot_file=spectrum_plot.name
    )


if __name__ == "__main__":
    run_problem4()

