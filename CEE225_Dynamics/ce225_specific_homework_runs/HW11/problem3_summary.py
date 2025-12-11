"""
Step 3 Summary Page Generator
=============================

Creates a comprehensive summary page for Step 3 with equations,
displacement plots, and mode shapes, organized according to the problem statement.
"""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotting_utils import Colors, get_axis_style


def create_problem3_summary(mass_matrix, mode_shapes, natural_freqs, damping_ratios,
                           floor_heights, time, ug_ddot, meas_ground, q, u, u_ddot, V_base, M_base,
                           analyzer, output_dir):
    """
    Create comprehensive Step 3 summary page organized by problem parts (a)-(e).
    
    Parameters:
    -----------
    mass_matrix : np.ndarray
        Mass matrix m [kg]
    mode_shapes : np.ndarray
        Mass-orthonormal mode shape matrix Φ
    natural_freqs : np.ndarray
        Natural frequencies [rad/s]
    damping_ratios : np.ndarray
        Damping ratios [-]
    floor_heights : np.ndarray
        Floor heights [m]
    time : np.ndarray
        Time vector [s]
    q : np.ndarray
        Modal coordinates q_n(t) [inches] (n_modes x n_steps)
    u : np.ndarray
        Floor displacements u_j(t) [inches] (n_floors x n_steps)
    u_ddot : np.ndarray
        Floor accelerations ü_j(t) [inches/s²] (n_floors x n_steps)
    V_base : np.ndarray
        Base shear [kips] (n_steps,)
    M_base : np.ndarray
        Base moment [kip-ft] (n_steps,)
    analyzer : ModalResponseAnalyzer
        Analyzer instance with computed modal properties
    output_dir : Path
        Output directory
    """
    n_floors, n_modes = mode_shapes.shape
    f_n = natural_freqs / (2 * np.pi)  # [Hz]
    
    # Create HTML content
    html_content = generate_html_template()
    
    # Add system properties section
    html_content += generate_system_properties_section(
        mass_matrix, mode_shapes, f_n, natural_freqs, damping_ratios, analyzer
    )
    
    # Ground motion section (placed after system properties)
    html_content += generate_ground_motion_section(ug_ddot, meas_ground)
    
    # Add problem parts (a) through (e) organized by problem statement
    html_content += generate_problem_part_a(q, time, analyzer)
    html_content += generate_problem_part_b(u, time, q, mode_shapes, floor_heights, output_dir)
    html_content += generate_problem_part_c(u_ddot, time)
    html_content += generate_problem_part_d(V_base, time, analyzer)
    html_content += generate_problem_part_e(M_base, time, analyzer)
    
    # Add summary statistics
    html_content += generate_summary_statistics_section(u, u_ddot, V_base, M_base, time)
    
    # Close HTML
    html_content += """
		</div>
	</body>
</html>
"""
    
    # Write HTML file
    output_path = output_dir / 'step3_modal_response.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  Generated: {output_path}")


def generate_ground_motion_section(ug_ddot, meas_ground):
    """Generate section showing ground motion (applied vs measured table)."""
    applied_max = float(np.max(np.abs(ug_ddot))) if ug_ddot is not None else 0.0
    measured_max = float(np.max(np.abs(meas_ground))) if meas_ground is not None else None
    stats_html = "<ul>"
    stats_html += f"<li>Applied ground accel |üg|<sub>max</sub>: {applied_max:.2f} in/s²</li>"
    if measured_max is not None:
        stats_html += f"<li>Measured table accel |üg|<sub>max</sub>: {measured_max:.2f} in/s²</li>"
    stats_html += "</ul>"
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>Ground Acceleration Input</h2>
\t\t\t\t<p>The applied ground motion (from the input file) and the measured table acceleration are shown below.</p>
\t\t\t\t<div class="plot-container">
\t\t\t\t\t<iframe src="./problem3_ground_motion.html" width="100%" height="420px" style="border: none;"></iframe>
\t\t\t\t</div>
\t\t\t\t<h3>Key Values</h3>
\t\t\t\t{stats_html}
\t\t\t</section>
"""


def generate_html_template():
    """Generate HTML template header."""
    return """<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Step 3: Modal Response Analysis Summary</title>
		<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
		<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
		<script>
			window.MathJax = {
				tex: {
					inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
					displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
					processEscapes: true,
					processEnvironments: true
				}
			};
		</script>
		<style>
			body {
				font-family: 'Arial', sans-serif;
				line-height: 1.6;
				color: #2c3e50;
				max-width: 1200px;
				margin: 0 auto;
				padding: 2rem;
				background-color: #f9fafb;
			}
			h1 {
				color: #003262;
				border-bottom: 3px solid #FDB515;
				padding-bottom: 0.5rem;
				margin-bottom: 2rem;
			}
			h2 {
				color: #003262;
				margin-top: 2rem;
				margin-bottom: 1rem;
				border-left: 4px solid #FDB515;
				padding-left: 1rem;
			}
			h3 {
				color: #003262;
				margin-top: 1.5rem;
				margin-bottom: 0.75rem;
			}
			.section {
				background: white;
				border: 1px solid #e5e7eb;
				border-radius: 8px;
				padding: 1.5rem;
				margin: 2rem 0;
			}
			.equation-block {
				background: #f9fafb;
				border-left: 4px solid #003262;
				padding: 1rem;
				margin: 1rem 0;
				overflow-x: auto;
			}
			table {
				width: 100%;
				border-collapse: collapse;
				margin: 1rem 0;
			}
			th, td {
				border: 1px solid #e5e7eb;
				padding: 0.75rem;
				text-align: left;
			}
			th {
				background-color: #003262;
				color: white;
			}
			tr:nth-child(even) {
				background-color: #f9fafb;
			}
			.plot-container {
				margin: 1.5rem 0;
				border: 1px solid #e5e7eb;
				border-radius: 8px;
				overflow: hidden;
			}
		</style>
	</head>
	<body>
		<div class="container">
			<h1>Step 3: Modal Response Analysis Summary</h1>
			<p style="font-size: 1.1em; color: #6b7280; margin-bottom: 2rem;">
				This page summarizes the modal response analysis of the 3-story MDOF building 
				subjected to the <strong>100% Loma Prieta at Palo Alto ground motion</strong>. 
				All responses are computed using modal superposition with mass-orthonormal mode shapes 
				and Newmark's method for numerical integration.
			</p>
"""


def generate_system_properties_section(mass_matrix, mode_shapes, f_n, omega_n, zeta, analyzer):
    """Generate system properties section with matrices."""
    n_floors, n_modes = mode_shapes.shape
    
    # Mass matrix
    mass_rows = []
    for i in range(n_floors):
        row_values = ' & '.join([f'{mass_matrix[i,j]:.0f}' for j in range(n_floors)])
        mass_rows.append(row_values + ' \\\\')
    mass_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(mass_rows)
    
    # Mode shape matrix (×10^-2 format)
    phi_rows = []
    for i in range(n_floors):
        row_values = ' & '.join([f'{mode_shapes[i,j]*100:.3f}' for j in range(n_modes)])
        phi_rows.append(row_values + ' \\\\')
    phi_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(phi_rows)
    
    # Natural frequencies and periods
    periods = 1.0 / f_n
    
    # Modal properties table
    modal_props_rows = []
    for n in range(n_modes):
        modal_props_rows.append(
            f"<tr><td>Mode {n+1}</td><td>{f_n[n]:.2f}</td><td>{periods[n]:.3f}</td><td>{omega_n[n]:.2f}</td>"
            f"<td>{zeta[n]*100:.2f}</td><td>{analyzer.Gamma[n]:.4f}</td><td>{analyzer.M_eff[n]:.1f}</td><td>{analyzer.h_star[n]:.2f}</td></tr>"
        )
    modal_props_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(modal_props_rows)
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>System Properties</h2>
\t\t\t\t<p>The following system properties were used for the modal response analysis:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Mass Matrix:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\t\\mathbf{{m}} = \\begin{{bmatrix}}
\t\t\t\t\t{mass_str}
\t\t\t\t\t\\end{{bmatrix}} \\quad \\text{{[kg]}}
\t\t\t\t\t\\]
\t\t\t\t</div>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Mass-Orthonormal Mode Shape Matrix:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\t\\boldsymbol{{\\Phi}} = \\begin{{bmatrix}}
\t\t\t\t\t{phi_str}
\t\t\t\t\t\\end{{bmatrix}} \\times 10^{{-2}}
\t\t\t\t\t\\]
\t\t\t\t</div>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Mode</th>
\t\t\t\t\t\t\t<th>$f_n$ [Hz]</th>
\t\t\t\t\t\t\t<th>$T_n$ [s]</th>
\t\t\t\t\t\t\t<th>$\\omega_n$ [rad/s]</th>
\t\t\t\t\t\t\t<th>$\\zeta_n$ [%]</th>
\t\t\t\t\t\t\t<th>$\\Gamma_n$ [-]</th>
\t\t\t\t\t\t\t<th>$M_n^*$ [kg]</th>
\t\t\t\t\t\t\t<th>$h_n^*$ [m]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t{modal_props_str}
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t</section>
"""


def generate_problem_part_a(q, time, analyzer):
    """Generate section for Problem Part (a): Modal displacement responses q_n(t)."""
    n_modes = q.shape[0]
    
    # Compute statistics
    max_q = [np.max(np.abs(q[n, :])) for n in range(n_modes)]
    min_q = [np.min(q[n, :]) for n in range(n_modes)]
    max_q_idx = [np.argmax(np.abs(q[n, :])) for n in range(n_modes)]
    
    # Create table rows
    modal_disp_rows = []
    for n in range(n_modes):
        modal_disp_rows.append(
            f"<tr><td>Mode {n+1} (q<sub>{n+1}</sub>)</td>"
            f"<td>{max_q[n]:.3f}</td><td>{min_q[n]:.3f}</td>"
            f"<td>{time[max_q_idx[n]]:.2f}</td></tr>"
        )
    modal_disp_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(modal_disp_rows)
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>(a) Modal Displacement Responses q<sub>n</sub>(t)</h2>
\t\t\t\t<p>The displacement response for each mode is computed using:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Governing Equation:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\t\\ddot{{D}}_n(t) + 2\\zeta_n\\omega_n\\dot{{D}}_n(t) + \\omega_n^2 D_n(t) = -\\ddot{{u}}_g(t)
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\twhere $D_n(t)$ is the modal displacement, $\\zeta_n$ is the damping ratio, 
\t\t\t\t\t\t$\\omega_n$ is the natural frequency, and $\\ddot{{u}}_g(t)$ is the ground acceleration.
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Modal Coordinate:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\tq_n(t) = \\Gamma_n D_n(t) \\quad \\text{{[in]}}
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\twhere $\\Gamma_n = L_n / M_n$ is the participation factor, with 
\t\t\t\t\t\t$L_n = \\boldsymbol{{\\phi}}_n^{{T}} \\mathbf{{m}} \\boldsymbol{{\\iota}}$ and 
\t\t\t\t\t\t$M_n = \\boldsymbol{{\\phi}}_n^{{T}} \\mathbf{{m}} \\boldsymbol{{\\phi}}_n = 1$ (mass-orthonormal).
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t<p><strong>Note:</strong> The modal equations were integrated using Newmark's method for SDOF systems.</p>
\t\t\t\t<div class="plot-container">
\t\t\t\t\t<iframe src="problem3_modal_displacements.html" width="100%" height="900px" style="border: none;"></iframe>
\t\t\t\t</div>
\t\t\t\t<h3>Response Statistics</h3>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Response</th>
\t\t\t\t\t\t\t<th>Maximum [in]</th>
\t\t\t\t\t\t\t<th>Minimum [in]</th>
\t\t\t\t\t\t\t<th>Time of Max [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t{modal_disp_str}
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t</section>
"""


def generate_problem_part_b(u, time, q, mode_shapes, floor_heights, output_dir):
    """Generate section for Problem Part (b): Floor displacement responses u_j(t) in inches."""
    n_floors = u.shape[0]
    
    # Compute statistics
    max_u = [np.max(np.abs(u[j, :])) for j in range(n_floors)]
    min_u = [np.min(u[j, :]) for j in range(n_floors)]
    max_u_idx = [np.argmax(np.abs(u[j, :])) for j in range(n_floors)]
    
    # Create table rows
    floor_disp_rows = []
    for j in range(n_floors):
        floor_disp_rows.append(
            f"<tr><td>Floor {j+1} (u<sub>{j+1}</sub>)</td>"
            f"<td>{max_u[j]:.3f}</td><td>{min_u[j]:.3f}</td>"
            f"<td>{time[max_u_idx[j]]:.2f}</td></tr>"
        )
    floor_disp_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(floor_disp_rows)
    
    # Create displacement plots with mode shapes
    displacement_plot_html = create_displacement_plots_with_mode_shapes(
        time, u, q, mode_shapes, floor_heights, output_dir
    )
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>(b) Floor Displacement Responses u<sub>j</sub>(t) [inches]</h2>
\t\t\t\t<p>The displacement response of each floor is computed using modal superposition:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Governing Equation:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\tu_j(t) = \\sum_{{n=1}}^{{3}} \\Gamma_n \\phi_{{jn}} D_n(t) = \\sum_{{n=1}}^{{3}} \\phi_{{jn}} q_n(t) \\quad \\text{{[in]}}
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\twhere $u_j(t)$ is the displacement at floor $j$ and $\\phi_{{jn}}$ is the 
\t\t\t\t\t\tmode shape component at floor $j$ for mode $n$.
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t{displacement_plot_html}
\t\t\t\t<h3>Response Statistics</h3>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Response</th>
\t\t\t\t\t\t\t<th>Maximum [in]</th>
\t\t\t\t\t\t\t<th>Minimum [in]</th>
\t\t\t\t\t\t\t<th>Time of Max [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t{floor_disp_str}
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t\t<p style="margin-top: 1rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t<strong>Note:</strong> The left column shows the displacement time history for each floor. 
\t\t\t\t\tThe right column shows the scaled mode shape contributions, where each mode's contribution 
\t\t\t\t\tis scaled to match the maximum displacement magnitude for visual comparison.
\t\t\t\t</p>
\t\t\t</section>
"""


def generate_problem_part_c(u_ddot, time):
    """Generate section for Problem Part (c): Floor acceleration responses ü_j(t) in inches/s²."""
    n_floors = u_ddot.shape[0]
    
    # Compute statistics
    max_u_ddot = [np.max(np.abs(u_ddot[j, :])) for j in range(n_floors)]
    min_u_ddot = [np.min(u_ddot[j, :]) for j in range(n_floors)]
    max_u_ddot_idx = [np.argmax(np.abs(u_ddot[j, :])) for j in range(n_floors)]
    
    # Create table rows
    floor_acc_rows = []
    for j in range(n_floors):
        floor_acc_rows.append(
            f"<tr><td>Floor {j+1} (ü<sub>{j+1}</sub>)</td>"
            f"<td>{max_u_ddot[j]:.2f}</td><td>{min_u_ddot[j]:.2f}</td>"
            f"<td>{time[max_u_ddot_idx[j]]:.2f}</td></tr>"
        )
    floor_acc_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(floor_acc_rows)
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>(c) Floor Acceleration Responses ü<sub>j</sub>(t) [inches/s²]</h2>
\t\t\t\t<p>The acceleration response of each floor includes both relative and ground acceleration:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Governing Equation:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\t\\ddot{{u}}_j(t) = \\sum_{{n=1}}^{{3}} \\Gamma_n \\phi_{{jn}} \\ddot{{D}}_n(t) + \\ddot{{u}}_g(t) \\quad \\text{{[in/s²]}}
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\tTotal floor acceleration includes both relative acceleration from modal response 
\t\t\t\t\t\tand the ground acceleration component.
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t<div class="plot-container">
\t\t\t\t\t<iframe src="problem3_floor_accelerations.html" width="100%" height="900px" style="border: none;"></iframe>
\t\t\t\t</div>
\t\t\t\t<h3>Response Statistics</h3>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Response</th>
\t\t\t\t\t\t\t<th>Maximum [in/s²]</th>
\t\t\t\t\t\t\t<th>Minimum [in/s²]</th>
\t\t\t\t\t\t\t<th>Time of Max [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t{floor_acc_str}
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t\t<p style="margin-top: 1rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t<strong>Note:</strong> Comparison with measured floor accelerations (from Item (v)) 
\t\t\t\t\tcan be performed by overlaying the measured data on these computed responses.
\t\t\t\t</p>
\t\t\t</section>
"""


def generate_problem_part_d(V_base, time, analyzer):
    """Generate section for Problem Part (d): Base shear V_b(t) in kips."""
    max_V = np.max(np.abs(V_base))
    min_V = np.min(V_base)
    max_V_idx = np.argmax(np.abs(V_base))
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>(d) Base Shear V<sub>b</sub>(t) [kips]</h2>
\t\t\t\t<p>The base shear is computed using modal superposition:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Governing Equation:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\tV_b(t) = \\sum_{{n=1}}^{{3}} V_{{b,n}}^{{\\text{{st}}}} A_n(t), \\quad A_n(t) = \\omega_n^2 D_n(t) \\quad \\text{{[kips]}}
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\twhere $V_{{b,n}}^{{\\text{{st}}}}$ is the modal static base shear and 
\t\t\t\t\t\t$A_n(t)$ is the pseudo-acceleration for mode $n$.
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t<div class="plot-container">
\t\t\t\t\t<iframe src="problem3_base_shear.html" width="100%" height="500px" style="border: none;"></iframe>
\t\t\t\t</div>
\t\t\t\t<h3>Response Statistics</h3>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Response</th>
\t\t\t\t\t\t\t<th>Maximum [kips]</th>
\t\t\t\t\t\t\t<th>Minimum [kips]</th>
\t\t\t\t\t\t\t<th>Time of Max [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t<tr><td>Base Shear (V<sub>b</sub>)</td>
\t\t\t\t\t\t\t<td>{max_V:.2f}</td><td>{min_V:.2f}</td>
\t\t\t\t\t\t\t<td>{time[max_V_idx]:.2f}</td></tr>
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t</section>
"""


def generate_problem_part_e(M_base, time, analyzer):
    """Generate section for Problem Part (e): Base overturning moment M_b(t) in kip-ft."""
    max_M = np.max(np.abs(M_base))
    min_M = np.min(M_base)
    max_M_idx = np.argmax(np.abs(M_base))
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>(e) Base Overturning Moment M<sub>b</sub>(t) [kip-ft]</h2>
\t\t\t\t<p>The base overturning moment is computed using modal superposition:</p>
\t\t\t\t<div class="equation-block">
\t\t\t\t\t<p><strong>Governing Equation:</strong></p>
\t\t\t\t\t\\[
\t\t\t\t\tM_b(t) = \\sum_{{n=1}}^{{3}} M_{{b,n}}^{{\\text{{st}}}} A_n(t) \\quad \\text{{[kip-ft]}}
\t\t\t\t\t\\]
\t\t\t\t\t<p style="margin-top: 0.5rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\twhere $M_{{b,n}}^{{\\text{{st}}}}$ is the modal static base moment and 
\t\t\t\t\t\t$A_n(t) = \\omega_n^2 D_n(t)$ is the pseudo-acceleration for mode $n$.
\t\t\t\t\t</p>
\t\t\t\t</div>
\t\t\t\t<div class="plot-container">
\t\t\t\t\t<iframe src="problem3_base_moment.html" width="100%" height="500px" style="border: none;"></iframe>
\t\t\t\t</div>
\t\t\t\t<h3>Response Statistics</h3>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Response</th>
\t\t\t\t\t\t\t<th>Maximum [kip-ft]</th>
\t\t\t\t\t\t\t<th>Minimum [kip-ft]</th>
\t\t\t\t\t\t\t<th>Time of Max [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t<tr><td>Base Moment (M<sub>b</sub>)</td>
\t\t\t\t\t\t\t<td>{max_M:.2f}</td><td>{min_M:.2f}</td>
\t\t\t\t\t\t\t<td>{time[max_M_idx]:.2f}</td></tr>
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t</section>
"""


def generate_summary_statistics_section(u, u_ddot, V_base, M_base, time):
    """Generate summary statistics section."""
    n_floors = u.shape[0]
    
    # Compute statistics
    max_u = np.max(np.abs(u), axis=1)
    max_u_ddot = np.max(np.abs(u_ddot), axis=1)
    max_V = np.max(np.abs(V_base))
    max_M = np.max(np.abs(M_base))
    
    # Find times of maximum responses
    idx_max_u = [np.argmax(np.abs(u[j, :])) for j in range(n_floors)]
    idx_max_u_ddot = [np.argmax(np.abs(u_ddot[j, :])) for j in range(n_floors)]
    idx_max_V = np.argmax(np.abs(V_base))
    idx_max_M = np.argmax(np.abs(M_base))
    
    stats_rows = []
    for j in range(n_floors):
        stats_rows.append(
            f"<tr><td>Floor {j+1}</td><td>{max_u[j]:.3f}</td><td>{time[idx_max_u[j]]:.2f}</td>"
            f"<td>{max_u_ddot[j]:.2f}</td><td>{time[idx_max_u_ddot[j]]:.2f}</td></tr>"
        )
    
    stats_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(stats_rows)
    
    return f"""
\t\t\t<section class="section">
\t\t\t\t<h2>Summary of Maximum Responses</h2>
\t\t\t\t<p>The following table summarizes the maximum absolute responses for each floor and the base:</p>
\t\t\t\t<table>
\t\t\t\t\t<thead>
\t\t\t\t\t\t<tr>
\t\t\t\t\t\t\t<th>Floor</th>
\t\t\t\t\t\t\t<th>Max Displacement [in]</th>
\t\t\t\t\t\t\t<th>Time [s]</th>
\t\t\t\t\t\t\t<th>Max Acceleration [in/s²]</th>
\t\t\t\t\t\t\t<th>Time [s]</th>
\t\t\t\t\t\t</tr>
\t\t\t\t\t</thead>
\t\t\t\t\t<tbody>
\t\t\t\t\t\t{stats_str}
\t\t\t\t\t</tbody>
\t\t\t\t</table>
\t\t\t\t<div style="margin-top: 1.5rem;">
\t\t\t\t\t<p><strong>Base Response:</strong></p>
\t\t\t\t\t<ul>
\t\t\t\t\t\t<li>Maximum Base Shear: <strong>{max_V:.2f} kips</strong> (at t = {time[idx_max_V]:.2f} s)</li>
\t\t\t\t\t\t<li>Maximum Base Moment: <strong>{max_M:.2f} kip-ft</strong> (at t = {time[idx_max_M]:.2f} s)</li>
\t\t\t\t\t</ul>
\t\t\t\t</div>
\t\t\t</section>
"""


def create_displacement_plots_with_mode_shapes(time, u, q, mode_shapes, floor_heights, output_dir):
    """Create displacement plots with modal contributions overlaid (single column)."""
    n_floors, n_modes = mode_shapes.shape
    
    fig = make_subplots(
        rows=n_floors, cols=1,
        subplot_titles=[f'Floor {j+1} Displacement and Modal Contributions' for j in range(n_floors)],
        vertical_spacing=0.08
    )
    
    # Colors
    total_color = 'black'
    mode_colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    floor_names = [f'Floor {j+1}' for j in range(n_floors)]
    
    for j in range(n_floors):
        # Total displacement
        fig.add_trace(
            go.Scatter(
                x=time,
                y=u[j, :],
                mode='lines',
                name=f'{floor_names[j]} total',
                line=dict(color=total_color, width=2),
                showlegend=(j == 0),
                legendgroup='total',
                hovertemplate=(
                    f'<b>{floor_names[j]} total</b><br>'
                    'Time: %{x:.2f} s<br>'
                    'Displacement: %{y:.3f} in<extra></extra>'
                )
            ),
            row=j+1, col=1
        )
        # Modal contributions (phi_jn * q_n)
        for n in range(n_modes):
            contrib = mode_shapes[j, n] * q[n, :]
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=contrib,
                    mode='lines',
                    name=f'Mode {n+1} contrib (Floor {j+1})',
                    line=dict(color=mode_colors[n], width=1.5, dash='dot'),
                    showlegend=(j == 0),
                    legendgroup=f'mode{n+1}',
                    hovertemplate=(
                        f'<b>Mode {n+1} @ Floor {j+1}</b><br>'
                        'Time: %{x:.2f} s<br>'
                        'φ·q: %{y:.3f} in<extra></extra>'
                    )
                ),
                row=j+1, col=1
            )
    
    fig.update_layout(
        title=dict(
            text='Floor Displacements with Modal Contributions',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=350 * n_floors,
        showlegend=True,
        hovermode='x unified'
    )
    
    for row in range(1, n_floors + 1):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Displacement [in]")
    
    # Save plot
    plot_filename = 'problem3_displacement_mode_shapes.html'
    plot_path = output_dir / plot_filename
    fig.write_html(str(plot_path), include_plotlyjs='cdn')
    
    # Return iframe embed code (use relative path)
    return f'<div class="plot-container"><iframe src="{plot_filename}" width="100%" height="{350*n_floors}px" style="border: none;"></iframe></div>'
