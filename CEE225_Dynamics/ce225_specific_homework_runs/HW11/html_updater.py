"""Updates HTML files with computed mode shape matrices."""

import re
import numpy as np


def update_mode_shape_matrix_in_html(all_mode_shapes, output_dir, all_mode_shapes_mass_normalized=None, mass_matrix=None):
    """Update step1_mode_shapes.html with actual mode shape matrix values."""
    psi_matrix = np.column_stack(all_mode_shapes)
    
    html_file = output_dir / 'step1_mode_shapes.html'
    
    if not html_file.exists():
        print(f"[WARNING] HTML file not found: {html_file}")
        return
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    n_floors, n_modes = psi_matrix.shape
    
    # Original matrix
    matrix_rows = []
    for i in range(n_floors):
        row_values = ' & '.join([f'{psi_matrix[i,j]:.4f}' for j in range(n_modes)])
        matrix_rows.append(row_values + ' \\\\')
    matrix_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(matrix_rows)
    
    # Post-processing section (mass matrix + mass-normalized mode shapes)
    post_processing_section = ""
    if all_mode_shapes_mass_normalized is not None and mass_matrix is not None:
        # Mass matrix
        mass_matrix = np.asarray(mass_matrix)
        mass_rows = []
        for i in range(mass_matrix.shape[0]):
            row_values = ' & '.join([f'{mass_matrix[i,j]:.0f}' for j in range(mass_matrix.shape[1])])
            mass_rows.append(row_values + ' \\\\')
        mass_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(mass_rows)
        
        # Mass-normalized matrix (multiplied by 100, 3 decimals)
        psi_mass_norm = np.column_stack(all_mode_shapes_mass_normalized)
        mass_norm_rows = []
        for i in range(n_floors):
            # Multiply by 100 and format with 3 decimals
            row_values = ' & '.join([f'{psi_mass_norm[i,j]*100:.3f}' for j in range(n_modes)])
            mass_norm_rows.append(row_values + ' \\\\')
        mass_norm_str = '\n\t\t\t\t\t\t\t\t\t\t'.join(mass_norm_rows)
        
        post_processing_section = f"""
\t\t\t\t\t\t\t\t<p style="margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid #e5e7eb;">
\t\t\t\t\t\t\t\t\t<strong>Post-Processing: Mass-Normalized Mode Shapes</strong><br>
\t\t\t\t\t\t\t\t\t<span style="font-size: 0.9em; color: #6b7280;">The mode shapes have been post-processed using mass-weighted orthogonalization. 
\t\t\t\t\t\t\t\t\tThis post-processing is necessary for two reasons:</span>
\t\t\t\t\t\t\t\t</p>
\t\t\t\t\t\t\t\t<ol style="font-size: 0.9em; color: #6b7280; margin-left: 1.5rem; margin-top: 0.5rem;">
\t\t\t\t\t\t\t\t\t<li style="margin-bottom: 0.5rem;">To be consistent with the masses provided for each floor, which are assumed to have less uncertainty than the mode shapes extracted from acceleration data.</li>
\t\t\t\t\t\t\t\t\t<li style="margin-bottom: 0.5rem;">To ensure the final $\\boldsymbol{{\\Phi}}$ diagonalizes $\\mathbf{{m}}$, i.e., $\\boldsymbol{{\\Phi}}^{{T}} \\mathbf{{m}} \\boldsymbol{{\\Phi}} = \\mathbf{{I}}$ (mass-orthonormal).</li>
\t\t\t\t\t\t\t\t</ol>
\t\t\t\t\t\t\t\t<div style="text-align: center; margin: 1.5rem 0;">
\t\t\t\t\t\t\t\t\t\\[
\t\t\t\t\t\t\t\t\t\\mathbf{{m}} = \\begin{{bmatrix}}
\t\t\t\t\t\t\t\t\t{mass_str}
\t\t\t\t\t\t\t\t\t\\end{{bmatrix}} \\quad \\text{{[kg]}} \\qquad
\t\t\t\t\t\t\t\t\t\\boldsymbol{{\\Phi}}_{{\\text{{mass-norm}}}} = \\begin{{bmatrix}}
\t\t\t\t\t\t\t\t\t{mass_norm_str}
\t\t\t\t\t\t\t\t\t\\end{{bmatrix}} \\times 10^{{-2}}
\t\t\t\t\t\t\t\t\t\\]
\t\t\t\t\t\t\t\t</div>
\t\t\t\t\t\t\t\t<p style="margin-top: 1rem; font-size: 0.9em; color: #6b7280;">
\t\t\t\t\t\t\t\t\tThe mass-normalized mode shapes are shown in the time variation plots below (dashed teal lines) 
\t\t\t\t\t\t\t\t\tfor comparison with the original extracted mode shapes (solid lines). Values are scaled for visual comparison.
\t\t\t\t\t\t\t\t</p>"""
    
    matrix_section = f"""\t\t\t\t\t\t<section id="mode_shape_matrix">
\t\t\t\t\t\t\t<h2>Mode Shape Matrix</h2>
\t\t\t\t\t\t\t<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:1.5rem; margin:2rem 0;">
\t\t\t\t\t\t\t\t<p>The extracted mode shapes are summarized in the mode shape matrix $\\boldsymbol{{\\Phi}}$, where each column represents a mode:</p>
\t\t\t\t\t\t\t\t<div style="text-align: center; margin: 1.5rem 0;">
\t\t\t\t\t\t\t\t\t\\[
\t\t\t\t\t\t\t\t\t\\boldsymbol{{\\Phi}} = \\begin{{bmatrix}}
\t\t\t\t\t\t\t\t\t{matrix_str}
\t\t\t\t\t\t\t\t\t\\end{{bmatrix}}
\t\t\t\t\t\t\t\t\t\\]
\t\t\t\t\t\t\t\t</div>
\t\t\t\t\t\t\t\t<p style="margin-top: 1rem;">where $\\phi_{{i,j}}$ represents the normalized mode shape component at floor $i$ for mode $j$. The matrix is normalized such that the maximum absolute value in each column is 1.0. The actual numerical values are displayed in the individual mode analyses below.</p>
\t\t\t\t\t\t\t\t{post_processing_section}
\t\t\t\t\t\t\t</div>
\t\t\t\t\t\t</section>
"""
    
    pattern = r'(\s*<section id="mode_shape_matrix">.*?</section>\s*)'
    if re.search(pattern, html_content, re.DOTALL):
        def replacer(match):
            return matrix_section
        html_content = re.sub(pattern, replacer, html_content, flags=re.DOTALL)
    else:
        insert_pattern = r'(<p style="font-size: 1\.1em; color: #6b7280; margin-bottom: 2rem;">.*?</p>\s*)'
        def replacer(match):
            return match.group(0) + '\n' + matrix_section
        html_content = re.sub(insert_pattern, replacer, html_content, flags=re.DOTALL)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[SUCCESS] Updated mode shape matrix in: {html_file}")
    print("\nMode Shape Matrix Φ (Original):")
    print("=" * 50)
    for mode_idx in range(n_modes):
        mode_values = ', '.join([f'{psi_matrix[i, mode_idx]:8.4f}' for i in range(n_floors)])
        print(f"Mode {mode_idx + 1}: [{mode_values}]")
    print("=" * 50)
    
    if all_mode_shapes_mass_normalized is not None:
        psi_mass_norm = np.column_stack(all_mode_shapes_mass_normalized)
        print("\nMode Shape Matrix Φ (Mass-Normalized, Post-Processing) [×10⁻²]:")
        print("=" * 50)
        for mode_idx in range(n_modes):
            mode_values = ', '.join([f'{psi_mass_norm[i, mode_idx]*100:8.3f}' for i in range(n_floors)])
        print(f"Mode {mode_idx + 1}: [{mode_values}]")
    print("=" * 50)

