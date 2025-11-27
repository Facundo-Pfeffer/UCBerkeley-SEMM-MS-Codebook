import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'CEE231_SolidMechanics', 'ce231_specific_homework_runs'))
    from plotly_templates import (
        UCBerkeleyColors as Colors,
        get_axis_style,
        get_plot_layout_style,
    )
except ImportError:
    # Fallback: define colors locally if import fails
    class Colors:
        BERKELEY_BLUE = '#003262'
        CALIFORNIA_GOLD = '#FDB515'
        FOUNDERS_ROCK = '#3B7EA1'
        MEDALIST = '#C4820E'
        TEXT_DARK = '#2C3E50'
        TEXT_LIGHT = '#7F8C8D'
        GRID = 'rgba(200, 200, 200, 0.3)'
        BG_LIGHT = 'rgba(250, 250, 250, 0.98)'
        BG_WHITE = 'white'
    
    def get_axis_style():
        return dict(
            showgrid=True,
            gridcolor=Colors.GRID,
            gridwidth=1,
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.3)',
            zerolinewidth=2,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            tickcolor='black',
            tickfont=dict(size=12, family='Arial, sans-serif')
        )

# ------------------------------------------------------------
# physical / system data
# ------------------------------------------------------------
g = 386.0                     # in/s^2
w_story = 100.0               # kips
m_story = w_story / g         # kip·s^2/in  (≈ 0.259)
k_col = 326.32                # kips/in (per column)
k_story = 2.0 * k_col         # kips/in (two columns in parallel)
zeta = 0.05                   # 5% modal damping for all modes

# lumped mass matrix
M = np.diag([m_story, m_story, m_story / 2.0])

# unnormalized modes (columns)
Phi = np.array([
    [0.50,   1.00,   0.50],
    [0.867,  0.00,  -0.867],
    [1.00,  -1.00,   1.00]
])

print("="*70)
print("P3: Rotational Slab Supported on Antisymmetric Columns")
print("="*70)

# ------------------------------------------------------------
# Part (a): Verify orthogonality properties
# ------------------------------------------------------------
print("\n(a) Verifying orthogonality properties...")

# Mass orthogonality: M_n = Phi^T M Phi (should be diagonal)
M_modal_unnorm = Phi.T @ M @ Phi
print("\nModal mass matrix (unnormalized):")
print(M_modal_unnorm)
print("\nOff-diagonal terms (should be ~0):")
off_diag_mass = M_modal_unnorm - np.diag(np.diag(M_modal_unnorm))
print(f"  Max off-diagonal: {np.max(np.abs(off_diag_mass)):.2e}")
print(f"  Mass orthogonality: {'✓ Verified' if np.max(np.abs(off_diag_mass)) < 1e-3 else '✗ Failed'}")

# Stiffness orthogonality: K_n = Phi^T K Phi (should be diagonal)
K = k_story * np.array([
    [ 2., -1.,  0.],
    [-1.,  2., -1.],
    [ 0., -1.,  1.]
])
K_modal_unnorm = Phi.T @ K @ Phi
print("\nModal stiffness matrix (unnormalized):")
print(K_modal_unnorm)
off_diag_stiff = K_modal_unnorm - np.diag(np.diag(K_modal_unnorm))
print(f"  Max off-diagonal: {np.max(np.abs(off_diag_stiff)):.2e}")
print(f"  Stiffness orthogonality: {'✓ Verified' if np.max(np.abs(off_diag_stiff)) < 1e-3 else '✗ Failed'}")

# ------------------------------------------------------------
# Part (b): Normalize modes so modal mass = 1
# ------------------------------------------------------------
print("\n" + "="*70)
print("(b) Mass-normalizing modes...")

# modal masses M_r = phi_r^T M phi_r
modal_masses = np.array([phi @ (M @ phi) for phi in Phi.T])
print(f"\nModal masses (unnormalized): M_1 = {modal_masses[0]:.4f}, "
      f"M_2 = {modal_masses[1]:.4f}, M_3 = {modal_masses[2]:.4f} kip·s²/in")

# scale factors so that phi_hat_r^T M phi_hat_r = 1
alpha = 1.0 / np.sqrt(modal_masses)
print(f"\nNormalization factors: α_1 = {alpha[0]:.4f}, "
      f"α_2 = {alpha[1]:.4f}, α_3 = {alpha[2]:.4f}")

Phi_hat = Phi * alpha          # mass-normalized modes
print("\nMass-normalized mode matrix Φ̂:")
print(Phi_hat)

# check mass-orthonormality (should be identity)
M_modal = Phi_hat.T @ M @ Phi_hat
print("\nVerification: Φ̂^T M Φ̂ (should be identity):")
print(M_modal)
print(f"  Max deviation from identity: {np.max(np.abs(M_modal - np.eye(3))):.2e}")
print(f"  Mass normalization: {'✓ Verified' if np.max(np.abs(M_modal - np.eye(3))) < 1e-3 else '✗ Failed'}")

# modal stiffness and natural frequencies
K_modal = Phi_hat.T @ K @ Phi_hat
omega_n = np.sqrt(np.diag(K_modal))        # rad/s
freq_n  = omega_n / (2.0 * np.pi)          # Hz

print(f"\nNatural frequencies:")
print(f"  ω₁ = {omega_n[0]:.2f} rad/s ({freq_n[0]:.2f} Hz)")
print(f"  ω₂ = {omega_n[1]:.2f} rad/s ({freq_n[1]:.2f} Hz)")
print(f"  ω₃ = {omega_n[2]:.2f} rad/s ({freq_n[2]:.2f} Hz)")

# ------------------------------------------------------------
# Part (c): Frequency response (0 to 15 Hz)
# ------------------------------------------------------------
print("\n" + "="*70)
print("(c) Computing frequency response (0 to 15 Hz)...")

# eccentric shaker forcing
W_e = 40.0 / 1000.0   # each weight = 40 lb = 0.04 kip
e = 10.0              # in
m_e = W_e / g         # kip·s^2/in (mass of each rotating weight)

# frequency grid - avoid exact 0 to prevent numerical issues
f_vals = np.linspace(0.0, 15.0, 1500)  # Hz
f_vals[0] = 0.01
omega = 2.0 * np.pi * f_vals           # rad/s

# horizontal force from two counter-rotating masses: P0 = 2 m_e e ω²
P0 = 2.0 * m_e * e * omega**2      # kips

# roof DOF is the third component of each mode
phi3 = Phi_hat[2, :]
print(f"\nMass-normalized mode components at roof DOF:")
print(f"  φ̂₃₁ = {phi3[0]:.4f}, φ̂₃₂ = {phi3[1]:.4f}, φ̂₃₃ = {phi3[2]:.4f}")

# modal force amplitudes: P_r = phi3_r * P0
P_r = np.outer(P0, phi3)

# solve for modal response amplitudes (complex)
# Q_hat = P_r / (ω_r² - ω² + 2i ζ_r ω_r ω)
Q_hat = np.zeros_like(P_r, dtype=complex)

for r in range(3):
    w_r = omega_n[r]
    z_r = zeta
    denom = (w_r**2 - omega**2) + 2j * z_r * w_r * omega
    Q_hat[:, r] = P_r[:, r] / denom

# roof displacement from each mode
u3_r_hat = Q_hat * phi3

# total roof displacement (sum over all modes)
u3_hat_total = np.sum(u3_r_hat, axis=1)
U3_total = np.abs(u3_hat_total)        # in
U3_modes = np.abs(u3_r_hat)            # individual mode contributions

# acceleration from displacement: A = ω² U
A3_total = omega**2 * U3_total         # in/s²
A3_total_g = A3_total / g              # convert to g's

# ------------------------------------------------------------
# Part (d): Ground floor story shear at 10 Hz
# ------------------------------------------------------------
print("\n" + "="*70)
print("(d) Computing ground floor story shear at 10 Hz...")

f_excite = 10.0  # Hz
omega_excite = 2.0 * np.pi * f_excite  # rad/s

# find frequency index for 10 Hz
idx_10hz = np.argmin(np.abs(f_vals - f_excite))
omega_10hz = omega[idx_10hz]

# compute modal responses at 10 Hz
Q_hat_10hz = np.zeros(3, dtype=complex)
for r in range(3):
    w_r = omega_n[r]
    z_r = zeta
    P_r_10hz = phi3[r] * 2.0 * m_e * e * omega_10hz**2
    denom = (w_r**2 - omega_10hz**2) + 2j * z_r * w_r * omega_10hz
    Q_hat_10hz[r] = P_r_10hz / denom

# get modal amplitudes and phases
Q_amp = np.abs(Q_hat_10hz)
Q_phase = np.angle(Q_hat_10hz)

# convert to physical displacements: u = Φ̂ q
# need u₁ and u₂ for story shear calculation
u_hat_10hz = Phi_hat @ Q_hat_10hz
u1_amp = np.abs(u_hat_10hz[0])
u1_phase = np.angle(u_hat_10hz[0])
u2_amp = np.abs(u_hat_10hz[1])
u2_phase = np.angle(u_hat_10hz[1])

# ground floor story shear: V₁ = 2k_col (2u₁ - u₂)
V1_hat = 2.0 * k_col * (2.0 * u_hat_10hz[0] - u_hat_10hz[1])
V1_max = np.abs(V1_hat)

print(f"\nAt f = {f_excite} Hz (ω = {omega_10hz:.2f} rad/s):")
print(f"  Modal amplitudes: |Q₁| = {Q_amp[0]:.4e}, |Q₂| = {Q_amp[1]:.4e}, |Q₃| = {Q_amp[2]:.4e}")
print(f"  Floor 1 displacement amplitude: |u₁| = {u1_amp:.4e} in")
print(f"  Floor 2 displacement amplitude: |u₂| = {u2_amp:.4e} in")
print(f"  Ground floor story shear amplitude: V₁,max = {V1_max:.2f} kips")

# ------------------------------------------------------------
# create plots
# ------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=(
        "Roof Displacement Amplitude U₃(f)",
        "Roof Acceleration Amplitude A₃(f)"
    )
)

mode_names = ["Mode 1", "Mode 2", "Mode 3"]
# Dark red works better than yellow for visibility against blue background
DARK_RED = '#8B0000'
mode_colors = [Colors.BERKELEY_BLUE, Colors.FOUNDERS_ROCK, DARK_RED]

# displacement – per mode
for r in range(3):
    fig.add_trace(
        go.Scatter(
            x=f_vals,
            y=U3_modes[:, r],
            mode="lines",
            name=f"{mode_names[r]} (disp.)",
            legendgroup=f"mode{r+1}",
            line=dict(color=mode_colors[r], dash="dot", width=2.5),
            hovertemplate=(
                "f = %{x:.2f} Hz<br>"
                "U₃⁽" + str(r+1) + "⁾ = %{y:.3e} in<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=1
    )

# displacement – total
fig.add_trace(
    go.Scatter(
        x=f_vals,
        y=U3_total,
        mode="lines",
        name="Total (disp.)",
        legendgroup="total_disp",
        line=dict(color=Colors.TEXT_DARK, width=3.5),
        hovertemplate=(
            "f = %{x:.2f} Hz<br>"
            "U₃ = %{y:.3e} in<br>"
            "<extra></extra>"
        ),
    ),
    row=1, col=1
)

# Mark 10 Hz point
fig.add_trace(
    go.Scatter(
        x=[f_excite],
        y=[U3_total[idx_10hz]],
        mode="markers",
        name="10 Hz",
        marker=dict(size=12, color=Colors.MEDALIST, symbol="diamond", 
                   line=dict(width=2, color='white')),
        legendgroup="markers",
        hovertemplate=(
            f"f = {f_excite} Hz<br>"
            "U₃ = %{y:.3e} in<br>"
            "<extra></extra>"
        ),
    ),
    row=1, col=1
)

# acceleration – total (in g)
fig.add_trace(
    go.Scatter(
        x=f_vals,
        y=A3_total_g,
        mode="lines",
        name="Total (accel.)",
        legendgroup="total_acc",
        line=dict(color=Colors.TEXT_DARK, width=3.5),
        hovertemplate=(
            "f = %{x:.2f} Hz<br>"
            "A₃ = %{y:.3e} g<br>"
            "<extra></extra>"
        ),
    ),
    row=2, col=1
)

# Mark 10 Hz point on acceleration plot
fig.add_trace(
    go.Scatter(
        x=[f_excite],
        y=[A3_total_g[idx_10hz]],
        mode="markers",
        name="10 Hz",
        marker=dict(size=12, color=Colors.MEDALIST, symbol="diamond",
                   line=dict(width=2, color='white')),
        legendgroup="markers",
        showlegend=False,
        hovertemplate=(
            f"f = {f_excite} Hz<br>"
            "A₃ = %{y:.3e} g<br>"
            "<extra></extra>"
        ),
    ),
    row=2, col=1
)

# apply axis styling
axis_style = get_axis_style()

# Update axes with professional styling
fig.update_xaxes(
    title=dict(
        text="<b>Excitation frequency f [Hz]</b>",
        font=dict(size=14, color=Colors.TEXT_DARK, family='Arial, sans-serif')
    ),
    row=2, col=1,
    **axis_style
)
fig.update_yaxes(
    title=dict(
        text="<b>Displacement amplitude U₃ [in]</b>",
        font=dict(size=14, color=Colors.TEXT_DARK, family='Arial, sans-serif')
    ),
    row=1, col=1,
    **axis_style
)
fig.update_yaxes(
    title=dict(
        text="<b>Acceleration amplitude A₃ [g]</b>",
        font=dict(size=14, color=Colors.TEXT_DARK, family='Arial, sans-serif')
    ),
    row=2, col=1,
    **axis_style
)

# format subplot titles
for i, annotation in enumerate(fig['layout']['annotations']):
    if i < 2:
        annotation['font'] = dict(size=16, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif', weight='bold')
        annotation['x'] = 0.5
        annotation['xanchor'] = 'center'

# layout
fig.update_layout(
    title=dict(
        text=(
            "<b>Frequency Response at Roof due to Eccentric Mass Shaker</b><br>"
            "<sub>5% modal damping, mass-normalized modes | "
            f"V₁,max at 10 Hz = {V1_max:.2f} kips</sub>"
        ),
        font=dict(size=20, color=Colors.BERKELEY_BLUE, family='Arial, sans-serif'),
        x=0.5,
        xanchor="center",
        y=0.98
    ),
    plot_bgcolor=Colors.BG_LIGHT,
    paper_bgcolor=Colors.BG_WHITE,
    font=dict(family='Arial, sans-serif', size=12),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        bordercolor=Colors.BERKELEY_BLUE
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.20,
        xanchor="center",
        x=0.5,
        font=dict(size=11, family='Arial, sans-serif'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=Colors.GRID,
        borderwidth=1
    ),
    margin=dict(l=80, r=40, t=120, b=100),
    height=750
)

print("\n" + "="*70)
print("All parts completed. Saving HTML file...")
print("="*70)

# save HTML to highlighted_htmls folder (gets deployed via GitHub Actions)
output_dir = os.path.join(os.path.dirname(__file__), '..', 'highlighted_htmls')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'HW10_FrequencyResponse.html')

fig.write_html(
    output_path,
    include_plotlyjs='cdn',
    config={'displayModeBar': True, 'displaylogo': False},
    div_id='frequency-response-plot'
)

print(f"\n[SUCCESS] HTML file saved to: {output_path}")
print(f"\n📊 The plot will be available at:")
print(f"   https://[your-username].github.io/[repo-name]/CEE225_Dynamics/highlighted_htmls/HW10_FrequencyResponse.html")
print(f"\n   Example: https://facundopfeffer.github.io/SEMM-Fall-2025-Homework-Facundo-Pfeffer/CEE225_Dynamics/highlighted_htmls/HW10_FrequencyResponse.html")
print(f"\n   After pushing to GitHub, the file will be automatically deployed via GitHub Actions.")
print("\n" + "="*70)

# Also show the plot
fig.show()
