#!/usr/bin/env python3
"""
Problem #3: Modal Response Analysis
===================================

Determines and plots the response of the 3-story MDOF building to the
100% Loma Prieta at Palo Alto ground motion.

Author: Facundo L. Pfeffer
Course: CEE225 - Structural Dynamics
University of California, Berkeley
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modal_response_analyzer import ModalResponseAnalyzer
from data_loader import DataLoader
from plotting_utils import Colors, get_axis_style
from problem3_summary import create_problem3_summary


def load_ground_motion(data_dir=None):
    """
    Load ground motion data from CSV file.
    
    Returns:
    --------
    time : np.ndarray
        Time vector [s]
    ug_ddot : np.ndarray
        Ground acceleration [m/s²]
    """
    if data_dir is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir / 'input_files'
    else:
        data_dir = Path(data_dir)
    
    csv_path = data_dir / 'ground_motion_excitation.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground motion file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Look for time and ground acceleration columns
    time_col = None
    acc_col = None
    
    # Check for common column names
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            time_col = col
        elif 'table' in col_lower and 'acc' in col_lower:
            # tableAccX_filtered is likely the ground/table acceleration
            acc_col = col
    
    if time_col is None:
        raise ValueError("Time column not found in ground motion CSV")
    if acc_col is None:
        # Try alternative names
        for col in df.columns:
            if 'ground' in col.lower() or ('ug' in col.lower() and 'acc' in col.lower()):
                acc_col = col
                break
        if acc_col is None:
            raise ValueError("Ground acceleration column not found in CSV")
    
    time = df[time_col].values
    ug_ddot = df[acc_col].values
    
    # Convert to SI units if needed (assuming input might be in g or in/s²)
    # Check typical range - if max is around 0.1-0.5, probably in g
    # If max is around 100-500, probably in in/s²
    max_acc = np.max(np.abs(ug_ddot))
    if max_acc < 10:
        ug_ddot = ug_ddot * 9.81  # Convert g to m/s²
        print("Note: Converted ground motion from g to m/s²")
    elif max_acc > 50 and max_acc < 1000:
        ug_ddot = ug_ddot * 0.0254  # Convert in/s² to m/s²
        print("Note: Converted ground motion from in/s² to m/s²")
    
    return time, ug_ddot


def load_measured_floor_accelerations(data_dir=None):
    """
    Load measured floor accelerations for comparison.
    
    Returns:
    --------
    time : np.ndarray
        Time vector [s]
    acc_measured : list of np.ndarray (floors)
    acc_ground : np.ndarray
        Measured accelerations for each floor [m/s²]
    """
    from pathlib import Path
    import pandas as pd
    try:
        data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'input_files'
        csv_path = data_dir / 'ground_motion_excitation.csv'
        df = pd.read_csv(csv_path)
        time = df['time'].values
        acc_all = [df[col].values for col in ['L1AccX_filtered', 'L2AccX_filtered', 'L3AccX_filtered']]
        acc_ground = df['tableAccX_filtered'].values
    except Exception as exc:
        print(f"Measured acceleration data not loaded: {exc}")
        return None, None, None
    
    time = np.asarray(time, dtype=float)
    acc_measured = [np.asarray(a, dtype=float) for a in acc_all]
    acc_ground = np.asarray(acc_ground, dtype=float)
    
    # Unit handling: these columns are in g (per user), convert to m/s²
    acc_measured = [a * 9.81 for a in acc_measured]
    acc_ground = acc_ground * 9.81
    print("Measured floor and ground accelerations converted from g to m/s² (ground_motion_excitation.csv)")
    
    return time, acc_measured, acc_ground


def run_problem3_analysis(mass_matrix, mode_shapes, natural_freqs, damping_ratios,
                          floor_heights, output_dir):
    """
    Run Problem 3 modal response analysis.
    
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
    output_dir : Path
        Output directory for plots
    """
    print("\nSystem Properties:")
    f_n = natural_freqs / (2 * np.pi)
    print(f"  Natural frequencies: {f_n} Hz")
    print(f"  Natural periods: {1/f_n} s")
    print(f"  Damping ratios: {damping_ratios*100}%")
    
    # Load ground motion
    print("\nLoading ground motion data...")
    try:
        time, ug_ddot = load_ground_motion()
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        print(f"  Time range: {time[0]:.2f} to {time[-1]:.2f} s")
        print(f"  Time step: {dt:.4f} s")
        print(f"  Number of steps: {len(time)}")
        print(f"  Max ground acceleration: {np.max(np.abs(ug_ddot)):.2f} m/s²")
    except Exception as e:
        print(f"Error loading ground motion: {e}")
        raise
    
    # Initialize modal response analyzer
    analyzer = ModalResponseAnalyzer(
        mass_matrix=mass_matrix,
        mode_shapes=mode_shapes,
        natural_freqs=natural_freqs,
        damping_ratios=damping_ratios,
        floor_heights=floor_heights
    )
    
    print("\nModal Properties:")
    print(f"  Participation factors Γ: {analyzer.Gamma}")
    print(f"  Effective modal masses M*: {analyzer.M_eff} kg")
    print(f"  Effective modal heights h*: {analyzer.h_star} m")
    
    # Solve modal equations
    print("\nSolving modal equations...")
    D, D_dot, D_ddot = analyzer.solve_modal_equations(ug_ddot, time, dt)
    D_ddot_inch = D_ddot * 39.3701  # m/s² to in/s² (modal relative accelerations)
    D_ddot_rel = D_ddot.copy()  # relative modal accels for diagnostics
    
    # Compute modal coordinates q_n(t) = Γ_n D_n(t)
    q = analyzer.compute_modal_coordinates(D)
    
    # Compute floor responses (include ground motion for total accelerations)
    print("Computing floor responses...")
    u, u_dot, u_ddot = analyzer.compute_floor_responses(D, D_dot, D_ddot, ug_ddot=ug_ddot)
    
    # Load measured floor accelerations (item v)
    meas_time, meas_acc, meas_ground = load_measured_floor_accelerations()
    if meas_time is not None and meas_acc is not None:
        meas_acc = [np.asarray(a) for a in meas_acc]
        meas_ground = np.asarray(meas_ground) if meas_ground is not None else None
    else:
        meas_acc = None
        meas_ground = None
    
    # Compute base shear and moment
    print("Computing base shear and moment...")
    V_base = analyzer.compute_base_shear(D)
    M_base = analyzer.compute_base_moment(D)
    
    # Convert units for output
    # Displacements: m to inches
    u_inch = u * 39.3701  # m to inches
    q_inch = q * 39.3701  # m to inches
    
    # Accelerations: m/s² to inches/s²
    u_ddot_inch = u_ddot * 39.3701  # m/s² to in/s²
    if meas_acc is not None:
        meas_acc_inch = [a * 39.3701 for a in meas_acc]
        meas_ground_inch = meas_ground * 39.3701 if meas_ground is not None else None
        # Simple double integration to get measured displacements (baseline assumed zero)
        meas_disp_m = [_integrate_acc_to_disp(meas_time, a) for a in meas_acc]
        meas_disp_inch = [d * 39.3701 for d in meas_disp_m]
    else:
        meas_acc_inch = None
        meas_ground_inch = None
        meas_disp_inch = None
    
    # Base shear: N to kips
    V_base_kips = V_base / 4448.22  # N to kips
    
    # Base moment: N·m to kip-ft
    M_base_kipft = M_base * 0.737562  # N·m to kip-ft
    
    # Create plots
    print("\nGenerating plots...")
    
    # (a) Modal displacement responses q_n(t)
    plot_modal_displacements(time, q_inch, output_dir)
    
    # Ground motion plot (applied vs measured table)
    plot_ground_motion(time, ug_ddot * 39.3701,
                       meas_time=meas_time,
                       meas_ground_inch=meas_ground_inch,
                       output_dir=output_dir)
    
    # (a-extra) Modal accelerations vs input acceleration (overlay)
    plot_mode_accelerations_vs_input(time, ug_ddot * 39.3701, D_ddot_inch, output_dir)
    
    # (b) Floor displacement responses u_j(t)
    plot_floor_displacements(time, u_inch, output_dir,
                             measured_time=meas_time, measured_disp=meas_disp_inch)
    
    # (c) Floor acceleration responses ü_j(t)
    plot_floor_accelerations(time, u_ddot_inch, output_dir,
                             measured_time=meas_time, measured_acc=meas_acc_inch)
    
    # Diagnostics: floor 3 relative vs absolute, sign, and peak ratios
    if meas_acc_inch is not None and meas_ground_inch is not None:
        u_rel_m = np.zeros_like(u_ddot)  # relative floor accels (exclude ground)
        for j in range(u_rel_m.shape[0]):
            for n in range(D_ddot.shape[0]):
                u_rel_m[j, :] += analyzer.Phi[j, n] * analyzer.Gamma[n] * D_ddot[n, :]
        u_rel_inch = u_rel_m * 39.3701
        meas_rel_inch = [meas_acc_inch[j] - meas_ground_inch for j in range(len(meas_acc_inch))]
        plot_floor3_diagnostics(
            time=time,
            u_abs_inch=u_ddot_inch,
            u_rel_inch=u_rel_inch,
            meas_time=meas_time,
            meas_abs_inch=meas_acc_inch,
            meas_rel_inch=meas_rel_inch,
            output_dir=output_dir
        )
        print_peak_ratios(u_ddot_inch, meas_acc_inch, time)

    # (d) Base shear
    plot_base_shear(time, V_base_kips, output_dir)
    
    # (e) Base overturning moment
    plot_base_moment(time, M_base_kipft, output_dir)
    
    # Create comprehensive summary page
    print("\nCreating Problem 3 summary page...")
    create_problem3_summary(
        mass_matrix=mass_matrix,
        mode_shapes=mode_shapes,
        natural_freqs=natural_freqs,
        damping_ratios=damping_ratios,
        floor_heights=floor_heights,
        time=time,
        ug_ddot=ug_ddot * 39.3701,
        meas_ground=meas_ground_inch,
        q=q_inch,
        u=u_inch,
        u_ddot=u_ddot_inch,
        V_base=V_base_kips,
        M_base=M_base_kipft,
        analyzer=analyzer,
        output_dir=output_dir
    )
    
    print("\nProblem 3 analysis complete!")


def main():
    """Main analysis function."""
    print("=" * 70)
    print("Problem #3: Modal Response Analysis")
    print("=" * 70)
    
    # System properties from LaTeX section (for standalone execution)
    # Mass matrix [kg]
    m = np.array([
        [1180, 0, 0],
        [0, 1180, 0],
        [0, 0, 910]
    ])
    
    # Mass-orthonormal mode shapes (×10^-2)
    Phi = np.array([
        [0.771, -1.916, 2.051],
        [1.755, -1.331, -1.903],
        [2.495, 1.982, 0.914]
    ]) * 1e-2  # Convert from ×10^-2 to actual values
    
    # Natural frequencies [Hz]
    f_n = np.array([2.00, 7.20, 13.75])
    omega_n = 2 * np.pi * f_n  # [rad/s]
    
    # Damping ratios [-]
    zeta = np.array([0.0113, 0.0157, 0.0093])  # 1.13%, 1.57%, 0.93%
    
    # Floor heights [m] - assuming 3.5m per story (adjust as needed)
    floor_heights = np.array([3.5, 7.0, 10.5])  # [m]
    
    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / 'highlighted_htmls'
    output_dir.mkdir(exist_ok=True)
    
    # Run analysis
    run_problem3_analysis(
        mass_matrix=m,
        mode_shapes=Phi,
        natural_freqs=omega_n,
        damping_ratios=zeta,
        floor_heights=floor_heights,
        output_dir=output_dir
    )


def plot_modal_displacements(time, q, output_dir):
    """Plot modal displacement responses q_n(t) [inches]."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'Mode {n+1} Displacement Response q_{n+1}(t)' for n in range(3)],
        vertical_spacing=0.16
    )
    
    colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    
    for n in range(3):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=q[n, :],
                mode='lines',
                name=f'Mode {n+1}',
                line=dict(color=colors[n], width=1.5),
                showlegend=True,
                hovertemplate=(
                    f"<b>Mode {n+1}</b><br>"
                    "Time: %{x:.2f} s<br>"
                    "Modal displacement q%{customdata}: %{y:.4f} in<br>"
                    "Meaning: modal contribution before floor superposition."
                    "<extra></extra>"
                ),
                customdata=np.full_like(time, n+1, dtype=float)
            ),
            row=n+1, col=1
        )
    
    fig.update_layout(
        title=dict(
            text='(a) Modal Displacement Responses q<sub>n</sub>(t)',
            x=0.5,
            y=0.99,
            pad=dict(t=20, b=10),
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif', weight='bold')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=900,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            x=1.02,
            xanchor='left',
            y=1.0,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.15)',
            borderwidth=1
        ),
        margin=dict(l=80, r=200, t=200, b=100)
    )

    # Shift subplot titles upward to sit above the plotting area
    for ann in fig['layout']['annotations']:
        ann['y'] += 0.02
        ann['yanchor'] = 'bottom'
    
    for row in range(1, 4):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Displacement [in]")
    
    output_path = output_dir / 'problem3_modal_displacements.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def _integrate_acc_to_disp(time, acc):
    """Double integrate acceleration → displacement using trapezoidal rule."""
    acc = np.asarray(acc, dtype=float)
    time = np.asarray(time, dtype=float)
    if len(time) < 2:
        return np.zeros_like(time)
    
    vel = np.concatenate(([0.0], _cumtrapz(acc, time)))
    disp = np.concatenate(([0.0], _cumtrapz(vel, time)))
    return disp


def _cumtrapz(y, x):
    """Cumulative trapezoidal integration without SciPy."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros_like(x)
    dx = np.diff(x)
    y_mid = 0.5 * (y[:-1] + y[1:])
    return np.cumsum(y_mid * dx)


def plot_floor_displacements(time, u, output_dir, measured_time=None, measured_disp=None):
    """Plot floor displacement responses u_j(t) [inches]."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'Floor {j+1} Displacement Response u_{j+1}(t)' for j in range(3)],
        vertical_spacing=0.08
    )
    
    colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    floor_names = ['Floor 1', 'Floor 2', 'Floor 3']
    
    for j in range(3):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=u[j, :],
                mode='lines',
                name=floor_names[j],
                line=dict(color=colors[j], width=2),
                showlegend=True,
                hovertemplate=(
                    f"<b>{floor_names[j]}</b><br>"
                    "Time: %{x:.2f} s<br>"
                    "Floor displacement u%{customdata}: %{y:.3f} in<br>"
                    "Meaning: relative to moving ground, converted to inches."
                    "<extra></extra>"
                ),
                customdata=np.full_like(time, j+1, dtype=float)
            ),
            row=j+1, col=1
        )
        
        # Overlay measured displacement if available (derived from measured accelerations)
        if measured_time is not None and measured_disp is not None:
            fig.add_trace(
                go.Scatter(
                    x=measured_time,
                    y=measured_disp[j],
                    mode='markers',
                    name=f'{floor_names[j]} (measured)',
                    marker=dict(color=colors[j], size=5, symbol='circle-open'),
                    line=dict(color=colors[j], width=1, dash='dot'),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{floor_names[j]} measured</b><br>"
                        "Time: %{x:.2f} s<br>"
                        "Disp (via ∫∫acc): %{y:.3f} in<br>"
                        "Computed by double-integrating measured accel."
                        "<extra></extra>"
                    )
                ),
                row=j+1, col=1
            )
    
    fig.update_layout(
        title=dict(
            text='(b) Floor Displacement Responses u<sub>j</sub>(t)',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=900,
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.08),
        margin=dict(l=80, r=140, t=90, b=80)
    )
    
    for row in range(1, 4):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Displacement [in]")
    
    output_path = output_dir / 'problem3_floor_displacements.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_floor_accelerations(time, u_ddot, output_dir, measured_time=None, measured_acc=None):
    """Plot floor acceleration responses ü_j(t) [inches/s²]."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'Floor {j+1} Acceleration Response ü_{j+1}(t)' for j in range(3)],
        vertical_spacing=0.16
    )
    
    colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    floor_names = ['Floor 1', 'Floor 2', 'Floor 3']
    
    for j in range(3):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=u_ddot[j, :],
                mode='lines',
                name=floor_names[j],
                line=dict(color=colors[j], width=2),
                showlegend=True,
                hovertemplate=(
                    f"<b>{floor_names[j]}</b><br>"
                    "Time: %{x:.2f} s<br>"
                    "Total accel ü%{customdata}: %{y:.2f} in/s²<br>"
                    "Includes relative modal accel + ground motion."
                    "<extra></extra>"
                ),
                customdata=np.full_like(time, j+1, dtype=float)
            ),
            row=j+1, col=1
        )
        
        if measured_time is not None and measured_acc is not None:
            fig.add_trace(
                go.Scatter(
                    x=measured_time,
                    y=measured_acc[j],
                    mode='markers',
                    name=f'{floor_names[j]} (measured)',
                    marker=dict(color=colors[j], size=5, symbol='diamond-open'),
                    line=dict(color=colors[j], width=1, dash='dot'),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{floor_names[j]} measured</b><br>"
                        "Time: %{x:.2f} s<br>"
                        "Accel: %{y:.2f} in/s²<br>"
                        "Unit source: item (v) file, converted from g."
                        "<extra></extra>"
                    )
                ),
                row=j+1, col=1
            )
    
    fig.update_layout(
        title=dict(
            text='(c) Floor Acceleration Responses ü<sub>j</sub>(t)',
            x=0.5,
            y=0.995,
            pad=dict(t=10, b=20),
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif', weight='bold')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=900,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            x=1.02,
            xanchor='left',
            y=1.0,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.15)',
            borderwidth=1
        ),
        margin=dict(l=80, r=200, t=200, b=100)
    )

    # Shift subplot titles upward to sit above the plotting area
    for ann in fig['layout']['annotations']:
        ann['y'] += 0.02
        ann['yanchor'] = 'bottom'
    
    for row in range(1, 4):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Acceleration [in/s²]")
    
    output_path = output_dir / 'problem3_floor_accelerations.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_base_shear(time, V_base, output_dir):
    """Plot base shear V_b(t) [kips]."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=V_base,
            mode='lines',
            name='Base Shear',
            line=dict(color=Colors.BERKELEY_BLUE, width=2),
            hovertemplate=(
                "<b>Base shear</b><br>"
                "Time: %{x:.2f} s<br>"
                "V_b: %{y:.2f} kips<br>"
                "Sign convention: positive pushes to +X at base."
                "<extra></extra>"
            )
        )
    )
    
    fig.update_layout(
        title=dict(
            text='(d) Base Shear V<sub>b</sub>(t)',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=500,
        xaxis=get_axis_style(),
        yaxis=get_axis_style(),
        xaxis_title="Time [s]",
        yaxis_title="Base Shear [kips]",
        hovermode='x unified',
        margin=dict(l=80, r=140, t=90, b=70)
    )
    
    output_path = output_dir / 'problem3_base_shear.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_base_moment(time, M_base, output_dir):
    """Plot base overturning moment M_b(t) [kip-ft]."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=M_base,
            mode='lines',
            name='Base Moment',
            line=dict(color=Colors.BERKELEY_BLUE, width=2),
            hovertemplate=(
                "<b>Base overturning moment</b><br>"
                "Time: %{x:.2f} s<br>"
                "M_b: %{y:.2f} kip-ft<br>"
                "Tip: slope of V_b influences rotations at the base."
                "<extra></extra>"
            )
        )
    )
    
    fig.update_layout(
        title=dict(
            text='(e) Base Overturning Moment M<sub>b</sub>(t)',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=500,
        xaxis=get_axis_style(),
        yaxis=get_axis_style(),
        xaxis_title="Time [s]",
        yaxis_title="Base Moment [kip-ft]",
        hovermode='x unified',
        margin=dict(l=80, r=140, t=90, b=70)
    )
    
    output_path = output_dir / 'problem3_base_moment.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_floor3_diagnostics(time, u_abs_inch, u_rel_inch, meas_time, meas_abs_inch, meas_rel_inch, output_dir):
    """Diagnostics for Floor 3: absolute vs relative, sign flip check."""
    idx = 2  # Floor 3 (0-indexed)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Floor 3: Absolute acceleration (computed vs measured, sign check)",
            "Floor 3: Relative acceleration (computed vs measured-ground, sign check)"
        ],
        vertical_spacing=0.12
    )
    color = Colors.BERKELEY_BLUE
    # Absolute
    fig.add_trace(
        go.Scatter(
            x=time, y=u_abs_inch[idx, :],
            mode='lines', name='Computed abs', line=dict(color=color, width=2)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=meas_time, y=meas_abs_inch[idx],
            mode='markers', name='Measured abs', marker=dict(color=color, symbol='diamond-open', size=5)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=meas_time, y=-meas_abs_inch[idx],
            mode='markers', name='Measured abs (flipped)', marker=dict(color='red', symbol='x', size=5)
        ), row=1, col=1
    )
    # Relative
    fig.add_trace(
        go.Scatter(
            x=time, y=u_rel_inch[idx, :],
            mode='lines', name='Computed relative', line=dict(color=color, width=2, dash='solid')
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=meas_time, y=meas_rel_inch[idx],
            mode='markers', name='Measured rel = meas - ground', marker=dict(color=color, symbol='circle-open', size=5)
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=meas_time, y=-meas_rel_inch[idx],
            mode='markers', name='Measured rel (flipped)', marker=dict(color='red', symbol='cross', size=5)
        ), row=2, col=1
    )
    fig.update_layout(
        height=800,
        title=dict(text="Floor 3 Diagnostics: absolute vs relative, sign check", x=0.5, font=dict(size=18, color=Colors.TEXT_DARK)),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(get_axis_style(), row=1, col=1, title_text="Time [s]")
    fig.update_yaxes(get_axis_style(), row=1, col=1, title_text="Acceleration [in/s²]")
    fig.update_xaxes(get_axis_style(), row=2, col=1, title_text="Time [s]")
    fig.update_yaxes(get_axis_style(), row=2, col=1, title_text="Acceleration [in/s²]")
    output_path = output_dir / 'problem3_floor3_diagnostics.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def print_peak_ratios(u_abs_inch, meas_abs_inch, time):
    """Print peak ratios Floor3:2:1 for computed and measured (absolute)."""
    comp_peaks = [np.max(np.abs(u_abs_inch[j, :])) for j in range(3)]
    meas_peaks = [np.max(np.abs(meas_abs_inch[j])) for j in range(3)]
    def ratio(vals):
        return [vals[2]/vals[1], vals[1]/vals[0], vals[2]/vals[0]] if all(v!=0 for v in vals) else [np.nan]*3
    comp_ratio = ratio(comp_peaks)
    meas_ratio = ratio(meas_peaks)
    print("\nPeak abs acceleration (in/s²):")
    print(f"  Computed floors 1/2/3: {comp_peaks}")
    print(f"  Measured floors 1/2/3: {meas_peaks}")
    print("Peak ratios (Floor3:2:1, 2:1, 3:1):")
    print(f"  Computed: {comp_ratio}")
    print(f"  Measured: {meas_ratio}")


def plot_mode_accelerations_vs_input(time, ug_ddot_inch, D_ddot_inch, output_dir):
    """Plot modal accelerations (relative) vs ground acceleration input [in/s²]."""
    n_modes = D_ddot_inch.shape[0]
    colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    
    fig = go.Figure()
    
    # Ground input
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ug_ddot_inch,
            mode='lines',
            name='Ground accel (input)',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate=(
                "<b>Ground acceleration</b><br>"
                "Time: %{x:.2f} s<br>"
                "ü_g: %{y:.2f} in/s²"
                "<extra></extra>"
            )
        )
    )
    
    # Modal accelerations
    for n in range(n_modes):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=D_ddot_inch[n, :],
                mode='lines',
                name=f'Mode {n+1} ü_rel',
                line=dict(color=colors[n], width=1.5),
                hovertemplate=(
                    f"<b>Mode {n+1} relative accel</b><br>"
                    "Time: %{x:.2f} s<br>"
                    "ü_rel: %{y:.2f} in/s²<br>"
                    "From modal equation: D¨ + 2ζω D˙ + ω²D = -ü_g"
                    "<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title=dict(
            text='Modal Accelerations vs Ground Acceleration',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=500,
        xaxis=get_axis_style(),
        yaxis=get_axis_style(),
        xaxis_title="Time [s]",
        yaxis_title="Acceleration [in/s²]",
        hovermode='x unified'
    )
    
    output_path = output_dir / 'problem3_mode_accel_vs_input.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_ground_motion(time, ug_ddot_inch, meas_time, meas_ground_inch, output_dir):
    """Plot ground acceleration (table) [in/s²]."""
    fig = go.Figure()
    if meas_time is not None and meas_ground_inch is not None:
        # Show only measured table points (requested)
        fig.add_trace(
            go.Scatter(
                x=meas_time, y=meas_ground_inch,
                mode='markers', name='Table acceleration (measured)',
                marker=dict(color=Colors.BERKELEY_BLUE, size=4, symbol='diamond'),
                hovertemplate="t=%{x:.2f}s<br>üg (table)=%{y:.2f} in/s²<extra></extra>"
            )
        )
    else:
        # Fallback to applied if measured missing
        fig.add_trace(
            go.Scatter(
                x=time, y=ug_ddot_inch,
                mode='markers', name='Ground acceleration (applied)',
                marker=dict(color=Colors.BERKELEY_BLUE, size=4, symbol='diamond'),
                hovertemplate="t=%{x:.2f}s<br>üg=%{y:.2f} in/s²<extra></extra>"
            )
        )
    fig.update_layout(
        title=dict(text='Ground Acceleration (Table)', x=0.5, font=dict(size=18, color=Colors.TEXT_DARK)),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=400,
        xaxis=get_axis_style(),
        yaxis=get_axis_style(),
        xaxis_title="Time [s]",
        yaxis_title="Acceleration [in/s²]",
        hovermode='x unified'
    )
    output_path = output_dir / 'problem3_ground_motion.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


if __name__ == '__main__':
    main()

