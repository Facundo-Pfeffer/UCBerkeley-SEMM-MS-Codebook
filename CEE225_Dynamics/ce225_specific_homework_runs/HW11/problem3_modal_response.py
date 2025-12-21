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
    acc_measured : list of np.ndarray
        Measured accelerations for each floor [m/s²]
    """
    data_loader = DataLoader(data_dir)
    
    # Try to load from a file that might contain measured data
    # For now, return None - user can specify file
    return None, None


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
    
    # Compute modal coordinates q_n(t) = Γ_n D_n(t)
    q = analyzer.compute_modal_coordinates(D)
    
    # Compute floor responses (include ground motion for total accelerations)
    print("Computing floor responses...")
    u, u_dot, u_ddot = analyzer.compute_floor_responses(D, D_dot, D_ddot, ug_ddot=ug_ddot)
    
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
    
    # Base shear: N to kips
    V_base_kips = V_base / 4448.22  # N to kips
    
    # Base moment: N·m to kip-ft
    M_base_kipft = M_base * 0.737562  # N·m to kip-ft
    
    # Create plots
    print("\nGenerating plots...")
    
    # (a) Modal displacement responses q_n(t)
    plot_modal_displacements(time, q_inch, output_dir)
    
    # (b) Floor displacement responses u_j(t)
    plot_floor_displacements(time, u_inch, output_dir)
    
    # (c) Floor acceleration responses ü_j(t)
    plot_floor_accelerations(time, u_ddot_inch, output_dir)
    
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
        vertical_spacing=0.08
    )
    
    colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
    
    for n in range(3):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=q[n, :],
                mode='lines',
                name=f'Mode {n+1}',
                line=dict(color=colors[n], width=2),
                showlegend=(n == 0)
            ),
            row=n+1, col=1
        )
    
    fig.update_layout(
        title=dict(
            text='(a) Modal Displacement Responses q<sub>n</sub>(t)',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=900,
        showlegend=True
    )
    
    for row in range(1, 4):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Displacement [in]")
    
    output_path = output_dir / 'problem3_modal_displacements.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_floor_displacements(time, u, output_dir):
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
                showlegend=(j == 0)
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
        showlegend=True
    )
    
    for row in range(1, 4):
        fig.update_xaxes(get_axis_style(), row=row, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=row, col=1, title_text="Displacement [in]")
    
    output_path = output_dir / 'problem3_floor_displacements.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


def plot_floor_accelerations(time, u_ddot, output_dir):
    """Plot floor acceleration responses ü_j(t) [inches/s²]."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'Floor {j+1} Acceleration Response ü_{j+1}(t)' for j in range(3)],
        vertical_spacing=0.08
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
                showlegend=(j == 0)
            ),
            row=j+1, col=1
        )
    
    fig.update_layout(
        title=dict(
            text='(c) Floor Acceleration Responses ü<sub>j</sub>(t)',
            x=0.5,
            font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
        ),
        plot_bgcolor=Colors.BG_LIGHT,
        paper_bgcolor=Colors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=900,
        showlegend=True
    )
    
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
            line=dict(color=Colors.BERKELEY_BLUE, width=2)
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
        yaxis_title="Base Shear [kips]"
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
            line=dict(color=Colors.BERKELEY_BLUE, width=2)
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
        yaxis_title="Base Moment [kip-ft]"
    )
    
    output_path = output_dir / 'problem3_base_moment.html'
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Generated: {output_path}")


if __name__ == '__main__':
    main()

