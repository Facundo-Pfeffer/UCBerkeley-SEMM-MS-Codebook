import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_torsional_stiffness(b, c, mu, num_terms):
    """
    Compute torsional stiffness k_T as a function of the number of terms.
    
    Parameters:
    -----------
    b : float
        Cross-sectional dimension [mm]
    c : float
        Cross-sectional dimension [mm]
    mu : float
        Shear modulus [kN/mm²]
    num_terms : int
        Number of terms to evaluate (e.g., num_terms=50 means 50 odd terms: n=1,3,5,...,99)
    
    Returns:
    --------
    n_odd_values : np.ndarray
        Array of odd n values [1, 3, 5, 7, ..., (2*num_terms-1)]
    kT_values : np.ndarray
        Cumulative torsional stiffness at each term
    term_numbers : np.ndarray
        Array of term numbers [1, 2, 3, ..., num_terms]
    """
    n_odd_values = []  # The actual odd n values (1, 3, 5, ...)
    kT_values = []
    term_numbers = []  # The term count (1st term, 2nd term, ...)
    kT = 0.0
    
    for i in range(num_terms):
        term_number = i + 1  # 1st term, 2nd term, 3rd term, etc.
        n_odd = 2 * i + 1  # odd values only: 1, 3, 5, 7, ...
        
        sin_term = np.sin(n_odd * np.pi / 2)
        
        if np.isclose(sin_term, 0.0):
            continue

        tanh_term = np.tanh(n_odd * np.pi * c / (2 * b))
        bracket = c - (2 * b / (n_odd * np.pi)) * tanh_term
        term = (32 * mu * b**3 * sin_term**2) / (n_odd**4 * np.pi**4) * bracket
        kT += term

        n_odd_values.append(n_odd)
        kT_values.append(kT)
        term_numbers.append(term_number)
    
    return np.array(n_odd_values), np.array(kT_values), np.array(term_numbers)


def plot_convergence_analysis(b, c, mu, num_terms=500):
    """
    Create a professional Plotly visualization showing convergence of torsional stiffness.
    
    Parameters:
    -----------
    b : float
        Cross-sectional dimension [mm]
    c : float
        Cross-sectional dimension [mm]
    mu : float
        Shear modulus [kN/mm²]
    num_terms : int
        Number of terms to evaluate (default: 500)
        Note: num_terms=50 means 50 odd terms with n values from 1 to 99
    """
    print("=" * 70)
    print("CEE 231 - Assignment 7: Convergence Analysis of Torsional Stiffness")
    print("=" * 70)
    print(f"\nInput Parameters:")
    print(f"  μ  = {mu} kN/mm²")
    print(f"  b  = {b} mm")
    print(f"  c  = {c} mm")
    print(f"  Number of terms to analyze: {num_terms}")
    print(f"  (This corresponds to odd n values from 1 to {2*num_terms-1})\n")
    
    # Compute convergence data
    n_odd_values, kT_values, term_numbers = compute_torsional_stiffness(b, c, mu, num_terms)
    
    # Compute relative change between consecutive terms
    relative_change = np.abs(np.diff(kT_values) / kT_values[1:]) * 100
    
    # Final converged value
    kT_final = kT_values[-1]
    n_max = n_odd_values[-1]
    print(f"Converged torsional stiffness k_T ≈ {kT_final:.4f} kN·mm²")
    print(f"Number of terms used: {len(term_numbers)}")
    print(f"Maximum n value reached: {n_max} (odd)")
    print(f"Final relative change: {relative_change[-1]:.2e}%\n")
    
    # Create subplot figure with two plots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Torsional Stiffness Convergence',
            'Relative Change Between Consecutive Terms'
        ),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"type": "scatter", "secondary_y": False}]]
    )
    
    # Main convergence plot
    fig.add_trace(
        go.Scatter(
            x=term_numbers,
            y=kT_values,
            mode='lines+markers',
            name='k<sub>T</sub>(N)',
            line=dict(color='#2E86AB', width=2.5),
            marker=dict(size=4, color='#2E86AB', symbol='circle'),
            customdata=n_odd_values,
            hovertemplate='<b>Term</b>: %{x}<br><b>n (odd)</b>: %{customdata}<br><b>k<sub>T</sub></b>: %{y:.4f} kN·mm²<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add horizontal line for converged value
    fig.add_hline(
        y=kT_final,
        line_dash="dash",
        line_color="rgba(220, 53, 69, 0.6)",
        annotation_text=f"Converged: {kT_final:.4f} kN·mm²",
        annotation_position="right",
        row=1, col=1
    )
    
    # Relative change plot (log scale)
    fig.add_trace(
        go.Scatter(
            x=term_numbers[1:],
            y=relative_change,
            mode='lines+markers',
            name='Relative Change',
            line=dict(color='#A23B72', width=2),
            marker=dict(size=4, color='#A23B72', symbol='diamond'),
            customdata=n_odd_values[1:],
            hovertemplate='<b>Term</b>: %{x}<br><b>n (odd)</b>: %{customdata}<br><b>Change</b>: %{y:.2e}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Terms (N)", row=1, col=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_xaxes(title_text="Number of Terms (N)", row=2, col=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(title_text="k<sub>T</sub> [kN·mm²]", row=1, col=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(
        title_text="Relative Change [%]",
        type="log",
        row=2, col=1,
        gridcolor='rgba(200,200,200,0.3)'
    )
    
    # Update layout for professional appearance
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Convergence Analysis: Torsional Stiffness</b><br>"
                f"<sub>μ = {mu} kN/mm², b = {b} mm, c = {c} mm</sub>"
            ),
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        template="plotly_white",
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Display the figure
    fig.show()
    
    # Export as HTML
    output_filename = f"CEE231_HW7_Convergence_Analysis.html"
    fig.write_html(output_filename)
    print(f"✓ Plot saved to: {output_filename}")
    print("=" * 70)


def main():
    """Main function to run the convergence analysis."""
    # Input parameters (from HW5)
    b = 10.0   # mm
    c = 20.0   # mm
    mu = 82.0  # kN/mm²
    
    # Number of terms to evaluate (sufficiently large for convergence)
    # num_terms = 50 means we evaluate 50 terms with odd n values: 1, 3, 5, ..., 99
    # num_terms = 500 means we evaluate 500 terms with odd n values: 1, 3, 5, ..., 999
    num_terms = 50
    
    # Run convergence analysis with professional Plotly visualization
    plot_convergence_analysis(b, c, mu, num_terms)


if __name__ == "__main__":
    main()
