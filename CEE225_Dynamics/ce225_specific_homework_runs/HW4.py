import numpy as np
import plotly.graph_objects as go


def run_assignment_1(cycles_number):

    # Define the function u(t) / (u_st)_0
    def response(t, a, omega_n):
        factor = 1 / (1 + (a / omega_n)**2)
        return factor * ((a / omega_n) * np.sin(omega_n * t) - np.cos(omega_n * t) + np.exp(-a * t))

    # Define parameters
    omega_n = 2 * np.pi  # Natural frequency (rad/s) → makes T = 1s
    T = 1 / (omega_n / (2 * np.pi))  # Period = 1s
    a_values = [0.02 * omega_n, 0.2 * omega_n, 2.0 * omega_n]  # a/omega_n = 0.02, 0.2, 2.0
    colors = ['royalblue', 'darkorange', 'firebrick']
    labels = [r"$\frac{a}{\omega_n}=0.02$", r"$\frac{a}{\omega_n}=0.2$", r"$\frac{a}{\omega_n}=2.0$"]

    # Time domain: from 0 to 5T
    discretization = cycles_number * 1000
    t = np.linspace(0, cycles_number * T, discretization)
    t_T = t / T  # Time normalized by period

    # Create the plot
    fig = go.Figure()

    for a, color, label in zip(a_values, colors, labels):
        u = response(t, a, omega_n)
        fig.add_trace(go.Scatter(
            x=t_T,
            y=u,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate="t/T = %{x:.2f}<br>u(t)/(u_st)₀ = %{y:.3f}<extra></extra>"
        ))

    # Customize layout
    fig.update_layout(
        title=r"$\text{Normalized Response}~\frac{u(t)}{(u_{st})_0}~\text{vs.}~\frac{t}{T}$",
        xaxis_title=r"$\huge\frac{t}{T}$",
        yaxis_title=r"$\huge\frac{u(t)}{(u_{st})_0}$",
        template="plotly_white",
        font=dict(size=20),
        legend=dict(x=0.90, y=0.98, xanchor='right', yanchor='top', bgcolor="rgba(255,255,255,0.7)"),
        autosize=True,
        height=650,
        margin=dict(l=100, r=40, t=80, b=80),
        title_x=0.5,
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=20)
        )
    )

    fig.show(config={
        "responsive": True,
        "mathjax": {
            "scale": 150  # Increase scale to make LaTeX text larger
        }
    })


def run_assignment_2(cycles_number):
    # Define the function u(t) = v(t - (1/omega_n) * sin(omega_n * t))
    def response(t, v, omega_n):
        return v * (t - (1 / omega_n) * np.sin(omega_n * t))

    # Define the linear function vt for comparison
    def linear_response(t, v):
        return v * t

    # Define parameters (reusing logic from assignment 1)
    omega_n = 2 * np.pi  # Natural frequency (rad/s) → makes T = 1s
    T = 1 / (omega_n / (2 * np.pi))  # Period = 1s
    v_values = [0.5, 1.0, 2.0]  # Different v values
    colors = ['royalblue', 'darkorange', 'firebrick']
    labels = [r"$v=0.5[in/s]$.", r"$v=1.0[in/s]$.", r"$v=2.0[in/s]$."]

    # Time domain: from 0 to cycles_number*T
    discretization = cycles_number * 1000
    t = np.linspace(0, cycles_number * T, discretization)
    t_T = t / T  # Time normalized by period

    # Create the plot
    fig = go.Figure()

    for v, color, label in zip(v_values, colors, labels):
        # Add the full response u(t) = v(t - (1/omega_n) * sin(omega_n * t))
        u = response(t, v, omega_n)
        fig.add_trace(go.Scatter(
            x=t_T,
            y=u,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate="t/T = %{x:.2f}<br>u(t) [in]= %{y:.3f}<extra></extra>"
        ))

        # Add the linear response vt as dashed lines
        u_linear = linear_response(t, v)
        fig.add_trace(go.Scatter(
            x=t_T,
            y=u_linear,
            mode='lines',
            name=f"{label} (linear)",
            line=dict(color=color, width=2, dash='dash'),
            hovertemplate="t/T = %{x:.2f}<br>vt = %{y:.3f}<extra></extra>",
            showlegend=False  # Don't clutter legend
        ))

    # Reusing styling from assignment 1)
    fig.update_layout(
        title=r"$\text{Response}~u(t) = v\left(t - \frac{1}{\omega_n}\sin(\omega_n t)\right)~\text{vs.}~\frac{t}{T}$",
        xaxis_title=r"$\huge\frac{t}{T}$",
        yaxis_title=r"$\huge u(t)[in]$",
        template="plotly_white",
        font=dict(size=20),
        legend=dict(
            x=0.02,  # small left offset
            y=0.98,
            xanchor='left',  # allow box to expand to the right
            yanchor='top',
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=1,
            bordercolor="rgba(0,0,0,0.15)",
            font=dict(size=20),
            itemsizing="trace"
        ),
        autosize=True,
        height=650,
        margin=dict(l=100, r=40, t=100, b=80),
        title_x=0.5,
        xaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title=dict(font=dict(size=32)),
            tickfont=dict(size=20)
        )
    )

    fig.show(config={
        "responsive": True,
        # "mathjax": {
            "scale": 150  # Increase scale to make LaTeX text larger
        # }
    })


if __name__ == "__main__":
    run_assignment_1(50)
    run_assignment_2(3)