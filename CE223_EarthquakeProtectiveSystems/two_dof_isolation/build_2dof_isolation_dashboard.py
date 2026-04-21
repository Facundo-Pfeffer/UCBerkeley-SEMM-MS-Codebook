from __future__ import annotations

"""
CE223 – 2-DOF Base-Isolated Building (Kobe KBU090) — Part (b)

This script builds a fully self-contained HTML report that:
- Defines the 2-DOF base-isolated model (u_b, u_s relative to ground),
- Runs direct MDOF Newmark time-history analysis under Kobe RSN1108 KBU090,
- Computes peak engineering demands:
    * max story drift: max |u_s - u_b|
    * max isolation displacement: max |u_b|
    * base shear coefficient: max |V_b| / (M g)
- Compares against fixed-base SDOF response (T_s, ζ_s),
- Computes modal properties (T_n, Φ, Γ_n, ζ_n) using classical modal damping,
- Builds response spectra for the modal damping ratios and estimates peaks via SRSS RSA.

Run:
    python build_2dof_isolation_dashboard.py

Output:
    CE223_EarthquakeProtectiveSystems/highlighted_htmls/CE223_2DOF_Isolation_Kobe090.html
"""

import math
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
CE223_DIR = BASE_DIR.parent
INPUT_GM_DIR = CE223_DIR / "input_ground_motion"
HIGHLIGHTED_HTML_DIR = CE223_DIR / "highlighted_htmls"
HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)

GM_FILENAME = "RSN1108_KOBE_KBU090.AT2"

G_SI = 9.80665  # m/s^2


def load_peer_at2(path: Path) -> tuple[np.ndarray, float, str]:
    """
    Load a PEER .AT2 file as (ug_ddot [m/s^2], dt [s], label).

    Assumes acceleration values are in units of g.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground motion file not found: {path}")

    dt = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "DT=" in line.upper():
                # Example: "NPTS=  3200, DT= 0.0100 SEC"
                up = line.upper()
                try:
                    dt_part = up.split("DT=")[1]
                    dt_str = dt_part.split("SEC")[0].replace(",", " ").strip().split()[0]
                    dt = float(dt_str)
                except Exception:
                    dt = None
                if dt is not None:
                    break

    if dt is None:
        raise ValueError(f"Could not parse DT from header of {path}")

    acc_g = np.loadtxt(path, comments="%")
    acc_g = np.asarray(acc_g, dtype=float).ravel()
    if acc_g.size < 2:
        raise ValueError(f"Ground motion file {path} has too few samples.")

    ug_ddot = acc_g * G_SI
    return ug_ddot, dt, path.name


def newmark_linear_mdoff_base_excitation(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    ug_ddot: np.ndarray,
    dt: float,
    r: np.ndarray | None = None,
    beta: float = 1.0 / 4.0,
    gamma: float = 1.0 / 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve: M u¨ + C u˙ + K u = -M r ü_g(t), with u relative to ground.
    r = [1, 1, ..., 1]^T.

    Returns (u, u_dot, u_ddot), each shaped (n_steps, dof).
    """
    ug_ddot = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug_ddot.size
    dof = int(M.shape[0])
    if r is None:
        r_vec = np.ones(dof, dtype=float)
    else:
        r_vec = np.asarray(r, dtype=float).ravel()
        if r_vec.size != dof:
            raise ValueError(f"Influence vector r must have length {dof}, got {r_vec.size}")

    u = np.zeros((n, dof), dtype=float)
    ud = np.zeros((n, dof), dtype=float)
    udd = np.zeros((n, dof), dtype=float)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    Keff = K + a0 * M + a1 * C
    mr = M @ r_vec

    for i in range(1, n):
        p_i = -mr * ug_ddot[i]
        p_eff = (
            p_i
            + M @ (a0 * u[i - 1] + a2 * ud[i - 1] + a3 * udd[i - 1])
            + C @ (a1 * u[i - 1] + a4 * ud[i - 1] + a5 * udd[i - 1])
        )

        u[i] = np.linalg.solve(Keff, p_eff)
        udd[i] = a0 * (u[i] - u[i - 1]) - a2 * ud[i - 1] - a3 * udd[i - 1]
        ud[i] = ud[i - 1] + dt * ((1.0 - gamma) * udd[i - 1] + gamma * udd[i])

    return u, ud, udd


def sdof_response_spectrum(
    ug_ddot: np.ndarray,
    dt: float,
    periods: np.ndarray,
    zeta: float,
    substeps: int = 5,
) -> dict[str, np.ndarray]:
    """
    Linear elastic SDOF response spectrum (relative coordinates):

        u¨ + 2 ζ ω u˙ + ω^2 u = -ü_g(t)

    Returns arrays for:
      - Sd [m]   (peak |u|)
      - Sa [g]   (pseudo-acc = ω^2 Sd / g)
    """
    periods = np.asarray(periods, dtype=float).ravel()
    Sd = np.zeros_like(periods)
    Sa_g = np.zeros_like(periods)

    ug_ddot = np.asarray(ug_ddot, dtype=float).ravel()
    t = np.arange(ug_ddot.size, dtype=float) * float(dt)

    substeps = int(substeps)
    if substeps < 1:
        substeps = 1
    dt_int = float(dt) / float(substeps)
    t_int = np.arange((ug_ddot.size - 1) * substeps + 1, dtype=float) * dt_int
    ug_ddot_int = np.interp(t_int, t, ug_ddot)

    for j, T in enumerate(periods):
        omega = 2.0 * math.pi / max(T, 1e-6)
        M = np.array([[1.0]])
        K = np.array([[omega * omega]])
        C = np.array([[2.0 * zeta * omega]])

        u, _, _ = newmark_linear_mdoff_base_excitation(M, C, K, ug_ddot_int, dt_int, r=np.array([1.0]))
        Sd_j = float(np.max(np.abs(u[:, 0])))
        Sd[j] = Sd_j
        Sa_g[j] = (omega * omega * Sd_j) / G_SI

    return {"T": periods, "Sd": Sd, "Sa_g": Sa_g}


def build_isolated_matrices(
    mb: float,
    ms: float,
    Ts: float,
    Tb: float,
    zeta_s: float,
    zeta_b: float,
) -> dict[str, np.ndarray | float]:
    """
    DOFs: u = [u_b, u_s]^T (isolation displacement relative to ground; superstructure displacement relative to isolation level).

    This coordinate convention matches the CEE223 lecture material:
      - u_b is the deformation of (k_b, c_b)
      - u_s is the deformation of (k_s, c_s)
    and therefore the influence vector is r = [1, 0]^T.
    """
    m = ms
    Mtot = mb + m
    omega_s = 2.0 * math.pi / Ts
    omega_b = 2.0 * math.pi / Tb

    ks = m * omega_s**2
    kb = Mtot * omega_b**2

    cs = 2.0 * zeta_s * omega_s * m
    cb = 2.0 * zeta_b * omega_b * Mtot

    # Preface coordinate form (symmetric M, diagonal C and K)
    # M := m_b + m
    # [ M  m ] [ü_b] + [c_b  0] [u̇_b] + [k_b  0] [u_b] = - [M] ü_g
    # [ m  m ] [ü_s]   [ 0  c_s] [u̇_s]   [ 0  k_s] [u_s]   - [m] ü_g
    M = np.array([[Mtot, m], [m, m]], dtype=float)
    K = np.array([[kb, 0.0], [0.0, ks]], dtype=float)
    C = np.array([[cb, 0.0], [0.0, cs]], dtype=float)
    r = np.array([1.0, 0.0], dtype=float)

    return dict(
        M=M,
        C=C,
        K=K,
        r=r,
        m=m,
        Mtot=Mtot,
        ks=ks,
        kb=kb,
        cs=cs,
        cb=cb,
        omega_s=omega_s,
        omega_b=omega_b,
    )


def modal_properties(
    M: np.ndarray, C: np.ndarray, K: np.ndarray, r: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Undamped modes (K φ = ω^2 M φ), with mode shapes scaled so that the
    base-displacement component is 1 (φ_{b,n} = 1). For each mode:

      m_n = φ_n^T M φ_n          (modal mass)
      Γ_n = (φ_n^T M r) / m_n    (participation factor for base excitation)
      ζ_n = (φ_n^T C φ_n) / (2 ω_n m_n)
      M*_n = Γ_n^2 m_n           (effective modal mass for base excitation)
    """
    dof = M.shape[0]
    r_vec = np.asarray(r, dtype=float).ravel()
    if r_vec.size != dof:
        raise ValueError(f"Influence vector r must have length {dof}, got {r_vec.size}")
    A = np.linalg.solve(M, K)
    lam, vec = np.linalg.eig(A)
    lam = np.real(lam)
    vec = np.real(vec)
    idx = np.argsort(np.sqrt(np.maximum(lam, 0.0)))
    lam = lam[idx]
    vec = vec[:, idx]

    omegas = np.sqrt(np.maximum(lam, 0.0))

    # Scale eigenvectors so that base DOF displacement is 1 for every mode
    Phi = np.zeros_like(vec)
    for i in range(dof):
        v = vec[:, i]
        base_val = v[0]
        scale = 1.0 / base_val if base_val != 0.0 else 1.0
        Phi[:, i] = v * scale

    # Modal mass, participation factors, damping ratios, and effective modal masses
    m_modal = np.array([float(Phi[:, i].T @ M @ Phi[:, i]) for i in range(dof)])
    num = np.array([float(Phi[:, i].T @ M @ r_vec) for i in range(dof)])
    Gamma = num / m_modal

    zeta = np.array(
        [float(Phi[:, i].T @ C @ Phi[:, i]) / (2.0 * omegas[i] * m_modal[i]) for i in range(dof)]
    )
    M_eff = Gamma**2 * m_modal
    M_tot = float(r_vec.T @ M @ r_vec)

    return dict(
        omega=omegas,
        T=2.0 * math.pi / omegas,
        Phi=Phi,
        Gamma=Gamma,
        zeta=zeta,
        M_eff=M_eff,
        M_tot=M_tot,
    )


def peaks_from_time_history(
    u: np.ndarray,
    ud: np.ndarray,
    kb: float,
    cb: float,
    Mtot: float,
) -> dict[str, float]:
    u_b = u[:, 0]  # isolation displacement (relative to ground)
    u_s = u[:, 1]  # superstructure deformation (relative to isolation level)
    ud_b = ud[:, 0]

    drift = u_s
    iso = u_b
    Vb = kb * u_b + cb * ud_b
    Cv = np.max(np.abs(Vb)) / (Mtot * G_SI)

    return dict(
        drift_max=float(np.max(np.abs(drift))),
        iso_max=float(np.max(np.abs(iso))),
        Cv_max=float(Cv),
    )


def main() -> None:
    # -------- Given data --------
    Ts = 0.5
    Tb = 2.0
    zeta_s = 0.02
    zeta_b = 0.15

    # Use arbitrary absolute mass scale; results for displacements and coefficients are mass-scale invariant
    mb = 1.0
    ms = 1.5 * mb

    gm_path = INPUT_GM_DIR / GM_FILENAME
    ug_ddot, dt, gm_label = load_peer_at2(gm_path)
    t = np.arange(ug_ddot.size) * dt
    ug_g = ug_ddot / G_SI

    mats = build_isolated_matrices(mb=mb, ms=ms, Ts=Ts, Tb=Tb, zeta_s=zeta_s, zeta_b=zeta_b)
    M, C, K = mats["M"], mats["C"], mats["K"]
    r = mats["r"]
    kb, cb, Mtot = float(mats["kb"]), float(mats["cb"]), float(mats["Mtot"])
    ks, cs = float(mats["ks"]), float(mats["cs"])
    m = float(mats["m"])

    # -------- Direct MDOF time-history --------
    u_dir, ud_dir, udd_dir = newmark_linear_mdoff_base_excitation(M, C, K, ug_ddot, dt, r=r)
    peaks_dir = peaks_from_time_history(u_dir, ud_dir, kb=kb, cb=cb, Mtot=Mtot)

    # -------- Fixed-base SDOF time-history --------
    omega_s = float(mats["omega_s"])
    M_fb = np.array([[m]])
    K_fb = np.array([[ks]])
    C_fb = np.array([[cs]])
    u_fb, ud_fb, _ = newmark_linear_mdoff_base_excitation(M_fb, C_fb, K_fb, ug_ddot, dt, r=np.array([1.0]))
    drift_fb = float(np.max(np.abs(u_fb[:, 0])))
    V_fb = ks * u_fb[:, 0] + cs * ud_fb[:, 0]
    # For the fixed-base comparison SDOF, the base shear coefficient is normalized by the superstructure mass m
    Cv_fb = float(np.max(np.abs(V_fb)) / (m * G_SI))

    # -------- Modal properties + modal time-history (2-mode superposition) --------
    mp = modal_properties(M, C, K, r=r)
    Phi = mp["Phi"]
    omegas = mp["omega"]
    Tn = mp["T"]
    Gamma = mp["Gamma"]
    zeta_n = mp["zeta"]
    M_eff = mp["M_eff"]
    M_tot = mp["M_tot"]

    # Modal time-history (classical damping approximation)
    q = np.zeros((ug_ddot.size, 2), dtype=float)
    qd = np.zeros((ug_ddot.size, 2), dtype=float)
    for i in range(2):
        Mq = np.array([[1.0]])
        Kq = np.array([[omegas[i] ** 2]])
        Cq = np.array([[2.0 * zeta_n[i] * omegas[i]]])
        q_i, qd_i, _ = newmark_linear_mdoff_base_excitation(Mq, Cq, Kq, Gamma[i] * ug_ddot, dt, r=np.array([1.0]))
        q[:, i] = q_i[:, 0]
        qd[:, i] = qd_i[:, 0]

    u_modal = (Phi @ q.T).T
    ud_modal = (Phi @ qd.T).T
    peaks_modal = peaks_from_time_history(u_modal, ud_modal, kb=kb, cb=cb, Mtot=Mtot)

    # -------- Response spectra for modal damping ratios --------
    # Response spectrum grid (higher resolution for smoother plotted curves)
    periods = np.linspace(0.05, 5.0, 320)
    # Use a smaller internal integration step for response spectra (substepping)
    # Keep modest to avoid long build times while improving accuracy.
    spec_substeps = 2
    spec_mode1 = sdof_response_spectrum(ug_ddot, dt, periods, float(zeta_n[0]), substeps=spec_substeps)
    spec_mode2 = sdof_response_spectrum(ug_ddot, dt, periods, float(zeta_n[1]), substeps=spec_substeps)
    spec_fb = sdof_response_spectrum(ug_ddot, dt, periods, zeta_s, substeps=spec_substeps)

    # Interpolate spectra at modal periods
    Sd1 = float(np.interp(Tn[0], spec_mode1["T"], spec_mode1["Sd"]))
    Sd2 = float(np.interp(Tn[1], spec_mode2["T"], spec_mode2["Sd"]))
    Sa1_g = float(np.interp(Tn[0], spec_mode1["T"], spec_mode1["Sa_g"]))
    Sa2_g = float(np.interp(Tn[1], spec_mode2["T"], spec_mode2["Sa_g"]))

    # Fixed-base RSA (single-mode; SRSS is trivial)
    Sd_fb = float(np.interp(Ts, spec_fb["T"], spec_fb["Sd"]))
    Sa_fb_g = float(np.interp(Ts, spec_fb["T"], spec_fb["Sa_g"]))
    drift_fb_rsa = Sd_fb
    # Fixed-base RSA base shear coefficient, normalized by m
    Cv_fb_rsa = float((m * (Sa_fb_g * G_SI)) / (m * G_SI))

    # RSA (SRSS) for drift and isolation displacement using spectral displacement
    # q_n,peak = |Γ_n| * Sd(Tn, ζn)
    q1 = abs(float(Gamma[0])) * Sd1
    q2 = abs(float(Gamma[1])) * Sd2
    # Modal contributions to response quantities (in the CEE223 lecture coordinate convention):
    #   u_b is isolation deformation; u_s is superstructure deformation (i.e., drift)
    ub1 = float(Phi[0, 0]) * q1
    ub2 = float(Phi[0, 1]) * q2
    drift1 = float(Phi[1, 0]) * q1
    drift2 = float(Phi[1, 1]) * q2

    iso_rsa = math.sqrt(ub1**2 + ub2**2)
    drift_rsa = math.sqrt(drift1**2 + drift2**2)

    # RSA base shear coefficient using effective modal mass and pseudo-acceleration
    # Vb_n ≈ M*_n * Sa_n, combine SRSS
    Vb1 = float(M_eff[0]) * (Sa1_g * G_SI)
    Vb2 = float(M_eff[1]) * (Sa2_g * G_SI)
    Cv_rsa = math.sqrt((Vb1 / (M_tot * G_SI)) ** 2 + (Vb2 / (M_tot * G_SI)) ** 2)

    # -------- Figures --------
    # Ground motion
    fig_gm = go.Figure()
    fig_gm.add_trace(go.Scatter(x=t, y=ug_g, mode="lines", name="ü_g(t) [g]", line=dict(color="rgb(0,55,95)", width=2.2)))
    fig_gm.update_layout(
        template="plotly_white",
        height=380,
        title=dict(
            text=f"Input ground motion — {gm_label} (Kobe 1995, 090 component)",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="Acceleration [g]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )

    # Time histories: isolator displacement and drift (direct vs modal)
    drift_dir = u_dir[:, 1]
    drift_mod = u_modal[:, 1]
    fig_th = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Isolation displacement u_b(t)", "Story drift Δ(t) = u_s(t)"),
    )
    fig_th.add_trace(
        go.Scatter(x=t, y=u_dir[:, 0] * 100.0, mode="lines", name="Direct time history (MDOF)", line=dict(color="rgb(0,55,95)", width=2.2)),
        row=1,
        col=1,
    )
    fig_th.add_trace(go.Scatter(x=t, y=u_modal[:, 0] * 100.0, mode="lines", name="Modal superposition", line=dict(color="rgb(220,38,38)", width=2.0, dash="dash")), row=1, col=1)
    fig_th.add_trace(
        go.Scatter(
            x=t,
            y=drift_dir * 100.0,
            mode="lines",
            name="Direct time history (MDOF)",
            line=dict(color="rgb(0,55,95)", width=2.2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig_th.add_trace(go.Scatter(x=t, y=drift_mod * 100.0, mode="lines", name="Modal superposition", line=dict(color="rgb(220,38,38)", width=2.0, dash="dash"), showlegend=False), row=2, col=1)
    fig_th.update_layout(
        template="plotly_white",
        height=520,
        title=dict(
            text="Time-history response (isolated building): direct vs modal superposition",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )
    fig_th.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig_th.update_yaxes(title_text="u_b [cm]", row=1, col=1)
    fig_th.update_yaxes(title_text="Δ [cm]", row=2, col=1)

    # Response spectra (pseudo-acceleration)
    fig_spec = go.Figure()
    blue = "rgb(0,114,189)"   # MATLAB-like blue for mode 1
    red = "rgb(220,38,38)"    # red for mode 2
    fig_spec.add_trace(
        go.Scatter(
            x=spec_mode1["T"],
            y=spec_mode1["Sa_g"],
            mode="lines",
            name=f"Sa(T), ζ₁={zeta_n[0]*100:.1f}%",
            line=dict(color=blue, width=2.2),
        )
    )
    fig_spec.add_trace(
        go.Scatter(
            x=spec_mode2["T"],
            y=spec_mode2["Sa_g"],
            mode="lines",
            name=f"Sa(T), ζ₂={zeta_n[1]*100:.1f}%",
            line=dict(color=red, width=2.2),
        )
    )
    fig_spec.add_trace(
        go.Scatter(
            x=spec_fb["T"],
            y=spec_fb["Sa_g"],
            mode="lines",
            name="Sa(T), ζ=2% fixed-base",
            line=dict(color="rgb(0,0,0)", width=2.0, dash="dash"),
        )
    )
    fig_spec.add_trace(
        go.Scatter(
            x=[Tn[0]],
            y=[Sa1_g],
            mode="markers",
            name="Mode 1 ordinate",
            marker=dict(size=10, color=blue, symbol="diamond"),
        )
    )
    fig_spec.add_trace(
        go.Scatter(
            x=[Tn[1]],
            y=[Sa2_g],
            mode="markers",
            name="Mode 2 ordinate",
            marker=dict(size=10, color=red, symbol="diamond"),
        )
    )
    fig_spec.update_layout(
        template="plotly_white",
        height=380,
        title=dict(
            text="Response spectra (pseudo-acceleration) for modal damping ratios",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        xaxis=dict(title="Period T [s]", range=[0.0, 5.0]),
        yaxis=dict(title="Sa(T) [g]"),
        legend=dict(orientation="h", yanchor="top", y=-0.30, x=0.5, xanchor="center"),
        margin=dict(t=80, b=150),
    )

    # Mode shape table (numeric, mass-normalized) – preferred over a bar chart
    table_phi = "\n".join(
        [
            "<tr>"
            f"<td>{dof_name}</td>"
            f"<td>{Phi[i,0]: .6f}</td>"
            f"<td>{Phi[i,1]: .6f}</td>"
            "</tr>"
            for i, dof_name in enumerate(["u_b (isolation deformation)", "u_s (superstructure deformation / drift)"])
        ]
    )

    # -------- HTML helpers --------
    def _to_div(fig: go.Figure, include_js: bool = False) -> str:
        return to_html(fig, include_plotlyjs=include_js, full_html=False, config=dict(displayModeBar=True, responsive=True))

    # Embed Plotly once (first figure)
    gm_html = _to_div(fig_gm, include_js=True)
    th_html = _to_div(fig_th, include_js=False)
    spec_html = _to_div(fig_spec, include_js=False)

    # Tables
    def fmt_cm(x_m: float) -> str:
        return f"{x_m*100.0:.2f}"

    table_modal = "\n".join(
        [
            "<tr>"
            f"<td>{i+1}</td>"
            f"<td>{Tn[i]:.4f}</td>"
            f"<td>{omegas[i]:.4f}</td>"
            f"<td>{zeta_n[i]*100.0:.2f}%</td>"
            f"<td>{Gamma[i]:.4f}</td>"
            f"<td>{M_eff[i]/M_tot*100.0:.1f}%</td>"
            "</tr>"
            for i in range(2)
        ]
    )

    # Peak comparison table
    rows = []
    rows.append(
        ("Isolated (direct time history, MDOF)", fmt_cm(peaks_dir["drift_max"]), fmt_cm(peaks_dir["iso_max"]), f"{peaks_dir['Cv_max']:.3f}")
    )
    rows.append(
        (
            "Isolated (modal time history, 2 modes)",
            fmt_cm(peaks_modal["drift_max"]),
            fmt_cm(peaks_modal["iso_max"]),
            f"{peaks_modal['Cv_max']:.3f}",
        )
    )
    rows.append(("Isolated (response spectrum analysis, SRSS)", fmt_cm(drift_rsa), fmt_cm(iso_rsa), f"{Cv_rsa:.3f}"))
    rows.append(("Fixed-base (time history, SDOF)", fmt_cm(drift_fb), "0.00", f"{Cv_fb:.3f}"))
    rows.append(("Fixed-base (response spectrum analysis)", fmt_cm(drift_fb_rsa), "0.00", f"{Cv_fb_rsa:.3f}"))

    table_peaks = "\n".join(
        [
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{drift_cm}</td>"
            f"<td>{iso_cm}</td>"
            f"<td>{cv}</td>"
            "</tr>"
            for (name, drift_cm, iso_cm, cv) in rows
        ]
    )

    html = dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>CE223 — 2-DOF Isolation System — Kobe KBU090</title>
          <style>
            :root {{
              --ucb-blue: #003262;
              --ucb-gold: #FDB515;
              --border: rgba(0,50,98,0.18);
              --bg: #f8fafc;
              --text: #2C3E50;
              --muted: #6b7280;
            }}
            * {{ box-sizing: border-box; }}
            body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; color: var(--text); background: var(--bg); line-height: 1.6; }}
            .wrap {{ max-width: 1040px; margin: 0 auto; padding: 2.2rem 1.5rem 4rem; }}
            header {{ text-align: left; padding-bottom: 1.2rem; border-bottom: 3px solid var(--ucb-blue); margin-bottom: 1.7rem; }}
            header h1 {{ margin: 0 0 0.35rem; color: var(--ucb-blue); font-size: 1.85rem; }}
            header p {{ margin: 0; color: var(--muted); font-size: 1.05rem; }}
            /* Single-column layout (full-width cards) */
            .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.25rem; }}
            .card {{ background: white; border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 2px 12px rgba(0,50,98,0.06); padding: 1.35rem 1.5rem; }}
            .card h2 {{ margin: 0 0 0.75rem; color: var(--ucb-blue); font-size: 1.25rem; }}
            .card h3 {{ margin: 1.1rem 0 0.5rem; color: #3B7EA1; font-size: 1.05rem; }}
            .card p {{ margin: 0 0 0.75rem; color: #374151; }}
            .plot {{ border: 1px solid var(--border); border-radius: 10px; background: white; padding: 0.6rem; margin-top: 0.8rem; }}
            table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 0; }}
            th, td {{ border: 1px solid #e5e7eb; padding: 0.55rem 0.7rem; text-align: left; }}
            th {{ background: var(--ucb-blue); color: white; }}
            tr:nth-child(even) {{ background: #f9fafb; }}
            .eq {{ background: #f9fafb; border-left: 4px solid var(--ucb-blue); padding: 0.9rem 1rem; margin: 0.75rem 0; overflow-x: auto; }}
            .pill {{ display:inline-block; background: rgba(253,181,21,0.18); border: 1px solid rgba(253,181,21,0.45); color: #7a4f00; padding: 0.2rem 0.5rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }}

            /* Plotly: keep modebar buttons, avoid title overlap */
            .js-plotly-plot .modebar {{
              top: 56px !important; /* below the plot title area */
            }}
            @media (max-width: 720px) {{
              .js-plotly-plot .modebar {{
                top: 74px !important;
              }}
            }}
          </style>
          <script>
            window.MathJax = {{
              tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
              }},
              options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
              }}
            }};
          </script>
          <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body>
          <div class="wrap">
            <header>
              <h1>2‑DOF Base‑Isolated Building — Kobe KBU090 (Part b)</h1>
              <p>
                This report evaluates a linear 2‑DOF base‑isolated building model under the 090 component of the 1995 Kobe University record
                (<strong>{gm_label}</strong>). The objective is to quantify three engineering demands—<strong>story drift</strong>,
                <strong>isolation displacement</strong>, and <strong>base shear coefficient</strong>—and compare them against the corresponding
                <strong>fixed‑base</strong> response.
              </p>
              <p style="margin-top:0.6rem;">
                Three complementary procedures are shown:
                (i) <strong>direct multi‑degree‑of‑freedom (MDOF) time‑history</strong> integration (Newmark),
                (ii) <strong>modal superposition time‑history</strong> (2 modes), and
                (iii) <strong>response spectrum analysis (RSA)</strong> using the <strong>square‑root‑of‑sum‑of‑squares (SRSS)</strong> rule.
              </p>
              <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.9rem;">
                <a href="CE223_MDOF_Newmark_Demo.html" style="text-decoration:none; border:1px solid rgba(0,50,98,0.18); padding:0.45rem 0.7rem; border-radius:999px; color:#003262; background:#ffffff;">
                  Open Newmark MDOF demo (non‑classical damping)
                </a>
                <a href="../../cee223-earthquake-protective-systems.html" style="text-decoration:none; border:1px solid rgba(0,50,98,0.18); padding:0.45rem 0.7rem; border-radius:999px; color:#003262; background:#ffffff;">
                  Back to CE223 main page
                </a>
              </div>
            </header>

            <div class="grid">
              <div class="card">
                <h2>1) Model definition and reported demands</h2>
                <p>
                  Using the coordinate convention from the CEE223 lecture material, the generalized displacement vector is
                  $\\mathbf{{u}}(t) = [u_b(t),\\;u_s(t)]^T$ where:
                  (i) $u_b(t)$ is the <em>isolation deformation</em> (base displacement relative to the ground), and
                  (ii) $u_s(t)$ is the <em>superstructure deformation</em> (superstructure displacement relative to the isolation level).
                </p>
                <div class="eq">
                $$\\mathbf{{M}}\\ddot{{\\mathbf{{u}}}} + \\mathbf{{C}}\\dot{{\\mathbf{{u}}}} + \\mathbf{{K}}\\mathbf{{u}} = -\\mathbf{{M}}\\boldsymbol{{\\iota}}\\,\\ddot{{u}}_g(t),\\qquad \\boldsymbol{{\\iota}}=[1,0]^T$$
                </div>
                <div class="eq">
                $$\\mathbf{{M}}=\\begin{{bmatrix}}M & m\\\\ m & m\\end{{bmatrix}},\\quad
                \\mathbf{{C}}=\\begin{{bmatrix}}c_b & 0\\\\ 0 & c_s\\end{{bmatrix}},\\quad
                \\mathbf{{K}}=\\begin{{bmatrix}}k_b & 0\\\\ 0 & k_s\\end{{bmatrix}},\\quad
                M=m_b+m$$
                </div>
                <p>
                  The right‑hand side is the <strong>effective inertia load</strong> induced by the ground acceleration $\\ddot{{u}}_g(t)$.
                  Because $u_s$ is already defined relative to the isolation level, the ground motion influences the system through
                  the isolation coordinate only; this is why the influence vector is $\\boldsymbol{{\\iota}}=[1,0]^T$.
                </p>
                <p><strong>Engineering demands reported:</strong></p>
                <ul style="margin:0.5rem 0 0; padding-left: 1.2rem;">
                  <li><strong>Story drift</strong>: $\\Delta(t)=u_s(t)$ (superstructure deformation relative to the isolation level)</li>
                  <li><strong>Isolation displacement</strong>: $u_b(t)$</li>
                  <li><strong>Base shear coefficient</strong>: $C_V = \\max\\,|V_b|/(Mg)$ with $V_b=k_b u_b + c_b \\dot{{u}}_b$</li>
                </ul>
                <p style="margin-top:0.7rem;">
                  Coordinate meaning (CEE223 lecture convention): $u_b$ is the deformation of the isolation layer, and $u_s$ is the deformation of the superstructure.
                  The corresponding absolute displacements are $u_b^{{\\mathrm{{abs}}}}=u_g+u_b$ and $u_s^{{\\mathrm{{abs}}}}=u_g+u_b+u_s$.
                  Because ground motion enters directly in the absolute base coordinate but not in the relative drift coordinate, the influence vector is
                  $\\boldsymbol{{\\iota}}=[1,0]^T$ (not $[1,1]^T$).
                </p>
                <h3>Given properties</h3>
                <p>
                  Mass ratio: $m_s = 3m_b/2$. Target periods: $T_s=0.5$ s (superstructure), $T_b=2.0$ s (isolation).
                  Damping ratios: $\\zeta_s=0.02$ and $\\zeta_b=0.15$.
                </p>
                <p style="color: var(--muted); margin-bottom:0;">
                  Note on scaling: the absolute mass scale cancels from these results—peak displacements and the coefficient $C_V$ are invariant to uniformly scaling all masses.
                </p>
              </div>

              <div class="card">
                <h2>2) Ground motion input</h2>
                <p>
                  The input is the Kobe University record <strong>{gm_label}</strong> (090 component). The file is a PEER-format
                  <code>.AT2</code> record with acceleration values in units of $g$, converted here to SI units for computation.
                  The plot below shows $\\ddot{{u}}_g(t)$ in $g$ so amplitudes are immediately interpretable.
                </p>
                <p style="color: var(--muted); margin-top:0;">
                  All response spectra on this page are generated directly from this same record.
                </p>
                <div class="plot">{gm_html}</div>
              </div>
            </div>

            <div class="card" style="margin-top: 1rem;">
              <h2>3) Modal properties and damping used for RSA</h2>
              <p>
                The undamped modes are obtained from the generalized eigenproblem
                $\\mathbf{{K}}\\phi_n = \\omega_n^2\\mathbf{{M}}\\phi_n$ and then scaled so that the <strong>base displacement component</strong>
                of each mode is one, $\\phi_{{b,n}} = 1$. This matches the coordinate convention used in the CEE223 lecture material and makes modal contributions
                to base and story deformations directly readable from the components of $\\phi_n$.
              </p>
              <div style="margin-top: 0.6rem;">
                <h3>Mode shapes</h3>
                <p style="margin-top:0;">
                  We begin by reporting the eigenvectors (mode shapes) because they are the building blocks for participation factors, effective modal mass,
                  and the modal time-history/RSA calculations that follow. The mode shape matrix $\\Phi=[\\phi_1\\;\\phi_2]$ is listed below; each column is a mode
                  (base-normalized, $\\phi_{{b,n}}=1$), and each row corresponds to a DOF in the CEE223 lecture coordinate convention.
                </p>
                <table>
                  <thead>
                    <tr>
                      <th>DOF</th>
                      <th>Mode 1: φ<sub>1</sub></th>
                      <th>Mode 2: φ<sub>2</sub></th>
                    </tr>
                  </thead>
                  <tbody>
                    {table_phi}
                  </tbody>
                </table>
              </div>
              <p>
                For base excitation, the key scalar is the <strong>participation factor</strong> $\\Gamma_n$, which maps ground acceleration into modal forcing
                through the influence vector $\\boldsymbol{{\\iota}}=[1,0]^T$ defined in Section 1.
                To write <em>independent</em> modal equations (one scalar ODE per mode), we make the common <strong>(approximately) classical damping</strong>
                assumption: in modal coordinates, the damping matrix is nearly diagonal, i.e., $\\Phi^T\\mathbf{{C}}\\Phi \\approx \\mathrm{{diag}}(c_1,c_2)$.
                Real structures are generally not exactly classical, so this step should be understood as an approximation adopted for tractability.
              </p>
              <div class="eq">
                $$M_n = \\phi_n^T\\mathbf{{M}}\\phi_n,\\qquad
                \\Gamma_n = \\dfrac{{\\phi_n^T \\mathbf{{M}}\\boldsymbol{{\\iota}}}}{{M_n}},\\qquad
                \\zeta_n \\approx \\dfrac{{\\phi_n^T\\mathbf{{C}}\\phi_n}}{{2\\omega_n\\,M_n}},\\qquad
                M_n^* = \\Gamma_n^2 M_n$$
              </div>
              <p style="margin-top:0;">
                $M_n^*/M$ (effective modal mass ratio) quantifies how much of the base-excited response is carried by each mode, where
                $M = \\boldsymbol{{\\iota}}^T\\mathbf{{M}}\\boldsymbol{{\\iota}}$ is the effective total mass associated with the influence vector.
                In an isolation regime, Mode 1 should dominate.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Mode</th>
                    <th>$T_n$ [s]</th>
                    <th>$\\omega_n$ [rad/s]</th>
                    <th>$\\zeta_n$</th>
                    <th>$\\Gamma_n$</th>
                    <th>$M_n^*/M$</th>
                  </tr>
                </thead>
                <tbody>
                  {table_modal}
                </tbody>
              </table>
              <p style="color: var(--muted); margin: 0.5rem 0 0;">
                Classical damping note: writing independent modal equations,
                $q_n'' + 2\\zeta_n\\omega_n q_n' + \\omega_n^2 q_n = -\\Gamma_n\\,\\ddot{{u}}_g(t)$,
                implicitly assumes that the damping matrix is diagonal in modal coordinates
                (so-called classical damping). Real structures are generally not exactly classical,
                but for lightly coupled low-order systems this approximation is standard in practice.
              </p>
              <div class="plot" style="margin-top: 0.8rem;">{spec_html}</div>
              <p style="color: var(--muted); margin: 0.2rem 0 0;">
                The spectrum plot shows pseudo-acceleration $S_a(T)$ for the modal damping ratios and the fixed-base 2% curve, with markers at $T_1$ and $T_2$.
              </p>
            </div>

            <div class="card" style="margin-top: 1rem;">
              <h2>4) Direct vs modal time-history response (sanity check)</h2>
              <p>
                The benchmark is the <strong>direct physical time-history</strong> solution of the coupled 2‑DOF equations using Newmark’s method:
                at each time step we solve a <em>2×2 coupled linear system</em> for $\\mathbf{{u}}(t)$.
                In parallel, a <strong>modal superposition</strong> time history is computed by integrating each modal coordinate as an
                independent <strong>single‑degree‑of‑freedom (SDOF)</strong> equation and reconstructing $\\mathbf{{u}}(t)$.
              </p>
              <div class="eq">
                $$q_n'' + 2\\zeta_n\\omega_n q_n' + \\omega_n^2 q_n = -\\Gamma_n\\,\\ddot{{u}}_g(t),\\qquad \\mathbf{{u}}(t)=\\sum_{{n=1}}^{{2}}\\phi_n\\,q_n(t)$$
              </div>
              <p style="margin-top:0;">
                The plots below show <strong>isolation displacement</strong> $u_b(t)$ and <strong>story drift</strong> $\\Delta(t)=u_s(t)$
                for the direct and modal solutions.
                This comparison isolates two modeling choices: (a) truncating to the first two modes (here, the full system has only two), and
                (b) treating damping as modal (classical) to obtain independent modal ODEs. Good agreement indicates these choices are adequate for this record.
              </p>
              <div class="plot">{th_html}</div>
            </div>

            <div class="card" style="margin-top: 1rem;">
              <h2>5) Peak demand summary and comparison to fixed-base</h2>
              <p>
                The table below reports peak values for the isolated building from: (i) direct 2‑DOF time history,
                (ii) modal time history, and (iii) response spectrum analysis (RSA) with SRSS modal combination. The <strong>fixed‑base</strong> entry is the corresponding SDOF response
                with $T_s=0.5$ s and $\\zeta_s=2\\%$ under the same record.
              </p>
              <p style="margin-top:0;">
                Fixed-base is modeled as an SDOF because setting $u_b\\equiv 0$ removes the isolation deformation coordinate, leaving only the
                superstructure deformation relative to the ground. This is equivalent to constraining the base DOF in the 2‑DOF model.
                For RSA, fixed-base has a single mode, so SRSS combination is trivial.
              </p>
              <p style="margin-top:0;">
                <strong>Why (ii) and (iii) differ:</strong> both are built on the same modal properties $(T_n,\\zeta_n,\\Gamma_n)$, but they answer different questions.
                Modal time history integrates $q_n(t)$ under the <em>actual</em> record and reconstructs $\\mathbf{{u}}(t)$, so peak response depends on modal phasing in time.
                RSA replaces the record with spectral ordinates $S_d(T_n,\\zeta_n)$ (or $S_a$) and then combines <em>modal peaks</em> statistically (SRSS),
                discarding time sequencing and assuming peaks do not occur simultaneously.
              </p>
              <p style="margin-top:0;">
                Units: displacements in <strong>cm</strong>. The base shear coefficient $C_V$ is dimensionless.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>$\\max\\,|\\Delta|$ [cm]</th>
                    <th>$\\max\\,|u_b|$ [cm]</th>
                    <th>$\\max\\,C_V$</th>
                  </tr>
                </thead>
                <tbody>
                  {table_peaks}
                </tbody>
              </table>
              <h3>Interpretation</h3>
              <p style="margin-top:0;">
                Isolation reduces drift and base shear primarily by shifting the dominant response to the longer first-mode period ($T_1\\approx 2$ s)
                and by concentrating damping in the isolation layer (here $\\zeta_b=15\\%$). The main trade-off is increased base displacement $|u_b|$,
                which becomes the governing design check for moat clearance and isolator deformation capacity.
              </p>
              <p style="color: var(--muted); margin-top: 0.8rem;">
                RSA note: SRSS combines modal peak contributions assuming statistical independence. For closely spaced modes, CQC would be preferred; here
                the two periods are well separated and Mode 1 dominates the effective modal mass.
              </p>
            </div>

            <div class="card" style="margin-top: 1rem;">
              <h2>References</h2>
              <ul style="margin:0.5rem 0 0; padding-left: 1.2rem;">
                <li><strong>Chopra (2014)</strong>: modal participation factors, response spectrum analysis, and modal combination rules.</li>
                <li><strong>CEE223 lectures (2025)</strong>: isolation modeling assumptions and the coordinate convention used on this page.</li>
                <li>
                  <strong>CEE225 lectures (2025)</strong>: Newmark integration and modal superposition notation, aligned with the CEE225 final project case study
                  (interactive 3‑story MDOF response analysis under recorded ground motion).
                  See the <a href="../../cee225-dynamics.html">CEE225 Structural Dynamics page</a> and the
                  <a href="../../CEE225_Dynamics/highlighted_htmls/final_project_menu.html">CEE225 final project menu</a>.
                </li>
              </ul>
              <p style="color: var(--muted); margin-top:0.7rem;">
                Citations: <em>Chopra, A.K.</em> (2014). <em>Dynamics of Structures</em> (4th ed.). Pearson.
                <em>Konstantinidis, D.</em> (2025). Lectures for CEE223: Earthquake Protective Systems, UC Berkeley.
                <em>DeJong, M.</em> (2025). Lectures for CEE225: Structural Dynamics, UC Berkeley.
              </p>
            </div>

            <div class="card" style="margin-top: 1rem;">
              <h2>PDF attachment (placeholder)</h2>
              <p>
                A PDF version of this analysis will be embedded here in the final submission.
                For now, this section is intentionally left as a placeholder.
              </p>
              <div class="plot" style="height: 420px; display:flex; align-items:center; justify-content:center; color: var(--muted);">
                <div style="text-align:center;">
                  <div style="font-weight:800; color: var(--ucb-blue); margin-bottom:0.35rem;">PDF placeholder</div>
                  <div>Drop the exported PDF here when ready.</div>
                </div>
              </div>
            </div>
          </div>
        </body>
        </html>
        """
    ).strip()

    out = HIGHLIGHTED_HTML_DIR / "CE223_2DOF_Isolation_Kobe090.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

