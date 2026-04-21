from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html


THIS_DIR = Path(__file__).resolve().parent
HIGHLIGHTED_HTML_DIR = THIS_DIR / ".." / "highlighted_htmls"
HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)

G_SI = 9.80665


@dataclass
class NewmarkParams:
    beta: float = 1.0 / 4.0
    gamma: float = 1.0 / 2.0


def newmark_mdoff_forced(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    p_t: np.ndarray,
    dt: float,
    params: NewmarkParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical Newmark-β constant-average-acceleration scheme for an n-DOF linear system

        M ü + C u̇ + K u = p(t)

    where p_t is an array of nodal loads at each time step.
    """
    if params is None:
        params = NewmarkParams()
    beta = params.beta
    gamma = params.gamma

    p_t = np.asarray(p_t, dtype=float)
    if p_t.ndim == 1:
        p_t = p_t[:, None]
    n_steps, dof = p_t.shape

    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)

    u = np.zeros((n_steps, dof), dtype=float)
    ud = np.zeros((n_steps, dof), dtype=float)
    udd = np.zeros((n_steps, dof), dtype=float)

    # Initial acceleration from static equilibrium
    udd[0] = np.linalg.solve(M, p_t[0] - C @ ud[0] - K @ u[0])

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    K_eff = K + a0 * M + a1 * C

    for i in range(1, n_steps):
        # Effective load at step i
        p_eff = (
            p_t[i]
            + M @ (a0 * u[i - 1] + a2 * ud[i - 1] + a3 * udd[i - 1])
            + C @ (a1 * u[i - 1] + a4 * ud[i - 1] + a5 * udd[i - 1])
        )

        u[i] = np.linalg.solve(K_eff, p_eff)
        udd[i] = a0 * (u[i] - u[i - 1]) - a2 * ud[i - 1] - a3 * udd[i - 1]
        ud[i] = ud[i - 1] + dt * ((1.0 - gamma) * udd[i - 1] + gamma * udd[i])

    return u, ud, udd


def newmark_sdof_forced(
    m: float,
    c: float,
    k: float,
    p_t: np.ndarray,
    dt: float,
    params: NewmarkParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper around newmark_mdoff_forced for SDOF."""
    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])
    u, ud, udd = newmark_mdoff_forced(M, C, K, p_t, dt, params=params)
    return u[:, 0], ud[:, 0], udd[:, 0]


def build_demo() -> str:
    """
    Build an educational dashboard for Newmark MDOF using a 2-DOF shear frame
    subjected to a unit load pulse at the roof.
    """
    # Toy 2-DOF shear frame
    m1 = 1.0
    m2 = 1.0
    k1 = 20.0
    k2 = 10.0
    # Strongly non-classical: different damping per story (e.g. isolator vs superstructure).
    # No single Rayleigh form αM+βK can match both; modal approximation will visibly diverge.
    zeta_story1 = 0.02   # e.g. superstructure
    zeta_story2 = 0.22   # e.g. isolator or story with supplemental damper

    M = np.array([[m1, 0.0], [0.0, m2]], dtype=float)
    K = np.array([[k1 + k2, -k2], [-k2, k2]], dtype=float)

    omega1_phys = math.sqrt(k1 / m1)
    omega2_phys = math.sqrt(k2 / m2)
    c1 = 2.0 * zeta_story1 * omega1_phys * m1
    c2 = 2.0 * zeta_story2 * omega2_phys * m2
    C = np.array([[c1, 0.0], [0.0, c2]], dtype=float)

    # Time discretization and loading: short sustained pulse at DOF 2
    dt = 0.01
    t_end = 8.0
    t = np.arange(0.0, t_end + dt, dt)
    n = t.size

    p = np.zeros((n, 2), dtype=float)
    pulse_indices = (t >= 0.0) & (t <= 0.5)
    p[pulse_indices, 1] = 1.0  # sustained unit load at roof DOF over a short window

    # Direct MDOF Newmark
    u_mdof, ud_mdof, _ = newmark_mdoff_forced(M, C, K, p, dt)

    # Modal superposition Newmark under the classical damping approximation
    # (all modes retained; any discrepancy with the MDOF result is due to
    # the non-classical structure of C in physical coordinates).
    A_mat = np.linalg.solve(M, K)
    lam, Phi = np.linalg.eig(A_mat)
    lam = np.real(lam)
    Phi = np.real(Phi)

    idx = np.argsort(np.sqrt(np.maximum(lam, 0.0)))
    lam = lam[idx]
    Phi = Phi[:, idx]
    omegas = np.sqrt(np.maximum(lam, 0.0))

    # Mass-normalize modes
    for i in range(2):
        phi = Phi[:, i]
        m_norm = math.sqrt(float(phi.T @ M @ phi))
        Phi[:, i] = phi / m_norm if m_norm > 0 else phi

    # Modal damping and participation
    zeta_n = np.empty(2)
    Gamma = np.empty(2)
    for i in range(2):
        phi = Phi[:, i]
        zeta_n[i] = float(phi.T @ C @ phi) / (2.0 * omegas[i] * float(phi.T @ M @ phi))
        Gamma[i] = float(phi.T @ p[:, :].T[:, pulse_indices].mean(axis=1))  # not used directly here

    # Transform load to modal coordinates: p_modal = Φ^T p
    p_modal = (Phi.T @ p.T).T  # shape (n, 2)

    q = np.zeros_like(p_modal)
    qd = np.zeros_like(p_modal)
    for i in range(2):
        mi = 1.0  # after mass-normalization
        ki = omegas[i] ** 2 * mi
        ci = 2.0 * zeta_n[i] * omegas[i] * mi
        qi, qdi, _ = newmark_sdof_forced(mi, ci, ki, p_modal[:, i], dt)
        q[:, i] = qi
        qd[:, i] = qdi

    u_modal = (Phi @ q.T).T
    ud_modal = (Phi @ qd.T).T

    # Figures: time histories at each DOF
    fig_dof1 = go.Figure()
    fig_dof1.add_trace(
        go.Scatter(x=t, y=u_mdof[:, 0], mode="lines", name="Direct time history (MDOF)", line=dict(color="rgb(0,55,95)", width=2.2))
    )
    fig_dof1.add_trace(
        go.Scatter(
            x=t,
            y=u_modal[:, 0],
            mode="lines",
            name="Modal time history (exact classical damping)",
            line=dict(color="rgb(220,38,38)", width=2.0, dash="dash"),
        )
    )
    fig_dof1.update_layout(
        template="plotly_white",
        height=360,
        title=dict(
            text="DOF 1 displacement: direct MDOF vs modal superposition",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="u₁(t) [m]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )

    fig_dof2 = go.Figure()
    fig_dof2.add_trace(
        go.Scatter(x=t, y=u_mdof[:, 1], mode="lines", name="Direct time history (MDOF)", line=dict(color="rgb(0,55,95)", width=2.2))
    )
    fig_dof2.add_trace(
        go.Scatter(
            x=t,
            y=u_modal[:, 1],
            mode="lines",
            name="Modal time history (exact classical damping)",
            line=dict(color="rgb(220,38,38)", width=2.0, dash="dash"),
        )
    )
    fig_dof2.update_layout(
        template="plotly_white",
        height=360,
        title=dict(
            text="DOF 2 displacement: direct MDOF vs modal superposition",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="u₂(t) [m]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )

    fig_load = go.Figure()
    fig_load.add_trace(
        go.Scatter(x=t, y=p[:, 1], mode="lines", name="Applied roof load p₂(t)", line=dict(color="rgb(0,55,95)", width=2.2))
    )
    fig_load.update_layout(
        template="plotly_white",
        height=320,
        title=dict(
            text="Toy loading for Newmark MDOF demo: unit pulse at DOF 2",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=16),
            automargin=True,
        ),
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="p₂(t) [N]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )

    def _to_div(fig: go.Figure, include_js: bool = False) -> str:
        return to_html(fig, include_plotlyjs=include_js, full_html=False, config=dict(displayModeBar=True, responsive=True))

    load_html = _to_div(fig_load, include_js=True)
    dof1_html = _to_div(fig_dof1, include_js=False)
    dof2_html = _to_div(fig_dof2, include_js=False)

    html = dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>CE223 — Newmark MDOF Demo (2-DOF shear frame)</title>
          <style>
            :root {{
              --ucb-blue: #003262;
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
            .card {{ background:#ffffff; border-radius: 14px; padding: 1.4rem 1.5rem 1.3rem; margin-bottom: 1.4rem; box-shadow: 0 10px 30px rgba(15,23,42,0.08); border:1px solid var(--border); }}
            .card h2 {{ margin-top: 0; color: var(--ucb-blue); font-size: 1.35rem; }}
            .card h3 {{ margin-top: 0.4rem; color: var(--ucb-blue); font-size: 1.1rem; }}
            .eq {{ background:#f9fafb; border-left: 3px solid var(--ucb-blue); padding: 0.65rem 0.9rem; border-radius: 0.375rem; margin: 0.75rem 0; font-size: 0.98rem; }}
            table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 0; }}
            th, td {{ padding: 0.45rem 0.6rem; border: 1px solid rgba(148,163,184,0.45); font-size: 0.92rem; }}
            th {{ background: rgba(0,50,98,0.96); color: #ffffff; text-align: left; }}
            code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9rem; }}
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
              <h1>Newmark MDOF demo — 2‑DOF shear frame</h1>
              <p>
                To demonstrate the <strong>power and flexibility of numerical multi‑degree‑of‑freedom (MDOF)</strong> analysis,
                we use a <strong>non‑classical damping matrix</strong> and compare the direct time‑history solution with the
                classical-damping approximation. This could be the case of a <strong>base‑isolated building</strong> (high damping
                in the isolator, lower damping in the superstructure), or a structure with <strong>story‑level supplemental dampers</strong>
                that do not combine into a Rayleigh form. The same Newmark formulation handles both without modal decomposition.
              </p>
              <p style="margin-top:0.6rem;">
                This page walks through a small 2‑DOF shear-frame example: we report $\\mathbf{{M}}$, $\\mathbf{{K}}$, and $\\mathbf{{C}}$,
                apply a short sustained load, and show how the coupled equations are integrated step-by-step. The result is a compact
                reference for the practising engineer who wants to see the algorithm in action.
              </p>
              <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.9rem;">
                <a href="CE223_2DOF_Isolation_Kobe090.html" style="text-decoration:none; border:1px solid var(--border); padding:0.45rem 0.7rem; border-radius:999px; color:var(--ucb-blue); background:#ffffff;">
                  Open 2‑DOF base‑isolation dashboard
                </a>
                <a href="../../cee223-earthquake-protective-systems.html" style="text-decoration:none; border:1px solid var(--border); padding:0.45rem 0.7rem; border-radius:999px; color:var(--ucb-blue); background:#ffffff;">
                  Back to CE223 main page
                </a>
              </div>
            </header>

            <div class="card">
              <h2>1) Real structure, matrices, and loading</h2>
              <p>
                The model corresponds to a <strong>two‑story shear building</strong>—for example a short office or laboratory
                with one lateral degree of freedom per floor. Lumped masses and story stiffnesses are $m_1=m_2=1$, $k_1=20$, $k_2=10$
                (units arbitrary but consistent). Mass and stiffness matrices are
              </p>
              <div class="eq">
              $$\\mathbf{{M}} = \\begin{{bmatrix}} m_1 & 0 \\\\ 0 & m_2 \\end{{bmatrix}},\\qquad
                \\mathbf{{K}} = \\begin{{bmatrix}} k_1 + k_2 & -k_2 \\\\ -k_2 & k_2 \\end{{bmatrix}}.$$
              </div>
              <p>
                Damping is taken as <strong>diagonal in physical coordinates</strong>, $\\mathbf{{C}}=\\operatorname{{diag}}(c_1,c_2)$,
                with <strong>different effective damping per story</strong>—e.g. story&nbsp;1 at about $2\\%$ and story&nbsp;2 at about $22\\%$—
                as might occur with a soft isolator or a story with supplemental dampers. No single Rayleigh form $\\alpha\\mathbf{{M}}+\\beta\\mathbf{{K}}$
                can match both; the system is strongly non‑classically damped:
              </p>
              <div class="eq">
              $$\\mathbf{{C}} = \\begin{{bmatrix}} c_1 & 0 \\\\ 0 & c_2 \\end{{bmatrix}}.$$
              </div>
              <p>
                The structure is subjected to a <strong>sustained unit load</strong> at the roof over a short window:
                $p_2(t)=1$ for $0\\le t\\le 0.5$ s and $p_2(t)=0$ afterwards, with $p_1(t)\\equiv 0$. The plot below shows this loading.
              </p>
              <div class="plot">{load_html}</div>
            </div>

            <div class="card">
              <h2>2) Why non‑classical damping? Approximate modal damping</h2>
              <p>
                Because $\\mathbf{{C}}$ is diagonal in physical coordinates and not of the form $\\alpha \\mathbf{{M}}+\\beta \\mathbf{{K}}$,
                the system is <strong>non‑classically damped</strong>: the undamped mode shapes do not diagonalize $\\mathbf{{C}}$.
                This is exactly the situation where direct MDOF time‑history shines—no need to force a Rayleigh fit or to use
                complex modal analysis. To show the contrast, we still run a <em>modal</em> time‑history using approximate
                modal damping ratios $\\zeta_n$ from the diagonal of $\\boldsymbol{{\\Phi}}^T \\mathbf{{C}} \\boldsymbol{{\\Phi}}$
                (mass‑normalized $\\boldsymbol{{\\Phi}}$). Those $\\zeta_n$ are only approximate; the reference response is the direct MDOF result.
              </p>
              <p>
                The undamped modal properties come from $\\mathbf{{K}}\\boldsymbol{{\\phi}}_n = \\omega_n^2 \\mathbf{{M}}\\boldsymbol{{\\phi}}_n$
                with mass normalization $\\boldsymbol{{\\phi}}_n^T \\mathbf{{M}}\\boldsymbol{{\\phi}}_n = 1$. The approximate modal damping ratio is
              </p>
              <div class="eq">
              $$\\zeta_n \\approx \\frac{{\\boldsymbol{{\\phi}}_n^T\\mathbf{{C}}\\boldsymbol{{\\phi}}_n}}{{2\\omega_n\\,\\boldsymbol{{\\phi}}_n^T\\mathbf{{M}}\\boldsymbol{{\\phi}}_n}}.$$
              </div>
              <p>
                Because $\\mathbf{{C}}$ is not exactly classical, the true modal equations would contain velocity-coupling
                terms; the modal time‑history here drops those terms so we can see the impact of the classical-damping approximation.
              </p>
            </div>

            <div class="card">
              <h2>3) Newmark MDOF update rule</h2>
              <p>
                For the MDOF system $\\mathbf{{M}}\\ddot{{\\mathbf{{u}}}} + \\mathbf{{C}}\\dot{{\\mathbf{{u}}}} + \\mathbf{{K}}\\mathbf{{u}} = \\mathbf{{p}}(t)$,
                the constant-average-acceleration Newmark scheme with parameters $(\\beta,\\gamma)=(1/4,1/2)$ defines a set of predictor coefficients
                $a_0,\\dots,a_5$ in terms of the time step $\\Delta t$. These are the standard Newmark coefficients used to construct the effective
                stiffness and effective load at each step.
              </p>
              <p>
                The <strong>effective stiffness matrix</strong> and <strong>effective load vector</strong> at step $i$ are
              </p>
              <div class="eq">
              $$\\mathbf{{K}}_\\text{{eff}} = \\mathbf{{K}} + a_0\\mathbf{{M}} + a_1\\mathbf{{C}},$$
              $$\\mathbf{{p}}_\\text{{eff}}^{(i)} = \\mathbf{{p}}^{(i)} + \\mathbf{{M}}\\left(a_0\\mathbf{{u}}^{(i-1)} + a_2\\dot{{\\mathbf{{u}}}}^{(i-1)} + a_3\\ddot{{\\mathbf{{u}}}}^{(i-1)}\\right)
                + \\mathbf{{C}}\\left(a_1\\mathbf{{u}}^{(i-1)} + a_4\\dot{{\\mathbf{{u}}}}^{(i-1)} + a_5\\ddot{{\\mathbf{{u}}}}^{(i-1)}\\right).$$
              </div>
              <p>
                At each time step, the algorithm solves the <strong>coupled 2×2 system</strong>
                $\\mathbf{{K}}_\\text{{eff}}\\Delta\\mathbf{{u}}^{(i)} = \\mathbf{{p}}_\\text{{eff}}^{(i)}$ to obtain $\\mathbf{{u}}^{(i)}$.
                The velocities and accelerations are then updated from the Newmark formulas.
              </p>
              <p>
                Below is the core of the algorithm in real Python (constant-average-acceleration: $\\beta=1/4$, $\\gamma=1/2$).
                The effective stiffness $\\mathbf{{K}}_\\text{{eff}}$ is formed once; each step solves one linear system and updates
                displacement, velocity, and acceleration. No modal decomposition is required—so it works for any $\\mathbf{{C}}$.
              </p>
              <pre class="eq" style="font-size:0.85rem; overflow-x:auto;"><code># Newmark coefficients (β=1/4, γ=1/2)
a0 = 1.0 / (beta * dt**2)
a1 = gamma / (beta * dt)
a2 = 1.0 / (beta * dt)
a3 = 1.0 / (2*beta) - 1.0
a4 = gamma/beta - 1.0
a5 = dt * (gamma/(2*beta) - 1.0)

K_eff = K + a0*M + a1*C   # formed once

# Initial acceleration from M ü = p - C u̇ - K u at t=0
udd[0] = np.linalg.solve(M, p_t[0] - C @ ud[0] - K @ u[0])

for i in range(1, n_steps):
    p_eff = (p_t[i]
        + M @ (a0*u[i-1] + a2*ud[i-1] + a3*udd[i-1])
        + C @ (a1*u[i-1] + a4*ud[i-1] + a5*udd[i-1]))
    u[i] = np.linalg.solve(K_eff, p_eff)
    udd[i] = a0*(u[i] - u[i-1]) - a2*ud[i-1] - a3*udd[i-1]
    ud[i] = ud[i-1] + dt*((1-gamma)*udd[i-1] + gamma*udd[i])</code></pre>
            </div>

            <div class="card">
              <h2>4) Direct MDOF vs modal time-history — what you get</h2>
              <p>
                With <strong>non‑classical damping</strong>, the direct MDOF solution in physical coordinates is the
                reference; the modal time‑history is an approximation that assumes classical damping (diagonal $\\zeta_n$
                in modal space). Any difference between the two curves comes from the neglected velocity-coupling terms
                in the modal equations—a reminder of why numerical MDOF is so useful when damping is not Rayleigh.
              </p>
              <p>
                For this example the gap is small but visible: the modal approximation captures the main response, while
                fine details differ. In practice, for base‑isolated or damper‑enhanced structures, running the direct
                MDOF formulation (as above) gives the engineer a single, consistent reference without modal approximations.
              </p>
              <div class="plot">{dof1_html}</div>
              <div class="plot" style="margin-top: 1.0rem;">{dof2_html}</div>
              <p style="color: var(--muted); margin-top:0.8rem;">
                This demo underpins the CE223 isolation dashboard: there we assume classical damping and use modal superposition;
                here we relax that assumption and show that the same Newmark engine handles both cases—the difference in results
                is due to the damping model, not the time‑integration method.
              </p>
            </div>

          </div>
        </body>
        </html>
        """
    ).strip()

    return html


def main() -> None:
    html = build_demo()
    out = HIGHLIGHTED_HTML_DIR / "CE223_MDOF_Newmark_Demo.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

