#!/usr/bin/env python3
"""
Generate the load-unload cycle plot (Figure 1.3) for CE221 HW2.
Uses xara to run load-up then unload; outputs Plotly HTML (fig_load_unload.html).
"""
import os
import numpy as np
import plotly.graph_objects as go

try:
    import xara
except ImportError:
    raise ImportError("xara is required. Install with: pip install xara")

# --- 1. Model Setup ---
k = 1000
P = 10.0

m = xara.Model(ndm=1, ndf=1)
m.node(1, 0.0)
m.node(2, 0.0)
m.fix(1, 1)
m.uniaxialMaterial("Elastic", 1, k)
m.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
m.pattern("Plain", 1, "Linear")
m.load(2, P, pattern=1)
m.system("BandGeneral")
m.numberer("Plain")
m.constraints("Plain")
m.algorithm("Linear")
m.analysis("Static")

# --- 2. Phase 1: Loading Up ---
n_steps_load = 20
dLambda_load = 1.0 / n_steps_load
m.integrator("LoadControl", dLambda_load)

U_load, F_load = [0.0], [0.0]
for i in range(n_steps_load):
    ok = m.analyze(1)
    if ok != 0:
        raise RuntimeError(f"Loading failed at step {i+1}")
    u_val = m.nodeDisp(2, 1)
    lf = m.getTime()
    U_load.append(u_val)
    F_load.append(lf * P)

# --- 3. Phase 2: Unloading ---
P_min = P / 2
n_steps_unload = 5
current_lf = m.getTime()
target_lf = P_min / P
delta_lf = target_lf - current_lf
dLambda_unload = delta_lf / n_steps_unload
m.integrator("LoadControl", dLambda_unload)

U_unload, F_unload = [U_load[-1]], [F_load[-1]]
for i in range(n_steps_unload):
    ok = m.analyze(1)
    if ok != 0:
        raise RuntimeError(f"Unloading failed at step {i+1}")
    u_val = m.nodeDisp(2, 1)
    lf = m.getTime()
    U_unload.append(u_val)
    F_unload.append(lf * P)

U_load = np.array(U_load)
F_load = np.array(F_load)
U_unload = np.array(U_unload)
F_unload = np.array(F_unload)

# --- 4. Plotly figure ---
FONT_FAMILY = "Arial"
FONT_TITLE, FONT_AXIS, FONT_TICK = 20, 15, 15
FIG_W, FIG_H = 800, 500

fig = go.Figure()

# Loading: solid blue
fig.add_trace(
    go.Scatter(
        x=U_load,
        y=F_load,
        mode="lines+markers",
        name="Loading",
        line=dict(color="#1e3a8a", width=2),
        marker=dict(size=8, color="#1e3a8a"),
        hovertemplate="u = %{x:.4f}<br>F = %{y:.4f}<extra></extra>",
    )
)
# Unloading: dashed orange with markers
fig.add_trace(
    go.Scatter(
        x=U_unload,
        y=F_unload,
        mode="lines+markers",
        name="Unloading",
        line=dict(color="#ea580c", width=2, dash="dash"),
        marker=dict(size=8, color="#ea580c"),
        hovertemplate="u = %{x:.4f}<br>F = %{y:.4f}<extra></extra>",
    )
)

fig.update_layout(
    title=dict(
        text=f"Load–Unload Cycle (k = {k})",
        font=dict(size=FONT_TITLE, color="#1e293b", family=FONT_FAMILY),
        x=0.5,
        xanchor="center",
        y=0.99,
        yanchor="top",
    ),
    xaxis=dict(
        title=dict(text="Displacement u (Node 2)", font=dict(size=FONT_AXIS, color="#475569", family=FONT_FAMILY)),
        tickfont=dict(size=FONT_TICK, color="#64748b", family=FONT_FAMILY),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.25)",
        zeroline=True,
        showline=True,
        linecolor="#cbd5e1",
        linewidth=1.2,
        mirror=True,
    ),
    yaxis=dict(
        title=dict(text="Force (P)", font=dict(size=FONT_AXIS, color="#475569", family=FONT_FAMILY)),
        tickfont=dict(size=FONT_TICK, color="#64748b", family=FONT_FAMILY),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.25)",
        zeroline=True,
        showline=True,
        linecolor="#cbd5e1",
        linewidth=1.2,
        mirror=True,
    ),
    legend=dict(
        orientation="h",
        y=1.06,
        x=0.5,
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0)",
        bordercolor="rgba(255,255,255,0)",
        font=dict(size=FONT_TICK, color="#334155", family=FONT_FAMILY),
        itemsizing="constant",
        itemwidth=30,
    ),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="white",
    margin=dict(t=100, b=60, l=78, r=48),
    width=FIG_W,
    height=FIG_H,
    font=dict(family=FONT_FAMILY, size=FONT_TICK, color="#334155"),
    hoverlabel=dict(bgcolor="white", font_size=14, font_family=FONT_FAMILY),
)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "fig_load_unload.html")
fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
print(f"Wrote {out_path}")
