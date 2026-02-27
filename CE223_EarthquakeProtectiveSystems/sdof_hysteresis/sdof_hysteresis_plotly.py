from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


# Color palette aligned with site branding used in
# CE225 highlighted HTMLs (deep Berkeley blue, Cal gold, accent red, etc.).
CE_BLUE = "rgb(0, 50, 98)"       # #003262
CE_BLACK = "rgb(0, 0, 0)"
CE_RED = "rgb(197, 48, 48)"      # #c53030
CE_GREEN = "rgb(22, 163, 74)"    # Tailwind-ish green for secondary accents
CE_SLATE = "rgb(107, 114, 128)"  # #6b7280 for neutral curves if needed


@dataclass
class HysteresisStyle:
    width: int = 950
    height: int = 600
    template: str = "plotly_white"
    line_width: float = 2.5
    grid_color: str = "rgba(0, 0, 0, 0.15)"
    font_family: str = "Arial"
    font_size: int = 16
    title_size: int = 22
    paper_bgcolor: str = "rgb(255, 255, 255)"
    plot_bgcolor: str = "rgb(250, 250, 250)"
    axis_line_color: str = "rgb(0, 0, 0)"


def create_sdof_hysteresis_figure(
    *,
    u: Mapping[str, np.ndarray],
    f: Mapping[str, np.ndarray],
    title: str,
    style: HysteresisStyle | None = None,
) -> go.Figure:
    """
    Build a Plotly figure comparing f–u hysteresis loops for several models.

    Parameters
    ----------
    u, f:
        Dictionaries keyed by model label with displacement and force arrays
        of equal length (per model).
    title:
        Figure title.
    """
    style = style or HysteresisStyle()
    fig = go.Figure()

    # Explicit styling per model label (stable + readable).
    # We intentionally draw Model B first and Model A second so that the
    # dashed blue line reveals the green underneath when they overlap.
    preferred_order: Sequence[str] = [
        "Model B (hysteretic)",
        "Model A (viscous)",
        "Model C (fractional)",
    ]
    ordered_labels = [lbl for lbl in preferred_order if lbl in u] + [
        lbl for lbl in u.keys() if lbl not in preferred_order
    ]

    def _trace_style(label: str) -> dict:
        if "Model A" in label:
            return dict(color=CE_BLUE, dash="dash", width=2.8, opacity=0.95, legendrank=1)
        if "Model B" in label:
            return dict(color=CE_GREEN, dash="solid", width=2.8, opacity=0.95, legendrank=2)
        if "Model C" in label:
            return dict(color=CE_RED, dash="solid", width=2.8, opacity=0.95, legendrank=3)
        # Fallback for any extra series
        return dict(color=CE_BLACK, dash="solid", width=2.4, opacity=0.9, legendrank=10)

    for label in ordered_labels:
        ui = np.asarray(u[label], dtype=float)
        fi = np.asarray(f[label], dtype=float)
        if ui.shape != fi.shape:
            raise ValueError(f"u and f must have same shape for {label}")
        ts = _trace_style(label)
        fig.add_trace(
            go.Scatter(
                x=ui,
                y=fi,
                mode="lines",
                name=label,
                line=dict(color=ts["color"], width=float(ts["width"]), dash=ts["dash"]),
                opacity=float(ts["opacity"]),
                legendrank=int(ts["legendrank"]),
                hovertemplate="u = %{x:.4f}<br>f = %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=style.title_size, family=style.font_family, color="#1e293b"),
            pad=dict(t=6, b=10),
        ),
        xaxis=dict(
            title=dict(text="Displacement u(t) [m]", font=dict(color=style.axis_line_color)),
            zeroline=True,
            zerolinecolor=style.axis_line_color,
            zerolinewidth=0.8,
            showline=True,
            linecolor=style.axis_line_color,
            linewidth=1,
            gridcolor=style.grid_color,
            mirror=True,
        ),
        yaxis=dict(
            title=dict(text="Internal force f(t) [N]", font=dict(color=style.axis_line_color)),
            zeroline=True,
            zerolinecolor=style.axis_line_color,
            zerolinewidth=0.8,
            showline=True,
            linecolor=style.axis_line_color,
            linewidth=1,
            gridcolor=style.grid_color,
            mirror=True,
        ),
        template=style.template,
        # Let Plotly size the figure responsively within the container.
        autosize=True,
        height=style.height,
        paper_bgcolor=style.paper_bgcolor,
        plot_bgcolor=style.plot_bgcolor,
        margin=dict(l=80, r=40, t=125, b=70),
        font=dict(family=style.font_family, size=style.font_size, color=style.axis_line_color),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=style.font_family),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.085,
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        showlegend=True,
    )

    return fig


def save_figure_html(fig: go.Figure, path: str | Path) -> None:
    """Save a standalone HTML file with the given figure."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)

