from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import plotly.graph_objects as go


# MATLAB-like default blue (RGB 0–255): [0, 114, 189] -> hex #0072BD
MATLAB_BLUE = "rgb(0, 114, 189)"
# Darker blue for hysteresis plots (closer to MATLAB printed output)
MATLAB_BLUE_DARK = "rgb(0, 55, 95)"
MATLAB_ORANGE = "rgb(217, 83, 25)"
MATLAB_YELLOW = "rgb(237, 177, 32)"


@dataclass
class PlotStyle:
    """Styling configuration for isolator force–displacement plots (MATLAB-like)."""

    width: int = 900
    height: int = 550
    template: str = "plotly_white"
    line_width: float = 2.0
    marker_size: int = 3
    grid_color: str = "rgba(0, 0, 0, 0.15)"
    font_family: str = "Arial"
    font_size: int = 16
    title_size: int = 22
    paper_bgcolor: str = "rgb(255, 255, 255)"
    plot_bgcolor: str = "rgb(250, 250, 250)"
    axis_line_color: str = "rgb(0, 0, 0)"
    cloud_color: str = MATLAB_BLUE_DARK


def create_force_displacement_figure(
    points: np.ndarray,
    *,
    title: str,
    style: PlotStyle | None = None,
    meta: Mapping[str, object] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    u0_point: tuple[float, float] | None = None,
    p0_point: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Build a Plotly figure for an isolator force–displacement cloud.

    Parameters
    ----------
    points:
        Array of shape (N, 2) with columns [disp, force].
    title:
        Figure title.
    style:
        Optional PlotStyle override.
    meta:
        Optional metadata dict to render in the subtitle / annotations
        (e.g. known max displacement, strain level).
    """
    style = style or PlotStyle()
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points array, got shape {points.shape}")

    x = points[:, 0]
    y = points[:, 1]

    fig = go.Figure()

    # Main cloud (MATLAB-like blue)
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            name="Digitized cloud",
            marker=dict(
                size=style.marker_size,
                color=style.cloud_color,
                opacity=0.85,
            ),
            hovertemplate="u = %{x:.4f} in<br>F = %{y:.4f} kips<extra></extra>",
        )
    )

    # Optional highlight for U0 (max |u|) – MATLAB green
    if u0_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[u0_point[0]],
                y=[u0_point[1]],
                mode="markers",
                name="U₀",
                marker=dict(
                    size=10,
                    color="rgb(0, 128, 0)",
                    symbol="circle-open",
                    line=dict(width=2, color="rgb(0, 128, 0)"),
                ),
                hovertemplate="U₀ point<br>u = %{x:.4f} in<br>F = %{y:.4f} kips<extra></extra>",
            )
        )

    # Optional highlight for P0 (max |F|) – MATLAB red
    if p0_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[p0_point[0]],
                y=[p0_point[1]],
                mode="markers",
                name="P₀",
                marker=dict(
                    size=10,
                    color=MATLAB_ORANGE,
                    symbol="x",
                    line=dict(width=2, color=MATLAB_ORANGE),
                ),
                hovertemplate="P₀ point<br>u = %{x:.4f} in<br>F = %{y:.4f} kips<extra></extra>",
            )
        )

    subtitle = None
    if meta:
        parts: list[str] = []
        if "strain" in meta:
            parts.append(f"Strain ≈ {meta['strain']}")
        if "max_disp" in meta:
            parts.append(f"u_max = {meta['max_disp']} in")
        if "notes" in meta:
            parts.append(str(meta["notes"]))
        subtitle = " · ".join(parts) if parts else None

    full_title = title if subtitle is None else f"{title}<br><span style='font-size:0.85em;color:#64748b;'>{subtitle}</span>"

    fig.update_layout(
        title=dict(
            text=full_title,
            x=0.5,
            xanchor="center",
            font=dict(size=style.title_size, family=style.font_family, color="#1e293b"),
        ),
        xaxis=dict(
            title=dict(text="Displacement u [in]", font=dict(color=style.axis_line_color)),
            zeroline=True,
            zerolinecolor=style.axis_line_color,
            zerolinewidth=0.8,
            showline=True,
            linecolor=style.axis_line_color,
            linewidth=1,
            gridcolor=style.grid_color,
            mirror=True,
            range=list(x_range) if x_range is not None else None,
            tickfont=dict(color=style.axis_line_color),
        ),
        yaxis=dict(
            title=dict(text="Force F [kips]", font=dict(color=style.axis_line_color)),
            zeroline=True,
            zerolinecolor=style.axis_line_color,
            zerolinewidth=0.8,
            showline=True,
            linecolor=style.axis_line_color,
            linewidth=1,
            gridcolor=style.grid_color,
            mirror=True,
            range=list(y_range) if y_range is not None else None,
            tickfont=dict(color=style.axis_line_color),
        ),
        template=style.template,
        width=style.width,
        height=style.height,
        paper_bgcolor=style.paper_bgcolor,
        plot_bgcolor=style.plot_bgcolor,
        margin=dict(l=80, r=40, t=90, b=70),
        font=dict(family=style.font_family, size=style.font_size, color=style.axis_line_color),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=style.font_family),
        # Put legend on top without taking vertical space (overlay inside plot area)
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.65)",
            bordercolor="rgba(0, 0, 0, 0.20)",
            borderwidth=1,
            font=dict(size=12, family=style.font_family, color=style.axis_line_color),
        ),
        showlegend=True,
    )

    return fig


def save_figure_html(
    fig: go.Figure,
    path: str | Path,
    *,
    include_plotlyjs: str = "cdn",
    full_html: bool = True,
) -> None:
    """Small wrapper around write_html with sane defaults."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=full_html)

