from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path


@dataclass
class PlotConfig:
    """Configuration class for plot styling and layout."""
    title_font_size: int = 22
    title_font_family: str = "Arial Black"
    width: int = 950
    height: int = 600
    line_width: float = 2.5
    marker_size: int = 6
    grid_color: str = 'rgba(200,200,200,0.4)'
    legend_position = "top"
    template: str = "plotly_white"



class SolutionPlotter:
    """Professional plotter class for dynamic response analysis."""

    def __init__(self, title, config: Optional[PlotConfig] = None):
        self.title = title
        self.config = config or PlotConfig()
        self.config.title = title

    def _extract_data(self, sol_points: List[Any]) -> tuple[List[float], List[float], List[float], List[float]]:
        """Extract time and displacement data from solution points."""
        t_vals = [p.t for p in sol_points]
        u_vals = [p.u for p in sol_points]
        v_vals = [p.v for p in sol_points if hasattr(p, 'v')]
        a_vals = [p.a for p in sol_points if hasattr(p, 'a')]
        return t_vals, u_vals, v_vals, a_vals

    def _generate_hover_text(self, sol_points: List[Any]) -> List[str]:
        """Generate hover text for each solution point."""
        hover_texts = []
        for p in sol_points:
            meta = p.metadata or {}
            constants = meta.get('Constants', {})
            method = meta.get('Method', None)

            hover_text = (
                f"<b>t:</b> {p.t:.3f}s<br>"
                f"<b>u:</b> {p.u:.4f} in<br>"
                + (f"<b>v:</b> {p.v:.4f} in/s<br>" if hasattr(p, 'v') and p.v else "")
                + (f"<b>a:</b> {p.a:.4f} in/s²<br>" if hasattr(p, 'a') and p.a else "")
                + (f"<b>Method:</b> {method}<br>" if method else "")
            )
            if meta.get('NetError') is not None:
                hover_text += (
                    f"<b>NetError:</b> {meta['NetError']:.2f}<br>"
                    f"<b>ErrorPercentage:</b> {meta['PercentageError']:.2f}%<br>"
                    f"<b>Method:</b> {method}<br>"
                )
            hover_texts.append(hover_text)
        return hover_texts

    def _create_trace(self, label: str, t_vals: List[float], u_vals: List[float],
                      hover_texts: List[str]) -> go.Scatter:
        """Create a Plotly trace for a solution."""
        return go.Scatter(
            x=t_vals,
            y=u_vals,
            mode='lines+markers',
            name=label,
            line=dict(width=self.config.line_width),
            marker=dict(size=self.config.marker_size, symbol="circle"),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts
        )

    def _apply_layout(self, fig: go.Figure,
                      x_label="Time [s]",
                      y_label="Displacement [in]",
                      log_x: bool = False,
                      log_y: bool = False) -> None:
        """Apply layout styling to the figure."""
        fig.update_layout(
            title=dict(
                text=self.config.title,
                font=dict(size=self.config.title_font_size,
                          family=self.config.title_font_family),
                x=0.5
            ),
            xaxis=dict(
                title=x_label,
                gridcolor=self.config.grid_color,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                type='log' if log_x else 'linear'
            ),
            yaxis=dict(
                title=y_label,
                gridcolor=self.config.grid_color,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                type='log' if log_y else 'linear'
            ),
            legend=dict(
                title="Solutions",
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.6)"
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=60, r=40, t=80, b=80)
        )

    def _apply_log_layout(self, fig: go.Figure,
                          x_label="Period [s]",
                          y_label="Pseudo-acceleration [in/s²]",
                          log_x: bool = True,
                          log_y: bool = True,
                          x_range: Optional[List[float]] = None,
                          y_range: Optional[List[float]] = None) -> None:
        """Apply a clean log-log layout with minor gridlines and hidden minor ticks."""

        # Major tick values (example: adjust based on your data)
        x_major = [0.01, 0.1, 1, 10]
        y_major = [10, 100, 1000, 10000, 100000]

        fig.update_layout(
            title=dict(
                text=self.config.title,
                font=dict(size=self.config.title_font_size,
                          family=self.config.title_font_family),
                x=0.5
            ),
            xaxis=dict(
                title=x_label,
                type='log' if log_x else 'linear',
                showgrid=True,
                gridcolor=self.config.grid_color,
                gridwidth=1,
                minor=dict(
                    showgrid=True,
                    gridcolor='rgba(220,220,220,0.3)',
                    gridwidth=0.5,
                    tickmode='linear',
                    tick0=0.01,
                    dtick=0.1
                ),
                tickvals=x_major,
                tickfont=dict(size=12),
                tickangle=0,
                showline=True,
                linewidth=1,
                linecolor='black',
                range=x_range
            ),
            yaxis=dict(
                title=y_label,
                type='log' if log_y else 'linear',
                showgrid=True,
                gridcolor=self.config.grid_color,
                gridwidth=1,
                minor=dict(
                    showgrid=True,
                    gridcolor='rgba(220,220,220,0.3)',
                    gridwidth=0.5,
                    tickmode='linear',
                    tick0=1,
                    dtick=0.1
                ),
                tickvals=y_major,
                tickfont=dict(size=12),
                showline=True,
                linewidth=1,
                linecolor='black',
                range=y_range
            ),
            legend=dict(
                title="Solutions",
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="black",
                borderwidth=1
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=60, r=40, t=80, b=80)
        )

    def _generate_filename(self, base_filename: str) -> str:
        """Generate timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{base_filename}"

    def plot_displacement_vs_time(self, solutions: Dict[str, List[Any]],
                                  filename: str = "u_vs_t.html",
                                  save_to_file: bool = True) -> go.Figure:
        """
        Creates an interactive Plotly graph for displacement vs time.

        Args:
            solutions: Dictionary mapping solution labels to lists of solution points
            filename: Base filename for output (will be timestamped)
            save_to_file: Whether to save the plot to an HTML file

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add traces for each solution
        for label, sol_points in solutions.items():
            t_vals, u_vals, v_vals, a_vals = self._extract_data(sol_points)
            hover_texts = self._generate_hover_text(sol_points)
            trace = self._create_trace(label, t_vals, u_vals, hover_texts)
            fig.add_trace(trace)

        # Apply layout
        self._apply_layout(fig)

        # Save to file if requested
        if save_to_file:
            output_name = self._generate_filename(filename)
            fig.write_html(output_name, include_plotlyjs='cdn', full_html=True)

        return fig

    def plot_acceleration_vs_time(self, solutions: Dict[str, List[Any]],
                                  filename: str = "a_vs_t.html",
                                  save_to_file: bool = True,
                                  use_log_x: bool = False,
                                  use_log_y: bool = False) -> go.Figure:
        """
        Creates an interactive Plotly graph for acceleration vs time.

        Args:
            solutions: Dictionary mapping solution labels to lists of solution points
            filename: Base filename for output (will be timestamped)
            save_to_file: Whether to save the plot to an HTML file
            use_log_x: Use logarithmic scale on x-axis
            use_log_y: Use logarithmic scale on y-axis

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        for label, sol_points in solutions.items():
            t_vals, u_vals, v_vals, a_vals = self._extract_data(sol_points)
            hover_texts = self._generate_hover_text(sol_points)
            trace = self._create_trace(label, t_vals, a_vals, hover_texts)
            fig.add_trace(trace)

        self._apply_log_layout(fig,
                           x_label="Time [s]",
                           y_label="Pseudo-acceleration, Normalized to 96 in/s²")

        if save_to_file:
            output_name = self._generate_filename(filename)
            fig.write_html(output_name, include_plotlyjs='cdn', full_html=True)

        return fig


def plot_displacement_vs_time(title, solutions: Dict[str, List[Any]],
                              filename: str = "u_vs_t.html") -> go.Figure:
    """
    Convenience function to maintain backward compatibility.

    Args:
        solutions: Dictionary mapping solution labels to lists of solution points
        filename: Base filename for output

    Returns:
        Plotly Figure object
    """
    plotter = SolutionPlotter(title)
    return plotter.plot_displacement_vs_time(solutions, filename)

def plot_pseudo_acceleration_vs_time(
        title, solutions: Dict[str, List[Any]],
        filename: str = "a_vs_t.html", **kwargs) -> go.Figure:
    plotter = SolutionPlotter(title)
    return plotter.plot_acceleration_vs_time(solutions, filename, **kwargs)