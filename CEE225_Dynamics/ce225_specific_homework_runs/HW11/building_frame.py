"""Building frame visualization for mode shape analysis."""

import numpy as np
import plotly.graph_objects as go


class BuildingFrame:
    """Building frame visualization with configurable number of stories."""
    
    def __init__(self, num_stories, floor_heights=None, column_spacing=1.0, scale_factor=1.0):
        """
        Parameters:
        -----------
        num_stories : int
            Number of stories
        floor_heights : array-like, optional
            Floor heights. If None, uses [1, 2, ..., num_stories]
        column_spacing : float
            Horizontal spacing between columns
        scale_factor : float
            Scale factor for displacement visualization
        """
        self.num_stories = num_stories
        self.column_spacing = column_spacing
        self.scale_factor = scale_factor
        
        if floor_heights is None:
            self.floor_heights = np.array([i + 1 for i in range(num_stories)], dtype=float)
        else:
            self.floor_heights = np.asarray(floor_heights, dtype=float)
            if len(self.floor_heights) != num_stories:
                raise ValueError(f"floor_heights length ({len(self.floor_heights)}) must match num_stories ({num_stories})")
        
        self.base_y = 0.0
        self.top_y = self.floor_heights[-1]
        self.n_seg_points = 40
    
    def get_traces(self, mode_shape, color):
        """Generate Plotly traces for building frame with mode shape displacement."""
        mode_shape = np.asarray(mode_shape, dtype=float)
        if len(mode_shape) != self.num_stories:
            raise ValueError(f"mode_shape length ({len(mode_shape)}) must match num_stories ({self.num_stories})")
        
        traces = []
        traces.extend(self._draw_undeformed_frame())
        left_displaced, right_displaced, heights_list = self._compute_displaced_positions(mode_shape)
        traces.extend(self._draw_displaced_floors(mode_shape, left_displaced, right_displaced, heights_list, color))
        traces.extend(self._draw_deflected_columns(mode_shape, left_displaced, right_displaced, heights_list, color))
        
        return traces
    
    def _draw_undeformed_frame(self):
        """Draw undeformed frame structure."""
        traces = []
        for x_pos in [-self.column_spacing / 2, self.column_spacing / 2]:
            traces.append(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[self.base_y, self.top_y],
                    mode="lines",
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        for h in self.floor_heights:
            traces.append(
                go.Scatter(
                    x=[-self.column_spacing / 2, self.column_spacing / 2],
                    y=[h, h],
                    mode="lines",
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        return traces
    
    def _compute_displaced_positions(self, mode_shape):
        """Compute displaced positions of floors."""
        left_displaced = []
        right_displaced = []
        heights_list = []
        
        for i, (height, displacement) in enumerate(zip(self.floor_heights, mode_shape)):
            x_left_orig = -self.column_spacing / 2
            x_right_orig = self.column_spacing / 2
            
            x_left_disp = x_left_orig + displacement * self.scale_factor
            x_right_disp = x_right_orig + displacement * self.scale_factor
            
            left_displaced.append(x_left_disp)
            right_displaced.append(x_right_disp)
            heights_list.append(height)
        
        return left_displaced, right_displaced, heights_list
    
    def _draw_displaced_floors(self, mode_shape, left_displaced, right_displaced, heights_list, color):
        """Draw displaced floor markers and beams."""
        traces = []
        
        for i, (height, displacement) in enumerate(zip(heights_list, mode_shape)):
            for x_disp in [left_displaced[i], right_displaced[i]]:
                traces.append(
                    go.Scatter(
                        x=[x_disp],
                        y=[height],
                        mode="markers",
                        marker=dict(size=10, symbol="circle", color=color),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Floor {i+1}</b><br>"
                            f"Height: {height:.1f}<br>"
                            f"Displacement: {displacement:.4f}<br>"
                            f"Normalized Amplitude: {displacement:.4f}<extra></extra>"
                        ),
                    )
                )
            traces.append(
                go.Scatter(
                    x=[left_displaced[i], right_displaced[i]],
                    y=[height, height],
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                    )
                )
            traces.append(
                go.Scatter(
                    x=[-self.column_spacing / 2 - 0.2],
                    y=[height],
                    mode="text",
                    text=[f"F{i+1}"],
                    textposition="middle right",
                    textfont=dict(
                        size=14, color="black", family="Arial, sans-serif", weight="bold"
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        return traces
    
    def _draw_deflected_columns(self, mode_shape, left_displaced, right_displaced, heights_list, color):
        """Draw deflected column shapes using fixed-fixed beam interpolation."""
        traces = []
        
        if not left_displaced:
            return traces
        
        node_heights = [self.base_y] + heights_list
        node_disp = [0.0] + list(mode_shape)
        
        y_smooth = []
        x_left_smooth = []
        x_right_smooth = []
        
        for j in range(len(node_heights) - 1):
            z0 = node_heights[j]
            z1 = node_heights[j + 1]
            L = z1 - z0
            
            u0 = node_disp[j]
            u1 = node_disp[j + 1]
            
            xi = np.linspace(0.0, 1.0, self.n_seg_points)
            shape = 3.0 * xi**2 - 2.0 * xi**3
            u = u0 + (u1 - u0) * shape
            
            z = z0 + xi * L
            x_left = -self.column_spacing / 2 + u * self.scale_factor
            x_right = self.column_spacing / 2 + u * self.scale_factor
            
            y_smooth.append(z)
            x_left_smooth.append(x_left)
            x_right_smooth.append(x_right)
        
        y_smooth = np.concatenate(y_smooth)
        x_left_smooth = np.concatenate(x_left_smooth)
        x_right_smooth = np.concatenate(x_right_smooth)
        for x_smooth in [x_left_smooth, x_right_smooth]:
            traces.append(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hovertemplate=(
                        "<b>Deflected Shape</b><br>"
                        "Fixed-fixed segment under floor displacements<extra></extra>"
                    ),
                )
            )
        for x_disp in [left_displaced, right_displaced]:
            traces.append(
                go.Scatter(
                    x=x_disp,
                    y=heights_list,
                    mode="markers",
                    marker=dict(size=12, symbol="circle", color=color),
                    showlegend=False,
                    hovertemplate="<b>Floor Displacement</b><extra></extra>",
                )
            )
        
        return traces

