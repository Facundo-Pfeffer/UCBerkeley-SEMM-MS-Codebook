"""Interactive plotting for mode shape analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from building_frame import BuildingFrame
from plotting_utils import Colors, get_axis_style


class ModeShapePlotter:
    """Creates interactive visualizations for mode shape analysis."""
    
    def __init__(self, num_floors, floor_heights=None, floor_colors=None, floor_symbols=None):
        """If floor_colors/symbols are None, uses default palette that cycles."""
        self.num_floors = num_floors
        self.building_frame = BuildingFrame(num_floors, floor_heights)
        
        if floor_colors is None:
            default_colors = [Colors.BERKELEY_BLUE, '#808080', '#8B0000']
            self.floor_colors = [default_colors[i % len(default_colors)] for i in range(num_floors)]
        else:
            self.floor_colors = floor_colors
        
        if floor_symbols is None:
            default_symbols = ['circle', 'square', 'diamond']
            self.floor_symbols = [default_symbols[i % len(default_symbols)] for i in range(num_floors)]
        else:
            self.floor_symbols = floor_symbols
        
        self.floor_names = [f'Floor {i+1}' for i in range(num_floors)]
    
    def plot_mode_shape(self, mode_number, mode_shape, statistics, output_path, 
                       mode_shape_mass_normalized=None):
        """
        Create visualization for a single mode.
        
        Parameters:
        -----------
        mode_number : int
            Mode number
        mode_shape : array-like
            Original mode shape
        statistics : dict
            Statistics dictionary
        output_path : Path or str
            Output file path
        mode_shape_mass_normalized : array-like, optional
            Mass-normalized/orthogonalized mode shape to compare
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'<b>Mode {mode_number} - Building Frame</b><br>'
                '<span style="font-size:11px;">Grey: Structure | Colored: Displacement</span>',
                '<b>Gaussian Distributions</b>',
                None,
                '<b>Mode Shape Variation Over Time</b>' + 
                ('<br><span style="font-size:11px;">Solid: Original | Dashed Teal: Mass-Normalized (Post-Processing, Scaled for Comparison)</span>' 
                 if mode_shape_mass_normalized is not None else '')
            ),
            specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                   [None, {"type": "scatter"}]],
            horizontal_spacing=0.15,
            vertical_spacing=0.15,
            column_widths=[0.45, 0.55]
        )
        
        # Original mode shape (solid, blue)
        building_traces = self.building_frame.get_traces(mode_shape, Colors.BERKELEY_BLUE)
        for trace in building_traces:
            fig.add_trace(trace, row=1, col=1)
        
        self._add_gaussian_plots(fig, mode_number, statistics, row=1, col=2)
        self._add_time_variation_plot(fig, mode_number, statistics, row=2, col=2,
                                     mode_shape_mass_normalized=mode_shape_mass_normalized)
        self._update_layout(fig, mode_number, has_mass_normalized=mode_shape_mass_normalized is not None)
        self._update_axes(fig, statistics)
        
        output_path_str = str(output_path)
        fig.write_html(output_path_str, include_plotlyjs='cdn')
        print(f"[SUCCESS] Generated: {output_path_str}")
        
        return fig
    
    def plot_combined_mode_shapes(self, all_mode_shapes, all_statistics, output_path,
                                  all_mode_shapes_mass_normalized=None):
        """Create combined visualization showing all modes."""
        num_modes = len(all_mode_shapes)
        
        fig = make_subplots(
            rows=1, cols=num_modes,
            subplot_titles=[f'Mode {i+1} - Building Frame' for i in range(num_modes)],
            horizontal_spacing=0.15
        )
        
        colors = [Colors.BERKELEY_BLUE, Colors.CALIFORNIA_GOLD, Colors.FOUNDERS_ROCK]
        
        for mode_idx, (mode_shape, stats) in enumerate(zip(all_mode_shapes, all_statistics)):
            mode_num = mode_idx + 1
            color = colors[mode_idx % len(colors)]
            
            # Original mode shape only
            building_traces = self.building_frame.get_traces(mode_shape, color)
            for trace in building_traces:
                fig.add_trace(trace, row=1, col=mode_num)
        
        fig.update_layout(
            title=dict(
                text='All Mode Shapes - 3-Story MDOF Building Comparison',
                x=0.5,
                font=dict(size=20, color=Colors.TEXT_DARK, family='Arial, sans-serif')
            ),
            plot_bgcolor=Colors.BG_LIGHT,
            paper_bgcolor=Colors.BG_WHITE,
            font=dict(family='Arial, sans-serif', size=12),
            height=600,
            showlegend=False
        )
        
        for col in range(1, num_modes + 1):
            fig.update_xaxes(
                get_axis_style(), row=1, col=col,
                title_text="Mode Shape\n[Normalized to max=1]",
                range=[-1.75, 1.75]
            )
            fig.update_yaxes(
                get_axis_style(), row=1, col=col,
                title_text="Floor Number", range=[0, self.building_frame.top_y + 0.5]
            )
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=16, color=Colors.BERKELEY_BLUE,
                                      family='Arial, sans-serif')
        
        output_path_str = str(output_path)
        fig.write_html(output_path_str, include_plotlyjs='cdn')
        print(f"[SUCCESS] Generated: {output_path_str}")
        
        return fig
    
    def _add_gaussian_plots(self, fig, mode_number, statistics, row, col):
        """Add Gaussian distribution plots."""
        std_all = statistics['std_mode_shape']
        mode_shape_ref = statistics['mode_shape_reference']
        
        mean_min = np.min(mode_shape_ref)
        mean_max = np.max(mode_shape_ref)
        std_max = np.max(std_all)
        x_range = [mean_min - 3 * std_max, mean_max + 3 * std_max]
        x_range_abs = max(abs(x_range[0]), abs(x_range[1]))
        x_range = [-x_range_abs, x_range_abs]
        x_gauss = np.linspace(x_range[0], x_range[1], 500)
        
        for floor_idx in range(self.num_floors):
            mean_value = mode_shape_ref[floor_idx]
            
            std_val = std_all[floor_idx]
            if std_val < 1e-10:
                std_val = 1e-10
            
            gauss_pdf = (1 / (std_val * np.sqrt(2 * np.pi))) * \
                        np.exp(-0.5 * ((x_gauss - mean_value) / std_val) ** 2)
            
            fig.add_trace(
                go.Scatter(
                    x=x_gauss,
                    y=gauss_pdf,
                    mode='lines',
                    name=self.floor_names[floor_idx],
                    legendgroup=f'Mode {mode_number}',
                    line=dict(color=self.floor_colors[floor_idx], width=3),
                    opacity=0.9,
                    showlegend=True,
                    hovertemplate=(
                        f'<b>{self.floor_names[floor_idx]}</b><br>'
                        f'DOF {floor_idx + 1}<br>'
                        f'Mean: {mean_value:.4f}<br>'
                        f'Std Dev: {std_val:.4f}<br>'
                        f'<extra></extra>'
                    ),
                ),
                row=row, col=col
            )
    
    def _add_time_variation_plot(self, fig, mode_number, statistics, row, col,
                                mode_shape_mass_normalized=None):
        """Add time variation plot."""
        all_mode_shapes = statistics['all_mode_shapes']
        time = statistics['time']
        mode_shape_ref = statistics['mode_shape_reference']
        
        sign_ref = np.sign(mode_shape_ref)
        sign_ref[sign_ref == 0] = 1
        
        time_range_size = 1.0
        time_max = time[-1] if len(time) > 0 else 1.0
        n_ranges = int(np.ceil(time_max / time_range_size))
        
        for i in range(self.num_floors):
            floor_values = all_mode_shapes[:, i]
            
            mean_inst = np.mean(floor_values)
            if sign_ref[i] != 0 and np.sign(mean_inst) != sign_ref[i]:
                floor_values = -floor_values
            
            time_grouped = []
            values_grouped = []
            
            for range_idx in range(n_ranges):
                t_start = range_idx * time_range_size
                t_end = min((range_idx + 1) * time_range_size, time_max)
                
                mask = (time >= t_start) & (time < t_end)
                if np.any(mask):
                    time_grouped.append(np.mean(time[mask]))
                    values_grouped.append(np.mean(floor_values[mask]))
            
            time_grouped = np.array(time_grouped)
            values_grouped = np.array(values_grouped)
            
            fig.add_trace(
                go.Scatter(
                    x=time_grouped,
                    y=values_grouped,
                    mode='markers',
                    name=self.floor_names[i],
                    legendgroup=f'Mode {mode_number}',
                    marker=dict(
                        color=self.floor_colors[i],
                        size=6,
                        opacity=0.8,
                        symbol=self.floor_symbols[i],
                        line=dict(width=1, color='white')
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{self.floor_names[i]}</b><br>'
                        f'Time Range: %{{x:.1f}} s (grouped average)<br>'
                        f'Average Mode Shape Value: %{{y:.4f}}<br>'
                        f'Calculated Mode Shape: {mode_shape_ref[i]:.4f}<br>'
                        f'<extra></extra>'
                    ),
                ),
                row=row, col=col
            )
            
            # Original reference line
            ref_value = mode_shape_ref[i]
            fig.add_trace(
                go.Scatter(
                    x=[time[0], time[-1]] if len(time) > 0 else [0, 1],
                    y=[ref_value, ref_value],
                    mode='lines',
                    name=f'{self.floor_names[i]} Calculated (Original)',
                    legendgroup=f'Mode {mode_number}',
                    line=dict(color=self.floor_colors[i], width=3, dash='solid'),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{self.floor_names[i]} Calculated Mode Shape (Original)</b><br>'
                        f'Value: {ref_value:.4f}<br>'
                        f'<extra></extra>'
                    ),
                ),
                row=row, col=col
            )
    
            # Mass-normalized reference line if provided (scaled for visual comparison)
            if mode_shape_mass_normalized is not None:
                mass_norm_value = mode_shape_mass_normalized[i]
                
                # Scale mass-normalized to match original for visual comparison
                # Find scaling factor based on max absolute values
                mode_shape_ref = statistics['mode_shape_reference']
                max_original = np.max(np.abs(mode_shape_ref))
                max_mass_norm = np.max(np.abs(mode_shape_mass_normalized))
                
                if max_mass_norm > 1e-12:
                    scale_factor = max_original / max_mass_norm
                    mass_norm_value_scaled = mass_norm_value * scale_factor
                else:
                    mass_norm_value_scaled = mass_norm_value
                    scale_factor = 1.0
                
                # Use a professional teal color for better contrast
                mass_norm_color = '#20B2AA'  # Light sea green - professional teal
                
                fig.add_trace(
                    go.Scatter(
                        x=[time[0], time[-1]] if len(time) > 0 else [0, 1],
                        y=[mass_norm_value_scaled, mass_norm_value_scaled],
                        mode='lines',
                        name=f'{self.floor_names[i]} Mass-Normalized',
                        legendgroup=f'Mode {mode_number}',
                        line=dict(color=mass_norm_color, width=2.5, dash='dash'),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>{self.floor_names[i]} Mass-Normalized Mode Shape</b><br>'
                            f'Scaled Value (for comparison): {mass_norm_value_scaled:.4f}<br>'
                            f'Actual Value: {mass_norm_value:.4f}<br>'
                            f'Scale Factor: {scale_factor:.4f}<br>'
                            f'<extra></extra>'
                        ),
                    ),
                    row=row, col=col
                )
    
    def _update_layout(self, fig, mode_number, has_mass_normalized=False):
        """Update figure layout."""
        title_text = f'Mode {mode_number} Shape Analysis - 3-Story MDOF Building'
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=18, color=Colors.TEXT_DARK, family='Arial, sans-serif')
            ),
            plot_bgcolor=Colors.BG_LIGHT,
            paper_bgcolor=Colors.BG_WHITE,
            font=dict(family='Arial, sans-serif', size=12),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                groupclick="toggleitem",
                font=dict(size=12, family='Arial, sans-serif'),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1,
                tracegroupgap=5
            )
        )
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE,
                                      family='Arial, sans-serif')
    
    def _update_axes(self, fig, statistics):
        """Update axis configurations."""
        std_all = statistics['std_mode_shape']
        mode_shape_ref = statistics['mode_shape_reference']
        
        mean_min = np.min(mode_shape_ref)
        mean_max = np.max(mode_shape_ref)
        std_max = np.max(std_all)
        x_range = [mean_min - 3 * std_max, mean_max + 3 * std_max]
        x_range_abs = max(abs(x_range[0]), abs(x_range[1]))
        x_range = [-x_range_abs, x_range_abs]
        
        fig.update_xaxes(
            get_axis_style(), row=1, col=1,
            title_text="Mode Shape (Normalized)",
            range=[-1.75, 1.75]
        )
        fig.update_yaxes(
            get_axis_style(), row=1, col=1,
            title_text="Floor Height", range=[0, self.building_frame.top_y + 0.5]
        )
        fig.update_xaxes(
            get_axis_style(), row=1, col=2,
            title_text="Mode Shape Value", range=x_range
        )
        fig.update_yaxes(
            get_axis_style(), row=1, col=2,
            title_text="Probability Density [%]"
        )
        fig.update_xaxes(
            get_axis_style(), row=2, col=2,
            title_text="Time [s]"
        )
        fig.update_yaxes(
            get_axis_style(), row=2, col=2,
            title_text="Normalized Mode Shape Value"
        )

