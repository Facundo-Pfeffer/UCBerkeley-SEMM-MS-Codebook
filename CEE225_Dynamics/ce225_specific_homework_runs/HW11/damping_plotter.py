"""Plotting for damping analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotting_utils import Colors, get_axis_style


class DampingPlotter:
    """Creates visualizations for damping analysis."""
    
    def __init__(self, num_floors):
        """Initialize damping plotter."""
        self.num_floors = num_floors
        self.floor_colors = [Colors.BERKELEY_BLUE, '#808080', '#8B0000']
        self.floor_names = [f'Floor {i+1}' for i in range(num_floors)]
    
    def plot_damping_analysis(self, mode_number, time, acc_data, damping_results, mode_shape, output_path):
        """Create comprehensive damping visualization for a single mode."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'<b>Mode {mode_number} - Full Time History</b>',
                f'<b>Mode {mode_number} - Filtered Decay Portion</b>',
                f'<b>Mode {mode_number} - Envelope Decay</b>',
                f'<b>Mode {mode_number} - Damping Estimates by Floor</b>'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            horizontal_spacing=0.15,
            vertical_spacing=0.15
        )
        
        for floor_idx in range(self.num_floors):
            color = self.floor_colors[floor_idx % len(self.floor_colors)]
            floor_name = self.floor_names[floor_idx]
            
            if mode_shape is not None:
                modal_acc = acc_data[floor_idx] * mode_shape[floor_idx]
            else:
                modal_acc = acc_data[floor_idx]
            
            floor_result = damping_results.get(f'floor_{floor_idx + 1}', {})
            
            self._add_full_history(fig, time, modal_acc, floor_name, color, row=1, col=1)
            
            if floor_result.get('decay_start_time') is not None:
                self._add_decay_with_peaks(fig, floor_result, floor_name, color, row=1, col=2)
                self._add_envelope_plot(fig, floor_result, floor_name, color, row=2, col=1, damping_results=damping_results)
        
        self._add_damping_comparison(fig, damping_results, row=2, col=2)
        self._update_layout(fig, mode_number, damping_results)
        
        output_path_str = str(output_path)
        fig.write_html(output_path_str, include_plotlyjs='cdn')
        print(f"[SUCCESS] Generated: {output_path_str}")
        
        return fig
    
    def _add_full_history(self, fig, time, acc, floor_name, color, row, col):
        """Add full time history plot."""
        fig.add_trace(
            go.Scatter(
                x=time,
                y=acc,
                mode='lines',
                name=floor_name,
                line=dict(color=color, width=2),
                opacity=0.7,
                showlegend=True,
                hovertemplate=f'<b>{floor_name}</b><br>Time: %{{x:.3f}} s<br>Acceleration: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_decay_with_peaks(self, fig, floor_result, floor_name, color, row, col):
        """Add decay portion with marked peaks."""
        decay_time = floor_result.get('decay_time')
        decay_acc = floor_result.get('decay_acc')
        peaks = floor_result.get('peaks')
        peak_times = floor_result.get('peak_times')
        decay_start = floor_result.get('decay_start_time')
        
        if decay_time is None or decay_acc is None:
            return
        
        fig.add_trace(
            go.Scatter(
                x=decay_time,
                y=decay_acc,
                mode='lines',
                name=f'{floor_name} Decay',
                line=dict(color=color, width=2),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f'<b>{floor_name} Decay</b><br>Time: %{{x:.3f}} s<br>Acceleration: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        if peaks is not None and peak_times is not None and len(peaks) > 0:
            damping_ratio = floor_result.get("damping_ratio")
            damping_str = f"{damping_ratio:.4f}" if damping_ratio is not None else "N/A"
            
            peak_numbers = [f"Peak {i+1}" for i in range(len(peaks))]
            
            fig.add_trace(
                go.Scatter(
                    x=peak_times,
                    y=peaks,
                    mode='markers',
                    name=f'{floor_name} Peaks',
                    marker=dict(color=color, size=8, symbol='diamond'),
                    text=peak_numbers,
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{floor_name} %{{text}}</b><br>'
                        f'Time: %{{x:.3f}} s<br>'
                        f'Amplitude: %{{y:.4f}}<br>'
                        f'Damping: {damping_str}<extra></extra>'
                    )
                ),
                row=row, col=col
            )
    
    def _add_envelope_plot(self, fig, floor_result, floor_name, color, row, col, damping_results=None):
        """Add envelope decay plot."""
        decay_time = floor_result.get('decay_time')
        envelope = floor_result.get('envelope')
        damping_ratio = floor_result.get('damping_ratio')
        peak_times = floor_result.get('peak_times')
        decay_start = floor_result.get('decay_start_time')
        
        if decay_time is None or envelope is None:
            return
        
        damping_str = f"{damping_ratio:.4f}" if damping_ratio is not None else "N/A"
        
        fig.add_trace(
            go.Scatter(
                x=decay_time,
                y=envelope,
                mode='lines',
                name=f'{floor_name} Envelope',
                line=dict(color=color, width=3),
                opacity=0.8,
                showlegend=False,
                hovertemplate=(
                    f'<b>{floor_name} Envelope</b><br>'
                    f'Time: %{{x:.3f}} s<br>'
                    f'Envelope: %{{y:.4f}}<br>'
                    f'ζ = {damping_str}<extra></extra>'
                )
            ),
            row=row, col=col
        )
        
        if damping_ratio and len(decay_time) > 0 and len(envelope) > 0:
            t0 = decay_time[0]
            A0 = envelope[0]
            
            if damping_results is not None:
                natural_freq = damping_results.get('natural_freq')
            else:
                natural_freq = None
            
            if natural_freq is None and peak_times is not None and len(peak_times) > 1:
                periods = np.diff(peak_times)
                natural_freq = 1.0 / np.mean(periods)
            elif natural_freq is None:
                natural_freq = 1.0
            
            omega_n = 2 * np.pi * natural_freq
            
            t_fit = np.linspace(decay_time[0], decay_time[-1], 200)
            envelope_fit = A0 * np.exp(-damping_ratio * omega_n * (t_fit - t0))
            
            # Exponential fit explanation:
            # Position: A0 is the envelope amplitude at decay start (t0)
            # Decay: A(t) = A0 * exp(-ζ * ωn * (t - t0))
            # If fit appears horizontal, it means:
            # - Very low damping (ζ << 1) → slow decay
            # - Short decay window → limited visible decay
            # - On log scale, slow decay can appear nearly flat
            
            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=envelope_fit,
                    mode='lines',
                    name=f'{floor_name} Fit',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=False,
                    hovertemplate=f'<b>Exponential Fit</b><br>ζ = {damping_ratio:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    def _add_damping_comparison(self, fig, damping_results, row, col):
        """Add bar chart comparing damping estimates across floors."""
        floor_dampings = []
        floor_labels = []
        
        for floor_idx in range(self.num_floors):
            floor_result = damping_results.get(f'floor_{floor_idx + 1}', {})
            damping = floor_result.get('damping_ratio')
            if damping is not None:
                floor_dampings.append(damping)
                floor_labels.append(f'Floor {floor_idx + 1}')
        
        if floor_dampings:
            mean_damping = damping_results.get('mean_damping', np.mean(floor_dampings))
            
            fig.add_trace(
                go.Bar(
                    x=floor_labels,
                    y=floor_dampings,
                    name='Damping Ratio',
                    marker=dict(color=self.floor_colors[:len(floor_dampings)]),
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>ζ = %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.add_hline(
                y=mean_damping,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_damping:.4f}",
                row=row, col=col
            )
    
    def _update_layout(self, fig, mode_number, damping_results):
        """Update figure layout."""
        mean_damping = damping_results.get('mean_damping')
        std_damping = damping_results.get('std_damping')
        
        title_text = f'Mode {mode_number} Damping Analysis - Logarithmic Decrement'
        if mean_damping is not None:
            title_text += f'<br><span style="font-size:14px;">Mean ζ = {mean_damping:.4f}'
            if std_damping is not None:
                title_text += f' ± {std_damping:.4f}'
            title_text += '</span>'
        
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
                x=1.02
            )
        )
        
        fig.update_xaxes(get_axis_style(), row=1, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=1, col=1, title_text="Modal Acceleration")
        fig.update_xaxes(get_axis_style(), row=1, col=2, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=1, col=2, title_text="Modal Acceleration")
        fig.update_xaxes(get_axis_style(), row=2, col=1, title_text="Time [s]")
        fig.update_yaxes(get_axis_style(), row=2, col=1, title_text="Envelope Amplitude", type="log")
        fig.update_xaxes(get_axis_style(), row=2, col=2, title_text="Floor")
        fig.update_yaxes(get_axis_style(), row=2, col=2, title_text="Damping Ratio ζ")
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color=Colors.BERKELEY_BLUE,
                                      family='Arial, sans-serif')

