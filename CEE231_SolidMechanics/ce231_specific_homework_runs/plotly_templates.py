#!/usr/bin/env python3
"""
Shared Plotly Templates and Styling for CEE 231 Dashboards
===========================================================
Provides consistent styling, colors, and configurations for all interactive dashboards.

Author: Facundo Pfeffer
Course: CEE 231 - Solid Mechanics
University of California, Berkeley
"""

# ============================================================================
# COLOR SCHEMES
# ============================================================================

class UCBerkeleyColors:
    """Official UC Berkeley brand colors."""
    BERKELEY_BLUE = '#003262'
    CALIFORNIA_GOLD = '#FDB515'
    FOUNDERS_ROCK = '#3B7EA1'
    MEDALIST = '#C4820E'
    BAY_FOG = '#DDD5C7'
    LAWRENCE = '#00B0DA'
    WELLMAN_TILE = '#D9661F'
    ROSE_GARDEN = '#EE1F60'
    SOUTH_HALL = '#6C3302'
    SATHER_GATE = '#B9D3B6'
    
    # Additional utility colors
    TEXT_DARK = '#2C3E50'
    TEXT_LIGHT = '#7F8C8D'
    GRID = 'rgba(200, 200, 200, 0.3)'
    BG_LIGHT = 'rgba(250, 250, 250, 0.98)'
    BG_WHITE = 'white'

# ============================================================================
# AXIS STYLING
# ============================================================================

def get_axis_style(gridcolor=UCBerkeleyColors.GRID, 
                   linecolor='black',
                   zeroline=True):
    """
    Standard axis styling for professional plots.
    
    Args:
        gridcolor: Color for grid lines
        linecolor: Color for axis lines
        zeroline: Whether to show zero line
    
    Returns:
        dict: Axis styling configuration
    """
    return dict(
        showgrid=True,
        gridcolor=gridcolor,
        gridwidth=1,
        zeroline=zeroline,
        zerolinecolor='rgba(0, 0, 0, 0.3)',
        zerolinewidth=2,
        showline=True,
        linewidth=2,
        linecolor=linecolor,
        mirror=True,
        ticks='outside',
        tickwidth=1.5,
        tickcolor=linecolor,
        tickfont=dict(size=12, family='Arial, sans-serif')
    )

# ============================================================================
# PLOT STYLING
# ============================================================================

def get_plot_layout_style(height=500, width=None, margin=None):
    """
    Standard layout styling for plots.
    
    Args:
        height: Plot height in pixels
        width: Plot width in pixels (None for auto)
        margin: Custom margins dict (None for default)
    
    Returns:
        dict: Layout styling configuration
    """
    if margin is None:
        margin = dict(l=80, r=40, t=80, b=60)
    
    layout = dict(
        plot_bgcolor=UCBerkeleyColors.BG_LIGHT,
        paper_bgcolor=UCBerkeleyColors.BG_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
        height=height,
        margin=margin,
        hovermode='closest'
    )
    
    if width is not None:
        layout['width'] = width
    
    return layout

# ============================================================================
# LINE STYLES
# ============================================================================

def get_line_style(color, width=3, dash=None):
    """
    Standard line styling.
    
    Args:
        color: Line color
        width: Line width
        dash: Dash pattern ('solid', 'dash', 'dot', 'dashdot')
    
    Returns:
        dict: Line styling configuration
    """
    style = dict(color=color, width=width)
    if dash:
        style['dash'] = dash
    return style

# ============================================================================
# MARKER STYLES
# ============================================================================

def get_marker_style(color, size=10, symbol='circle', 
                     line_color='white', line_width=2):
    """
    Standard marker styling.
    
    Args:
        color: Marker color
        size: Marker size
        symbol: Marker symbol
        line_color: Marker border color
        line_width: Marker border width
    
    Returns:
        dict: Marker styling configuration
    """
    return dict(
        size=size,
        color=color,
        symbol=symbol,
        line=dict(width=line_width, color=line_color)
    )

# ============================================================================
# TEXT STYLES
# ============================================================================

def get_text_style(color=UCBerkeleyColors.TEXT_DARK, size=11, 
                   family='Arial, sans-serif'):
    """
    Standard text styling.
    
    Args:
        color: Text color
        size: Font size
        family: Font family
    
    Returns:
        dict: Text styling configuration
    """
    return dict(size=size, color=color, family=family)

# ============================================================================
# TITLE STYLES
# ============================================================================

def get_title_style(text, size=20, color=UCBerkeleyColors.BERKELEY_BLUE,
                    x=0.5, xanchor='center'):
    """
    Standard title styling.
    
    Args:
        text: Title text (can include HTML tags)
        size: Font size
        color: Text color
        x: Horizontal position (0-1)
        xanchor: Horizontal anchor
    
    Returns:
        dict: Title styling configuration
    """
    return dict(
        text=text,
        font=dict(size=size, color=color, family='Arial, sans-serif'),
        x=x,
        xanchor=xanchor
    )

# ============================================================================
# SLIDER STYLES
# ============================================================================

def get_slider_style(steps, active_index, 
                     prefix="<b>Parameter = </b>", suffix="",
                     y=-0.05, x=0.1, length=0.8):
    """
    Standard slider styling.
    
    Args:
        steps: List of slider step dicts
        active_index: Index of initially active step
        prefix: Prefix text for current value
        suffix: Suffix text for current value
        y: Vertical position
        x: Horizontal position (left anchor)
        length: Slider length (0-1)
    
    Returns:
        dict: Slider styling configuration
    """
    return dict(
        active=active_index,
        yanchor="top",
        y=y,
        xanchor="left",
        x=x,
        currentvalue=dict(
            prefix=prefix,
            suffix=suffix,
            visible=True,
            xanchor="left",
            font=dict(size=14, color=UCBerkeleyColors.BERKELEY_BLUE,
                     family='Arial, sans-serif')
        ),
        pad=dict(b=10, t=10),
        len=length,
        steps=steps,
        bgcolor='rgba(0, 50, 98, 0.05)',
        bordercolor=UCBerkeleyColors.BERKELEY_BLUE,
        borderwidth=2,
        tickcolor=UCBerkeleyColors.BERKELEY_BLUE,
        font=dict(size=11, family='Arial, sans-serif')
    )

# ============================================================================
# TABLE STYLES
# ============================================================================

def get_table_header_style(fill_color=UCBerkeleyColors.BERKELEY_BLUE,
                           font_color='white', font_size=13):
    """
    Standard table header styling.
    
    Args:
        fill_color: Header background color
        font_color: Header text color
        font_size: Header font size
    
    Returns:
        dict: Table header styling configuration
    """
    return dict(
        fill_color=fill_color,
        font=dict(color=font_color, size=font_size, family='Arial, sans-serif'),
        align='center',
        height=35
    )

def get_table_cells_style(fill_color='rgba(250, 250, 250, 0.95)',
                          font_size=12, align='left', height=30):
    """
    Standard table cells styling.
    
    Args:
        fill_color: Cell background color
        font_size: Cell font size
        align: Text alignment
        height: Row height
    
    Returns:
        dict: Table cells styling configuration
    """
    return dict(
        fill_color=fill_color,
        font=dict(size=font_size, family='Arial, sans-serif'),
        align=align,
        height=height
    )

# ============================================================================
# SUBPLOT CONFIGURATIONS
# ============================================================================

def get_subplot_config(rows, cols, subplot_titles=None,
                       horizontal_spacing=0.12, vertical_spacing=0.15):
    """
    Standard subplot configuration.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        subplot_titles: List of subplot titles
        horizontal_spacing: Horizontal spacing between subplots
        vertical_spacing: Vertical spacing between subplots
    
    Returns:
        dict: Subplot configuration
    """
    config = dict(
        rows=rows,
        cols=cols,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing
    )
    
    if subplot_titles:
        # Add bold formatting to titles
        formatted_titles = [f'<b>{title}</b>' for title in subplot_titles]
        config['subplot_titles'] = formatted_titles
    
    return config

def format_subplot_titles(fig, color=UCBerkeleyColors.BERKELEY_BLUE, 
                         size=14):
    """
    Format subplot titles in an existing figure.
    
    Args:
        fig: Plotly figure object
        color: Title color
        size: Font size
    """
    # Subplot titles are stored in annotations
    for annotation in fig['layout']['annotations']:
        if 'text' in annotation and '<b>' in annotation.get('text', ''):
            annotation['font'] = dict(
                size=size, 
                color=color,
                family='Arial, sans-serif'
            )

# ============================================================================
# HOVER TEMPLATE STYLES
# ============================================================================

def format_hover_template(x_label, y_label, x_format='.3f', y_format='.4e'):
    """
    Create formatted hover template.
    
    Args:
        x_label: Label for x-axis value
        y_label: Label for y-axis value
        x_format: Format string for x value
        y_format: Format string for y value
    
    Returns:
        str: Formatted hover template
    """
    return f'<b>{x_label}</b> = %{{x:{x_format}}}<br><b>{y_label}</b> = %{{y:{y_format}}}<extra></extra>'

# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def save_figure(fig, filename, include_plotlyjs='cdn', auto_open=False):
    """
    Save figure to HTML with standard configuration.
    
    Args:
        fig: Plotly figure object
        filename: Output filename
        include_plotlyjs: How to include plotly.js ('cdn', True, False)
        auto_open: Whether to open in browser automatically
    """
    fig.write_html(filename, include_plotlyjs=include_plotlyjs, auto_open=auto_open)
    print(f"[SUCCESS] Generated: {filename}")

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class DashboardPresets:
    """Preset configurations for common dashboard types."""
    
    @staticmethod
    def mechanical_response():
        """Configuration for mechanical response dashboards."""
        return {
            'colors': {
                'stress': UCBerkeleyColors.BERKELEY_BLUE,
                'strain': UCBerkeleyColors.CALIFORNIA_GOLD,
                'modulus': UCBerkeleyColors.FOUNDERS_ROCK,
                'hysteresis': UCBerkeleyColors.MEDALIST
            },
            'layout': get_plot_layout_style(height=900, width=1500,
                                           margin=dict(t=120, b=120, l=80, r=80))
        }
    
    @staticmethod
    def single_plot():
        """Configuration for single plot dashboards."""
        return {
            'colors': {
                'primary': UCBerkeleyColors.BERKELEY_BLUE,
                'secondary': UCBerkeleyColors.FOUNDERS_ROCK,
                'accent': UCBerkeleyColors.CALIFORNIA_GOLD
            },
            'layout': get_plot_layout_style(height=600, width=1200,
                                           margin=dict(t=80, b=100, l=80, r=60))
        }

