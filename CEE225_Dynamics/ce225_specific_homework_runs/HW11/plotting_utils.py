"""
Plotting Utilities Module
==========================
Common utilities for plotting (colors, axis styles, etc.)
"""


class Colors:
    """Plot color palette (shared across all plots)."""
    # Darker palette order: blue, crimson/red, green, black
    BERKELEY_BLUE = '#003262'     # Dark blue
    CALIFORNIA_GOLD = '#8B0000'   # Dark red/crimson
    FOUNDERS_ROCK = '#2E7D32'     # Slightly lighter dark green for contrast
    MEDALIST = '#000000'          # Black (spare/general use)
    ORANGE = '#E67E22'            # Orange (for extended palettes)
    PURPLE = '#6A1B9A'            # Purple (for extended palettes)
    TEXT_DARK = '#2C3E50'
    TEXT_LIGHT = '#7F8C8D'
    GRID = 'rgba(200, 200, 200, 0.3)'
    BG_LIGHT = 'rgba(250, 250, 250, 0.98)'
    BG_WHITE = 'white'


def get_axis_style():
    """Return standard axis styling configuration."""
    return dict(
        showgrid=True,
        gridcolor=Colors.GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(0, 0, 0, 0.3)',
        zerolinewidth=2,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=1.5,
        tickcolor='black',
        tickfont=dict(size=12, family='Arial, sans-serif')
    )





