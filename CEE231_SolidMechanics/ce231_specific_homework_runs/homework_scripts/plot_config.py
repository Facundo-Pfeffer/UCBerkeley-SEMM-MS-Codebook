"""
Shared plotting configuration for CEE231 homework dashboards.
Contains color schemes, axis styling, and common layout parameters.
"""

# Professional color scheme (UC Berkeley palette)
COLORS = {
    'stress': '#003262',      # Berkeley Blue
    'modulus': '#3B7EA1',     # Founder's Rock
    'strain': '#FDB515',      # California Gold
    'hysteresis': '#C4820E',  # California Gold (darker)
    'annotation': '#003262',  # Berkeley Blue
    'grid': 'rgba(200, 200, 200, 0.3)',
    'zeroline': 'rgba(0, 0, 0, 0.3)',
    'background_plot': 'rgba(250, 250, 250, 0.95)',
    'background_paper': 'white'
}

# Professional axis styling
AXIS_STYLE = {
    'showgrid': True,
    'gridcolor': COLORS['grid'],
    'gridwidth': 1,
    'zeroline': True,
    'zerolinecolor': COLORS['zeroline'],
    'zerolinewidth': 1.5,
    'showline': True,
    'linewidth': 1.5,
    'linecolor': 'black',
    'mirror': True,
    'ticks': 'outside',
    'tickwidth': 1.5,
    'tickcolor': 'black',
    'tickfont': {'size': 12, 'family': 'Arial, sans-serif'}
}

# Table styling
TABLE_HEADER_STYLE = {
    'align': 'center',
    'fill_color': COLORS['stress'],
    'font': {'color': 'white', 'size': 12, 'family': 'Arial, sans-serif'},
    'height': 32
}

TABLE_CELL_STYLE = {
    'align': ['left', 'left'],
    'fill_color': COLORS['background_plot'],
    'font': {'size': 11, 'family': 'Arial, sans-serif'}
}

# Layout defaults
LAYOUT_DEFAULTS = {
    'title_font': {'size': 16, 'family': 'Arial, sans-serif', 'color': COLORS['stress']},
    'plot_bgcolor': COLORS['background_plot'],
    'paper_bgcolor': COLORS['background_paper'],
    'font': {'family': 'Arial, sans-serif', 'size': 12}
}

# Slider styling
SLIDER_STYLE = {
    'bgcolor': 'rgba(0, 50, 98, 0.08)',
    'bordercolor': COLORS['stress'],
    'borderwidth': 1,
    'tickcolor': COLORS['stress'],
    'font': {'size': 9, 'family': 'Arial, sans-serif'}
}

# Annotation fonts
ANNOTATION_FONTS = {
    'title': {'size': 14, 'family': 'Arial, sans-serif', 'color': COLORS['stress']},
    'text': {'size': 11, 'family': 'Arial, sans-serif', 'color': '#444'}
}

