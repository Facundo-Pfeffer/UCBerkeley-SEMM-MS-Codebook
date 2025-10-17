import plotly.graph_objects as go

def _add_element_trace(fig, elem):
    """Add the element line trace (with markers at nodes)."""
    x_coords = [elem.initial_node.x, elem.final_node.x]
    y_coords = [elem.initial_node.y, elem.final_node.y]

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines+markers",
        line=dict(color="black"),
        marker=dict(color="black", size=8),
        showlegend=False,
        hoverinfo="skip"
    ))

def _add_element_label(fig, elem):
    mid_x = (elem.initial_node.x + elem.final_node.x) / 2
    mid_y = (elem.initial_node.y + elem.final_node.y) / 2

    fig.add_trace(go.Scatter(
        x=[mid_x],
        y=[mid_y],
        mode="markers+text",
        marker=dict(size=12, color="rgba(0,0,0,0)"),  # invisible marker
        text=[f"E{elem.id}"],
        textposition="top center",
        textfont=dict(color="blue"),
        hovertext=f"Element {elem.id}: {elem.initial_node.id}-{elem.final_node.id}",
        hoverinfo="text",
        showlegend=False
    ))

def _add_node_annotation(fig, node, xshift=10, yshift=10):
    fig.add_annotation(
        x=node.x,
        y=node.y,
        text=f"N{node.id}",
        showarrow=False,
        font=dict(color="red"),
        xshift=xshift,
        yshift=yshift
    )

def _add_free_dof(fig, dof, scale=1.0):
    x, y = dof.node.x, dof.node.y
    if dof.direction == 1:   # x-direction
        dx, dy = scale, 0
    elif dof.direction == 2: # y-direction
        dx, dy = 0, scale
    else:
        dx, dy = 0, 0

    fig.add_annotation(
        x=x+dx, y=y+dy, ax=x, ay=y,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red"
    )

def plot_elements_plotly(nodes, elements, free_dofs):
    fig = go.Figure()

    # add elements (line + label)
    for elem in elements:
        _add_element_trace(fig, elem)
        _add_element_label(fig, elem)

    # add node annotations
    for node in nodes:
        _add_node_annotation(fig, node)

    # add free DOF arrows
    if free_dofs:
        for dof in free_dofs:
            _add_free_dof(fig, dof, scale=1.0)  # scale controls arrow length

    # layout
    fig.update_layout(
        title="Truss / Frame Connectivity",
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y"),
        plot_bgcolor="white",
        showlegend=False
    )

    fig.show()
