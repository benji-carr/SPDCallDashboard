import plotly.graph_objects as go


I5_ROUTE = {
    "name": "I-5",
    "lat": [
        47.497,
        47.520,
        47.548,
        47.571,
        47.596,
        47.620,
        47.645,
        47.668,
        47.693,
        47.720,
        47.742,
    ],
    "lon": [
        -122.281,
        -122.301,
        -122.321,
        -122.321,
        -122.326,
        -122.323,
        -122.322,
        -122.322,
        -122.326,
        -122.329,
        -122.329,
    ],
    "line_width": 3,
    "line_color": "rgba(255, 255, 255, 0.75)",
}

SR99_ROUTE = {
    "name": "SR-99 / Aurora Ave",
    "lat": [
        47.515,
        47.545,
        47.575,
        47.600,
        47.620,
        47.642,
        47.662,
        47.684,
        47.706,
        47.729,
    ],
    "lon": [
        -122.335,
        -122.337,
        -122.339,
        -122.344,
        -122.348,
        -122.347,
        -122.347,
        -122.344,
        -122.344,
        -122.344,
    ],
    "line_width": 3,
    "line_color": "rgba(255, 220, 90, 0.82)",
}


ROAD_OVERLAYS = {
    "i5": I5_ROUTE,
    "sr99": SR99_ROUTE,
}


def add_road_overlays(
    fig: go.Figure,
    selected_overlays: list[str] | None = None,
) -> go.Figure:
    if selected_overlays is None:
        selected_overlays = ["i5", "sr99"]

    for overlay_id in selected_overlays:
        route = ROAD_OVERLAYS.get(overlay_id)

        if route is None:
            continue

        fig.add_trace(
            go.Scattermapbox(
                lat=route["lat"],
                lon=route["lon"],
                mode="lines",
                name=route["name"],
                legendgroup="road-overlays",
                showlegend=True,
                line={
                    "width": route["line_width"],
                    "color": route["line_color"],
                },
                hovertemplate=(
                    f"<b>{route['name']}</b><br>"
                    "Reference road overlay"
                    "<extra></extra>"
                ),
            )
        )

    return fig