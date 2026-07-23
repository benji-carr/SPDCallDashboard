from functools import lru_cache

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from spd_config import TIME_COLUMN
from spd_dashboard_data import load_dashboard_context
from spd_dashboard_figures import (
    make_daily_figure,
    make_map_figure,
    make_volume_response_scatter,
)
from spd_event_bins import (
    decode_bin_combo,
    make_bin_dropdown_options,
)


def get_default_map_date_range(context: dict) -> tuple[str, str]:
    valid_time = context["valid_time"].copy()

    valid_time[TIME_COLUMN] = pd.to_datetime(
        valid_time[TIME_COLUMN],
        errors="coerce",
    )

    valid_dates = valid_time[TIME_COLUMN].dropna().dt.normalize()

    if valid_dates.empty:
        raise ValueError("No valid dates available for default map date range")

    latest_available_day = valid_dates.max()
    default_start = latest_available_day - pd.Timedelta(days=29)

    return (
        default_start.strftime("%Y-%m-%d"),
        latest_available_day.strftime("%Y-%m-%d"),
    )


def clean_date_string(value) -> str | None:
    if value is None:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def extract_daily_visible_date_range(
    relayout_data,
    context: dict,
) -> tuple[str, str]:
    default_start, default_end = get_default_map_date_range(context)

    if not relayout_data:
        return default_start, default_end

    if relayout_data.get("xaxis.autorange") is True:
        return default_start, default_end

    start_value = None
    end_value = None

    if (
        "xaxis.range[0]" in relayout_data
        and "xaxis.range[1]" in relayout_data
    ):
        start_value = relayout_data["xaxis.range[0]"]
        end_value = relayout_data["xaxis.range[1]"]

    elif (
        "xaxis.range" in relayout_data
        and isinstance(relayout_data["xaxis.range"], list)
        and len(relayout_data["xaxis.range"]) >= 2
    ):
        start_value = relayout_data["xaxis.range"][0]
        end_value = relayout_data["xaxis.range"][1]

    start_date = clean_date_string(start_value)
    end_date = clean_date_string(end_value)

    if start_date is None or end_date is None:
        return default_start, default_end

    return start_date, end_date


def get_range_from_store(
    range_store_data,
    context: dict,
) -> tuple[str, str]:
    default_start, default_end = get_default_map_date_range(context)

    if not range_store_data:
        return default_start, default_end

    start_date = clean_date_string(range_store_data.get("start"))
    end_date = clean_date_string(range_store_data.get("end"))

    if start_date is None or end_date is None:
        return default_start, default_end

    return start_date, end_date


def count_map_points(fig) -> int:
    point_count = 0

    for trace in fig.data:
        trace_type = str(getattr(trace, "type", "")).lower()

        if trace_type in ["scattermapbox", "scattermap"]:
            lat_values = getattr(trace, "lat", None)

            if lat_values is not None:
                point_count += len(lat_values)

    return point_count


PANEL_STYLE = {
    "height": "100%",
    "width": "100%",
    "minHeight": "0",
    "minWidth": "0",
    "border": "1px solid #333333",
    "borderRadius": "8px",
    "overflow": "hidden",
    "backgroundColor": "#111111",
    "boxSizing": "border-box",
}

GRAPH_STYLE = {
    "height": "100%",
    "width": "100%",
}

LOADING_STYLE = {
    "height": "100%",
    "width": "100%",
}


def create_app() -> Dash:
    context = load_dashboard_context()

    bin_options = make_bin_dropdown_options()
    default_bin_value = bin_options[0]["value"]

    default_start, default_end = get_default_map_date_range(context)

    app = Dash(__name__)

    @lru_cache(maxsize=64)
    def cached_daily_figure(
        selected_bin_value: str,
        show_legend: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_daily_figure(
            context=context,
            selected_bins=selected_bins,
        )

        fig.update_layout(
            showlegend=show_legend,
            autosize=True,
            uirevision="preserve-daily-time-range",
        )

        return fig

    @lru_cache(maxsize=64)
    def cached_scatter_figure(
        selected_bin_value: str,
        show_legend: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_volume_response_scatter(
            context=context,
            selected_bins=selected_bins,
        )

        fig.update_layout(
            showlegend=show_legend,
            autosize=True,
            uirevision="preserve-scatter-view",
        )

        return fig

    @lru_cache(maxsize=128)
    def cached_map_figure(
        selected_bin_value: str,
        point_start_date: str,
        point_end_date: str,
        show_colorbar: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_map_figure(
            context=context,
            selected_bins=selected_bins,
            point_start_date=point_start_date,
            point_end_date=point_end_date,
            show_colorbar=show_colorbar,
        )

        fig.update_layout(
            autosize=True,
            uirevision="preserve-map-camera",
        )

        visible_point_count = count_map_points(fig)

        return fig, visible_point_count

    app.layout = html.Div(
        children=[
            dcc.Store(
                id="daily-visible-range-store",
                data={
                    "start": default_start,
                    "end": default_end,
                },
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H1(
                                "Seattle SPD Call Dashboard",
                                style={
                                    "margin": "0",
                                    "fontSize": "19px",
                                    "lineHeight": "21px",
                                    "color": "white",
                                },
                            ),
                            html.P(
                                (
                                    "Neighborhood call volume, daily trends, "
                                    "and response-time context by type of crime."
                                ),
                                style={
                                    "margin": "2px 0 0 0",
                                    "color": "#bbbbbb",
                                    "fontSize": "11px",
                                    "lineHeight": "13px",
                                },
                            ),
                        ],
                        className="title-block",
                        style={
                            "minWidth": "0",
                        },
                    ),

                    html.Div(
                        children=[
                            html.Label(
                                "Type of Crime",
                                style={
                                    "fontSize": "12px",
                                    "color": "#dddddd",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            dcc.Dropdown(
                                id="importance-bin-filter",
                                className="type-dropdown",
                                options=bin_options,
                                value=default_bin_value,
                                clearable=False,
                                style={
                                    "width": "320px",
                                    "color": "#111111",
                                    "fontSize": "13px",
                                },
                            ),
                        ],

                        className="type-control",

                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "gap": "10px",
                            "minWidth": "0",
                        },
                    ),

                    html.Div(
                        id="map-point-window-label",
                        children=f"Map points: {default_start} to {default_end}",
                        style={
                            "color": "#bbbbbb",
                            "fontSize": "11px",
                            "textAlign": "right",
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "minWidth": "0",
                        },
                    ),

                    html.Div(
                        "Mobile view shows the interactive map only.",
                        className="mobile-map-note",
                    ),
                ],
                className="top-bar",
                style={
                    "height": "52px",
                    "display": "grid",
                    "gridTemplateColumns": "minmax(250px, 1fr) minmax(330px, 420px) minmax(250px, 0.9fr)",
                    "alignItems": "center",
                    "gap": "12px",
                    "padding": "6px 10px",
                    "backgroundColor": "#151515",
                    "borderBottom": "1px solid #333333",
                    "boxSizing": "border-box",
                    "minWidth": "0",
                },
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="map-figure",
                                        className="map-graph",
                                        config={"responsive": True},
                                        style=GRAPH_STYLE,
                                    )
                                ],
                                type="default",
                                style=LOADING_STYLE,
                                parent_style=LOADING_STYLE,
                            ),
                        ],
                        className="map-panel dashboard-panel",
                        style={
                            **PANEL_STYLE,
                            "gridColumn": "1",
                            "gridRow": "1 / 3",
                        },
                    ),

                    html.Div(
                        children=[
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="daily-figure",
                                        config={"responsive": True},
                                        style=GRAPH_STYLE,
                                    )
                                ],
                                type="default",
                                style=LOADING_STYLE,
                                parent_style=LOADING_STYLE,
                            ),
                        ],
                        className="daily-panel dashboard-panel",
                        style={
                            **PANEL_STYLE,
                            "gridColumn": "2",
                            "gridRow": "1",
                        },
                    ),

                    html.Div(
                        children=[
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="scatter-figure",
                                        config={"responsive": True},
                                        style=GRAPH_STYLE,
                                    )
                                ],
                                type="default",
                                style=LOADING_STYLE,
                                parent_style=LOADING_STYLE,
                            ),
                        ],
                        className="scatter-panel dashboard-panel",
                        style={
                            **PANEL_STYLE,
                            "gridColumn": "2",
                            "gridRow": "2",
                        },
                    ),

                    html.Details(
                        children=[
                            html.Summary("Controls"),
                            html.Div(
                                children=[
                                    html.P(
                                        "Map point legend is always visible.",
                                        style={
                                            "margin": "0 0 8px 0",
                                            "fontSize": "11px",
                                            "lineHeight": "14px",
                                            "color": "#bbbbbb",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="legend-toggle",
                                        options=[
                                            {
                                                "label": " Map color scale",
                                                "value": "map_colorbar",
                                            },
                                            {
                                                "label": " Daily legend",
                                                "value": "daily",
                                            },
                                            {
                                                "label": " Scatter legend",
                                                "value": "scatter",
                                            },
                                        ],
                                        value=[],
                                        className="control-sidebar",
                                        style={
                                            "fontSize": "12px",
                                            "lineHeight": "1.8",
                                        },
                                    ),
                                ],
                                className="control-sidebar",
                            ),
                        ],
                        className="floating-control-panel",
                    ),
                ],
                className="dashboard-grid",
                style={
                    "position": "relative",
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1.2fr) minmax(0, 1fr)",
                    "gridTemplateRows": "minmax(0, 1fr) minmax(0, 1fr)",
                    "gap": "8px",
                    "height": "calc(100dvh - 52px)",
                    "width": "100%",
                    "padding": "8px",
                    "backgroundColor": "#111111",
                    "boxSizing": "border-box",
                    "minHeight": "0",
                    "minWidth": "0",
                    "overflow": "hidden",
                },
            ),
        ],

        className="app-shell",

        style={
            "height": "100dvh",
            "width": "100%",
            "backgroundColor": "#111111",
            "fontFamily": "Arial, sans-serif",
            "overflow": "hidden",
            "margin": "0",
            "padding": "0",
        },
    )

    @app.callback(
        Output("daily-visible-range-store", "data"),
        Input("daily-figure", "relayoutData"),
        State("daily-visible-range-store", "data"),
        prevent_initial_call=True,
    )
    def update_daily_visible_range_store(
        daily_relayout_data,
        current_range_data,
    ):
        if not daily_relayout_data:
            raise PreventUpdate

        start_date, end_date = extract_daily_visible_date_range(
            relayout_data=daily_relayout_data,
            context=context,
        )

        current_start, current_end = get_range_from_store(
            range_store_data=current_range_data,
            context=context,
        )

        if start_date == current_start and end_date == current_end:
            raise PreventUpdate

        return {
            "start": start_date,
            "end": end_date,
        }

    @app.callback(
        Output("daily-figure", "figure"),
        Input("importance-bin-filter", "value"),
        Input("legend-toggle", "value"),
    )
    def update_daily_figure(
        selected_bin_value,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        show_legend = "daily" in legend_values

        return cached_daily_figure(
            selected_bin_value=selected_bin_value,
            show_legend=show_legend,
        )

    @app.callback(
        Output("scatter-figure", "figure"),
        Input("importance-bin-filter", "value"),
        Input("legend-toggle", "value"),
    )
    def update_scatter_figure(
        selected_bin_value,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        show_legend = "scatter" in legend_values

        return cached_scatter_figure(
            selected_bin_value=selected_bin_value,
            show_legend=show_legend,
        )

    @app.callback(
        Output("map-figure", "figure"),
        Output("map-point-window-label", "children"),
        Input("importance-bin-filter", "value"),
        Input("daily-visible-range-store", "data"),
        Input("legend-toggle", "value"),
    )
    def update_map_figure(
        selected_bin_value,
        range_store_data,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        point_start_date, point_end_date = get_range_from_store(
            range_store_data=range_store_data,
            context=context,
        )

        show_colorbar = "map_colorbar" in legend_values

        fig, visible_point_count = cached_map_figure(
            selected_bin_value=selected_bin_value,
            point_start_date=point_start_date,
            point_end_date=point_end_date,
            show_colorbar=show_colorbar,
        )

        label = (
            f"Map points: {point_start_date} to {point_end_date}"
            f" | visible points: {visible_point_count:,}"
        )

        return fig, label

    return app


dashboard = create_app()
server = dashboard.server


if __name__ == "__main__":
    dashboard.run(debug=True)