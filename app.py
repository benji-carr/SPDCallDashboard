from functools import lru_cache
import itertools

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.spd_config import (
    TIME_COLUMN as CALL_TIME_COLUMN,
)

from dashboard.spd_event_bins import (
    decode_bin_combo,
    make_bin_dropdown_options,
)

from dashboard.spd_dashboard_data import (
    load_dashboard_context as load_calls_dashboard_context,
)

from dashboard.spd_dashboard_figures import (
    make_daily_figure as make_calls_daily_figure,
    make_map_figure as make_calls_map_figure,
    make_volume_response_scatter as make_calls_scatter_figure,
)

from dashboard.crime_dashboard_data import (
    TIME_COLUMN as CRIME_TIME_COLUMN,
    load_crime_dashboard_context,
)

from dashboard.crime_dashboard_figures import (
    TARGET_CRIME_CATEGORIES,
    make_daily_figure as make_crime_daily_figure,
    make_map_figure as make_crime_map_figure,
)


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


def encode_combo(combo: list[str]) -> str:
    return "||".join(combo)


def decode_combo(
    value: str | None,
    default_values: list[str],
) -> list[str]:
    if value is None:
        return default_values

    return value.split("||")


def make_combo_label(
    combo: list[str],
    all_values: list[str],
) -> str:
    if len(combo) == len(all_values):
        return "All selected categories"

    return " + ".join(combo)


def make_combo_options(
    values: list[str],
) -> list[dict[str, str]]:
    combos = []

    for r in range(1, len(values) + 1):
        for combo in itertools.combinations(values, r):
            combos.append(list(combo))

    ordered_combos = [
        values,
        *[
            combo
            for combo in combos
            if combo != values
        ],
    ]

    return [
        {
            "label": make_combo_label(combo, values),
            "value": encode_combo(combo),
        }
        for combo in ordered_combos
    ]


def make_page_nav(active_page: str) -> html.Div:
    return html.Div(
        children=[
            dcc.Link(
                "Home",
                href="/",
                className="page-nav-link",
            ),
            dcc.Link(
                "Crime Dashboard",
                href="/crime",
                className=(
                    "page-nav-link active"
                    if active_page == "crime"
                    else "page-nav-link"
                ),
            ),
            dcc.Link(
                "Calls Dashboard",
                href="/calls",
                className=(
                    "page-nav-link active"
                    if active_page == "calls"
                    else "page-nav-link"
                ),
            ),
        ],
        className="page-nav",
    )


def get_default_map_date_range(
    context: dict,
    time_column: str,
) -> tuple[str, str]:
    valid_time = context["valid_time"].copy()

    valid_time[time_column] = pd.to_datetime(
        valid_time[time_column],
        errors="coerce",
    )

    latest_day = valid_time[time_column].dropna().max().normalize()
    start_day = latest_day 

    return start_day.date().isoformat(), latest_day.date().isoformat()


def get_full_dashboard_date_range(
    context: dict,
    time_column: str,
) -> tuple[str, str]:
    valid_time = context["valid_time"].copy()

    valid_dates = pd.to_datetime(
        valid_time[time_column],
        errors="coerce",
    ).dropna()

    latest_day = valid_dates.max().normalize()
    earliest_available_day = valid_dates.min().normalize()
    earliest_analysis_day = earliest_available_day + pd.Timedelta(days=1)

    return (
        earliest_analysis_day.date().isoformat(),
        latest_day.date().isoformat(),
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
    default_start: str,
    default_end: str,
    full_start: str,
    full_end: str,
) -> tuple[str, str]:
    if not relayout_data:
        return default_start, default_end

    if relayout_data.get("xaxis.autorange") is True:
        return full_start, full_end

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
    default_start: str,
    default_end: str,
) -> tuple[str, str]:
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

def make_landing_page():
    return html.Div(
        className="site-shell landing-site-shell",
        children=[
            html.Header(
                className="landing-topbar",
                children=[
                    html.Div(
                        className="landing-topbar-inner",
                        children=[
                            html.A(
                                href="https://data.seattle.gov",
                                target="_blank",
                                rel="noopener noreferrer",
                                className="landing-brand-link",
                                title="Visit the City of Seattle Open Data Portal",
                                children=[
                                    html.Img(
                                        src="/assets/seattle-logo.png",
                                        className="landing-brand-logo",
                                        alt="Seattle logo",
                                    ),
                                    html.Span(
                                        "Seattle",
                                        className="landing-brand-text",
                                    ),
                                ],
                            ),
                            html.Nav(
                                className="landing-external-nav",
                                children=[
                                    html.A(
                                        "Open Data Program",
                                        href=(
                                            "https://www.seattle.gov/tech/"
                                            "reports-and-data/open-data"
                                        ),
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        className="landing-topbar-link",
                                    ),
                                    html.Span(
                                        className="landing-topbar-divider",
                                    ),
                                    html.A(
                                        href=(
                                            "https://www.linkedin.com/in/"
                                            "benji-carr-1a9b8c4/"
                                        ),
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        className="landing-linkedin-link",
                                        title="Visit Ben Carr on LinkedIn",
                                        children=[
                                            html.Img(
                                                src="/assets/linkedin-logo.png",
                                                className="landing-linkedin-logo",
                                                alt="LinkedIn",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            html.Main(
                className="landing-hero",
                children=[
                    html.Div(
                        className="landing-hero-content",
                        children=[
                            html.Section(
                                className="landing-intro",
                                children=[
                                    html.H1(
                                        "Seattle Public Safety Dashboards",
                                        className="landing-title",
                                    ),
                                    html.P(
                                        className="landing-subtitle",
                                        children=[
                                            (
                                                "Welcome to our independently "
                                                "developed Seattle public safety "
                                                "dashboard. Here you can explore "
                                                "Seattle reported crime data and "
                                                "police calls for service through "
                                                "interactive maps and time-series "
                                                "views. All data is from the "
                                            ),
                                            html.A(
                                                "City of Seattle Open Data Portal",
                                                href="https://data.seattle.gov",
                                                target="_blank",
                                                rel="noopener noreferrer",
                                                className="landing-inline-link",
                                            ),
                                            ".",
                                        ],
                                    ),
                                ],
                            ),
                            html.Section(
                                className="landing-card-grid",
                                children=[
                                    dcc.Link(
                                        href="/crime",
                                        className="landing-card-wrapper",
                                        children=[
                                            html.Div(
                                                className="landing-card",
                                                children=[
                                                    html.Div(
                                                        className=(
                                                            "landing-card-icon-frame"
                                                        ),
                                                        children=[
                                                            html.Img(
                                                                src="/assets/crime-dashboard-logo.png",
                                                                alt=(
                                                                    "Crime Dashboard"
                                                                ),
                                                                className=(
                                                                    "landing-card-"
                                                                    "icon-image"
                                                                ),
                                                            )
                                                        ],
                                                    ),
                                                    html.H2(
                                                        "Crime Dashboard",
                                                        className=(
                                                            "landing-card-title"
                                                        ),
                                                    ),
                                                    html.P(
                                                        (
                                                            "Reported crime offenses "
                                                            "by neighborhood, crime "
                                                            "category, and recent "
                                                            "time window."
                                                        ),
                                                        className=(
                                                            "landing-card-text"
                                                        ),
                                                    ),
                                                    html.Span(
                                                        "Open crime dashboard →",
                                                        className=(
                                                            "landing-card-link"
                                                        ),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dcc.Link(
                                        href="/calls",
                                        className="landing-card-wrapper",
                                        children=[
                                            html.Div(
                                                className="landing-card",
                                                children=[
                                                    html.Div(
                                                        className=(
                                                            "landing-card-icon-frame"
                                                        ),
                                                        children=[
                                                            html.Img(
                                                                src="/assets/call-dashboard-icon.png",
                                                                alt=(
                                                                    "Calls Dashboard"
                                                                ),
                                                                className=(
                                                                    "landing-card-"
                                                                    "icon-image"
                                                                ),
                                                            )
                                                        ],
                                                    ),
                                                    html.H2(
                                                        "Calls Dashboard",
                                                        className=(
                                                            "landing-card-title"
                                                        ),
                                                    ),
                                                    html.P(
                                                        (
                                                            "SPD calls for service by "
                                                            "event type, neighborhood, "
                                                            "daily volume, and response "
                                                            "patterns."
                                                        ),
                                                        className=(
                                                            "landing-card-text"
                                                        ),
                                                    ),
                                                    html.Span(
                                                        "Open calls dashboard →",
                                                        className=(
                                                            "landing-card-link"
                                                        ),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    html.A(
                                        href=(
                                            "https://www.linkedin.com/in/"
                                            "benji-carr-1a9b8c4/"
                                        ),
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        className="landing-card-wrapper",
                                        children=[
                                            html.Div(
                                                className="landing-card",
                                                children=[
                                                    html.Div(
                                                        className=(
                                                            "landing-card-icon-frame"
                                                        ),
                                                        children=[
                                                            html.Img(
                                                                src="/assets/contact-developers-icon.png",
                                                                alt=(
                                                                    "Contact the "
                                                                    "Developers"
                                                                ),
                                                                className=(
                                                                    "landing-card-"
                                                                    "icon-image"
                                                                ),
                                                            )
                                                        ],
                                                    ),
                                                    html.H2(
                                                        "Contact the Developers",
                                                        className=(
                                                            "landing-card-title"
                                                        ),
                                                    ),
                                                    html.P(
                                                        (
                                                            "Share feedback, ask "
                                                            "questions, or get in "
                                                            "touch about the "
                                                            "development of this "
                                                            "project."
                                                        ),
                                                        className=(
                                                            "landing-card-text"
                                                        ),
                                                    ),
                                                    html.Span(
                                                        "Contact us on LinkedIn →",
                                                        className=(
                                                            "landing-card-link"
                                                        ),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            ),
                            html.P(
                                (
                                    "This is an independently developed project "
                                    "and is not affiliated with or endorsed by the "
                                    "City of Seattle or the Seattle Police Department."
                                ),
                                className="landing-disclaimer",
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

def create_app() -> Dash:
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
    )

    calls_context = load_calls_dashboard_context()
    crime_context = load_crime_dashboard_context()

    def make_options_from_series(series: pd.Series) -> list[dict[str, str]]:
        values = (
            series
            .dropna()
            .astype("string")
            .str.strip()
            .sort_values()
            .unique()
            .tolist()
        )

        return [
            {
                "label": value.title(),
                "value": value,
            }
            for value in values
            if value not in ["", "nan", "none"]
        ]


    crime_subcategory_options = make_options_from_series(
        crime_context["event_mcpp"]["offense_sub_category"]
    )

    crime_neighborhood_options = make_options_from_series(
        crime_context["event_mcpp"]["mcpp_neighborhood"]
    )

    call_bin_options = make_bin_dropdown_options()
    default_call_bin_value = call_bin_options[0]["value"]

    crime_category_options = make_combo_options(TARGET_CRIME_CATEGORIES)
    default_crime_category_value = encode_combo(TARGET_CRIME_CATEGORIES)

    default_call_start, default_call_end = get_default_map_date_range(
        calls_context,
        CALL_TIME_COLUMN,
    )

    full_call_start, full_call_end = get_full_dashboard_date_range(
        calls_context,
        CALL_TIME_COLUMN,
    )

    default_crime_start, default_crime_end = get_default_map_date_range(
        crime_context,
        CRIME_TIME_COLUMN,
    )

    full_crime_start, full_crime_end = get_full_dashboard_date_range(
        crime_context,
        CRIME_TIME_COLUMN,
    )

    @lru_cache(maxsize=64)
    def cached_calls_daily_figure(
        selected_bin_value: str,
        show_legend: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_calls_daily_figure(
            context=calls_context,
            selected_bins=selected_bins,
        )

        fig.update_layout(
            showlegend=show_legend,
            autosize=True,
            uirevision="preserve-calls-daily-time-range",
        )

        return fig

    @lru_cache(maxsize=64)
    def cached_calls_scatter_figure(
        selected_bin_value: str,
        show_legend: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_calls_scatter_figure(
            context=calls_context,
            selected_bins=selected_bins,
        )

        fig.update_layout(
            showlegend=show_legend,
            autosize=True,
            uirevision="preserve-calls-scatter-view",
        )

        return fig

    @lru_cache(maxsize=128)
    def cached_calls_map_figure(
        selected_bin_value: str,
        point_start_date: str,
        point_end_date: str,
        show_colorbar: bool,
    ):
        selected_bins = decode_bin_combo(selected_bin_value)

        fig = make_calls_map_figure(
            context=calls_context,
            selected_bins=selected_bins,
            point_start_date=point_start_date,
            point_end_date=point_end_date,
            show_colorbar=show_colorbar,
        )

        fig.update_layout(
            autosize=True,
            # Keep the user's zoom and pan position when the map updates.
            uirevision="preserve-calls-map-camera",
            # Keep legend selections during ordinary map updates,
            # but reset them when the top-level crime-type dropdown changes.
            legend_uirevision=f"calls-map-legend-{selected_bin_value}",
        )

        visible_point_count = count_map_points(fig)

        return fig, visible_point_count

    @lru_cache(maxsize=64)
    def cached_crime_daily_figure(
        selected_category_value: str,
        show_legend: bool,
    ):
        selected_categories = decode_combo(
            selected_category_value,
            TARGET_CRIME_CATEGORIES,
        )

        fig = make_crime_daily_figure(
            context=crime_context,
            selected_bins=selected_categories,
        )

        fig.update_layout(
            showlegend=show_legend,
            autosize=True,
        )

        return fig

    def build_crime_map_figure(
        selected_category_value: str,
        point_start_date: str,
        point_end_date: str,
        show_colorbar: bool,
        point_filters: dict | None = None,
    ):
        selected_categories = decode_combo(
            selected_category_value,
            TARGET_CRIME_CATEGORIES,
        )

        fig = make_crime_map_figure(
            context=crime_context,
            selected_bins=selected_categories,
            point_start_date=point_start_date,
            point_end_date=point_end_date,
            show_colorbar=show_colorbar,
            point_filters=point_filters,
        )

        fig.update_layout(
            autosize=True,
        )

        visible_point_count = count_map_points(fig)

        return fig, visible_point_count

    def make_calls_page() -> html.Div:
        return html.Div(
            children=[
                make_page_nav("calls"),

                html.Div(
                    children=[
                        dcc.Store(
                            id="daily-visible-range-store",
                            data={
                                "start": default_call_start,
                                "end": default_call_end,
                            },
                        ),

                        dcc.Store(
                            id="fullscreen-figure-store",
                            data=None,
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
                                            options=call_bin_options,
                                            value=default_call_bin_value,
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
                                    children=(
                                        f"Map points: {default_call_start} "
                                        f"to {default_call_end}"
                                    ),
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
                                "gridTemplateColumns": (
                                    "minmax(250px, 1fr) "
                                    "minmax(330px, 420px) "
                                    "minmax(250px, 0.9fr)"
                                ),
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
                                        html.Button(
                                            "↗",
                                            id="expand-map-button",
                                            className="expand-button",
                                            title="Expand map",
                                        ),

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
                                        html.Button(
                                            "↗",
                                            id="expand-daily-button",
                                            className="expand-button",
                                            title="Expand daily chart",
                                        ),

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
                                        html.Button(
                                            "↗",
                                            id="expand-scatter-button",
                                            className="expand-button",
                                            title="Expand scatterplot",
                                        ),

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
                                "gridTemplateColumns": (
                                    "minmax(0, 1.2fr) minmax(0, 1fr)"
                                ),
                                "gridTemplateRows": (
                                    "minmax(0, 1fr) minmax(0, 1fr)"
                                ),
                                "gap": "8px",
                                "height": "calc(100dvh - 88px)",
                                "width": "100%",
                                "padding": "8px",
                                "backgroundColor": "#111111",
                                "boxSizing": "border-box",
                                "minHeight": "0",
                                "minWidth": "0",
                                "overflow": "hidden",
                            },
                        ),

                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            id="fullscreen-title",
                                            className="fullscreen-title",
                                        ),
                                        html.Button(
                                            "×",
                                            id="close-fullscreen-button",
                                            className="close-fullscreen-button",
                                            title="Close fullscreen view",
                                        ),
                                    ],
                                    className="fullscreen-header",
                                ),

                                dcc.Graph(
                                    id="fullscreen-figure",
                                    className="fullscreen-graph",
                                    config={"responsive": True},
                                    style={
                                        "height": "100%",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            id="fullscreen-overlay",
                            className="fullscreen-overlay hidden",
                        ),
                    ],
                    className="app-shell",
                    style={
                        "height": "calc(100dvh - 36px)",
                        "width": "100%",
                        "backgroundColor": "#111111",
                        "fontFamily": "Arial, sans-serif",
                        "overflow": "hidden",
                        "margin": "0",
                        "padding": "0",
                    },
                ),
            ],
            className="dashboard-page",
        )

    def make_crime_page() -> html.Div:
        return html.Div(
            children=[
                make_page_nav("crime"),

                html.Div(
                    children=[
                        dcc.Store(
                            id="crime-daily-visible-range-store",
                            data={
                                "start": default_crime_start,
                                "end": default_crime_end,
                            },
                        ),

                        dcc.Store(
                            id="crime-fullscreen-figure-store",
                            data=None,
                        ),

                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.H1(
                                            "Seattle Crime Dashboard",
                                            style={
                                                "margin": "0",
                                                "fontSize": "19px",
                                                "lineHeight": "21px",
                                                "color": "white",
                                            },
                                        ),
                                        html.P(
                                            (
                                                "Reported crime offenses by neighborhood, "
                                                "daily trends, and type of crime."
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
                                            id="crime-category-filter",
                                            className="type-dropdown",
                                            options=crime_category_options,
                                            value=default_crime_category_value,
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
                                    id="crime-map-point-window-label",
                                    children=(
                                        f"Map points: {default_crime_start} "
                                        f"to {default_crime_end}"
                                    ),
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
                                "gridTemplateColumns": (
                                    "minmax(250px, 1fr) "
                                    "minmax(330px, 420px) "
                                    "minmax(250px, 0.9fr)"
                                ),
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
                                        html.Button(
                                            "↗",
                                            id="crime-expand-map-button",
                                            className="expand-button",
                                            title="Expand crime map",
                                        ),

                                        dcc.Loading(
                                            children=[
                                                html.Div(
                                                    id="crime-map-graph-container",
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
                                        "gridRow": "1",
                                    },
                                ),

                                html.Div(
                                    children=[
                                        html.Button(
                                            "↗",
                                            id="crime-expand-daily-button",
                                            className="expand-button",
                                            title="Expand crime daily chart",
                                        ),

                                        dcc.Loading(
                                            children=[
                                                dcc.Graph(
                                                    id="crime-daily-figure",
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
                                                    id="crime-legend-toggle",
                                                    options=[
                                                        {
                                                            "label": " Map color scale",
                                                            "value": "map_colorbar",
                                                        },
                                                        {
                                                            "label": " Daily legend",
                                                            "value": "daily",
                                                        },
                                                    ],
                                                    value=[],
                                                    className="control-sidebar",
                                                    style={
                                                        "fontSize": "12px",
                                                        "lineHeight": "1.8",
                                                        "marginBottom": "10px",
                                                    },
                                                ),

                                                html.Hr(
                                                    style={
                                                        "borderColor": "rgba(255,255,255,0.18)",
                                                        "margin": "8px 0",
                                                    },
                                                ),

                                                html.P(
                                                    "Point filters",
                                                    style={
                                                        "margin": "8px 0 6px 0",
                                                        "fontSize": "12px",
                                                        "fontWeight": "bold",
                                                        "color": "#dddddd",
                                                    },
                                                ),

                                                html.Label(
                                                    "Offense sub-category",
                                                    style={
                                                        "fontSize": "11px",
                                                        "color": "#bbbbbb",
                                                    },
                                                ),

                                                dcc.Dropdown(
                                                    id="crime-point-subcategory-filter",
                                                    options=crime_subcategory_options,
                                                    value=[],
                                                    multi=True,
                                                    placeholder="All sub-categories",
                                                    style={
                                                        "color": "#111111",
                                                        "fontSize": "12px",
                                                        "marginBottom": "8px",
                                                    },
                                                ),

                                                html.Label(
                                                    "Neighborhood",
                                                    style={
                                                        "fontSize": "11px",
                                                        "color": "#bbbbbb",
                                                    },
                                                ),

                                                dcc.Dropdown(
                                                    id="crime-point-neighborhood-filter",
                                                    options=crime_neighborhood_options,
                                                    value=[],
                                                    multi=True,
                                                    placeholder="All neighborhoods",
                                                    style={
                                                        "color": "#111111",
                                                        "fontSize": "12px",
                                                        "marginBottom": "8px",
                                                    },
                                                ),

                                                html.Label(
                                                    "Text search",
                                                    style={
                                                        "fontSize": "11px",
                                                        "color": "#bbbbbb",
                                                    },
                                                ),

                                                dcc.Input(
                                                    id="crime-point-text-filter",
                                                    type="text",
                                                    debounce=True,
                                                    placeholder="Search ID, report #, block...",
                                                    style={
                                                        "width": "100%",
                                                        "fontSize": "12px",
                                                        "padding": "5px",
                                                        "boxSizing": "border-box",
                                                    },
                                                ),
                                            ],
                                            className="control-sidebar",
                                        ),
                                    ],
                                    className="floating-control-panel",
                                    style={
                                        "width": "330px",
                                    },
                                ),
                            ],
                            className="dashboard-grid crime-dashboard-grid",
                            style={
                                "position": "relative",
                                "display": "grid",
                                "gridTemplateColumns": (
                                    "minmax(0, 1.2fr) minmax(0, 1fr)"
                                ),
                                "gridTemplateRows": "minmax(0, 1fr)",
                                "gap": "8px",
                                "height": "calc(100dvh - 88px)",
                                "width": "100%",
                                "padding": "8px",
                                "backgroundColor": "#111111",
                                "boxSizing": "border-box",
                                "minHeight": "0",
                                "minWidth": "0",
                                "overflow": "hidden",
                            },
                        ),

                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            id="crime-fullscreen-title",
                                            className="fullscreen-title",
                                        ),
                                        html.Button(
                                            "×",
                                            id="crime-close-fullscreen-button",
                                            className="close-fullscreen-button",
                                            title="Close fullscreen view",
                                        ),
                                    ],
                                    className="fullscreen-header",
                                ),

                                dcc.Graph(
                                    id="crime-fullscreen-figure",
                                    className="fullscreen-graph",
                                    config={"responsive": True},
                                    style={
                                        "height": "100%",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            id="crime-fullscreen-overlay",
                            className="fullscreen-overlay hidden",
                        ),
                    ],
                    className="app-shell",
                    style={
                        "height": "calc(100dvh - 36px)",
                        "width": "100%",
                        "backgroundColor": "#111111",
                        "fontFamily": "Arial, sans-serif",
                        "overflow": "hidden",
                        "margin": "0",
                        "padding": "0",
                    },
                ),
            ],
            className="dashboard-page",
        )
    
    app.layout = html.Div(
        children=[
            dcc.Location(id="url"),
            html.Div(id="page-content"),
        ],
        className="site-shell",
    )

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def display_page(pathname: str):
        if pathname in [None, "/", ""]:
            return make_landing_page()

        if pathname in ["/crime", "/crime/"]:
            return make_crime_page()

        if pathname in ["/calls", "/calls/"]:
            return make_calls_page()

        return make_landing_page()

    @app.callback(
        Output("daily-visible-range-store", "data"),
        Input("daily-figure", "relayoutData"),
        State("daily-visible-range-store", "data"),
        prevent_initial_call=True,
    )
    def update_calls_daily_visible_range_store(
        daily_relayout_data,
        current_range_data,
    ):
        if not daily_relayout_data:
            raise PreventUpdate

        start_date, end_date = extract_daily_visible_date_range(
            relayout_data=daily_relayout_data,
            default_start=default_call_start,
            default_end=default_call_end,
            full_start=full_call_start,
            full_end=full_call_end,
        )

        current_start, current_end = get_range_from_store(
            range_store_data=current_range_data,
            default_start=default_call_start,
            default_end=default_call_end,
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
    def update_calls_daily_figure(
        selected_bin_value,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        show_legend = "daily" in legend_values

        return cached_calls_daily_figure(
            selected_bin_value=selected_bin_value,
            show_legend=show_legend,
        )

    @app.callback(
        Output("scatter-figure", "figure"),
        Input("importance-bin-filter", "value"),
        Input("legend-toggle", "value"),
    )
    def update_calls_scatter_figure(
        selected_bin_value,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        show_legend = "scatter" in legend_values

        return cached_calls_scatter_figure(
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
    def update_calls_map_figure(
        selected_bin_value,
        range_store_data,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        point_start_date, point_end_date = get_range_from_store(
            range_store_data=range_store_data,
            default_start=default_call_start,
            default_end=default_call_end,
        )

        show_colorbar = "map_colorbar" in legend_values

        fig, visible_point_count = cached_calls_map_figure(
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

    @app.callback(
        Output("crime-daily-visible-range-store", "data"),
        Input("crime-daily-figure", "relayoutData"),
        State("crime-daily-visible-range-store", "data"),
        prevent_initial_call=True,
    )
    def update_crime_daily_visible_range_store(
        daily_relayout_data,
        current_range_data,
    ):
        if not daily_relayout_data:
            raise PreventUpdate

        start_date, end_date = extract_daily_visible_date_range(
            relayout_data=daily_relayout_data,
            default_start=default_crime_start,
            default_end=default_crime_end,
            full_start=full_crime_start,
            full_end=full_crime_end,
        )

        current_start, current_end = get_range_from_store(
            range_store_data=current_range_data,
            default_start=default_crime_start,
            default_end=default_crime_end,
        )

        if start_date == current_start and end_date == current_end:
            raise PreventUpdate

        return {
            "start": start_date,
            "end": end_date,
        }

    @app.callback(
        Output("crime-daily-figure", "figure"),
        Input("crime-category-filter", "value"),
        Input("crime-legend-toggle", "value"),
    )
    def update_crime_daily_figure(
        selected_category_value,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        show_legend = "daily" in legend_values

        return cached_crime_daily_figure(
            selected_category_value=selected_category_value,
            show_legend=show_legend,
        )

    @app.callback(
        Output("crime-map-graph-container", "children"),
        Output("crime-map-point-window-label", "children"),
        Input("crime-category-filter", "value"),
        Input("crime-daily-visible-range-store", "data"),
        Input("crime-legend-toggle", "value"),
        Input("crime-point-subcategory-filter", "value"),
        Input("crime-point-neighborhood-filter", "value"),
        Input("crime-point-text-filter", "value"),
    )
    def update_crime_map_figure(
        selected_category_value,
        range_store_data,
        legend_values,
        selected_subcategories,
        selected_neighborhoods,
        text_filter,
    ):
        if legend_values is None:
            legend_values = []

        point_start_date, point_end_date = get_range_from_store(
            range_store_data=range_store_data,
            default_start=default_crime_start,
            default_end=default_crime_end,
        )

        point_filters = {
            "offense_sub_categories": selected_subcategories or [],
            "mcpp_neighborhoods": selected_neighborhoods or [],
            "text": text_filter or "",
        }

        show_colorbar = "map_colorbar" in legend_values

        fig, visible_point_count = build_crime_map_figure(
            selected_category_value=selected_category_value,
            point_start_date=point_start_date,
            point_end_date=point_end_date,
            show_colorbar=show_colorbar,
            point_filters=point_filters,
        )

        graph_key = (
            f"crime-map|{selected_category_value}|"
            f"{point_start_date}|{point_end_date}|"
            f"{show_colorbar}|"
            f"{','.join(point_filters['offense_sub_categories'])}|"
            f"{','.join(point_filters['mcpp_neighborhoods'])}|"
            f"{point_filters['text']}"
        )
        graph = html.Div(
            children=[
                dcc.Graph(
                    id="crime-map-figure",
                    className="map-graph",
                    figure=fig,
                    config={"responsive": True},
                    style=GRAPH_STYLE,
                )
            ],
            id={
                "type": "crime-map-graph-wrapper",
                "key": graph_key,
            },
            style=GRAPH_STYLE,
        )

        label = (
            f"Map points: {point_start_date} to {point_end_date}"
            f" | visible points: {visible_point_count:,}"
        )

        return graph, label

    @app.callback(
        Output("fullscreen-figure-store", "data"),
        Input("expand-map-button", "n_clicks"),
        Input("expand-daily-button", "n_clicks"),
        Input("expand-scatter-button", "n_clicks"),
        Input("close-fullscreen-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_fullscreen_store(
        map_clicks,
        daily_clicks,
        scatter_clicks,
        close_clicks,
    ):
        triggered_id = ctx.triggered_id

        if triggered_id == "close-fullscreen-button":
            return None

        if triggered_id == "expand-map-button":
            return "map"

        if triggered_id == "expand-daily-button":
            return "daily"

        if triggered_id == "expand-scatter-button":
            return "scatter"

        raise PreventUpdate

    @app.callback(
        Output("crime-fullscreen-figure-store", "data"),
        Input("crime-expand-map-button", "n_clicks"),
        Input("crime-expand-daily-button", "n_clicks"),
        Input("crime-close-fullscreen-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_crime_fullscreen_store(
        map_clicks,
        daily_clicks,
        close_clicks,
    ):
        triggered_id = ctx.triggered_id

        if triggered_id == "crime-close-fullscreen-button":
            return None

        if triggered_id == "crime-expand-map-button":
            return "map"

        if triggered_id == "crime-expand-daily-button":
            return "daily"

        raise PreventUpdate

    @app.callback(
        Output("fullscreen-overlay", "className"),
        Output("fullscreen-title", "children"),
        Output("fullscreen-figure", "figure"),
        Input("fullscreen-figure-store", "data"),
        Input("importance-bin-filter", "value"),
        Input("daily-visible-range-store", "data"),
        Input("legend-toggle", "value"),
    )
    def update_fullscreen_overlay(
        fullscreen_target,
        selected_bin_value,
        range_store_data,
        legend_values,
    ):
        if legend_values is None:
            legend_values = []

        if fullscreen_target is None:
            return "fullscreen-overlay hidden", "", {}

        if fullscreen_target == "map":
            point_start_date, point_end_date = get_range_from_store(
                range_store_data=range_store_data,
                default_start=default_call_start,
                default_end=default_call_end,
            )

            show_colorbar = "map_colorbar" in legend_values

            fig, visible_point_count = cached_calls_map_figure(
                selected_bin_value=selected_bin_value,
                point_start_date=point_start_date,
                point_end_date=point_end_date,
                show_colorbar=show_colorbar,
            )

            title = (
                f"Map view | {point_start_date} to {point_end_date}"
                f" | {visible_point_count:,} visible points"
            )

            return "fullscreen-overlay", title, fig

        if fullscreen_target == "daily":
            show_legend = "daily" in legend_values

            fig = cached_calls_daily_figure(
                selected_bin_value=selected_bin_value,
                show_legend=show_legend,
            )

            return "fullscreen-overlay", "Daily crime events", fig

        if fullscreen_target == "scatter":
            show_legend = "scatter" in legend_values

            fig = cached_calls_scatter_figure(
                selected_bin_value=selected_bin_value,
                show_legend=show_legend,
            )

            return "fullscreen-overlay", "Call volume vs. response time", fig

        raise PreventUpdate

    @app.callback(
        Output("crime-fullscreen-overlay", "className"),
        Output("crime-fullscreen-title", "children"),
        Output("crime-fullscreen-figure", "figure"),
        Input("crime-fullscreen-figure-store", "data"),
        Input("crime-category-filter", "value"),
        Input("crime-daily-visible-range-store", "data"),
        Input("crime-legend-toggle", "value"),
        Input("crime-point-subcategory-filter", "value"),
        Input("crime-point-neighborhood-filter", "value"),
        Input("crime-point-text-filter", "value"),
    )
    def update_crime_fullscreen_overlay(
        fullscreen_target,
        selected_category_value,
        range_store_data,
        legend_values,
        selected_subcategories,
        selected_neighborhoods,
        text_filter,
    ):
        if legend_values is None:
            legend_values = []

        if fullscreen_target is None:
            return "fullscreen-overlay hidden", "", {}

        if fullscreen_target == "map":
            point_start_date, point_end_date = get_range_from_store(
                range_store_data=range_store_data,
                default_start=default_crime_start,
                default_end=default_crime_end,
            )

            point_filters = {
                "offense_sub_categories": selected_subcategories or [],
                "mcpp_neighborhoods": selected_neighborhoods or [],
                "text": text_filter or "",
            }

            show_colorbar = "map_colorbar" in legend_values

            fig, visible_point_count = build_crime_map_figure(
                selected_category_value=selected_category_value,
                point_start_date=point_start_date,
                point_end_date=point_end_date,
                show_colorbar=show_colorbar,
                point_filters=point_filters,
            )

            title = (
                f"Map view | {point_start_date} to {point_end_date}"
                f" | {visible_point_count:,} visible points"
            )

            return "fullscreen-overlay", title, fig

        if fullscreen_target == "daily":
            show_legend = "daily" in legend_values

            fig = cached_crime_daily_figure(
                selected_category_value=selected_category_value,
                show_legend=show_legend,
            )

            return "fullscreen-overlay", "Daily crime events", fig

        raise PreventUpdate

    return app


dashboard = create_app()
server = dashboard.server


@server.route("/healthz")
def health_check():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    dashboard.run(debug=True)
