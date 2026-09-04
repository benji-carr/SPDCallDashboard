import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.spd_config import (
    PAPER_BG,
    PLOTLY_MAP_STYLE,
    PLOTLY_SEATTLE_CENTER,
    PLOTLY_TEMPLATE,
    PLOT_BG,
)
from dashboard.crime_dashboard_data import (
    EVENT_ID_COLUMN,
    ROW_ID_COLUMN,
    TIME_COLUMN,
    REPORT_TIME_COLUMN,
    LAT_COL,
    LON_COL,
    CATEGORY_COLUMN,
    SUB_CATEGORY_COLUMN,
)


TARGET_CRIME_CATEGORIES = [
    "other (includes drug and sex offenses)",
    "property crime",
    "violent crime",
]


CRIME_CATEGORY_COLOR_MAP = {
    "other (includes drug and sex offenses)": "#2F80ED",        # blue
    "property crime": "#27AE60",                                # green
    "violent crime": "#EB5757",                                 # red
}


def get_category_color(category_name: str) -> str:
    return CRIME_CATEGORY_COLOR_MAP.get(category_name, "#bbbbbb")


def get_combo_color(selected_categories: list[str]) -> str:
    if len(selected_categories) == 1:
        return get_category_color(selected_categories[0])

    return "#dddddd"


def make_crime_combo_label(category_combo: list[str]) -> str:
    if len(category_combo) == len(TARGET_CRIME_CATEGORIES):
        return "All selected categories"

    return " + ".join(category_combo)


def get_dataset_relative_daily_window(data: pd.DataFrame) -> dict:
    if "date" not in data.columns:
        raise ValueError("DataFrame is missing required column: date")

    valid_dates = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dropna()

    if valid_dates.empty:
        raise ValueError("No valid dates available for daily chart")

    latest_available_day = valid_dates.max().normalize()
    earliest_available_day = valid_dates.min().normalize()

    earliest_analysis_day = earliest_available_day + pd.Timedelta(days=1)
    plot_start_day = earliest_analysis_day

    plot_end_day = latest_available_day
    initial_view_start = latest_available_day - pd.Timedelta(days=1)

    if initial_view_start < plot_start_day:
        initial_view_start = plot_start_day

    return {
        "earliest_available_day": earliest_available_day,
        "latest_available_day": latest_available_day,
        "earliest_analysis_day": earliest_analysis_day,
        "plot_start_day": plot_start_day,
        "plot_end_day": plot_end_day,
        "initial_view_start": initial_view_start,
    }


def prepare_daily_event_data(
    context: dict,
    selected_bins: list[str],
) -> tuple[pd.DataFrame, dict]:
    valid_time = context["valid_time"].copy()

    required_columns = [
        TIME_COLUMN,
        EVENT_ID_COLUMN,
        ROW_ID_COLUMN,
        "event_importance_bin",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in valid_time.columns
    ]

    if missing_columns:
        raise ValueError(
            f"valid_time is missing required columns: {missing_columns}"
        )

    valid_time[TIME_COLUMN] = pd.to_datetime(
        valid_time[TIME_COLUMN],
        errors="coerce",
    )

    valid_time = valid_time[
        valid_time[TIME_COLUMN].notna()
        & valid_time[EVENT_ID_COLUMN].notna()
    ].copy()

    valid_time["date"] = valid_time[TIME_COLUMN].dt.normalize()

    window = get_dataset_relative_daily_window(valid_time)

    plot_start_day = window["plot_start_day"]
    plot_end_day = window["plot_end_day"]

    filtered = valid_time[
        valid_time["date"].between(
            plot_start_day,
            plot_end_day,
        )
        & valid_time["event_importance_bin"].isin(selected_bins)
    ].copy()

    date_index = pd.date_range(
        start=plot_start_day,
        end=plot_end_day,
        freq="D",
    )

    daily_volume = (
        filtered
        .groupby("date", as_index=False)
        .agg(
            reported_offenses=(EVENT_ID_COLUMN, "nunique"),
            unique_reports=(ROW_ID_COLUMN, "nunique"),
        )
        .set_index("date")
        .reindex(date_index)
        .fillna(0)
        .rename_axis("date")
        .reset_index()
    )

    daily_volume["reported_offenses"] = (
        daily_volume["reported_offenses"]
        .astype(int)
    )

    daily_volume["unique_reports"] = (
        daily_volume["unique_reports"]
        .astype(int)
    )

    daily_volume["rolling_7_day_avg"] = (
        daily_volume["reported_offenses"]
        .rolling(
            window=7,
            min_periods=7,
        )
        .mean()
    )

    return daily_volume, window

def make_plotly_safe_customdata(
    data: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    customdata = data[columns].copy()

    for column in columns:
        if pd.api.types.is_numeric_dtype(customdata[column]):
            customdata[column] = pd.to_numeric(
                customdata[column],
                errors="coerce",
            )
        else:
            customdata[column] = (
                customdata[column]
                .astype("object")
                .where(customdata[column].notna(), "Not available")
            )

    return customdata.to_numpy()


def make_daily_figure(
    context: dict,
    selected_bins: list[str],
) -> go.Figure:
    combo_label = make_crime_combo_label(selected_bins)
    combo_color = get_combo_color(selected_bins)

    daily_volume, window = prepare_daily_event_data(
        context=context,
        selected_bins=selected_bins,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_volume["date"],
            y=daily_volume["reported_offenses"],
            mode="lines",
            name="Daily reported offenses",
            line=dict(
                width=1.4,
                color=combo_color,
            ),
            opacity=0.45,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Daily reported offenses: %{y:,}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_volume["date"],
            y=daily_volume["rolling_7_day_avg"],
            mode="lines",
            name="7-day average",
            line=dict(
                width=3,
                color=combo_color,
            ),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "7-day average: %{y:,.1f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Daily Crime Events<br><sup>Type of Crime: {combo_label}</sup>",
            x=0.01,
            xanchor="left",
        ),
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        xaxis=dict(
            title=None,
            range=[
                window["initial_view_start"],
                window["plot_end_day"],
            ],
            rangeselector=dict(
                x=0.01,
                xanchor="left",
                y=1,
                yanchor="top",
                bgcolor="rgba(17, 17, 17, 0.85)",
                activecolor="rgba(255,255,255,0.18)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                font=dict(size=10),
                buttons=[
                    dict(
                        count=1,
                        label="1D",
                        step="day",
                        stepmode="backward",
                    ),
                    dict(
                        count=6,
                        label="1W",
                        step="day",
                        stepmode="backward",
                    ),
                    dict(
                        count=29,
                        label="1M",
                        step="day",
                        stepmode="backward",
                    ),
                    dict(
                        label="1Y",
                        step="all",
                    ),
                ],
            ),
            rangeslider=dict(
                visible=True,
                thickness=0.08,
            ),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(
                text="Reported offenses",
                standoff=12,
            ),
            automargin=True,
        ),
        margin={
            "l": 70,
            "r": 25,
            "t": 84,
            "b": 45,
        },
        hovermode="x unified",
        legend_title_text="Metric",
        showlegend=False,
    )

    return fig

def normalize_mcpp_neighborhood(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("&", "and", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_neighborhood_total_counts(
    event_mcpp: pd.DataFrame,
    unmappable_events: pd.DataFrame,
    selected_bins: list[str],
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> pd.DataFrame:
    mappable = event_mcpp.copy()

    mappable[TIME_COLUMN] = pd.to_datetime(
        mappable[TIME_COLUMN],
        errors="coerce",
    )

    mappable["mcpp_neighborhood"] = normalize_mcpp_neighborhood(
        mappable["mcpp_neighborhood"]
    )

    mappable["event_importance_bin"] = (
        mappable["event_importance_bin"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mappable = mappable[
        mappable[TIME_COLUMN].notna()
        & mappable[EVENT_ID_COLUMN].notna()
        & mappable["mcpp_neighborhood"].notna()
        & mappable["event_importance_bin"].isin(selected_bins)
        & mappable[TIME_COLUMN].dt.normalize().between(start_day, end_day)
    ].copy()

    mappable_counts = (
        mappable
        .groupby("mcpp_neighborhood", as_index=False)
        .agg(
            past_year_mappable_events=(EVENT_ID_COLUMN, "nunique"),
        )
    )

    if unmappable_events is None or unmappable_events.empty:
        unmappable_counts = pd.DataFrame(
            columns=[
                "mcpp_neighborhood",
                "past_year_unmappable_events",
            ]
        )

    else:
        unmappable = unmappable_events.copy()

        required_unmappable_columns = [
            EVENT_ID_COLUMN,
            TIME_COLUMN,
            "mcpp_neighborhood",
            "event_importance_bin",
        ]

        missing_unmappable_columns = [
            column
            for column in required_unmappable_columns
            if column not in unmappable.columns
        ]

        if missing_unmappable_columns:
            raise ValueError(
                "unmappable_events is missing required columns: "
                + ", ".join(missing_unmappable_columns)
            )

        unmappable[TIME_COLUMN] = pd.to_datetime(
            unmappable[TIME_COLUMN],
            errors="coerce",
        )

        unmappable["mcpp_neighborhood"] = normalize_mcpp_neighborhood(
            unmappable["mcpp_neighborhood"]
        )

        unmappable["event_importance_bin"] = (
            unmappable["event_importance_bin"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        unmappable = unmappable[
            unmappable[TIME_COLUMN].notna()
            & unmappable[EVENT_ID_COLUMN].notna()
            & unmappable["mcpp_neighborhood"].notna()
            & unmappable["event_importance_bin"].isin(selected_bins)
            & unmappable[TIME_COLUMN].dt.normalize().between(start_day, end_day)
        ].copy()

        unmappable_counts = (
            unmappable
            .groupby("mcpp_neighborhood", as_index=False)
            .agg(
                past_year_unmappable_events=(EVENT_ID_COLUMN, "nunique"),
            )
        )

    total_counts = mappable_counts.merge(
        unmappable_counts,
        on="mcpp_neighborhood",
        how="outer",
    )

    total_counts["past_year_mappable_events"] = (
        total_counts["past_year_mappable_events"]
        .fillna(0)
        .astype(int)
    )

    total_counts["past_year_unmappable_events"] = (
        total_counts["past_year_unmappable_events"]
        .fillna(0)
        .astype(int)
    )

    total_counts["past_year_total_events"] = (
        total_counts["past_year_mappable_events"]
        + total_counts["past_year_unmappable_events"]
    )

    return total_counts

def make_map_figure(
    context: dict,
    selected_bins: list[str],
    point_start_date: str | None = None,
    point_end_date: str | None = None,
    show_colorbar: bool = False,
    point_filters: dict | None = None,
) -> go.Figure:
    event_mcpp = context["event_mcpp"].copy()
    unmappable_events = context.get("unmappable_events", pd.DataFrame()).copy()
    mcpp_boundaries = context["mcpp_boundaries"].copy()
    neighborhood_population = context["neighborhood_population"].copy()

    combo_label = make_crime_combo_label(selected_bins)

    required_event_columns = [
        EVENT_ID_COLUMN,
        ROW_ID_COLUMN,
        TIME_COLUMN,
        REPORT_TIME_COLUMN,
        LAT_COL,
        LON_COL,
        "event_group",
        "event_importance_bin",
        "mcpp_neighborhood",
        "mcpp_precinct",
        CATEGORY_COLUMN,
        SUB_CATEGORY_COLUMN,
    ]

    missing_event_columns = [
        column
        for column in required_event_columns
        if column not in event_mcpp.columns
    ]

    if missing_event_columns:
        raise ValueError(
            f"event_mcpp is missing required columns: {missing_event_columns}"
        )

    required_boundary_columns = [
        "objectid",
        "plot_feature_id",
        "mcpp_neighborhood",
        "mcpp_precinct",
        "geometry",
    ]

    missing_boundary_columns = [
        column
        for column in required_boundary_columns
        if column not in mcpp_boundaries.columns
    ]

    if missing_boundary_columns:
        raise ValueError(
            f"mcpp_boundaries is missing required columns: {missing_boundary_columns}"
        )

    if "population" not in neighborhood_population.columns:
        raise ValueError(
            "neighborhood_population is missing required column: population"
        )

    if (
        "mcpp_neighborhood" not in neighborhood_population.columns
        and "dispatch_neighborhood" not in neighborhood_population.columns
    ):
        raise ValueError(
            "neighborhood_population must contain either "
            "'mcpp_neighborhood' or 'dispatch_neighborhood'."
        )

    event_mcpp[TIME_COLUMN] = pd.to_datetime(
        event_mcpp[TIME_COLUMN],
        errors="coerce",
    )

    event_mcpp[REPORT_TIME_COLUMN] = pd.to_datetime(
        event_mcpp[REPORT_TIME_COLUMN],
        errors="coerce",
    )

    event_mcpp[LAT_COL] = pd.to_numeric(
        event_mcpp[LAT_COL],
        errors="coerce",
    )

    event_mcpp[LON_COL] = pd.to_numeric(
        event_mcpp[LON_COL],
        errors="coerce",
    )

    event_mcpp["event_importance_bin"] = (
        event_mcpp["event_importance_bin"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    event_mcpp["mcpp_neighborhood"] = (
        event_mcpp["mcpp_neighborhood"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    event_mcpp = event_mcpp[
        event_mcpp[TIME_COLUMN].notna()
        & event_mcpp[EVENT_ID_COLUMN].notna()
        & event_mcpp[LAT_COL].notna()
        & event_mcpp[LON_COL].notna()
    ].copy()

    if event_mcpp.empty:
        fig = go.Figure()

        fig.update_layout(
            title=dict(
                text="Map unavailable<br><sup>No mappable offenses</sup>",
                x=0.01,
                xanchor="left",
            ),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            mapbox=dict(
                style=PLOTLY_MAP_STYLE,
                center=PLOTLY_SEATTLE_CENTER,
                zoom=10,
            ),
        )

        return fig

    latest_available_day = event_mcpp[TIME_COLUMN].dt.normalize().max()
    past_year_start = latest_available_day - pd.Timedelta(days=364)

    past_year_events = event_mcpp[
        event_mcpp[TIME_COLUMN]
        .dt.normalize()
        .between(
            past_year_start,
            latest_available_day,
        )
    ].copy()

    selected_events = past_year_events[
        past_year_events["event_importance_bin"].isin(selected_bins)
    ].copy()

    population_for_mcpp = neighborhood_population.copy()

    if "mcpp_neighborhood" in population_for_mcpp.columns:
        population_neighborhood_column = "mcpp_neighborhood"

    elif "dispatch_neighborhood" in population_for_mcpp.columns:
        population_neighborhood_column = "dispatch_neighborhood"

    else:
        raise ValueError(
            "neighborhood_population must contain either "
            "'mcpp_neighborhood' or 'dispatch_neighborhood'."
        )

    population_for_mcpp = population_for_mcpp[
        [
            population_neighborhood_column,
            "population",
        ]
    ].copy()

    population_for_mcpp = population_for_mcpp.rename(
        columns={
            population_neighborhood_column: "mcpp_neighborhood",
        }
    )

    population_for_mcpp["mcpp_neighborhood"] = (
        population_for_mcpp["mcpp_neighborhood"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    population_for_mcpp["population"] = pd.to_numeric(
        population_for_mcpp["population"],
        errors="coerce",
    )

    population_for_mcpp = population_for_mcpp.drop_duplicates(
        subset="mcpp_neighborhood"
    )

    base_gdf = mcpp_boundaries.copy()

    base_gdf["plot_feature_id"] = (
        base_gdf["plot_feature_id"]
        .astype(str)
    )

    base_gdf["mcpp_neighborhood"] = (
        base_gdf["mcpp_neighborhood"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if "mcpp_neighborhood_display" not in base_gdf.columns:
        base_gdf["mcpp_neighborhood_display"] = (
            base_gdf["mcpp_neighborhood"]
            .astype("string")
            .str.title()
        )

    base_gdf = base_gdf.drop(
        columns=["population"],
        errors="ignore",
    )

    base_gdf = base_gdf.merge(
        population_for_mcpp[
            [
                "mcpp_neighborhood",
                "population",
            ]
        ],
        on="mcpp_neighborhood",
        how="left",
    )

    base_gdf["population"] = pd.to_numeric(
        base_gdf["population"],
        errors="coerce",
    )

    base_geojson = json.loads(base_gdf.to_json())

    neighborhood_total_counts = prepare_neighborhood_total_counts(
        event_mcpp=event_mcpp,
        unmappable_events=unmappable_events,
        selected_bins=selected_bins,
        start_day=past_year_start,
        end_day=latest_available_day,
    )

    choropleth_gdf = base_gdf.merge(
        neighborhood_total_counts,
        on="mcpp_neighborhood",
        how="left",
    )

    for count_column in [
        "past_year_mappable_events",
        "past_year_unmappable_events",
        "past_year_total_events",
    ]:
        choropleth_gdf[count_column] = (
            choropleth_gdf[count_column]
            .fillna(0)
            .astype(int)
        )

    choropleth_gdf["past_year_total_events_per_1000"] = np.where(
        choropleth_gdf["population"].notna()
        & (choropleth_gdf["population"] > 0),
        (
            choropleth_gdf["past_year_total_events"]
            / choropleth_gdf["population"]
            * 1000
        ),
        np.nan,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=base_geojson,
            locations=choropleth_gdf["plot_feature_id"],
            z=choropleth_gdf["past_year_total_events_per_1000"],
            featureidkey="properties.plot_feature_id",
            colorscale="Viridis",
            marker={
                "opacity": 0.68,
                "line": {
                    "width": 0.4,
                    "color": "rgba(255,255,255,0.35)",
                },
            },
            colorbar={
                "title": "Total crime<br>events per 1,000",
                "x": 0.98,
                "y": 0.50,
                "len": 0.62,
            },
            showscale=show_colorbar,
            name="Past-year total events per 1,000 residents",
            showlegend=False,
            customdata=choropleth_gdf[
                [
                    "mcpp_neighborhood_display",
                    "population",
                    "past_year_total_events_per_1000",
                    "past_year_total_events",
                    "past_year_mappable_events",
                    "past_year_unmappable_events",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Type of Crime: {combo_label}<br>"
                "Population: %{customdata[1]:,.0f}<br>"
                "<br>"
                "<b>Total past-year events: %{customdata[3]:,}</b><br>"
                "Mappable point events: %{customdata[4]:,}<br>"
                "Unmappable neighborhood-assigned events: %{customdata[5]:,}<br>"
                "Total events per 1,000 residents: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
        )
    )

    point_events = selected_events.copy()

    if point_start_date is not None and point_end_date is not None:
        point_start = pd.to_datetime(point_start_date, errors="coerce")
        point_end = pd.to_datetime(point_end_date, errors="coerce")

        if pd.notna(point_start) and pd.notna(point_end):
            point_events = point_events[
                point_events[TIME_COLUMN]
                .dt.normalize()
                .between(
                    point_start.normalize(),
                    point_end.normalize(),
                )
            ].copy()

    else:
        point_start = latest_available_day
        point_end = latest_available_day

        point_events = point_events[
            point_events[TIME_COLUMN]
            .dt.normalize()
            .between(
                point_start,
                point_end,
            )
        ].copy()

    point_events = apply_point_filters(
        point_events=point_events,
        point_filters=point_filters,
    )

    point_metric_lookup = choropleth_gdf[
        [
            "mcpp_neighborhood",
            "population",
            "past_year_total_events",
            "past_year_mappable_events",
            "past_year_unmappable_events",
            "past_year_total_events_per_1000",
        ]
    ].copy()

    point_events["mcpp_neighborhood"] = (
        point_events["mcpp_neighborhood"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    point_events = point_events.merge(
        point_metric_lookup,
        on="mcpp_neighborhood",
        how="left",
    )

    point_events["offense_time_display"] = (
        point_events[TIME_COLUMN]
        .dt.strftime("%Y-%m-%d %H:%M")
        .fillna("Not available")
    )

    point_events["report_time_display"] = (
        point_events[REPORT_TIME_COLUMN]
        .dt.strftime("%Y-%m-%d %H:%M")
        .fillna("Not available")
    )

    if "mcpp_neighborhood_display" not in point_events.columns:
        point_events["mcpp_neighborhood_display"] = (
            point_events["mcpp_neighborhood"]
            .astype("string")
            .str.title()
        )

    point_events["mcpp_neighborhood_display"] = (
        point_events["mcpp_neighborhood_display"]
        .astype("string")
        .fillna("Not available")
    )

    point_events["population_display"] = np.where(
        point_events["population"].notna(),
        point_events["population"].round(0).astype("Int64").astype(str),
        "Not available",
    )

    point_events["population_display"] = (
        point_events["population_display"]
        .replace("<NA>", "Not available")
    )

    point_events["past_year_total_events_display"] = np.where(
        point_events["past_year_total_events"].notna(),
        point_events["past_year_total_events"].round(0).astype("Int64").astype(str),
        "Not available",
    )

    point_events["past_year_total_events_display"] = (
        point_events["past_year_total_events_display"]
        .replace("<NA>", "Not available")
    )

    point_events["past_year_mappable_events_display"] = np.where(
        point_events["past_year_mappable_events"].notna(),
        point_events["past_year_mappable_events"].round(0).astype("Int64").astype(str),
        "Not available",
    )

    point_events["past_year_mappable_events_display"] = (
        point_events["past_year_mappable_events_display"]
        .replace("<NA>", "Not available")
    )

    point_events["past_year_unmappable_events_display"] = np.where(
        point_events["past_year_unmappable_events"].notna(),
        point_events["past_year_unmappable_events"].round(0).astype("Int64").astype(str),
        "Not available",
    )

    point_events["past_year_unmappable_events_display"] = (
        point_events["past_year_unmappable_events_display"]
        .replace("<NA>", "Not available")
    )

    point_events["past_year_total_events_per_1000_display"] = np.where(
        point_events["past_year_total_events_per_1000"].notna(),
        point_events["past_year_total_events_per_1000"].round(1).astype(str),
        "Not available",
    )

    for bin_name in TARGET_CRIME_CATEGORIES:
        if bin_name not in selected_bins:
            continue

        bin_points = point_events[
            point_events["event_importance_bin"] == bin_name
        ].copy()

        if bin_points.empty:
            continue

        fig.add_trace(
            go.Scattermapbox(
                lat=bin_points[LAT_COL],
                lon=bin_points[LON_COL],
                mode="markers",
                name=bin_name,
                legendgroup=bin_name,
                showlegend=True,
                marker=dict(
                    size=8,
                    opacity=0.85,
                    color=get_category_color(bin_name),
                ),
                customdata=make_plotly_safe_customdata(
                    bin_points,
                    [
                        EVENT_ID_COLUMN,
                        ROW_ID_COLUMN,
                        "offense_time_display",
                        "report_time_display",
                        "event_importance_bin",
                        SUB_CATEGORY_COLUMN,
                        "mcpp_neighborhood_display",
                        "population_display",
                        "past_year_total_events_display",
                        "past_year_mappable_events_display",
                        "past_year_unmappable_events_display",
                        "past_year_total_events_per_1000_display",
                    ],
                ),
                hovertemplate=(
                    "<b>Offense ID:</b> %{customdata[0]}<br>"
                    "<b>Report number:</b> %{customdata[1]}<br>"
                    "<b>Offense time:</b> %{customdata[2]}<br>"
                    "<b>Report time:</b> %{customdata[3]}<br>"
                    f"<b>Selected Type of Crime:</b> {combo_label}<br>"
                    "<b>Point Type of Crime:</b> %{customdata[4]}<br>"
                    "<b>Offense sub-category:</b> %{customdata[5]}<br>"
                    "<b>Neighborhood:</b> %{customdata[6]}<br>"
                    "<br>"
                    "<b>Population:</b> %{customdata[7]}<br>"
                    "<b>Total past-year neighborhood events:</b> %{customdata[8]}<br>"
                    "<b>Mappable neighborhood events:</b> %{customdata[9]}<br>"
                    "<b>Unmappable neighborhood-assigned events:</b> %{customdata[10]}<br>"
                    "<b>Total events per 1,000 residents:</b> %{customdata[11]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Reported Crime Offenses Per 1,000 Residents In the Past Year"
                f"<br><sup>Type of Crime: {combo_label}</sup>"
            ),
            x=0.01,
            xanchor="left",
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        mapbox=dict(
            style=PLOTLY_MAP_STYLE,
            center=PLOTLY_SEATTLE_CENTER,
            zoom=10,
        ),
        legend=dict(
            title="Point Type",
            x=0.02,
            y=0.50,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(
                color="#dddddd",
                size=11,
            ),
        ),
        margin={
            "l": 0,
            "r": 0,
            "t": 58,
            "b": 0,
        },
        showlegend=True,
    )

    return fig

def apply_point_filters(
    point_events: pd.DataFrame,
    point_filters: dict | None,
) -> pd.DataFrame:
    if point_filters is None:
        return point_events

    filtered = point_events.copy()

    selected_subcategories = point_filters.get("offense_sub_categories", [])
    selected_neighborhoods = point_filters.get("mcpp_neighborhoods", [])
    text_filter = str(point_filters.get("text", "")).strip().lower()

    if selected_subcategories:
        filtered = filtered[
            filtered["offense_sub_category"].isin(selected_subcategories)
        ].copy()

    if selected_neighborhoods:
        filtered = filtered[
            filtered["mcpp_neighborhood"].isin(selected_neighborhoods)
        ].copy()

    if text_filter:
        searchable_text = (
            filtered["offense_id"].astype("string").fillna("")
            + " "
            + filtered["report_number"].astype("string").fillna("")
            + " "
            + filtered["block_address"].astype("string").fillna("")
            + " "
            + filtered["offense_sub_category"].astype("string").fillna("")
        ).str.lower()

        filtered = filtered[
            searchable_text.str.contains(
                text_filter,
                regex=False,
                na=False,
            )
        ].copy()

    return filtered


if __name__ == "__main__":
    from dashboard.crime_dashboard_data import (
        load_crime_dashboard_context,
    )

    context = load_crime_dashboard_context()

    fig = make_map_figure(
        context=context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        show_colorbar=True,
    )

    fig.show()

    daily_fig = make_daily_figure(
        context=context,
        selected_bins=TARGET_CRIME_CATEGORIES,
    )

    daily_fig.show()
