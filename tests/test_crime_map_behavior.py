import json

import pandas as pd
import pytest

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crime_dashboard_data import (
    EVENT_ID_COLUMN,
    TIME_COLUMN,
    load_crime_dashboard_context,
)
from crime_dashboard_figures import (
    TARGET_CRIME_CATEGORIES,
    make_map_figure,
)


POINT_TRACE_TYPES = {
    "scattermapbox",
    "scattermap",
}

CHOROPLETH_TRACE_TYPES = {
    "choroplethmapbox",
    "choroplethmap",
}


@pytest.fixture(scope="session")
def crime_context():
    return load_crime_dashboard_context()


def get_point_traces(fig):
    return [
        trace
        for trace in fig.data
        if str(getattr(trace, "type", "")).lower() in POINT_TRACE_TYPES
    ]


def get_choropleth_traces(fig):
    return [
        trace
        for trace in fig.data
        if str(getattr(trace, "type", "")).lower() in CHOROPLETH_TRACE_TYPES
    ]


def get_point_trace_counts(fig):
    counts = {}

    for trace in get_point_traces(fig):
        trace_name = getattr(trace, "name", "unnamed")
        lat_values = getattr(trace, "lat", None)

        if lat_values is None:
            counts[trace_name] = 0
        else:
            counts[trace_name] = len(lat_values)

    return counts


def get_nonzero_point_categories(fig):
    counts = get_point_trace_counts(fig)

    return {
        category
        for category, count in counts.items()
        if count > 0
    }


def get_default_point_window(crime_context):
    event_mcpp = crime_context["event_mcpp"].copy()

    event_mcpp[TIME_COLUMN] = pd.to_datetime(
        event_mcpp[TIME_COLUMN],
        errors="coerce",
    )

    latest_day = event_mcpp[TIME_COLUMN].dropna().max().normalize()
    start_day = latest_day - pd.Timedelta(days=29)

    return start_day.date().isoformat(), latest_day.date().isoformat()


def test_crime_map_returns_valid_map_figure(crime_context):
    start_date, end_date = get_default_point_window(crime_context)

    fig = make_map_figure(
        context=crime_context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        point_start_date=start_date,
        point_end_date=end_date,
        show_colorbar=True,
    )

    choropleth_traces = get_choropleth_traces(fig)
    point_traces = get_point_traces(fig)

    assert len(choropleth_traces) == 1
    assert len(point_traces) >= 1
    assert "mapbox" in fig.layout

    # This catches non-JSON-safe values like pd.NA inside customdata.
    json.loads(fig.to_json())


@pytest.mark.parametrize(
    "selected_categories",
    [
        ["all other"],
        ["property crime"],
        ["violent crime"],
        ["all other", "property crime"],
        ["property crime", "violent crime"],
        ["all other", "property crime", "violent crime"],
    ],
)
def test_crime_map_point_categories_match_selection(
    crime_context,
    selected_categories,
):
    start_date, end_date = get_default_point_window(crime_context)

    fig = make_map_figure(
        context=crime_context,
        selected_bins=selected_categories,
        point_start_date=start_date,
        point_end_date=end_date,
        show_colorbar=False,
    )

    nonzero_categories = get_nonzero_point_categories(fig)

    unexpected_categories = nonzero_categories - set(selected_categories)

    assert unexpected_categories == set()


def test_crime_map_colorbar_toggle_only_changes_choropleth_showscale(
    crime_context,
):
    start_date, end_date = get_default_point_window(crime_context)

    fig_without_colorbar = make_map_figure(
        context=crime_context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        point_start_date=start_date,
        point_end_date=end_date,
        show_colorbar=False,
    )

    fig_with_colorbar = make_map_figure(
        context=crime_context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        point_start_date=start_date,
        point_end_date=end_date,
        show_colorbar=True,
    )

    choropleth_without = get_choropleth_traces(fig_without_colorbar)[0]
    choropleth_with = get_choropleth_traces(fig_with_colorbar)[0]

    assert choropleth_without.showscale is False
    assert choropleth_with.showscale is True


def test_crime_map_repeated_category_switching_is_stable(
    crime_context,
):
    start_date, end_date = get_default_point_window(crime_context)

    selection_sequence = [
        ["all other", "property crime", "violent crime"],
        ["property crime"],
        ["violent crime"],
        ["all other"],
        ["property crime"],
    ]

    nonzero_category_sequence = []

    for selected_categories in selection_sequence:
        fig = make_map_figure(
            context=crime_context,
            selected_bins=selected_categories,
            point_start_date=start_date,
            point_end_date=end_date,
            show_colorbar=False,
        )

        nonzero_category_sequence.append(
            get_nonzero_point_categories(fig)
        )

    assert nonzero_category_sequence[1] <= {"property crime"}
    assert nonzero_category_sequence[2] <= {"violent crime"}
    assert nonzero_category_sequence[3] <= {"all other"}
    assert nonzero_category_sequence[4] <= {"property crime"}

    # The same selection should produce the same point categories later.
    assert nonzero_category_sequence[1] == nonzero_category_sequence[4]


def test_crime_map_shorter_point_window_has_no_more_points_than_longer_window(
    crime_context,
):
    event_mcpp = crime_context["event_mcpp"].copy()

    event_mcpp[TIME_COLUMN] = pd.to_datetime(
        event_mcpp[TIME_COLUMN],
        errors="coerce",
    )

    latest_day = event_mcpp[TIME_COLUMN].dropna().max().normalize()

    short_start = (latest_day - pd.Timedelta(days=7)).date().isoformat()
    long_start = (latest_day - pd.Timedelta(days=29)).date().isoformat()
    end_date = latest_day.date().isoformat()

    short_fig = make_map_figure(
        context=crime_context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        point_start_date=short_start,
        point_end_date=end_date,
        show_colorbar=False,
    )

    long_fig = make_map_figure(
        context=crime_context,
        selected_bins=TARGET_CRIME_CATEGORIES,
        point_start_date=long_start,
        point_end_date=end_date,
        show_colorbar=False,
    )

    short_points = sum(get_point_trace_counts(short_fig).values())
    long_points = sum(get_point_trace_counts(long_fig).values())

    assert short_points <= long_points

if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-vv",
                "-s",
                "--tb=long",
            ]
        )
    )