import numpy as np
import pandas as pd
import geopandas as gpd
import pytest

from shapely.geometry import (
    Point,
    Polygon,
)

from forecasting.features.predicthq import (
    ATTENDED_CATEGORIES,
    PredictHQDataError,
    aggregate_predicthq_features,
    build_complete_feature_panel,
    iter_month_windows,
    map_event_days_to_mcpp,
    normalize_predicthq_events,
    prepare_backtest_event_days,
    validate_feature_panel,
)


def make_event(
    event_id="E1",
    category="concerts",
    start_local="2024-07-04T18:00:00",
    end_local="2024-07-04T21:00:00",
    first_seen="2024-07-01T12:00:00Z",
    updated="2024-07-02T12:00:00Z",
    cancelled=None,
    postponed=None,
    deleted_reason=None,
    attendance=1000,
    rank=60,
    local_rank=70,
    geometry=None,
):
    if geometry is None:
        geometry = {
            "type": "Point",
            "coordinates": [
                -122.34,
                47.61,
            ],
        }

    state = (
        "deleted"
        if deleted_reason
        else "active"
    )

    return {
        "id": event_id,
        "title": f"Event {event_id}",
        "category": category,
        "state": state,
        "deleted_reason":
            deleted_reason,
        "rank": rank,
        "local_rank":
            local_rank,
        "phq_attendance":
            attendance,
        "start":
            "2024-07-05T01:00:00Z",
        "start_local":
            start_local,
        "end":
            "2024-07-05T04:00:00Z",
        "end_local":
            end_local,
        "first_seen":
            first_seen,
        "updated":
            updated,
        "cancelled":
            cancelled,
        "postponed":
            postponed,
        "timezone":
            "America/Los_Angeles",
        "duration": 10800,
        "scope": "locality",
        "country": "US",
        "geo": {
            "geometry":
                geometry,
            "address": {
                "formatted_address":
                    "Seattle, WA",
            },
        },
        "phq_labels": [],
    }


@pytest.fixture
def boundaries():
    return gpd.GeoDataFrame(
        {
            "mcpp_neighborhood": [
                "Belltown",
                "Downtown Commercial",
            ],
        },
        geometry=[
            Polygon(
                [
                    (-122.36, 47.59),
                    (-122.33, 47.59),
                    (-122.33, 47.63),
                    (-122.36, 47.63),
                ]
            ),
            Polygon(
                [
                    (-122.33, 47.59),
                    (-122.30, 47.59),
                    (-122.30, 47.63),
                    (-122.33, 47.63),
                ]
            ),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def target_panel():
    return pd.DataFrame(
        {
            "target_date": [
                pd.Timestamp(
                    "2024-07-04"
                ),
                pd.Timestamp(
                    "2024-07-04"
                ),
            ],
            "neighborhood": [
                "BELLTOWN",
                "DOWNTOWN COMMERCIAL",
            ],
            "calls": [
                10,
                20,
            ],
        }
    )


def test_month_windows_cover_range():
    windows = list(
        iter_month_windows(
            "2024-01-15",
            "2024-03-10",
        )
    )

    assert windows == [
        (
            pd.Timestamp(
                "2024-01-15"
            ),
            pd.Timestamp(
                "2024-01-31"
            ),
        ),
        (
            pd.Timestamp(
                "2024-02-01"
            ),
            pd.Timestamp(
                "2024-02-29"
            ),
        ),
        (
            pd.Timestamp(
                "2024-03-01"
            ),
            pd.Timestamp(
                "2024-03-10"
            ),
        ),
    ]


def test_normalize_point_geometry():
    df = (
        normalize_predicthq_events(
            [
                make_event()
            ]
        )
    )

    row = df.iloc[0]

    assert row[
        "event_id"
    ] == "E1"

    assert row[
        "geometry_type"
    ] == "Point"

    assert np.isclose(
        row["longitude"],
        -122.34,
    )

    assert np.isclose(
        row["latitude"],
        47.61,
    )

    assert row[
        "event_start_date"
    ] == pd.Timestamp(
        "2024-07-04"
    )


def test_normalization_deduplicates_event_ids():
    old = make_event(
        event_id="E1",
        updated=(
            "2024-07-01T12:00:00Z"
        ),
        rank=40,
    )

    new = make_event(
        event_id="E1",
        updated=(
            "2024-07-03T12:00:00Z"
        ),
        rank=70,
    )

    df = (
        normalize_predicthq_events(
            [old, new]
        )
    )

    assert len(df) == 1
    assert df.iloc[0]["rank"] == 70


def test_first_seen_after_forecast_is_excluded():
    event = make_event(
        event_id="LATE",
        first_seen=(
            "2024-07-04T12:00:00Z"
        ),
    )

    events = (
        normalize_predicthq_events(
            [event]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    assert result.empty


def test_event_known_before_forecast_survives():
    events = (
        normalize_predicthq_events(
            [
                make_event(
                    event_id="KNOWN"
                )
            ]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "known_by_forecast"
        ]
    )


def test_cancelled_before_forecast_is_excluded():
    event = make_event(
        event_id="CANCELLED",
        deleted_reason="cancelled",
        cancelled=(
            "2024-07-03T12:00:00Z"
        ),
    )

    events = (
        normalize_predicthq_events(
            [event]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    assert result.empty


def test_cancelled_after_forecast_is_retained():
    event = make_event(
        event_id="CANCELLED_LATE",
        deleted_reason="cancelled",
        cancelled=(
            "2024-07-05T12:00:00Z"
        ),
    )

    events = (
        normalize_predicthq_events(
            [event]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    assert len(result) == 1


def test_duplicate_deleted_event_is_excluded():
    event = make_event(
        event_id="DUP",
        deleted_reason="duplicate",
    )

    events = (
        normalize_predicthq_events(
            [event]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    assert result.empty


def test_events_outside_requested_window_return_empty():
    events = (
        normalize_predicthq_events(
            [
                make_event(
                    event_id="OUTSIDE",
                )
            ]
        )
    )

    result = (
        prepare_backtest_event_days(
            events,
            "2024-07-10",
            "2024-07-10",
        )
    )

    assert result.empty


def test_point_event_maps_to_mcpp(
    boundaries,
):
    events = (
        normalize_predicthq_events(
            [
                make_event()
            ]
        )
    )

    event_days = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    mapped, unmapped = (
        map_event_days_to_mcpp(
            event_days,
            boundaries,
            valid_neighborhoods={
                "BELLTOWN",
                "DOWNTOWN COMMERCIAL",
            },
        )
    )

    assert unmapped.empty
    assert len(mapped) == 1

    assert (
        mapped.iloc[0][
            "neighborhood"
        ]
        == "BELLTOWN"
    )


def test_area_event_can_map_to_multiple_mcpps(
    boundaries,
):
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.35, 47.60],
                [-122.31, 47.60],
                [-122.31, 47.62],
                [-122.35, 47.62],
                [-122.35, 47.60],
            ]
        ],
    }

    events = (
        normalize_predicthq_events(
            [
                make_event(
                    geometry=geometry,
                    attendance=1000,
                )
            ]
        )
    )

    event_days = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    mapped, _ = (
        map_event_days_to_mcpp(
            event_days,
            boundaries,
            valid_neighborhoods={
                "BELLTOWN",
                "DOWNTOWN COMMERCIAL",
            },
        )
    )

    assert set(
        mapped["neighborhood"]
    ) == {
        "BELLTOWN",
        "DOWNTOWN COMMERCIAL",
    }

    assert (
        mapped[
            "n_mcpp_neighborhoods"
        ]
        == 2
    ).all()

    assert np.isclose(
        mapped[
            "phq_attendance_daily_split"
        ].sum(),
        1000,
    )


def test_no_duplicate_event_date_neighborhood_keys(
    boundaries,
):
    events = (
        normalize_predicthq_events(
            [
                make_event()
            ]
        )
    )

    event_days = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    mapped, _ = (
        map_event_days_to_mcpp(
            event_days,
            boundaries,
        )
    )

    assert (
        mapped
        .duplicated(
            subset=[
                "event_id",
                "target_date",
                "neighborhood",
            ]
        )
        .sum()
        == 0
    )


def test_category_counts_sum_to_event_count(
    boundaries,
):
    records = [
        make_event(
            event_id="C",
            category="concerts",
        ),
        make_event(
            event_id="S",
            category="sports",
        ),
    ]

    events = (
        normalize_predicthq_events(
            records
        )
    )

    days = (
        prepare_backtest_event_days(
            events,
            "2024-07-04",
            "2024-07-04",
        )
    )

    mapped, _ = (
        map_event_days_to_mcpp(
            days,
            boundaries,
        )
    )

    features = (
        aggregate_predicthq_features(
            mapped
        )
    )

    row = features.iloc[0]

    category_columns = [
        (
            "phq_"
            f"{category.replace('-', '_')}"
            "_count"
        )
        for category
        in ATTENDED_CATEGORIES
    ]

    assert (
        row[
            category_columns
        ].sum()
        == row[
            "phq_event_count"
        ]
    )


def test_complete_panel_zero_fills_missing_days(
    target_panel,
):
    sparse = pd.DataFrame(
        {
            "target_date": [
                pd.Timestamp(
                    "2024-07-04"
                )
            ],
            "neighborhood": [
                "BELLTOWN"
            ],
            "phq_event_count": [
                1
            ],
        }
    )

    # Add required category columns.
    for category in (
        ATTENDED_CATEGORIES
    ):
        sparse[
            (
                "phq_"
                f"{category.replace('-', '_')}"
                "_count"
            )
        ] = 0

    sparse[
        "phq_concerts_count"
    ] = 1

    panel = (
        build_complete_feature_panel(
            sparse,
            target_panel,
            "2024-07-04",
            "2024-07-05",
        )
    )

    assert len(panel) == 4

    july_5 = panel.loc[
        panel[
            "target_date"
        ].eq(
            pd.Timestamp(
                "2024-07-05"
            )
        )
    ]

    assert (
        july_5[
            "phq_event_count"
        ]
        == 0
    ).all()


def test_final_panel_validation(
    target_panel,
):
    rows = []

    for date in pd.date_range(
        "2024-07-04",
        "2024-07-05",
    ):
        for neighborhood in [
            "BELLTOWN",
            "DOWNTOWN COMMERCIAL",
        ]:
            row = {
                "target_date":
                    date,

                "neighborhood":
                    neighborhood,

                "phq_event_count":
                    0,
            }

            for category in (
                ATTENDED_CATEGORIES
            ):
                row[
                    (
                        "phq_"
                        f"{category.replace('-', '_')}"
                        "_count"
                    )
                ] = 0

            rows.append(row)

    panel = pd.DataFrame(
        rows
    )

    validate_feature_panel(
        panel=panel,
        target_panel=
            target_panel,
        start_date=
            "2024-07-04",
        end_date=
            "2024-07-05",
    )
