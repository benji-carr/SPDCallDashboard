import numpy as np
import pandas as pd
import pytest

from permitted_events_features import (
    add_attendance_features,
    aggregate_permit_features,
    build_complete_permit_panel,
    build_permitted_event_features,
    explode_event_neighborhoods,
    expand_event_days,
    map_to_spd_neighborhoods,
    normalize_special_events,
    select_modeling_events,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def raw_events():
    """
    Small synthetic dataset using the original Seattle
    Special Events display-column names.
    """
    return pd.DataFrame(
        {
            "Application Date": [
                "2024-06-01",
                "2024-06-15",
                "2024-07-04",
                "2024-01-01",
            ],
            "Permit Status": [
                "Issued",
                "Issued",
                "Issued",
                "Issued",
            ],
            "Permit Type": [
                "Special Event",
                "Special Event",
                "Special Event",
                "Special Event",
            ],
            "Event Category": [
                "Community",
                "Commercial",
                "Community",
                "Community",
            ],
            "Event Sub-Category": [
                None,
                None,
                None,
                None,
            ],
            "Name of Event": [
                "Belltown Festival",
                "Downtown Event",
                "Late Application Event",
                "Long Running Market",
            ],
            "Year-Month-App#": [
                "E1",
                "E2",
                "E3",
                "E4",
            ],
            "Event Start Date": [
                "2024-07-04",
                "2024-07-10",
                "2024-07-04",
                "2024-02-01",
            ],
            "Event End Date": [
                "2024-07-04",
                "2024-07-12",
                "2024-07-04",
                "2024-02-20",
            ],
            "Event Location - Park": [
                None,
                None,
                None,
                None,
            ],
            "Event Location - Neighborhood": [
                "Belltown",
                "Downtown",
                "Capitol Hill",
                "Belltown",
            ],
            "Council District": [
                None,
                None,
                None,
                None,
            ],
            "Precinct": [
                None,
                None,
                None,
                None,
            ],
            "Organization": [
                None,
                None,
                None,
                None,
            ],
            "Attendance": [
                "1000",
                "3000",
                "500",
                "250",
            ],
        }
    )


@pytest.fixture
def target_panel():
    """
    Minimal target-panel geography needed by the synthetic events.
    """
    return pd.DataFrame(
        {
            "target_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
            ],
            "neighborhood": [
                "BELLTOWN",
                "DOWNTOWN COMMERCIAL",
                "CAPITOL HILL",
            ],
            "calls": [
                10,
                20,
                15,
            ],
        }
    )


# ============================================================
# Normalization
# ============================================================

def test_normalize_special_events(raw_events):
    result = normalize_special_events(raw_events)

    assert "event_id" in result.columns
    assert "application_date" in result.columns
    assert "event_start_date" in result.columns
    assert "event_end_date" in result.columns
    assert "attendance" in result.columns
    assert "event_duration_days" in result.columns

    assert pd.api.types.is_datetime64_any_dtype(
        result["application_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["event_start_date"]
    )

    assert pd.api.types.is_numeric_dtype(
        result["attendance"]
    )

    # E1 is a one-day event.
    e1 = result.loc[
        result["event_id"].eq("E1")
    ].iloc[0]

    assert e1["event_duration_days"] == 1

    # E2 spans July 10, 11, and 12.
    e2 = result.loc[
        result["event_id"].eq("E2")
    ].iloc[0]

    assert e2["event_duration_days"] == 3


def test_normalize_missing_required_column_raises(raw_events):
    bad = raw_events.drop(
        columns="Application Date"
    )

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        normalize_special_events(bad)


# ============================================================
# Event selection
# ============================================================

def test_select_modeling_events_excludes_long_events(
    raw_events,
):
    normalized = normalize_special_events(
        raw_events
    )

    selected = select_modeling_events(
        normalized
    )

    assert "E1" in set(selected["event_id"])
    assert "E2" in set(selected["event_id"])

    # 20-day synthetic market should be excluded.
    assert "E4" not in set(
        selected["event_id"]
    )

    assert (
        selected["event_duration_days"]
        .between(1, 7)
        .all()
    )


def test_select_modeling_events_excludes_invalid_duration():
    raw = pd.DataFrame(
        {
            "Application Date": ["2024-01-01"],
            "Year-Month-App#": ["BAD"],
            "Event Start Date": ["2024-02-10"],
            "Event End Date": ["2024-02-01"],
            "Event Location - Neighborhood": ["Belltown"],
            "Attendance": [100],
        }
    )

    normalized = normalize_special_events(raw)

    selected = select_modeling_events(
        normalized
    )

    assert selected.empty


# ============================================================
# Event-day expansion / leakage
# ============================================================

def test_expand_event_days_creates_correct_number_of_days(
    raw_events,
):
    normalized = normalize_special_events(
        raw_events
    )

    selected = select_modeling_events(
        normalized
    )

    # Remove E3 for this specific duration test.
    selected = selected.loc[
        selected["event_id"].isin(
            ["E1", "E2"]
        )
    ]

    result = expand_event_days(
        selected
    )

    e1 = result.loc[
        result["event_id"].eq("E1")
    ]

    e2 = result.loc[
        result["event_id"].eq("E2")
    ]

    assert len(e1) == 1
    assert len(e2) == 3


def test_expand_event_days_prevents_forecast_leakage(
    raw_events,
):
    normalized = normalize_special_events(
        raw_events
    )

    selected = select_modeling_events(
        normalized
    )

    result = expand_event_days(
        selected
    )

    assert (
        result["application_date"]
        <= result["forecast_origin"]
    ).all()

    # E3 was applied for ON the target date,
    # therefore it was not known one day earlier.
    assert "E3" not in set(
        result["event_id"]
    )


# ============================================================
# Source-neighborhood explosion
# ============================================================

def test_explode_event_neighborhoods():
    df = pd.DataFrame(
        {
            "event_id": ["A"],
            "target_date": [
                pd.Timestamp("2024-07-01")
            ],
            "event_location_neighborhood": [
                "Belltown;Downtown"
            ],
        }
    )

    result = explode_event_neighborhoods(
        df
    )

    assert len(result) == 2

    assert set(
        result["event_neighborhood"]
    ) == {
        "Belltown",
        "Downtown",
    }


# ============================================================
# Geography crosswalk
# ============================================================

def test_mapping_collapses_duplicate_crosswalk_paths():
    """
    Two different source geography labels both map to the
    same SPD destination. The destination must only appear once.
    """

    event_days = pd.DataFrame(
        {
            "event_id": [
                "A",
                "A",
            ],
            "target_date": [
                pd.Timestamp("2024-07-04"),
                pd.Timestamp("2024-07-04"),
            ],
            "event_neighborhood": [
                "SOURCE 1",
                "SOURCE 2",
            ],
            "attendance": [
                1000,
                1000,
            ],
        }
    )

    crosswalk = {
        "SOURCE 1": [
            "SPD A",
            "SPD B",
        ],
        "SOURCE 2": [
            "SPD A",
        ],
    }

    mapped, unmapped = (
        map_to_spd_neighborhoods(
            event_days,
            crosswalk=crosswalk,
        )
    )

    assert unmapped.empty

    assert len(mapped) == 2

    assert set(
        mapped["neighborhood"]
    ) == {
        "SPD A",
        "SPD B",
    }

    duplicate_count = (
        mapped
        .duplicated(
            subset=[
                "event_id",
                "target_date",
                "neighborhood",
            ]
        )
        .sum()
    )

    assert duplicate_count == 0


def test_unmapped_geography_is_reported():
    event_days = pd.DataFrame(
        {
            "event_id": ["A"],
            "target_date": [
                pd.Timestamp("2024-07-04")
            ],
            "event_neighborhood": [
                "UNKNOWN PLACE"
            ],
            "attendance": [100],
        }
    )

    mapped, unmapped = (
        map_to_spd_neighborhoods(
            event_days,
            crosswalk={},
        )
    )

    assert mapped.empty
    assert len(unmapped) == 1

    assert (
        unmapped.iloc[0][
            "event_neighborhood"
        ]
        == "UNKNOWN PLACE"
    )


# ============================================================
# Attendance
# ============================================================

def test_split_attendance_conserves_total():
    mapped = pd.DataFrame(
        {
            "event_id": [
                "A",
                "A",
                "A",
            ],
            "target_date": [
                pd.Timestamp("2024-07-04"),
                pd.Timestamp("2024-07-04"),
                pd.Timestamp("2024-07-04"),
            ],
            "neighborhood": [
                "N1",
                "N2",
                "N3",
            ],
            "attendance": [
                12000,
                12000,
                12000,
            ],
        }
    )

    result = add_attendance_features(
        mapped
    )

    assert (
        result[
            "n_dispatch_neighborhoods"
        ]
        == 3
    ).all()

    assert np.allclose(
        result["split_attendance"],
        4000,
    )

    assert np.isclose(
        result[
            "split_attendance"
        ].sum(),
        12000,
    )


def test_missing_attendance_remains_identifiable():
    mapped = pd.DataFrame(
        {
            "event_id": ["A"],
            "target_date": [
                pd.Timestamp("2024-07-04")
            ],
            "neighborhood": ["N1"],
            "attendance": [np.nan],
        }
    )

    result = add_attendance_features(
        mapped
    )

    row = result.iloc[0]

    assert row["attendance_known"] == 0
    assert row["attendance_missing"] == 1

    assert pd.isna(
        row["split_attendance"]
    )

    assert pd.isna(
        row["log_split_attendance"]
    )


# ============================================================
# Aggregation
# ============================================================

def test_aggregate_permit_count_uses_unique_event_ids():
    mapped = pd.DataFrame(
        {
            "event_id": [
                "A",
                "B",
            ],
            "target_date": [
                pd.Timestamp("2024-07-04"),
                pd.Timestamp("2024-07-04"),
            ],
            "neighborhood": [
                "BELLTOWN",
                "BELLTOWN",
            ],
            "attendance": [
                100,
                200,
            ],
            "attendance_known": [
                1,
                1,
            ],
            "attendance_missing": [
                0,
                0,
            ],
            "split_attendance": [
                100,
                200,
            ],
            "log_split_attendance": [
                np.log1p(100),
                np.log1p(200),
            ],
        }
    )

    result = aggregate_permit_features(
        mapped
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["se_permit_count"] == 2

    assert np.isclose(
        row[
            "se_split_attendance_sum"
        ],
        300,
    )


# ============================================================
# Complete panel
# ============================================================

def test_complete_panel_has_every_date_neighborhood_pair():
    permit_features = pd.DataFrame(
        {
            "target_date": [
                pd.Timestamp("2024-01-01")
            ],
            "neighborhood": [
                "N1"
            ],
            "se_permit_count": [1],
            "se_attendance_known_count": [1],
            "se_attendance_missing_count": [0],
            "se_total_attendance_known": [100],
            "se_split_attendance_sum": [100],
            "se_max_attendance_known": [100],
            "se_log_split_attendance_sum": [
                np.log1p(100)
            ],
        }
    )

    target_panel = pd.DataFrame(
        {
            "neighborhood": [
                "N1",
                "N2",
            ]
        }
    )

    result = build_complete_permit_panel(
        permit_features=permit_features,
        target_panel=target_panel,
        coverage_start=pd.Timestamp(
            "2024-01-01"
        ),
        coverage_end=pd.Timestamp(
            "2024-01-03"
        ),
    )

    # 3 days x 2 neighborhoods
    assert len(result) == 6

    assert (
        result
        .duplicated(
            subset=[
                "target_date",
                "neighborhood",
            ]
        )
        .sum()
        == 0
    )


def test_complete_panel_fills_no_event_days_with_zero():
    permit_features = pd.DataFrame(
        {
            "target_date": [
                pd.Timestamp("2024-01-01")
            ],
            "neighborhood": [
                "N1"
            ],
            "se_permit_count": [1],
            "se_attendance_known_count": [1],
            "se_attendance_missing_count": [0],
            "se_total_attendance_known": [100],
            "se_split_attendance_sum": [100],
            "se_max_attendance_known": [100],
            "se_log_split_attendance_sum": [
                np.log1p(100)
            ],
        }
    )

    target_panel = pd.DataFrame(
        {
            "neighborhood": ["N1"]
        }
    )

    result = build_complete_permit_panel(
        permit_features=permit_features,
        target_panel=target_panel,
        coverage_start=pd.Timestamp(
            "2024-01-01"
        ),
        coverage_end=pd.Timestamp(
            "2024-01-02"
        ),
    )

    no_event_day = result.loc[
        result["target_date"].eq(
            pd.Timestamp("2024-01-02")
        )
    ].iloc[0]

    assert no_event_day[
        "se_permit_count"
    ] == 0

    assert no_event_day[
        "se_split_attendance_sum"
    ] == 0


# ============================================================
# End-to-end pipeline
# ============================================================

def test_end_to_end_pipeline_has_no_duplicates_or_leakage(
    raw_events,
    target_panel,
):
    outputs = build_permitted_event_features(
        target_panel=target_panel,
        raw_events=raw_events,
    )

    mapped = outputs[
        "mapped_event_days"
    ]

    panel = outputs[
        "permit_panel"
    ]

    # No permit-date-neighborhood duplicates.
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

    # No panel duplicates.
    assert (
        panel
        .duplicated(
            subset=[
                "target_date",
                "neighborhood",
            ]
        )
        .sum()
        == 0
    )

    # Leakage rule.
    assert (
        mapped["application_date"]
        <= mapped["forecast_origin"]
    ).all()

    # Only safe contiguous events survived.
    assert (
        mapped[
            "event_duration_days"
        ]
        .between(1, 7)
        .all()
    )


def test_end_to_end_panel_features_are_nonnegative_and_finite(
    raw_events,
    target_panel,
):
    outputs = build_permitted_event_features(
        target_panel=target_panel,
        raw_events=raw_events,
    )

    panel = outputs[
        "permit_panel"
    ]

    feature_cols = [
        "se_permit_count",
        "se_attendance_known_count",
        "se_attendance_missing_count",
        "se_total_attendance_known",
        "se_split_attendance_sum",
        "se_max_attendance_known",
        "se_log_split_attendance_sum",
    ]

    assert not (
        panel[feature_cols]
        .isna()
        .any()
        .any()
    )

    assert (
        panel[feature_cols]
        >= 0
    ).all().all()

    assert np.isfinite(
        panel[feature_cols]
        .to_numpy()
    ).all()