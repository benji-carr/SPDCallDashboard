from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import wkt
from shapely.geometry import shape

from forecasting.paths import (
    GENERATED_FOLDS_PATH,
    PREDICTHQ_CHUNK_DIR,
    PREDICTHQ_OUTPUT_DIR,
    TARGET_PANEL_5Y_PATH,
)

PREDICTHQ_EVENTS_URL = (
    "https://api.predicthq.com/v1/events/"
)

SEATTLE_TZ = "America/Los_Angeles"

# Broad enough to capture all of Seattle.
# We spatially filter to actual SPD MCPP boundaries afterward.
SEATTLE_WITHIN = (
    "20mi@47.6062,-122.3321"
)

ATTENDED_CATEGORIES = (
    "community",
    "conferences",
    "concerts",
    "expos",
    "festivals",
    "performing-arts",
    "sports",
)


class PredictHQDataError(RuntimeError):
    pass


# ============================================================
# Date-window helpers
# ============================================================

def iter_month_windows(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> Iterable[
    tuple[pd.Timestamp, pd.Timestamp]
]:
    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    if end < start:
        raise ValueError(
            "end_date must be on or after start_date."
        )

    cursor = start

    while cursor <= end:
        month_end = (
            cursor
            + pd.offsets.MonthEnd(0)
        ).normalize()

        window_end = min(
            month_end,
            end,
        )

        yield (
            cursor,
            window_end,
        )

        cursor = (
            window_end
            + pd.Timedelta(days=1)
        )


def _local_date_from_utc(
    series: pd.Series,
) -> pd.Series:
    timestamps = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return (
        timestamps
        .dt.tz_convert(SEATTLE_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )


# ============================================================
# Event normalization
# ============================================================

def _event_geometry(
    event: dict,
):
    geo = event.get("geo") or {}

    geometry_data = (
        geo.get("geometry")
    )

    if not geometry_data:
        return None

    try:
        return shape(
            geometry_data
        )

    except Exception:
        return None


def _flatten_event(
    event: dict,
) -> dict:
    geometry = _event_geometry(
        event
    )

    longitude = np.nan
    latitude = np.nan

    if (
        geometry is not None
        and geometry.geom_type
        == "Point"
    ):
        longitude = float(
            geometry.x
        )

        latitude = float(
            geometry.y
        )

    parent_event = (
        event.get(
            "parent_event"
        )
        or {}
    )

    geo = (
        event.get("geo")
        or {}
    )

    address = (
        geo.get("address")
        or {}
    )

    return {
        "event_id":
            event.get("id"),

        "title":
            event.get("title"),

        "category":
            event.get("category"),

        "state":
            event.get("state"),

        "deleted_reason":
            event.get(
                "deleted_reason"
            ),

        "duplicate_of_id":
            event.get(
                "duplicate_of_id"
            ),

        "parent_event_id":
            parent_event.get(
                "parent_event_id"
            ),

        "rank":
            event.get("rank"),

        "local_rank":
            event.get(
                "local_rank"
            ),

        "phq_attendance":
            event.get(
                "phq_attendance"
            ),

        "start":
            event.get("start"),

        "start_local":
            event.get(
                "start_local"
            ),

        "end":
            event.get("end"),

        "end_local":
            event.get(
                "end_local"
            ),

        "first_seen":
            event.get(
                "first_seen"
            ),

        "updated":
            event.get("updated"),

        "cancelled":
            event.get(
                "cancelled"
            ),

        "postponed":
            event.get(
                "postponed"
            ),

        "timezone":
            event.get(
                "timezone"
            ),

        "duration_seconds":
            event.get(
                "duration"
            ),

        "scope":
            event.get("scope"),

        "country":
            event.get(
                "country"
            ),

        "location_confidence_score":
            event.get(
                "location_confidence_score"
            ),

        "geometry_type": (
            geometry.geom_type
            if geometry is not None
            else None
        ),

        "geometry_wkt": (
            geometry.wkt
            if geometry is not None
            else None
        ),

        "longitude":
            longitude,

        "latitude":
            latitude,

        "formatted_address":
            address.get(
                "formatted_address"
            ),

        "phq_labels_json":
            json.dumps(
                event.get(
                    "phq_labels"
                )
                or [],
                sort_keys=True,
            ),
    }


def normalize_predicthq_events(
    records: list[dict],
) -> pd.DataFrame:
    rows = [
        _flatten_event(record)
        for record in records
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    required = {
        "event_id",
        "category",
        "start",
        "first_seen",
        "updated",
        "geometry_wkt",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise PredictHQDataError(
            "Normalized PredictHQ data "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        "rank",
        "local_rank",
        "phq_attendance",
        "duration_seconds",
        "location_confidence_score",
        "longitude",
        "latitude",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    utc_columns = [
        "start",
        "end",
        "first_seen",
        "updated",
        "cancelled",
        "postponed",
    ]

    for column in utc_columns:
        df[
            f"{column}_utc"
        ] = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

    start_local = pd.to_datetime(
        df["start_local"],
        errors="coerce",
    ).dt.normalize()

    end_local = pd.to_datetime(
        df["end_local"],
        errors="coerce",
    ).dt.normalize()

    df[
        "event_start_date"
    ] = start_local.fillna(
        _local_date_from_utc(
            df["start"]
        )
    )

    df[
        "event_end_date"
    ] = end_local.fillna(
        _local_date_from_utc(
            df["end"]
        )
    )

    df[
        "event_end_date"
    ] = (
        df["event_end_date"]
        .fillna(
            df[
                "event_start_date"
            ]
        )
    )

    df[
        "first_seen_local_date"
    ] = _local_date_from_utc(
        df["first_seen"]
    )

    df[
        "updated_local_date"
    ] = _local_date_from_utc(
        df["updated"]
    )

    df[
        "cancelled_local_date"
    ] = _local_date_from_utc(
        df["cancelled"]
    )

    df[
        "postponed_local_date"
    ] = _local_date_from_utc(
        df["postponed"]
    )

    df[
        "event_span_days"
    ] = (
        df["event_end_date"]
        - df["event_start_date"]
    ).dt.days + 1

    # The same event may appear in several
    # monthly active-date chunks.
    df = (
        df
        .sort_values(
            [
                "event_id",
                "updated_utc",
            ],
            na_position="first",
        )
        .drop_duplicates(
            subset="event_id",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# PredictHQ API
# ============================================================

def fetch_predicthq_window(
    access_token: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    page_size: int = 100,
    timeout: int = 60,
    session=None,
) -> list[dict]:
    if not access_token:
        raise ValueError(
            "PredictHQ access token is required."
        )

    client = (
        session
        or requests.Session()
    )

    headers = {
        "Authorization":
            f"Bearer {access_token}",

        "Accept":
            "application/json",
    }

    base_params = {
        "active.gte":
            pd.Timestamp(
                start_date
            ).date().isoformat(),

        "active.lte":
            pd.Timestamp(
                end_date
            ).date().isoformat(),

        "active.tz":
            SEATTLE_TZ,

        "category":
            ",".join(
                ATTENDED_CATEGORIES
            ),

        "country":
            "US",

        "within":
            SEATTLE_WITHIN,

        # Use parent/non-umbrella events,
        # avoiding child-event double counts.
        "parent.include":
            "true",

        # Include deleted events so we can use
        # historical cancellation/postponement
        # timestamps.
        "state":
            "active,deleted,predicted",

        "sort":
            "start",

        "limit":
            page_size,
    }

    results = []
    offset = 0

    while True:
        params = {
            **base_params,
            "offset": offset,
        }

        response = client.get(
            PREDICTHQ_EVENTS_URL,
            headers=headers,
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

        if payload.get(
            "overflow",
            False,
        ):
            raise PredictHQDataError(
                "PredictHQ reports overflow. "
                "The subscription is truncating "
                "the result set."
            )

        page = payload.get(
            "results",
            [],
        )

        if not isinstance(
            page,
            list,
        ):
            raise PredictHQDataError(
                "PredictHQ response "
                "'results' is not a list."
            )

        results.extend(page)

        if not payload.get(
            "next"
        ):
            break

        if not page:
            raise PredictHQDataError(
                "PredictHQ returned a next "
                "page but no results."
            )

        offset += len(page)

    return results


def download_predicthq_events(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    chunk_dir: str | Path,
    access_token: str | None = None,
    refresh: bool = False,
    page_size: int = 100,
) -> pd.DataFrame:
    token = (
        access_token
        or os.getenv(
            "PREDICTHQ_ACCESS_TOKEN"
        )
    )

    if not token:
        raise ValueError(
            "Set PREDICTHQ_ACCESS_TOKEN "
            "before downloading data."
        )

    chunk_dir = Path(
        chunk_dir
    )

    chunk_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = []

    for (
        window_start,
        window_end,
    ) in iter_month_windows(
        start_date,
        end_date,
    ):
        path = (
            chunk_dir
            / (
                "predicthq_"
                f"{window_start:%Y%m%d}_"
                f"{window_end:%Y%m%d}"
                ".parquet"
            )
        )

        if (
            path.exists()
            and not refresh
        ):
            chunk = (
                pd.read_parquet(
                    path
                )
            )

        else:
            records = (
                fetch_predicthq_window(
                    access_token=token,
                    start_date=
                        window_start,
                    end_date=
                        window_end,
                    page_size=
                        page_size,
                )
            )

            chunk = (
                normalize_predicthq_events(
                    records
                )
            )

            chunk.to_parquet(
                path,
                index=False,
            )

        chunks.append(
            chunk
        )

    if not chunks:
        return pd.DataFrame()

    events = pd.concat(
        chunks,
        ignore_index=True,
    )

    if events.empty:
        return events

    events = (
        events
        .sort_values(
            [
                "event_id",
                "updated_utc",
            ],
            na_position="first",
        )
        .drop_duplicates(
            subset="event_id",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return events


# ============================================================
# Historical availability
# ============================================================

def prepare_backtest_event_days(
    events: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    if end < start:
        raise ValueError(
            "end_date must be on or after start_date."
        )

    df = events.copy()

    # Current duplicate / invalid records
    # should not become separate events.
    df = df.loc[
        ~df[
            "deleted_reason"
        ].isin(
            [
                "duplicate",
                "invalid",
            ]
        )
    ].copy()

    if df.empty:
        return df

    df = df.loc[
        df["event_id"].notna()
        & df[
            "event_start_date"
        ].notna()
        & df[
            "event_end_date"
        ].notna()
        & df[
            "first_seen_local_date"
        ].notna()
        & df[
            "geometry_wkt"
        ].notna()
        & (
            df[
                "event_span_days"
            ]
            >= 1
        )
    ].copy()

    if df.empty:
        return df

    df = df.loc[
        (
            df[
                "event_start_date"
            ]
            <= end
        )
        & (
            df[
                "event_end_date"
            ]
            >= start
        )
    ].copy()

    if df.empty:
        return df

    df[
        "target_date"
    ] = df.apply(
        lambda row:
        pd.date_range(
            start=max(
                row[
                    "event_start_date"
                ],
                start,
            ),
            end=min(
                row[
                    "event_end_date"
                ],
                end,
            ),
            freq="D",
        ),
        axis=1,
    )

    df = (
        df
        .explode(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    if df.empty:
        return df

    df[
        "target_date"
    ] = pd.to_datetime(
        df["target_date"]
    ).dt.normalize()

    df[
        "forecast_origin"
    ] = (
        df["target_date"]
        - pd.Timedelta(
            days=1
        )
    )

    # Necessary anti-leakage condition:
    # PredictHQ had to know the event
    # before the forecast was made.
    df[
        "known_by_forecast"
    ] = (
        df[
            "first_seen_local_date"
        ]
        <= df[
            "forecast_origin"
        ]
    )

    df[
        "cancelled_by_forecast"
    ] = (
        df[
            "cancelled_local_date"
        ].notna()
        & (
            df[
                "cancelled_local_date"
            ]
            <= df[
                "forecast_origin"
            ]
        )
    )

    df[
        "postponed_by_forecast"
    ] = (
        df[
            "postponed_local_date"
        ].notna()
        & (
            df[
                "postponed_local_date"
            ]
            <= df[
                "forecast_origin"
            ]
        )
    )

    # QA flag only.
    # Do NOT use this as a predictor.
    df[
        "updated_after_forecast"
    ] = (
        df[
            "updated_local_date"
        ].notna()
        & (
            df[
                "updated_local_date"
            ]
            > df[
                "forecast_origin"
            ]
        )
    )

    df = df.loc[
        df["known_by_forecast"]
        & ~df[
            "cancelled_by_forecast"
        ]
        & ~df[
            "postponed_by_forecast"
        ]
    ].copy()

    if df.empty:
        return df

    # Current-snapshot approximation only.
    # Do not use in the first strict-ish
    # count/category SARIMAX experiment.
    df[
        "phq_attendance_daily_equal"
    ] = np.where(
        df[
            "phq_attendance"
        ].notna()
        & (
            df[
                "event_span_days"
            ]
            > 0
        ),
        (
            df[
                "phq_attendance"
            ]
            / df[
                "event_span_days"
            ]
        ),
        np.nan,
    )

    return df


# ============================================================
# SPD geography
# ============================================================

def map_event_days_to_mcpp(
    event_days: pd.DataFrame,
    mcpp_boundaries:
        gpd.GeoDataFrame,
    valid_neighborhoods:
        set[str] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if event_days.empty:
        return (
            event_days.copy(),
            event_days.copy(),
        )

    boundaries = (
        mcpp_boundaries
        .copy()
    )

    if (
        "mcpp_neighborhood"
        not in boundaries.columns
    ):
        raise ValueError(
            "MCPP boundaries must contain "
            "'mcpp_neighborhood'."
        )

    if boundaries.crs is None:
        boundaries = (
            boundaries
            .set_crs(
                epsg=4326
            )
        )

    else:
        boundaries = (
            boundaries
            .to_crs(
                epsg=4326
            )
        )

    boundaries[
        "neighborhood"
    ] = (
        boundaries[
            "mcpp_neighborhood"
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        valid_neighborhoods
        is not None
    ):
        boundaries = (
            boundaries.loc[
                boundaries[
                    "neighborhood"
                ].isin(
                    valid_neighborhoods
                )
            ]
            .copy()
        )

    geometries = (
        event_days[
            "geometry_wkt"
        ]
        .map(
            lambda value: (
                wkt.loads(
                    value
                )
                if pd.notna(
                    value
                )
                else None
            )
        )
    )

    events_gdf = (
        gpd.GeoDataFrame(
            event_days.copy(),
            geometry=
                geometries,
            crs="EPSG:4326",
        )
    )

    # Polygon/area PredictHQ events may
    # legitimately intersect several MCPPs.
    joined = (
        gpd.sjoin(
            events_gdf,
            boundaries[
                [
                    "neighborhood",
                    "geometry",
                ]
            ],
            how="left",
            predicate="intersects",
        )
        .drop(
            columns=
                "index_right",
            errors="ignore",
        )
    )

    unmapped = (
        joined.loc[
            joined[
                "neighborhood"
            ].isna()
        ]
        .drop(
            columns="geometry",
            errors="ignore",
        )
        .copy()
    )

    mapped = (
        joined.loc[
            joined[
                "neighborhood"
            ].notna()
        ]
        .copy()
    )

    # Boundary overlap or area geometry
    # must not duplicate the same
    # event-date-neighborhood key.
    mapped = (
        mapped
        .drop_duplicates(
            subset=[
                "event_id",
                "target_date",
                "neighborhood",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    mapped[
        "n_mcpp_neighborhoods"
    ] = (
        mapped
        .groupby(
            [
                "event_id",
                "target_date",
            ]
        )[
            "neighborhood"
        ]
        .transform(
            "nunique"
        )
    )

    # Preserve total attendance if an
    # area event crosses several MCPPs.
    mapped[
        "phq_attendance_daily_split"
    ] = np.where(
        mapped[
            "phq_attendance_daily_equal"
        ].notna(),
        (
            mapped[
                "phq_attendance_daily_equal"
            ]
            / mapped[
                "n_mcpp_neighborhoods"
            ]
        ),
        np.nan,
    )

    mapped = mapped.drop(
        columns="geometry",
        errors="ignore",
    )

    return (
        mapped,
        unmapped,
    )


# ============================================================
# Daily features
# ============================================================

def aggregate_predicthq_features(
    mapped_event_days:
        pd.DataFrame,
) -> pd.DataFrame:
    if mapped_event_days.empty:
        return pd.DataFrame()

    base = (
        mapped_event_days
        .groupby(
            [
                "target_date",
                "neighborhood",
            ],
            as_index=False,
        )
        .agg(
            phq_event_count=(
                "event_id",
                "nunique",
            ),

            # Current-snapshot fields.
            # Save them, but don't make
            # them our first experiment.
            phq_rank_snapshot_sum=(
                "rank",
                "sum",
            ),

            phq_rank_snapshot_max=(
                "rank",
                "max",
            ),

            phq_local_rank_snapshot_sum=(
                "local_rank",
                "sum",
            ),

            phq_local_rank_snapshot_max=(
                "local_rank",
                "max",
            ),

            phq_attendance_snapshot_daily_split_sum=(
                "phq_attendance_daily_split",
                "sum",
            ),
        )
    )

    category_counts = (
        mapped_event_days
        .groupby(
            [
                "target_date",
                "neighborhood",
                "category",
            ]
        )["event_id"]
        .nunique()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    rename = {
        category: (
            "phq_"
            f"{category.replace('-', '_')}"
            "_count"
        )
        for category
        in ATTENDED_CATEGORIES
    }

    category_counts = (
        category_counts
        .rename(
            columns=rename
        )
    )

    for feature in (
        rename.values()
    ):
        if (
            feature
            not in
            category_counts.columns
        ):
            category_counts[
                feature
            ] = 0

    return base.merge(
        category_counts[
            [
                "target_date",
                "neighborhood",
                *rename.values(),
            ]
        ],
        on=[
            "target_date",
            "neighborhood",
        ],
        how="left",
        validate="one_to_one",
    )


def build_complete_feature_panel(
    sparse_features:
        pd.DataFrame,
    target_panel:
        pd.DataFrame,
    start_date:
        str | pd.Timestamp,
    end_date:
        str | pd.Timestamp,
) -> pd.DataFrame:
    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    neighborhoods = sorted(
        target_panel.loc[
            target_panel[
                "neighborhood"
            ].notna()
            & target_panel[
                "neighborhood"
            ].ne("NULL"),
            "neighborhood",
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    full_index = (
        pd.MultiIndex
        .from_product(
            [
                pd.date_range(
                    start,
                    end,
                    freq="D",
                ),
                neighborhoods,
            ],
            names=[
                "target_date",
                "neighborhood",
            ],
        )
    )

    panel = (
        sparse_features
        .set_index(
            [
                "target_date",
                "neighborhood",
            ]
        )
        .reindex(
            full_index
        )
        .reset_index()
    )

    feature_columns = [
        column
        for column
        in panel.columns
        if column.startswith(
            "phq_"
        )
    ]

    panel[
        feature_columns
    ] = (
        panel[
            feature_columns
        ]
        .fillna(0)
    )

    return panel


# ============================================================
# Validation
# ============================================================

def validate_feature_panel(
    panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    start_date:
        str | pd.Timestamp,
    end_date:
        str | pd.Timestamp,
) -> None:
    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    valid_neighborhoods = set(
        target_panel.loc[
            target_panel[
                "neighborhood"
            ].notna()
            & target_panel[
                "neighborhood"
            ].ne("NULL"),
            "neighborhood",
        ]
        .astype(str)
        .str.strip()
    )

    duplicates = (
        panel
        .duplicated(
            subset=[
                "target_date",
                "neighborhood",
            ]
        )
        .sum()
    )

    if duplicates:
        raise PredictHQDataError(
            "Duplicate date/neighborhood "
            "rows in feature panel."
        )

    expected_rows = (
        len(
            pd.date_range(
                start,
                end,
                freq="D",
            )
        )
        * len(
            valid_neighborhoods
        )
    )

    if (
        len(panel)
        != expected_rows
    ):
        raise PredictHQDataError(
            "Feature panel is not a "
            "complete date x neighborhood grid."
        )

    if (
        set(
            panel[
                "neighborhood"
            ]
        )
        != valid_neighborhoods
    ):
        raise PredictHQDataError(
            "Feature-panel neighborhoods "
            "do not match target panel."
        )

    if (
        panel[
            "target_date"
        ].min()
        != start
    ):
        raise PredictHQDataError(
            "Incorrect minimum date."
        )

    if (
        panel[
            "target_date"
        ].max()
        != end
    ):
        raise PredictHQDataError(
            "Incorrect maximum date."
        )

    feature_columns = [
        column
        for column
        in panel.columns
        if column.startswith(
            "phq_"
        )
    ]

    if (
        panel[
            feature_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise PredictHQDataError(
            "PredictHQ feature panel "
            "contains missing features."
        )

    values = (
        panel[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise PredictHQDataError(
            "PredictHQ feature panel "
            "contains non-finite values."
        )

    if (
        values < 0
    ).any():
        raise PredictHQDataError(
            "PredictHQ feature panel "
            "contains negative values."
        )

    category_columns = [
        (
            "phq_"
            f"{category.replace('-', '_')}"
            "_count"
        )
        for category
        in ATTENDED_CATEGORIES
    ]

    category_total = (
        panel[
            category_columns
        ]
        .sum(
            axis=1
        )
    )

    if not np.array_equal(
        category_total
        .to_numpy(),
        panel[
            "phq_event_count"
        ].to_numpy(),
    ):
        raise PredictHQDataError(
            "Category counts do not "
            "sum to phq_event_count."
        )


# ============================================================
# QA outputs
# ============================================================

def build_qa_summary(
    events: pd.DataFrame,
    event_days: pd.DataFrame,
    mapped_event_days:
        pd.DataFrame,
    unmapped_event_days:
        pd.DataFrame,
    feature_panel:
        pd.DataFrame,
) -> pd.Series:
    return pd.Series(
        {
            "unique_downloaded_events":
                (
                    events[
                        "event_id"
                    ].nunique()
                    if not
                    events.empty
                    else 0
                ),

            "event_day_rows_after_asof_filter":
                len(
                    event_days
                ),

            "mapped_event_day_rows":
                len(
                    mapped_event_days
                ),

            "unmapped_event_day_rows":
                len(
                    unmapped_event_days
                ),

            "mapped_unique_events":
                (
                    mapped_event_days[
                        "event_id"
                    ].nunique()
                    if not
                    mapped_event_days.empty
                    else 0
                ),

            "updated_after_forecast_share":
                (
                    mapped_event_days[
                        "updated_after_forecast"
                    ].mean()
                    if not
                    mapped_event_days.empty
                    else np.nan
                ),

            "attendance_missing_share":
                (
                    mapped_event_days[
                        "phq_attendance"
                    ].isna().mean()
                    if not
                    mapped_event_days.empty
                    else np.nan
                ),

            "panel_rows":
                len(
                    feature_panel
                ),

            "panel_min_date":
                feature_panel[
                    "target_date"
                ].min(),

            "panel_max_date":
                feature_panel[
                    "target_date"
                ].max(),
        }
    )


def monthly_event_counts(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "events",
            ]
        )

    return (
        events
        .assign(
            month=(
                events[
                    "event_start_date"
                ]
                .dt.to_period(
                    "M"
                )
            )
        )
        .groupby(
            "month",
            as_index=False,
        )
        .agg(
            events=(
                "event_id",
                "nunique",
            )
        )
    )


# ============================================================
# Full builder
# ============================================================

def build_predicthq_backtest_data(
    events: pd.DataFrame,
    target_panel: pd.DataFrame,
    mcpp_boundaries:
        gpd.GeoDataFrame,
    start_date:
        str | pd.Timestamp,
    end_date:
        str | pd.Timestamp,
) -> dict:
    valid_neighborhoods = set(
        target_panel.loc[
            target_panel[
                "neighborhood"
            ].notna()
            & target_panel[
                "neighborhood"
            ].ne("NULL"),
            "neighborhood",
        ]
        .astype(str)
        .str.strip()
    )

    event_days = (
        prepare_backtest_event_days(
            events=events,
            start_date=start_date,
            end_date=end_date,
        )
    )

    (
        mapped,
        unmapped,
    ) = map_event_days_to_mcpp(
        event_days=
            event_days,

        mcpp_boundaries=
            mcpp_boundaries,

        valid_neighborhoods=
            valid_neighborhoods,
    )

    sparse_features = (
        aggregate_predicthq_features(
            mapped
        )
    )

    feature_panel = (
        build_complete_feature_panel(
            sparse_features=
                sparse_features,

            target_panel=
                target_panel,

            start_date=
                start_date,

            end_date=
                end_date,
        )
    )

    validate_feature_panel(
        panel=
            feature_panel,

        target_panel=
            target_panel,

        start_date=
            start_date,

        end_date=
            end_date,
    )

    qa_summary = (
        build_qa_summary(
            events=events,
            event_days=
                event_days,
            mapped_event_days=
                mapped,
            unmapped_event_days=
                unmapped,
            feature_panel=
                feature_panel,
        )
    )

    return {
        "events":
            events,

        "event_days":
            event_days,

        "mapped_event_days":
            mapped,

        "unmapped_event_days":
            unmapped,

        "sparse_features":
            sparse_features,

        "feature_panel":
            feature_panel,

        "qa_summary":
            qa_summary,

        "monthly_event_counts":
            monthly_event_counts(
                events
            ),
    }


def save_outputs(
    outputs: dict,
    output_dir:
        str | Path,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs[
        "events"
    ].to_parquet(
        output_dir
        / "predicthq_events_snapshot.parquet",
        index=False,
    )

    outputs[
        "mapped_event_days"
    ].to_parquet(
        output_dir
        / "predicthq_mapped_event_days.parquet",
        index=False,
    )

    outputs[
        "unmapped_event_days"
    ].to_parquet(
        output_dir
        / "predicthq_unmapped_event_days.parquet",
        index=False,
    )

    outputs[
        "sparse_features"
    ].to_parquet(
        output_dir
        / "predicthq_sparse_features.parquet",
        index=False,
    )

    outputs[
        "feature_panel"
    ].to_parquet(
        output_dir
        / "predicthq_feature_panel.parquet",
        index=False,
    )

    outputs[
        "qa_summary"
    ].to_csv(
        output_dir
        / "predicthq_qa_summary.csv",
        header=["value"],
    )

    outputs[
        "monthly_event_counts"
    ].to_csv(
        output_dir
        / "predicthq_monthly_event_counts.csv",
        index=False,
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--target-panel",
        default=str(
            TARGET_PANEL_5Y_PATH
        ),
    )

    parser.add_argument(
        "--folds",
        default=str(
            GENERATED_FOLDS_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            PREDICTHQ_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--chunk-dir",
        default=str(
            PREDICTHQ_CHUNK_DIR
        ),
    )

    parser.add_argument(
        "--start",
        default=None,
    )

    parser.add_argument(
        "--end",
        default=None,
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    target_panel = (
        pd.read_parquet(
            args.target_panel
        )
    )

    if args.start and args.end:
        start_date = pd.Timestamp(
            args.start
        )

        end_date = pd.Timestamp(
            args.end
        )

    else:
        folds = (
            pd.read_parquet(
                args.folds
            )
        )

        start_date = (
            pd.Timestamp(
                args.start
            )
            if args.start
            else pd.Timestamp(
                folds[
                    "train_start"
                ].min()
            )
        )

        end_date = (
            pd.Timestamp(
                args.end
            )
            if args.end
            else pd.Timestamp(
                folds[
                    "val_end"
                ].max()
            )
        )

    from crime_dashboard_data import (
        load_mcpp_boundaries,
    )

    boundaries = (
        load_mcpp_boundaries()
    )

    events = (
        download_predicthq_events(
            start_date=
                start_date,

            end_date=
                end_date,

            chunk_dir=
                args.chunk_dir,

            refresh=
                args.refresh,
        )
    )

    if events.empty:
        raise PredictHQDataError(
            "PredictHQ returned no events. "
            "Check token permissions and "
            "subscription coverage."
        )

    outputs = (
        build_predicthq_backtest_data(
            events=events,
            target_panel=
                target_panel,
            mcpp_boundaries=
                boundaries,
            start_date=
                start_date,
            end_date=
                end_date,
        )
    )

    save_outputs(
        outputs,
        args.output_dir,
    )

    print(
        "\nPredictHQ backtest "
        "data complete"
    )

    print(
        outputs[
            "qa_summary"
        ]
    )

    print(
        "\nMonthly coverage"
    )

    print(
        outputs[
            "monthly_event_counts"
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
