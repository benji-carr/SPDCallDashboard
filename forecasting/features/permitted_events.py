from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# Configuration
# ============================================================

SPECIAL_EVENTS_URL = (
    "https://cos-data.seattle.gov/"
    "api/v3/views/dm95-f8w5/export.csv"
    "?accessType=DOWNLOAD"
)

MAX_CONTIGUOUS_EVENT_DAYS = 7

# The source becomes clearly incomplete during 2025.
PERMIT_COVERAGE_START = pd.Timestamp("2019-01-01")
PERMIT_COVERAGE_END = pd.Timestamp("2024-12-31")


CALENDAR_DATE_COLUMNS = [
    "application_date",
    "event_start_date",
    "event_end_date",
]


COLUMN_RENAME_MAP = {
    "Application Date": "application_date",
    "Permit Status": "permit_status",
    "Permit Type": "permit_type",
    "Event Category": "event_category",
    "Event Sub-Category": "event_sub_category",
    "Name of Event": "name_of_event",
    "Year-Month-App#": "event_id",
    "Event Start Date": "event_start_date",
    "Event End Date": "event_end_date",
    "Event Location - Park": "event_location_park",
    "Event Location - Neighborhood": "event_location_neighborhood",
    "Council District": "council_district",
    "Precinct": "precinct",
    "Organization": "organization",
    "Attendance": "attendance",
}


TEXT_COLUMNS = [
    "event_id",
    "permit_status",
    "permit_type",
    "event_category",
    "event_sub_category",
    "name_of_event",
    "event_location_neighborhood",
]


# ============================================================
# Geography crosswalk
# ============================================================

SPECIAL_EVENTS_TO_SPD = {
    "Alki / Admiral": [
        "ALKI",
        "NORTH ADMIRAL",
    ],

    "Ballard": [
        "BALLARD NORTH",
        "BALLARD SOUTH",
    ],

    "Beacon Hill": [
        "NORTH BEACON HILL",
        "MID BEACON HILL",
        "SOUTH BEACON HILL",
    ],

    "Belltown": [
        "BELLTOWN",
    ],

    "Broadview / Bitter Lake": [
        "BITTERLAKE",
    ],

    "Capitol Hill": [
        "CAPITOL HILL",
    ],

    "Cascade / Eastlake / South Lake Union": [
        "SLU/CASCADE",
        "EASTLAKE - EAST",
        "EASTLAKE - WEST",
    ],

    "Cedar Park / Meadowbrook": [
        "LAKECITY",
    ],

    "Central Area / Squire Park": [
        "CENTRAL AREA/SQUIRE PARK",
    ],

    "Columbia City": [
        "COLUMBIA CITY",
    ],

    "Delridge / North Delridge": [
        "NORTH DELRIDGE",
        "SOUTH DELRIDGE",
    ],

    "Downtown": [
        "DOWNTOWN COMMERCIAL",
    ],

    "Duwamish / SODO": [
        "SODO",
        "COMMERCIAL DUWAMISH",
    ],

    "Fauntleroy / Seaview": [
        "FAUNTLEROY SW",
    ],

    "First Hill": [
        "FIRST HILL",
    ],

    "Fremont": [
        "FREMONT",
    ],

    "Georgetown": [
        "GEORGETOWN",
    ],

    "Greenwood / Phinney Ridge": [
        "GREENWOOD",
        "PHINNEY RIDGE",
    ],

    "High Point": [
        "HIGH POINT",
    ],

    "Highland Park": [
        "HIGHLAND PARK",
    ],

    "International District": [
        "CHINATOWN/INTERNATIONAL DISTRICT",
    ],

    "Judkins Park": [
        "JUDKINS PARK/NORTH BEACON HILL",
    ],

    "Lake City / Olympic Hills / Victory Heights": [
        "LAKECITY",
    ],

    "Laurelhurst / Sand Point": [
        "SANDPOINT",
    ],

    "Madison Park": [
        "MADISON PARK",
    ],

    "Madrona / Leschi": [
        "MADRONA/LESCHI",
    ],

    "Magnolia": [
        "MAGNOLIA",
    ],

    "Miller Park": [
        "MILLER PARK",
    ],

    "Montlake / Portage Bay": [
        "MONTLAKE/PORTAGE BAY",
    ],

    "Mt. Baker / North Rainier": [
        "MOUNT BAKER",
        "CLAREMONT/RAINIER VISTA",
    ],

    "North Beacon Hill / Jefferson Park": [
        "NORTH BEACON HILL",
        "JUDKINS PARK/NORTH BEACON HILL",
    ],

    "North Capitol Hill": [
        "CAPITOL HILL",
    ],

    "Northgate / Maple Leaf": [
        "NORTHGATE",
    ],

    "Pioneer Square": [
        "PIONEER SQUARE",
    ],

    "Queen Anne": [
        "QUEEN ANNE",
    ],

    "Rainier Beach": [
        "RAINIER BEACH",
    ],

    "Ravenna / Bryant": [
        "ROOSEVELT/RAVENNA",
    ],

    "Roxhill / Westwood": [
        "ROXHILL/WESTWOOD/ARBOR HEIGHTS",
    ],

    "Seward Park": [
        "LAKEWOOD/SEWARD PARK",
    ],

    "South Beacon Hill / New Holly": [
        "SOUTH BEACON HILL",
        "NEW HOLLY",
    ],

    "South Park": [
        "SOUTH PARK",
    ],

    "Sunset Hill / Loyal Heights": [
        "BALLARD NORTH",
    ],

    "University District": [
        "UNIVERSITY",
    ],

    "Wallingford": [
        "WALLINGFORD",
    ],

    "West Seattle Junction / Genesee Hill": [
        "ALASKA JUNCTION",
        "GENESEE",
    ],

    "Whittier Heights": [
        "BALLARD NORTH",
    ],
}


AMBIGUOUS_SPECIAL_EVENT_NEIGHBORHOODS = {
    "Green Lake",
    "Haller Lake",
    "Interbay",
    "North Beach / Blue Ridge",
    "Riverview",
    "Wedgwood / View Ridge",
}


# ============================================================
# Download / normalization
# ============================================================

def download_special_events(
    url: str = SPECIAL_EVENTS_URL,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download the current Seattle Special Events permit dataset.
    """
    response = requests.get(
        url,
        timeout=timeout,
    )
    response.raise_for_status()

    return pd.read_csv(
        BytesIO(response.content)
    )


def normalize_special_events(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Seattle Special Events columns and dtypes.
    """
    df = (
        raw
        .rename(columns=COLUMN_RENAME_MAP)
        .copy()
    )

    required = {
        "event_id",
        "application_date",
        "event_start_date",
        "event_end_date",
        "event_location_neighborhood",
        "attendance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required Special Events columns: "
            f"{sorted(missing)}"
        )

    for col in CALENDAR_DATE_COLUMNS:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
        ).dt.normalize()

    df["attendance"] = pd.to_numeric(
        df["attendance"],
        errors="coerce",
    )

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
            )

    df["event_duration_days"] = (
        df["event_end_date"]
        - df["event_start_date"]
    ).dt.days + 1

    return df


# ============================================================
# Event selection
# ============================================================

def select_modeling_events(
    events: pd.DataFrame,
    max_contiguous_days: int = MAX_CONTIGUOUS_EVENT_DAYS,
    coverage_start: pd.Timestamp = PERMIT_COVERAGE_START,
    coverage_end: pd.Timestamp = PERMIT_COVERAGE_END,
) -> pd.DataFrame:
    """
    Select permit records whose active dates can be safely interpreted
    as contiguous event days.

    Rules:
    - application date must be known
    - duration must be between 1 and max_contiguous_days
    - event date must overlap trusted source coverage
    """
    mask = (
        events["application_date"].notna()
        & events["event_duration_days"].between(
            1,
            max_contiguous_days,
        )
        & (
            events["event_start_date"]
            <= coverage_end
        )
        & (
            events["event_end_date"]
            >= coverage_start
        )
    )

    return events.loc[mask].copy()


# ============================================================
# Event-day expansion
# ============================================================

def expand_event_days(
    events: pd.DataFrame,
    coverage_start: pd.Timestamp = PERMIT_COVERAGE_START,
    coverage_end: pd.Timestamp = PERMIT_COVERAGE_END,
) -> pd.DataFrame:
    """
    Expand safe contiguous permits to one row per active event day.

    Applies the one-day-ahead leakage rule:
        application_date <= target_date - 1 day
    """
    df = events.copy()

    df["target_date"] = df.apply(
        lambda row: pd.date_range(
            start=max(
                row["event_start_date"],
                coverage_start,
            ),
            end=min(
                row["event_end_date"],
                coverage_end,
            ),
            freq="D",
        ),
        axis=1,
    )

    df = (
        df
        .explode("target_date")
        .reset_index(drop=True)
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    ).dt.normalize()

    df["forecast_origin"] = (
        df["target_date"]
        - pd.Timedelta(days=1)
    )

    df["known_at_forecast"] = (
        df["application_date"]
        <= df["forecast_origin"]
    )

    df = df.loc[
        df["known_at_forecast"]
    ].copy()

    return df


# ============================================================
# Geography
# ============================================================

def explode_event_neighborhoods(
    event_days: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explode semicolon-delimited Special Events neighborhood values.
    """
    df = event_days.copy()

    df["event_neighborhood"] = (
        df["event_location_neighborhood"]
        .str.split(";")
    )

    df = (
        df
        .explode("event_neighborhood")
        .reset_index(drop=True)
    )

    df["event_neighborhood"] = (
        df["event_neighborhood"]
        .astype("string")
        .str.strip()
    )

    df = df.loc[
        df["event_neighborhood"].notna()
        & df["event_neighborhood"].ne("")
    ].copy()

    return df


def map_to_spd_neighborhoods(
    event_days: pd.DataFrame,
    crosswalk: dict[str, list[str]] = SPECIAL_EVENTS_TO_SPD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map Special Events neighborhoods to SPD dispatch neighborhoods.

    Returns
    -------
    mapped:
        One row per unique permit-date-SPD neighborhood.

    unmapped:
        Summary of source neighborhoods with no crosswalk entry.
    """
    df = event_days.copy()

    df["neighborhood"] = (
        df["event_neighborhood"]
        .map(crosswalk)
    )

    unmapped = (
        df.loc[
            df["neighborhood"].isna(),
            "event_neighborhood",
        ]
        .value_counts()
        .rename_axis("event_neighborhood")
        .reset_index(name="permit_day_rows")
    )

    mapped = (
        df.loc[
            df["neighborhood"].notna()
        ]
        .explode("neighborhood")
        .reset_index(drop=True)
    )

    key_cols = [
        "event_id",
        "target_date",
        "neighborhood",
    ]

    # Preserve source geography provenance.
    source_geography = (
        mapped
        .groupby(
            key_cols,
            as_index=False,
        )
        .agg(
            source_event_neighborhoods=(
                "event_neighborhood",
                lambda values: ";".join(
                    sorted(
                        set(
                            values.dropna()
                        )
                    )
                ),
            )
        )
    )

    # Many-to-many mappings can produce the same SPD destination
    # through multiple source labels. Collapse those duplicate paths.
    mapped = (
        mapped
        .drop_duplicates(
            subset=key_cols,
            keep="first",
        )
        .drop(
            columns="event_neighborhood"
        )
        .merge(
            source_geography,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    )

    duplicate_count = (
        mapped
        .duplicated(
            subset=key_cols
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate permit-date-SPD neighborhood rows remain "
            f"after crosswalk collapse: {duplicate_count}"
        )

    return mapped, unmapped


# ============================================================
# Attendance features
# ============================================================

def add_attendance_features(
    mapped_event_days: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add attendance features after final SPD geography is known.
    """
    df = mapped_event_days.copy()

    df["n_dispatch_neighborhoods"] = (
        df
        .groupby(
            [
                "event_id",
                "target_date",
            ]
        )["neighborhood"]
        .transform("nunique")
    )

    df["attendance_known"] = (
        df["attendance"].notna()
    ).astype("int8")

    df["attendance_missing"] = (
        df["attendance"].isna()
    ).astype("int8")

    df["split_attendance"] = np.where(
        df["attendance"].notna(),
        (
            df["attendance"]
            / df["n_dispatch_neighborhoods"]
        ),
        np.nan,
    )

    df["log_split_attendance"] = np.where(
        df["split_attendance"].notna(),
        np.log1p(
            df["split_attendance"]
        ),
        np.nan,
    )

    return df


# ============================================================
# Aggregation
# ============================================================

def aggregate_permit_features(
    mapped_event_days: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate mapped permit-day records to:
        target_date x SPD neighborhood
    """
    features = (
        mapped_event_days
        .groupby(
            [
                "target_date",
                "neighborhood",
            ],
            as_index=False,
        )
        .agg(
            se_permit_count=(
                "event_id",
                "nunique",
            ),

            se_attendance_known_count=(
                "attendance_known",
                "sum",
            ),

            se_attendance_missing_count=(
                "attendance_missing",
                "sum",
            ),

            se_total_attendance_known=(
                "attendance",
                "sum",
            ),

            se_split_attendance_sum=(
                "split_attendance",
                "sum",
            ),

            se_max_attendance_known=(
                "attendance",
                "max",
            ),

            se_log_split_attendance_sum=(
                "log_split_attendance",
                "sum",
            ),
        )
    )

    return features


# ============================================================
# Full panel
# ============================================================

def build_complete_permit_panel(
    permit_features: pd.DataFrame,
    target_panel: pd.DataFrame,
    coverage_start: pd.Timestamp = PERMIT_COVERAGE_START,
    coverage_end: pd.Timestamp = PERMIT_COVERAGE_END,
) -> pd.DataFrame:
    """
    Build a complete date x neighborhood permit feature panel.

    Zeros are created ONLY inside the trusted permit-data
    coverage window.
    """
    neighborhoods = (
        target_panel.loc[
            target_panel["neighborhood"].notna()
            & target_panel["neighborhood"].ne("NULL"),
            "neighborhood",
        ]
        .drop_duplicates()
        .sort_values()
    )

    dates = pd.date_range(
        coverage_start,
        coverage_end,
        freq="D",
    )

    complete_index = pd.MultiIndex.from_product(
        [
            dates,
            neighborhoods,
        ],
        names=[
            "target_date",
            "neighborhood",
        ],
    )

    panel = (
        permit_features
        .set_index(
            [
                "target_date",
                "neighborhood",
            ]
        )
        .reindex(complete_index)
        .reset_index()
    )

    count_columns = [
        "se_permit_count",
        "se_attendance_known_count",
        "se_attendance_missing_count",
    ]

    sum_columns = [
        "se_total_attendance_known",
        "se_split_attendance_sum",
        "se_log_split_attendance_sum",
    ]

    panel[
        count_columns + sum_columns
    ] = (
        panel[
            count_columns + sum_columns
        ]
        .fillna(0)
    )

    # No permit means maximum known attendance is naturally zero.
    panel["se_max_attendance_known"] = (
        panel["se_max_attendance_known"]
        .fillna(0)
    )

    return panel


# ============================================================
# QA
# ============================================================

def validate_spd_neighborhoods(
    mapped: pd.DataFrame,
    target_panel: pd.DataFrame,
) -> None:
    """
    Ensure crosswalk output contains only valid target neighborhoods.
    """
    valid = set(
        target_panel.loc[
            target_panel["neighborhood"].notna()
            & target_panel["neighborhood"].ne("NULL"),
            "neighborhood",
        ]
    )

    observed = set(
        mapped["neighborhood"]
        .dropna()
        .unique()
    )

    invalid = observed - valid

    if invalid:
        raise ValueError(
            "Mapped neighborhoods absent from target panel: "
            f"{sorted(invalid)}"
        )


def build_qa_summary(
    raw_events: pd.DataFrame,
    modeling_events: pd.DataFrame,
    event_days: pd.DataFrame,
    mapped_event_days: pd.DataFrame,
    permit_features: pd.DataFrame,
    unmapped: pd.DataFrame,
) -> pd.Series:
    """
    Produce a compact reproducibility / QA summary.
    """
    return pd.Series(
        {
            "raw_permit_rows":
                len(raw_events),

            "modeling_permit_rows":
                len(modeling_events),

            "excluded_permit_rows":
                len(raw_events)
                - len(modeling_events),

            "event_day_rows_before_geography":
                len(event_days),

            "mapped_event_day_rows":
                len(mapped_event_days),

            "unique_modeled_permits":
                mapped_event_days[
                    "event_id"
                ].nunique(),

            "unmapped_source_neighborhoods":
                len(unmapped),

            "feature_rows_with_activity":
                len(permit_features),

            "feature_min_date":
                permit_features[
                    "target_date"
                ].min(),

            "feature_max_date":
                permit_features[
                    "target_date"
                ].max(),
        }
    )


# ============================================================
# Main builder
# ============================================================

def build_permitted_event_features(
    target_panel: pd.DataFrame,
    raw_events: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Run the complete research feature-engineering pipeline.
    """
    events = normalize_special_events(
        raw_events
    )

    modeling_events = select_modeling_events(
        events
    )

    event_days = expand_event_days(
        modeling_events
    )

    event_days = explode_event_neighborhoods(
        event_days
    )

    mapped_event_days, unmapped = (
        map_to_spd_neighborhoods(
            event_days
        )
    )

    validate_spd_neighborhoods(
        mapped_event_days,
        target_panel,
    )

    mapped_event_days = (
        add_attendance_features(
            mapped_event_days
        )
    )

    permit_features = (
        aggregate_permit_features(
            mapped_event_days
        )
    )

    permit_panel = (
        build_complete_permit_panel(
            permit_features,
            target_panel,
        )
    )

    qa_summary = build_qa_summary(
        raw_events=events,
        modeling_events=modeling_events,
        event_days=event_days,
        mapped_event_days=mapped_event_days,
        permit_features=permit_features,
        unmapped=unmapped,
    )

    return {
        "events": events,
        "modeling_events": modeling_events,
        "event_days": event_days,
        "mapped_event_days": mapped_event_days,
        "permit_features": permit_features,
        "permit_panel": permit_panel,
        "unmapped": unmapped,
        "qa_summary": qa_summary,
    }


# ============================================================
# Saving
# ============================================================

def save_permitted_event_outputs(
    outputs: dict,
    output_dir: str | Path,
) -> None:
    """
    Save research artifacts to disk.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs["events"].to_parquet(
        output_dir / "special_events_normalized.parquet",
        index=False,
    )

    outputs["modeling_events"].to_parquet(
        output_dir / "special_events_modeling_records.parquet",
        index=False,
    )

    outputs["mapped_event_days"].to_parquet(
        output_dir / "special_events_mapped_event_days.parquet",
        index=False,
    )

    outputs["permit_features"].to_parquet(
        output_dir / "special_events_sparse_features.parquet",
        index=False,
    )

    outputs["permit_panel"].to_parquet(
        output_dir / "special_events_feature_panel.parquet",
        index=False,
    )

    outputs["unmapped"].to_csv(
        output_dir / "special_events_unmapped_neighborhoods.csv",
        index=False,
    )

    outputs["qa_summary"].to_csv(
        output_dir / "special_events_qa_summary.csv",
        header=["value"],
    )