import logging
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta


from spd_service import load_spd_call_dataset
from spd_snapshot import save_spd_call_snapshot
from spd_snapshot import load_spd_call_snapshot


DEFAULT_START_DATE = "2025-07-04"
TIME_COLUMN = "cad_event_original_time_queued"
DEDUPLICATION_KEY = ["call_sign_dispatch_id"]
DEFAULT_PAGE_SIZE = 5000
DEFAULT_MAX_PAGES = None
DEFAULT_TIMEOUT = 20.0
DEFAULT_ROLLING_WINDOW_DAYS = 365
DEFAULT_OVERLAP_DAYS = 14
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed")


def incremental_refresh_spd_call_snapshot(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[Path, Path]:
    output_directory = Path(output_directory)

    existing_df, metadata = load_spd_call_snapshot(output_directory)

    if TIME_COLUMN not in existing_df.columns:
        raise ValueError(f"Existing snapshot is missing {TIME_COLUMN}")

    missing_key_columns = [
        column
        for column in DEDUPLICATION_KEY
        if column not in existing_df.columns
    ]

    if missing_key_columns:
        raise ValueError(
            f"Existing snapshot is missing deduplication columns: {missing_key_columns}"
        )

    existing_df = existing_df.copy()

    existing_df[TIME_COLUMN] = pd.to_datetime(
        existing_df[TIME_COLUMN],
        errors="coerce",
    )

    latest_existing_timestamp = existing_df[TIME_COLUMN].max()

    if pd.isna(latest_existing_timestamp):
        raise ValueError("Existing snapshot has no valid timestamps")

    fetch_start_date = (
        latest_existing_timestamp.date() - timedelta(days=overlap_days)
    ).isoformat()

    logging.info(
        "Starting incremental SPD refresh from %s with overlap_days=%s",
        fetch_start_date,
        overlap_days,
    )

    new_df = load_spd_call_dataset(
        start_date=fetch_start_date,
        page_size=page_size,
        max_pages=None,
        timeout=timeout,
    )

    logging.info("Fetched %s recent SPD rows", len(new_df))

    combined_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True,
    )

    combined_df[TIME_COLUMN] = pd.to_datetime(
        combined_df[TIME_COLUMN],
        errors="coerce",
    )

    before_deduplication = len(combined_df)

    combined_df = combined_df.drop_duplicates(
        subset=DEDUPLICATION_KEY,
        keep="last",
    )

    logging.info(
        "Removed %s duplicate rows",
        before_deduplication - len(combined_df),
    )

    latest_combined_timestamp = combined_df[TIME_COLUMN].max()

    if pd.isna(latest_combined_timestamp):
        raise ValueError("Combined snapshot has no valid timestamps")

    cutoff_timestamp = latest_combined_timestamp - timedelta(
        days=rolling_window_days
    )

    combined_df = combined_df[
        combined_df[TIME_COLUMN] >= cutoff_timestamp
    ].copy()

    combined_df = combined_df.sort_values(
        TIME_COLUMN,
        ascending=True,
    ).reset_index(drop=True)

    logging.info(
        "Final rolling snapshot has %s rows from %s to %s",
        len(combined_df),
        combined_df[TIME_COLUMN].min(),
        combined_df[TIME_COLUMN].max(),
    )

    snapshot_path, metadata_path = save_spd_call_snapshot(
        combined_df,
        output_directory=output_directory,
        source_start_date=cutoff_timestamp.date().isoformat(),
    )

    logging.info("Saved SPD call snapshot to %s", snapshot_path)
    logging.info("Saved SPD call metadata to %s", metadata_path)

    return snapshot_path, metadata_path



def full_refresh_spd_call_snapshot(
    start_date: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = DEFAULT_MAX_PAGES,
    timeout: float = DEFAULT_TIMEOUT,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    logging.info(
        "Starting full SPD call snapshot refresh: start_date=%s",
        start_date,
    )

    df = load_spd_call_dataset(
        start_date=start_date,
        page_size=page_size,
        max_pages=max_pages,
        timeout=timeout,
    )

    snapshot_path, metadata_path = save_spd_call_snapshot(
        df,
        output_directory=output_directory,
        source_start_date=start_date,
    )

    logging.info("Saved full SPD snapshot with %s rows", len(df))

    return snapshot_path, metadata_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    incremental_refresh_spd_call_snapshot()