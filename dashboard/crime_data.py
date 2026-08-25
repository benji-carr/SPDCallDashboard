from typing import Any

import pandas as pd

from dashboard.crime_query import (
    CRIME_COLUMNS,
)


def crime_records_to_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    if not isinstance(records, list):
        raise ValueError("Top-level JSON is not a list")

    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Not all items in JSON object are dictionaries")

    df = pd.DataFrame.from_records(records)

    for column in CRIME_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    numeric_columns = [
        "latitude",
        "longitude",
    ]

    date_columns = [
        "report_date_time",
        "offense_date",
    ]

    text_columns = [
        "report_number",
        "offense_id",
        "block_address",
        "nibrs_offense_code",
        "census_block_2020",
    ]

    cat_columns = [
        "nibrs_group_a_b",
        "nibrs_crime_against_category",
        "offense_sub_category",
        "shooting_type_group",
        "beat",
        "precinct",
        "sector",
        "neighborhood",
        "reporting_area",
        "offense_category",
        "nibrs_offense_code_description",
    ]

    cleaned_df = df.copy()

    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    cleaned_df[date_columns] = cleaned_df[date_columns].apply(
        pd.to_datetime,
        errors="coerce",
    )

    cleaned_df[text_columns] = cleaned_df[text_columns].apply(
        lambda column: column.astype("string").str.strip()
    )

    cleaned_df[cat_columns] = cleaned_df[cat_columns].apply(
        lambda column: column.astype("string").str.strip().str.lower()
    )

    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df = cleaned_df.reindex(columns=CRIME_COLUMNS)

    return cleaned_df


if __name__ == "__main__":
    from dashboard.crime_client import (
        fetch_crime_page,
    )

    records = fetch_crime_page(
        start_date="2025-01-01",
        limit=25,
    )

    df = crime_records_to_dataframe(records)

    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())
    print(f"Unique offense IDs: {df['offense_id'].nunique():,}")
    print(f"Total rows: {len(df):,}")
    print(f"Duplicate offense IDs: {df['offense_id'].duplicated().sum():,}")
