from typing import Any

import pandas as pd


SPD_CALL_COLUMNS = [
    "cad_event_number",
    "cad_event_original_time_queued",
    "cad_event_arrived_time",
    "cad_event_clearance_description",
    "call_sign_dispatch_id",
    "call_type",
    "priority",
    "initial_call_type",
    "final_call_type",
    "cad_event_response_category",
    "dispatch_precinct",
    "dispatch_sector",
    "dispatch_beat",
    "dispatch_neighborhood",
    "dispatch_latitude",
    "dispatch_longitude",
    "count_of_officers",
    "event_group",
]


def spd_calls_to_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    if not isinstance(records, list):
        raise ValueError("Top-level JSON is not a list")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Not all items in JSON object are dictionaries")
    
    df = pd.DataFrame.from_records(records)

    for column in SPD_CALL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    numeric_columns = [
        "priority",
        "dispatch_latitude",
        "dispatch_longitude",
        "count_of_officers",
    ]

    date_columns = [
        "cad_event_original_time_queued",
        "cad_event_arrived_time",
    ]

    text_columns = [
        "cad_event_number",
        "cad_event_clearance_description",
        "call_sign_dispatch_id",
        "call_type",
        "initial_call_type",
        "final_call_type",
        "cad_event_response_category",
        "dispatch_precinct",
        "dispatch_sector",
        "dispatch_beat",
        "dispatch_neighborhood",
        "event_group",
    ]

    cat_columns = [
        "cad_event_clearance_description",
        "call_type",
        "initial_call_type",
        "final_call_type",
        "cad_event_response_category",
        "dispatch_precinct",
        "dispatch_sector",
        "dispatch_beat",
        "dispatch_neighborhood",
        "event_group",
    ]

    cleaned_df = df.copy()
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    cleaned_df[date_columns] = cleaned_df[date_columns].apply(pd.to_datetime, errors='coerce')

    cleaned_df[text_columns] = cleaned_df[text_columns].apply(
        lambda column: column.str.strip())
    cleaned_df[cat_columns] = cleaned_df[cat_columns].apply(
        lambda column: column.str.lower()
    )

    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df = cleaned_df.reindex(columns=SPD_CALL_COLUMNS)
    return cleaned_df


if __name__ == "__main__":
    from spd_client import fetch_spd_call_page

    records = fetch_spd_call_page(
        start_date="2025-01-01",
        limit=25,
    )

    df = spd_calls_to_dataframe(records)

    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())


