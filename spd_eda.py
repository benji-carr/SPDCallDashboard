from typing import Any

import pandas as pd


def summarize_spd_calls(df: pd.DataFrame) -> dict[str, Any]:
    required_columns = [
        "cad_event_number",
        "cad_event_original_time_queued",
        "event_group",
        "priority",
        "dispatch_precinct",
        "dispatch_neighborhood",
        "dispatch_latitude",
        "dispatch_longitude",
        "count_of_officers",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {missing_columns}"
        )
    
    result = {}
    result['total_calls'] = len(df)
    result['unique_events'] = df['cad_event_number'].nunique()
    
    if df['cad_event_original_time_queued'].isnull().all():
        result['date_min'] = None
        result['date_max'] = None
    else:
        result['date_min'] = df['cad_event_original_time_queued'].min()
        result['date_max'] = df['cad_event_original_time_queued'].max()
    
    result['unique_event_groups'] = df['event_group'].nunique()
    result['unique_neighborhoods'] = df['dispatch_neighborhood'].nunique()
    
    if df['priority'].isnull().all():
        result['median_priority'] = None
    else:    
        result['median_priority'] = float(df['priority'].median())

    if df['count_of_officers'].isnull().all():
        result['median_officers'] = None
    else:    
        result['median_officers'] = float(df['count_of_officers'].median())
    

    result['event_group_counts'] = df['event_group'].value_counts().to_dict()
    result['priority_counts'] = df['priority'].value_counts().to_dict()
    result['precinct_counts'] = df['dispatch_precinct'].value_counts().to_dict()
    result['missing_values'] = df.isna().sum().to_dict()
    result['mappable_records'] = int(df[["dispatch_latitude", "dispatch_longitude"]].notna().all(axis=1).sum())

    return result

if __name__ == "__main__":
    from spd_snapshot import load_spd_call_snapshot

    df, metadata = load_spd_call_snapshot("data/processed")

    summary = summarize_spd_calls(df)

    for key, value in summary.items():
        print(f"{key}: {value}")