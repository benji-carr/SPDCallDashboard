import pandas as pd


POINT_MAP_COLUMNS = [
    "cad_event_number",
    "cad_event_original_time_queued",
    "call_sign_dispatch_id",
    "event_group",
    "priority",
    "dispatch_neighborhood",
    "dispatch_precinct",
    "dispatch_sector",
    "dispatch_beat",
    "dispatch_latitude",
    "dispatch_longitude",
]


def prepare_spd_point_map_data(
    df: pd.DataFrame,
    max_points: int | None = None,
) -> pd.DataFrame:
    if not all(col in df.columns for col in POINT_MAP_COLUMNS):
        missing_columns = [col for col in POINT_MAP_COLUMNS if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    if max_points is not None:
        if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
            raise ValueError("max_points must be a positive integer")
    
    plot_df = df[POINT_MAP_COLUMNS].copy()

    plot_df["dispatch_latitude"] = pd.to_numeric(
        plot_df["dispatch_latitude"],
        errors="coerce",
    )

    plot_df["dispatch_longitude"] = pd.to_numeric(
        plot_df["dispatch_longitude"],
        errors="coerce",
    )

    plot_df["cad_event_original_time_queued"] = pd.to_datetime(
        plot_df["cad_event_original_time_queued"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=["dispatch_latitude", "dispatch_longitude"]
    )

    plot_df = plot_df.sort_values(
        "cad_event_original_time_queued",
        ascending=False,
    )

    if max_points is not None:
        plot_df = plot_df.head(max_points)

    return plot_df.reset_index(drop=True)


def aggregate_spd_calls_by_neighborhood(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "dispatch_neighborhood",
        "cad_event_number",
    ]
    if not all(col in df.columns for col in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    plot_df = df[required_columns].copy()
    plot_df = plot_df.dropna(subset=['dispatch_neighborhood'])
    summary_df = (
        plot_df
        .groupby("dispatch_neighborhood")
        .agg(
            call_count=("cad_event_number", "size"),
            unique_events=("cad_event_number", "nunique"),
        )
        .reset_index()
        .sort_values("call_count", ascending=False)
        .reset_index(drop=True)
    )

    return summary_df

if __name__ == "__main__":
    from spd_snapshot import load_spd_call_snapshot

    df, metadata = load_spd_call_snapshot("data/processed")

    point_df = prepare_spd_point_map_data(df, max_points=100)
    neighborhood_df = aggregate_spd_calls_by_neighborhood(df)

    print(point_df.head())
    print(point_df.dtypes)
    print(f"\nMappable rows: {len(point_df)}")

    print("\nNeighborhood aggregation:")
    print(neighborhood_df.head(20))