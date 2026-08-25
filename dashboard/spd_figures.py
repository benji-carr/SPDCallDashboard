import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def make_monthly_call_chart(df: pd.DataFrame) -> Figure:
    if "cad_event_original_time_queued" not in df.columns:
        raise ValueError("cad_event_original_time_queued column is missing from the DataFrame")
    
    plot_df = df[["cad_event_original_time_queued"]].copy()

    plot_df["cad_event_original_time_queued"] = pd.to_datetime(
        plot_df["cad_event_original_time_queued"],
        errors="coerce",
        )
    
    plot_df = plot_df.dropna(subset=["cad_event_original_time_queued"])

    plot_df["month"] = (
        plot_df["cad_event_original_time_queued"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    plot_df = (
        plot_df
        .groupby("month")
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )
    return px.line(
        plot_df,
        x="month",
        y="count",
        title="SPD Calls by Month",
        labels={
            "month": "Month",
            "count": "Calls",
            },
            )


def make_hourly_call_chart(df: pd.DataFrame) -> Figure:
    if "cad_event_original_time_queued" not in df.columns:
        raise ValueError("cad_event_original_time_queued column is missing from the DataFrame")
    
    plot_df = df[["cad_event_original_time_queued"]].copy()

    plot_df["cad_event_original_time_queued"] = pd.to_datetime(
        plot_df["cad_event_original_time_queued"],
        errors="coerce",
        )
    plot_df = plot_df.dropna(subset=["cad_event_original_time_queued"])

    plot_df["hour"] = plot_df["cad_event_original_time_queued"].dt.hour

    hourly_counts = (
        plot_df["hour"]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .rename_axis("hour")
        .reset_index(name="count")
        )
    
    return px.bar(
        hourly_counts,
        x="hour",
        y="count",
        title="SPD Calls by Hour of Day",
        labels={
            "hour": "Hour",
            "count": "Calls",
            },
            )


def make_event_group_bar_chart(
    df: pd.DataFrame,
    top_n: int = 10,
) -> Figure:
    if "event_group" not in df.columns:
        raise ValueError("event_group column is missing from the DataFrame")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    plot_df = (
        df["event_group"]
        .value_counts()
        .head(top_n)
        .rename_axis("event_group")
        .reset_index(name="count")
        )
    return px.bar(
        plot_df,
        x="event_group",
        y="count",
        title="Top SPD Event Groups",
        labels={
            "event_group": "Event Group",
            "count": "Calls",
            },
            )


def make_priority_bar_chart(df: pd.DataFrame) -> Figure:
    if "priority" not in df.columns:
        raise ValueError("priority column is missing from the DataFrame")
    
    plot_df = (
        df[["priority"]]
        .dropna()
        .copy()
        )

    plot_df = (
        plot_df["priority"]
        .value_counts()
        .rename_axis("priority")
        .reset_index(name="count")
        .sort_values("priority")
        )
    
    return px.bar(
        plot_df,
        x="priority",
        y="count",
        title="SPD Calls by Priority",
        labels={
            "priority": "Priority",
            "count": "Calls",
            },
            )

if __name__ == "__main__":
    from spd_snapshot import load_spd_call_snapshot

    df, metadata = load_spd_call_snapshot("data/processed")

    fig = make_monthly_call_chart(df)
    fig.show()

    fig = make_hourly_call_chart(df)
    fig.show()

    fig = make_event_group_bar_chart(df, top_n=10)
    fig.show()

    fig = make_priority_bar_chart(df)
    fig.show()