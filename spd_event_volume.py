import requests
import pandas as pd
from datetime import date, datetime


# -----------------------------
# Config
# -----------------------------

SPD_CALL_ENDPOINT = "https://data.seattle.gov/resource/33kz-ixgy.json"

EVENT_ID_COL = "cad_event_number"
EVENT_TIME_COL = "cad_event_original_time_queued"

#
ALLOWED_GROUP_BYS = {
    None,
    "event_group",
    "call_type",
    "initial_call_type",
    "final_call_type",
    "dispatch_neighborhood",
}


# -----------------------------
# Small helpers
# -----------------------------

def _format_socrata_datetime(value):
    """
    Converts a string/date/datetime into Socrata timestamp format.

    Examples:
        "2024-01-01" -> "2024-01-01T00:00:00.000"
        "2024-01-01T13:30:00" -> "2024-01-01T13:30:00.000"
    """
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.000")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%dT00:00:00.000")

    value = str(value)

    if "T" not in value:
        value = f"{value}T00:00:00.000"
    elif "." not in value:
        value = f"{value}.000"

    return value


def _floor_to_period_start(ts, grain):
    ts = pd.Timestamp(ts)

    if grain == "month":
        return ts.to_period("M").start_time

    if grain == "week":
        # Weeks start on Monday and end on Sunday
        return ts.to_period("W-SUN").start_time

    raise ValueError("grain must be 'month' or 'week'")


def _add_one_period(ts, grain):
    ts = pd.Timestamp(ts)

    if grain == "month":
        return ts + pd.DateOffset(months=1)

    if grain == "week":
        return ts + pd.DateOffset(weeks=1)

    raise ValueError("grain must be 'month' or 'week'")


def _iter_period_windows(start_ts, end_ts, grain):
    """
    Yields one month/week window at a time.

    Each yield gives:
        period_start, query_start, query_end
    """
    period_start = _floor_to_period_start(start_ts, grain)

    while period_start < end_ts:
        period_end = _add_one_period(period_start, grain)

        query_start = max(period_start, start_ts)
        query_end = min(period_end, end_ts)

        if query_start < query_end:
            yield period_start, query_start, query_end

        period_start = period_end


# -----------------------------
# Socrata queries
# -----------------------------

def fetch_spd_time_bounds(timeout=30.0):
    """
    Gets the full available date range of the dataset.
    This does not load raw rows.
    """
    params = {
        "$select": f"min({EVENT_TIME_COL}) as min_time, max({EVENT_TIME_COL}) as max_time",
        "$where": f"{EVENT_TIME_COL} IS NOT NULL AND {EVENT_ID_COL} IS NOT NULL",
    }

    response = requests.get(
        SPD_CALL_ENDPOINT,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("Expected one-row response for dataset time bounds.")

    return data[0]


def fetch_distinct_events_for_window(
    start_datetime,
    end_datetime,
    group_by=None,
    limit=50000,
    offset=0,
    timeout=30.0,
):
    """
    Fetches skinny distinct-event records for one bounded time window.

    If group_by is None:
        returns distinct cad_event_number

    If group_by is provided:
        returns distinct cad_event_number, group_by
    """
    if group_by not in ALLOWED_GROUP_BYS:
        raise ValueError(f"group_by must be one of {ALLOWED_GROUP_BYS}. Got {group_by}")

    start_ts = _format_socrata_datetime(start_datetime)
    end_ts = _format_socrata_datetime(end_datetime)

    select_cols = [EVENT_ID_COL]

    if group_by is not None:
        select_cols.append(group_by)

    params = {
        "$select": "distinct " + ", ".join(select_cols),
        "$where": (
            f"{EVENT_TIME_COL} >= '{start_ts}' "
            f"AND {EVENT_TIME_COL} < '{end_ts}' "
            f"AND {EVENT_ID_COL} IS NOT NULL"
        ),
        "$order": ", ".join(select_cols),
        "$limit": limit,
        "$offset": offset,
    }

    response = requests.get(
        SPD_CALL_ENDPOINT,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError("Expected Socrata response to be a list.")

    if not all(isinstance(row, dict) for row in data):
        raise ValueError("Expected every row to be a dictionary.")

    return data


def fetch_all_distinct_events_for_window(
    start_datetime,
    end_datetime,
    group_by=None,
    page_size=50000,
    timeout=30.0,
):
    """
    Pages through one month/week window until all distinct event records are collected.
    """
    all_records = []
    offset = 0

    while True:
        page = fetch_distinct_events_for_window(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            group_by=group_by,
            limit=page_size,
            offset=offset,
            timeout=timeout,
        )

        all_records.extend(page)

        if len(page) < page_size:
            break

        offset += page_size

    return all_records


# -----------------------------
# Counting logic
# -----------------------------

def _count_event_records(records, period_start, group_by=None):
    """
    Converts skinny distinct-event records into one compact count table.
    """
    if len(records) == 0:
        if group_by is None:
            return pd.DataFrame({
                "period_start": [period_start],
                "event_count": [0],
            })

        return pd.DataFrame(columns=["period_start", group_by, "event_count"])

    df = pd.DataFrame(records)

    if EVENT_ID_COL not in df.columns:
        df[EVENT_ID_COL] = pd.NA

    df = df.dropna(subset=[EVENT_ID_COL])

    if group_by is None:
        return pd.DataFrame({
            "period_start": [period_start],
            "event_count": [df[EVENT_ID_COL].nunique()],
        })

    if group_by not in df.columns:
        df[group_by] = pd.NA

    df[group_by] = (
        df[group_by]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("unknown")
    )

    out = (
        df.groupby(group_by, dropna=False)[EVENT_ID_COL]
        .nunique()
        .reset_index(name="event_count")
    )

    out.insert(0, "period_start", period_start)

    return out


# -----------------------------
# Main user-facing function
# -----------------------------

def load_spd_event_volume(
    grain="month",
    group_by=None,
    start_date=None,
    end_date=None,
    page_size=50000,
    timeout=30.0,
    verbose=True,
):
    """
    Loads compact SPD call-volume data using unique CAD events.

    This counts unique:
        cad_event_number

    It does NOT count:
        call_sign_dispatch_id

    Parameters
    ----------
    grain:
        "month" or "week"

    group_by:
        None,
        "event_group",
        "call_type",
        "initial_call_type",
        or "final_call_type"

    start_date:
        Optional inclusive start date.
        Example: "2020-01-01"

    end_date:
        Optional exclusive end date.
        Example: "2025-01-01" means include records before 2025-01-01.

    page_size:
        Socrata page size. 50000 is a good upper value.

    timeout:
        Request timeout in seconds.

    verbose:
        If True, prints progress period by period.

    Returns
    -------
    pandas.DataFrame
    """
    if grain not in {"month", "week"}:
        raise ValueError("grain must be 'month' or 'week'")

    if group_by not in ALLOWED_GROUP_BYS:
        raise ValueError(f"group_by must be one of {ALLOWED_GROUP_BYS}. Got {group_by}")

    # Resolve full dataset range only if needed
    if start_date is None or end_date is None:
        bounds = fetch_spd_time_bounds(timeout=timeout)

    if start_date is None:
        start_ts = pd.Timestamp(bounds["min_time"])
    else:
        start_ts = pd.Timestamp(start_date)

    if end_date is None:
        max_ts = pd.Timestamp(bounds["max_time"])

        # Include the full final month/week containing the latest record
        end_ts = _add_one_period(
            _floor_to_period_start(max_ts, grain),
            grain,
        )
    else:
        end_ts = pd.Timestamp(end_date)

    frames = []

    for period_start, query_start, query_end in _iter_period_windows(
        start_ts=start_ts,
        end_ts=end_ts,
        grain=grain,
    ):
        if verbose:
            print(f"Fetching {grain}: {period_start.date()}")

        records = fetch_all_distinct_events_for_window(
            start_datetime=query_start,
            end_datetime=query_end,
            group_by=group_by,
            page_size=page_size,
            timeout=timeout,
        )

        counts = _count_event_records(
            records=records,
            period_start=period_start,
            group_by=group_by,
        )

        frames.append(counts)

    if len(frames) == 0:
        if group_by is None:
            return pd.DataFrame(columns=["period_start", "event_count"])

        return pd.DataFrame(columns=["period_start", group_by, "event_count"])

    result = pd.concat(frames, ignore_index=True)

    result["period_start"] = pd.to_datetime(result["period_start"])

    if grain == "week":
        iso = result["period_start"].dt.isocalendar()
        result.insert(1, "year", iso["year"].astype(int))
        result.insert(2, "week_of_year", iso["week"].astype(int))

    sort_cols = ["period_start"]

    if group_by is not None:
        sort_cols.append(group_by)

    result = result.sort_values(sort_cols).reset_index(drop=True)

    return result