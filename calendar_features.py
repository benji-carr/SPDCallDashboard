# calendar_features.py

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# All 19 candidate calendar features
# ---------------------------------------------------------------------

CALENDAR_FEATURES = [
    # Cyclical calendar position
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "week_of_year_sin",
    "week_of_year_cos",

    # Generic calendar effects
    "is_weekend",
    "is_federal_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",

    # Specific days
    "is_new_years_eve",
    "is_new_years_day",
    "is_july_fourth",
    "is_halloween",
    "is_thanksgiving",
    "is_christmas_eve",
    "is_christmas_day",

    # Proximity to federal holidays
    "days_until_holiday",
    "days_since_holiday",
]


# Rather than immediately put all 19 into SARIMAX,
# we'll test coherent subsets.
FEATURE_SETS = {
    "seasonal_cycle": [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "week_of_year_sin",
        "week_of_year_cos",
        "is_weekend",
    ],

    "holiday_core": [
        "is_federal_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "days_until_holiday",
        "days_since_holiday",
    ],

    "special_days": [
        "is_new_years_eve",
        "is_new_years_day",
        "is_july_fourth",
        "is_halloween",
        "is_thanksgiving",
        "is_christmas_eve",
        "is_christmas_day",
    ],

    "calendar_compact": [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "week_of_year_sin",
        "week_of_year_cos",
        "is_weekend",
        "is_federal_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "is_new_years_eve",
        "is_new_years_day",
        "is_july_fourth",
        "is_halloween",
        "is_thanksgiving",
        "is_christmas_eve",
        "is_christmas_day",
    ],

    "weekly_cycle": [
        "day_of_week_sin",
        "day_of_week_cos",
    ],

    "annual_cycle": [
        "week_of_year_sin",
        "week_of_year_cos",
    ],

    "monthly_cycle": [
        "month_sin",
        "month_cos",
    ],

    "weekend_only": [
        "is_weekend",
    ],

    "weekly_annual_cycle": [
        "day_of_week_sin",
        "day_of_week_cos",
        "week_of_year_sin",
        "week_of_year_cos",
    ],


    "full_19": CALENDAR_FEATURES,
}


def _holiday_distances(
    dates: pd.DatetimeIndex,
    holiday_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return number of days until and since the nearest
    federal holiday.

    A holiday itself receives 0 for both.
    """

    dates_np = dates.values.astype("datetime64[D]")
    holidays_np = holiday_dates.values.astype("datetime64[D]")

    # Next holiday, including today.
    next_positions = np.searchsorted(
        holidays_np,
        dates_np,
        side="left",
    )

    next_positions = np.clip(
        next_positions,
        0,
        len(holidays_np) - 1,
    )

    next_dates = holidays_np[next_positions]

    days_until = (
        next_dates - dates_np
    ).astype("timedelta64[D]").astype(int)

    # Previous holiday, including today.
    previous_positions = (
        np.searchsorted(
            holidays_np,
            dates_np,
            side="right",
        )
        - 1
    )

    previous_positions = np.clip(
        previous_positions,
        0,
        len(holidays_np) - 1,
    )

    previous_dates = holidays_np[
        previous_positions
    ]

    days_since = (
        dates_np - previous_dates
    ).astype("timedelta64[D]").astype(int)

    return days_until, days_since


def build_calendar_features(
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Build leakage-safe calendar features for every date
    from start_date through end_date, inclusive.
    """

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "target_date": dates,
        }
    )

    # --------------------------------------------------------------
    # Cyclical encodings
    # --------------------------------------------------------------

    day_of_week = df[
        "target_date"
    ].dt.dayofweek

    month = df[
        "target_date"
    ].dt.month

    week_of_year = (
        df["target_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (month - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    df["week_of_year_sin"] = np.sin(
        2 * np.pi * (week_of_year - 1) / 52.1775
    )

    df["week_of_year_cos"] = np.cos(
        2 * np.pi * (week_of_year - 1) / 52.1775
    )

    df["is_weekend"] = (
        day_of_week >= 5
    ).astype("int8")

    # --------------------------------------------------------------
    # Federal holidays
    # --------------------------------------------------------------

    # Add one year on either side so nearest-holiday
    # distances work at the boundaries.
    years = range(
        start_date.year - 1,
        end_date.year + 2,
    )

    us_holidays = holidays.US(
        years=years,
        observed=True,
    )

    holiday_dates = pd.DatetimeIndex(
        sorted(
            pd.Timestamp(date).normalize()
            for date in us_holidays.keys()
        )
    )

    holiday_set = set(
        holiday_dates
    )

    df["is_federal_holiday"] = (
        df["target_date"]
        .isin(holiday_set)
        .astype("int8")
    )

    df["is_day_before_holiday"] = (
        (
            df["target_date"]
            + pd.Timedelta(days=1)
        )
        .isin(holiday_set)
        .astype("int8")
    )

    df["is_day_after_holiday"] = (
        (
            df["target_date"]
            - pd.Timedelta(days=1)
        )
        .isin(holiday_set)
        .astype("int8")
    )

    # --------------------------------------------------------------
    # Specific calendar days
    # --------------------------------------------------------------

    date = df["target_date"]

    df["is_new_years_eve"] = (
        (date.dt.month == 12)
        & (date.dt.day == 31)
    ).astype("int8")

    df["is_new_years_day"] = (
        (date.dt.month == 1)
        & (date.dt.day == 1)
    ).astype("int8")

    df["is_july_fourth"] = (
        (date.dt.month == 7)
        & (date.dt.day == 4)
    ).astype("int8")

    df["is_halloween"] = (
        (date.dt.month == 10)
        & (date.dt.day == 31)
    ).astype("int8")

    # Fourth Thursday of November.
    df["is_thanksgiving"] = (
        (date.dt.month == 11)
        & (date.dt.dayofweek == 3)
        & (date.dt.day >= 22)
        & (date.dt.day <= 28)
    ).astype("int8")

    df["is_christmas_eve"] = (
        (date.dt.month == 12)
        & (date.dt.day == 24)
    ).astype("int8")

    df["is_christmas_day"] = (
        (date.dt.month == 12)
        & (date.dt.day == 25)
    ).astype("int8")

    # --------------------------------------------------------------
    # Holiday proximity
    # --------------------------------------------------------------

    days_until, days_since = (
        _holiday_distances(
            dates,
            holiday_dates,
        )
    )

    df["days_until_holiday"] = days_until.astype(
        "int16"
    )

    df["days_since_holiday"] = days_since.astype(
        "int16"
    )

    # --------------------------------------------------------------
    # QA
    # --------------------------------------------------------------

    assert df["target_date"].is_unique

    assert not df[
        CALENDAR_FEATURES
    ].isna().any().any()

    return df


def build_calendar_features_from_panel(
    target_panel: pd.DataFrame,
) -> pd.DataFrame:

    return build_calendar_features(
        start_date=target_panel[
            "target_date"
        ].min(),

        end_date=target_panel[
            "target_date"
        ].max(),
    )


def save_calendar_features(
    target_panel: pd.DataFrame,
    output_path,
) -> pd.DataFrame:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = (
        build_calendar_features_from_panel(
            target_panel
        )
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    return features