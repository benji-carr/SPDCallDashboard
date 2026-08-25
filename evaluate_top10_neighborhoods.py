from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOP_K = 10


def prepare_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "fold",
        "target_date",
        "neighborhood",
        "actual",
        "prediction",
    }

    missing = (
        required
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Predictions are missing required columns: "
            f"{sorted(missing)}"
        )

    df = predictions.copy()

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    ).dt.normalize()

    for column in [
        "actual",
        "prediction",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df.duplicated(
        subset=[
            "target_date",
            "neighborhood",
        ]
    ).any():
        raise ValueError(
            "Duplicate target-date/neighborhood rows found."
        )

    if not np.isfinite(
        df[
            [
                "actual",
                "prediction",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite actual or predicted values found."
        )

    return df


def rank_day(
    day: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign deterministic actual and predicted ranks.

    Neighborhood name is used as a tie-breaker so results are
    fully reproducible.
    """

    actual_ranked = (
        day
        .sort_values(
            [
                "actual",
                "neighborhood",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    actual_ranked[
        "actual_rank"
    ] = (
        np.arange(
            1,
            len(actual_ranked) + 1,
        )
    )

    predicted_ranked = (
        day
        .sort_values(
            [
                "prediction",
                "neighborhood",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    predicted_ranked[
        "predicted_rank"
    ] = (
        np.arange(
            1,
            len(predicted_ranked) + 1,
        )
    )

    ranks = (
        actual_ranked[
            [
                "neighborhood",
                "actual_rank",
            ]
        ]
        .merge(
            predicted_ranked[
                [
                    "neighborhood",
                    "predicted_rank",
                ]
            ],
            on="neighborhood",
            validate="one_to_one",
        )
    )

    return day.merge(
        ranks,
        on="neighborhood",
        validate="one_to_one",
    )


def evaluate_day(
    day: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict:

    ranked = rank_day(
        day
    )

    actual_top = ranked.loc[
        ranked[
            "actual_rank"
        ]
        <= top_k
    ].copy()

    predicted_top = ranked.loc[
        ranked[
            "predicted_rank"
        ]
        <= top_k
    ].copy()

    actual_set = set(
        actual_top[
            "neighborhood"
        ]
    )

    predicted_set = set(
        predicted_top[
            "neighborhood"
        ]
    )

    overlap = (
        actual_set
        & predicted_set
    )

    overlap_count = len(
        overlap
    )

    # ----------------------------------------------------------
    # Rank accuracy for the TRUE top-K neighborhoods.
    #
    # A neighborhood actually ranked #3 but predicted #14
    # receives an absolute rank error of 11.
    # ----------------------------------------------------------

    actual_top[
        "absolute_rank_error"
    ] = (
        actual_top[
            "predicted_rank"
        ]
        - actual_top[
            "actual_rank"
        ]
    ).abs()

    mean_top_rank_error = (
        actual_top[
            "absolute_rank_error"
        ].mean()
    )

    median_top_rank_error = (
        actual_top[
            "absolute_rank_error"
        ].median()
    )

    # ----------------------------------------------------------
    # How much of the maximum possible top-K call volume did the
    # predicted top-K capture?
    # ----------------------------------------------------------

    actual_top_volume = (
        actual_top[
            "actual"
        ].sum()
    )

    predicted_top_actual_volume = (
        predicted_top[
            "actual"
        ].sum()
    )

    if actual_top_volume > 0:
        volume_capture = (
            predicted_top_actual_volume
            / actual_top_volume
        )
    else:
        volume_capture = np.nan

    # ----------------------------------------------------------
    # Overall ranking correlation across every neighborhood.
    # ----------------------------------------------------------

    rank_correlation = (
        ranked[
            [
                "actual_rank",
                "predicted_rank",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    # ----------------------------------------------------------
    # Was there an actual-volume tie at positions K/K+1?
    # Useful QA because a cutoff tie makes membership somewhat
    # arbitrary.
    # ----------------------------------------------------------

    sorted_actual = (
        ranked[
            "actual"
        ]
        .sort_values(
            ascending=False
        )
        .to_numpy()
    )

    cutoff_tie = False

    if len(
        sorted_actual
    ) > top_k:
        cutoff_tie = bool(
            sorted_actual[
                top_k - 1
            ]
            == sorted_actual[
                top_k
            ]
        )

    return {
        "fold":
            day[
                "fold"
            ].iloc[0],

        "target_date":
            day[
                "target_date"
            ].iloc[0],

        "top_k":
            top_k,

        "overlap_count":
            overlap_count,

        "top_k_accuracy":
            overlap_count
            / top_k,

        "mean_top10_rank_error":
            float(
                mean_top_rank_error
            ),

        "median_top10_rank_error":
            float(
                median_top_rank_error
            ),

        "top10_volume_capture":
            float(
                volume_capture
            ),

        "overall_rank_correlation":
            float(
                rank_correlation
            ),

        "actual_cutoff_tie":
            cutoff_tie,
    }


def evaluate_top10_predictions(
    predictions: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:

    df = prepare_predictions(
        predictions
    )

    rows = []

    for (
        target_date,
        day,
    ) in df.groupby(
        "target_date",
        sort=True,
    ):

        rows.append(
            evaluate_day(
                day=day,
                top_k=top_k,
            )
        )

    return pd.DataFrame(
        rows
    )


def summarize_top10(
    daily_results: pd.DataFrame,
) -> pd.DataFrame:

    summary = {
        "n_days":
            len(
                daily_results
            ),

        "mean_correct_top10":
            daily_results[
                "overlap_count"
            ].mean(),

        "median_correct_top10":
            daily_results[
                "overlap_count"
            ].median(),

        "mean_top10_accuracy_pct":
            (
                100
                * daily_results[
                    "top_k_accuracy"
                ].mean()
            ),

        "pct_days_10_of_10":
            (
                100
                * (
                    daily_results[
                        "overlap_count"
                    ]
                    >= 10
                ).mean()
            ),

        "pct_days_at_least_9":
            (
                100
                * (
                    daily_results[
                        "overlap_count"
                    ]
                    >= 9
                ).mean()
            ),

        "pct_days_at_least_8":
            (
                100
                * (
                    daily_results[
                        "overlap_count"
                    ]
                    >= 8
                ).mean()
            ),

        "pct_days_at_least_7":
            (
                100
                * (
                    daily_results[
                        "overlap_count"
                    ]
                    >= 7
                ).mean()
            ),

        "mean_top10_rank_error":
            daily_results[
                "mean_top10_rank_error"
            ].mean(),

        "median_top10_rank_error":
            daily_results[
                "median_top10_rank_error"
            ].median(),

        "mean_top10_volume_capture_pct":
            (
                100
                * daily_results[
                    "top10_volume_capture"
                ].mean()
            ),

        "median_top10_volume_capture_pct":
            (
                100
                * daily_results[
                    "top10_volume_capture"
                ].median()
            ),

        "mean_overall_rank_correlation":
            daily_results[
                "overall_rank_correlation"
            ].mean(),

        "pct_days_actual_cutoff_tie":
            (
                100
                * daily_results[
                    "actual_cutoff_tie"
                ].mean()
            ),
    }

    return pd.DataFrame(
        [
            summary
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "data/backtest/xgboost_v1/"
            "lags_rolling_calendar/"
            "predictions.parquet"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/backtest/xgboost_v1/"
            "top10_evaluation"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    predictions = pd.read_parquet(
        args.predictions
    )

    daily_results = (
        evaluate_top10_predictions(
            predictions=
                predictions,

            top_k=
                args.top_k,
        )
    )

    summary = summarize_top10(
        daily_results
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_results.to_parquet(
        output_dir
        / "daily_top10_results.parquet",
        index=False,
    )

    daily_results.to_csv(
        output_dir
        / "daily_top10_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "top10_summary.csv",
        index=False,
    )

    print(
        "\nTop-10 neighborhood evaluation\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nBy fold:\n"
    )

    fold_summary = (
        daily_results
        .groupby(
            "fold"
        )
        .agg(
            days=(
                "target_date",
                "count",
            ),

            mean_correct=(
                "overlap_count",
                "mean",
            ),

            mean_accuracy=(
                "top_k_accuracy",
                "mean",
            ),

            mean_volume_capture=(
                "top10_volume_capture",
                "mean",
            ),

            mean_rank_error=(
                "mean_top10_rank_error",
                "mean",
            ),

            mean_rank_correlation=(
                "overall_rank_correlation",
                "mean",
            ),
        )
    )

    fold_summary[
        "mean_accuracy"
    ] *= 100

    fold_summary[
        "mean_volume_capture"
    ] *= 100

    print(
        fold_summary.to_string()
    )


if __name__ == "__main__":
    main()