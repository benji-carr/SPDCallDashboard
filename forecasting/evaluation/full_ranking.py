from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def rank_day(
    day: pd.DataFrame,
) -> pd.DataFrame:

    actual_ranked = (
        day
        .sort_values(
            ["actual", "neighborhood"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    actual_ranked["actual_rank"] = (
        np.arange(
            1,
            len(actual_ranked) + 1,
        )
    )

    predicted_ranked = (
        day
        .sort_values(
            ["prediction", "neighborhood"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    predicted_ranked["predicted_rank"] = (
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

    ranked = day.merge(
        ranks,
        on="neighborhood",
        validate="one_to_one",
    )

    ranked[
        "absolute_rank_error"
    ] = (
        ranked[
            "predicted_rank"
        ]
        - ranked[
            "actual_rank"
        ]
    ).abs()

    return ranked


def evaluate_day(
    day: pd.DataFrame,
) -> dict:

    ranked = rank_day(
        day
    )

    errors = ranked[
        "absolute_rank_error"
    ]

    spearman = spearmanr(
        ranked[
            "actual_rank"
        ],
        ranked[
            "predicted_rank"
        ],
    ).statistic

    kendall = kendalltau(
        ranked[
            "actual_rank"
        ],
        ranked[
            "predicted_rank"
        ],
    ).statistic

    return {
        "fold":
            int(
                day[
                    "fold"
                ].iloc[0]
            ),

        "target_date":
            day[
                "target_date"
            ].iloc[0],

        "n_neighborhoods":
            len(
                ranked
            ),

        "mean_absolute_rank_error":
            float(
                errors.mean()
            ),

        "median_absolute_rank_error":
            float(
                errors.median()
            ),

        "pct_within_1_rank":
            float(
                100
                * (
                    errors <= 1
                ).mean()
            ),

        "pct_within_2_ranks":
            float(
                100
                * (
                    errors <= 2
                ).mean()
            ),

        "pct_within_3_ranks":
            float(
                100
                * (
                    errors <= 3
                ).mean()
            ),

        "pct_within_5_ranks":
            float(
                100
                * (
                    errors <= 5
                ).mean()
            ),

        "spearman":
            float(
                spearman
            ),

        "kendall_tau":
            float(
                kendall
            ),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    df = predictions.copy()

    df[
        "target_date"
    ] = pd.to_datetime(
        df[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    if df.duplicated(
        subset=[
            "target_date",
            "neighborhood",
        ]
    ).any():
        raise ValueError(
            "Duplicate prediction keys."
        )

    rows = []

    for _, day in df.groupby(
        "target_date",
        sort=True,
    ):
        rows.append(
            evaluate_day(
                day
            )
        )

    return pd.DataFrame(
        rows
    )


def summarize(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "n_days":
                    len(
                        daily
                    ),

                "mean_absolute_rank_error":
                    daily[
                        "mean_absolute_rank_error"
                    ].mean(),

                "median_daily_absolute_rank_error":
                    daily[
                        "median_absolute_rank_error"
                    ].median(),

                "mean_pct_within_1_rank":
                    daily[
                        "pct_within_1_rank"
                    ].mean(),

                "mean_pct_within_2_ranks":
                    daily[
                        "pct_within_2_ranks"
                    ].mean(),

                "mean_pct_within_3_ranks":
                    daily[
                        "pct_within_3_ranks"
                    ].mean(),

                "mean_pct_within_5_ranks":
                    daily[
                        "pct_within_5_ranks"
                    ].mean(),

                "mean_spearman":
                    daily[
                        "spearman"
                    ].mean(),

                "mean_kendall_tau":
                    daily[
                        "kendall_tau"
                    ].mean(),
            }
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    predictions = pd.read_parquet(
        args.predictions
    )

    daily = evaluate_predictions(
        predictions
    )

    summary = summarize(
        daily
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_parquet(
        output_dir
        / "daily_full_ranking.parquet",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "full_ranking_summary.csv",
        index=False,
    )

    print(
        "\nFull-neighborhood ranking results\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nBy fold\n"
    )

    print(
        daily
        .groupby(
            "fold"
        )[
            [
                "mean_absolute_rank_error",
                "pct_within_3_ranks",
                "spearman",
                "kendall_tau",
            ]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()