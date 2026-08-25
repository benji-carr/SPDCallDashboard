from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecasting.evaluation.top10 import (
    evaluate_top10_predictions,
    summarize_top10,
)
from forecasting.paths import (
    TARGET_PANEL_5Y_PATH,
    XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_PREDICTIONS_PATH,
    XGBOOST_TOP10_BASELINE_COMPARISON_DIR,
)


def build_naive_predictions(
    target_panel: pd.DataFrame,
    xgb_predictions: pd.DataFrame,
    lag: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Build lag-based forecasts for exactly the same
    target-date/neighborhood rows evaluated by XGBoost.

    lag=1:
        tomorrow looks like yesterday

    lag=7:
        tomorrow looks like the same day last week
    """

    target = target_panel.copy()

    target["target_date"] = pd.to_datetime(
        target["target_date"],
        errors="raise",
    ).dt.normalize()

    target = (
        target
        .sort_values(
            [
                "neighborhood",
                "target_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Construct naive prediction inside each neighborhood
    # ----------------------------------------------------------

    target["prediction"] = (
        target
        .groupby(
            "neighborhood",
            sort=False,
        )["calls"]
        .shift(lag)
    )

    target = target.rename(
        columns={
            "calls": "actual",
        }
    )

    # ----------------------------------------------------------
    # Restrict evaluation to the exact XGBoost validation keys
    # ----------------------------------------------------------

    evaluation_keys = (
        xgb_predictions[
            [
                "fold",
                "target_date",
                "neighborhood",
            ]
        ]
        .copy()
    )

    evaluation_keys[
        "target_date"
    ] = pd.to_datetime(
        evaluation_keys[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    predictions = (
        evaluation_keys
        .merge(
            target[
                [
                    "target_date",
                    "neighborhood",
                    "actual",
                    "prediction",
                ]
            ],
            on=[
                "target_date",
                "neighborhood",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if predictions[
        [
            "actual",
            "prediction",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{model_name} contains missing "
            "actuals or predictions."
        )

    predictions[
        "forecast_origin"
    ] = (
        predictions[
            "target_date"
        ]
        - pd.Timedelta(
            days=1
        )
    )

    predictions[
        "model"
    ] = model_name

    return predictions


def summary_to_row(
    daily_results: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    summary = summarize_top10(
        daily_results
    )

    summary.insert(
        0,
        "model",
        model_name,
    )

    return summary


def build_daily_comparison(
    results_by_model: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    comparison = None

    columns = [
        "target_date",
        "overlap_count",
        "top_k_accuracy",
        "mean_top10_rank_error",
        "top10_volume_capture",
        "overall_rank_correlation",
    ]

    for (
        model_name,
        results,
    ) in results_by_model.items():

        model_results = (
            results[
                columns
            ]
            .copy()
        )

        rename_map = {
            column:
                f"{column}_{model_name}"
            for column in columns
            if column != "target_date"
        }

        model_results = (
            model_results.rename(
                columns=rename_map
            )
        )

        if comparison is None:
            comparison = model_results
        else:
            comparison = (
                comparison.merge(
                    model_results,
                    on="target_date",
                    how="inner",
                    validate="one_to_one",
                )
            )

    return comparison


def print_pairwise_wins(
    daily_comparison: pd.DataFrame,
    candidate: str,
    baseline: str,
) -> None:

    print(
        f"\nXGBoost vs {baseline}"
    )

    xgb_overlap = daily_comparison[
        f"overlap_count_{candidate}"
    ]

    baseline_overlap = daily_comparison[
        f"overlap_count_{baseline}"
    ]

    print(
        "Days XGBoost identifies more "
        "correct top-10 neighborhoods:",
        f"{100 * (xgb_overlap > baseline_overlap).mean():.2f}%"
    )

    print(
        "Days tied on top-10 membership:",
        f"{100 * (xgb_overlap == baseline_overlap).mean():.2f}%"
    )

    print(
        "Days baseline identifies more:",
        f"{100 * (xgb_overlap < baseline_overlap).mean():.2f}%"
    )

    xgb_volume = daily_comparison[
        f"top10_volume_capture_{candidate}"
    ]

    baseline_volume = daily_comparison[
        f"top10_volume_capture_{baseline}"
    ]

    print(
        "Days XGBoost captures more "
        "top-10 call volume:",
        f"{100 * (xgb_volume > baseline_volume).mean():.2f}%"
    )

    print(
        "Mean volume-capture advantage:",
        (
            f"{100 * (xgb_volume - baseline_volume).mean():.3f} "
            "percentage points"
        ),
    )

    xgb_corr = daily_comparison[
        f"overall_rank_correlation_{candidate}"
    ]

    baseline_corr = daily_comparison[
        f"overall_rank_correlation_{baseline}"
    ]

    print(
        "Mean rank-correlation advantage:",
        f"{(xgb_corr - baseline_corr).mean():.4f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-panel",
        default=str(
            TARGET_PANEL_5Y_PATH
        ),
    )

    parser.add_argument(
        "--xgb-predictions",
        default=str(
            XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_PREDICTIONS_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            XGBOOST_TOP10_BASELINE_COMPARISON_DIR
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    target_panel = pd.read_parquet(
        args.target_panel
    )

    xgb_predictions = pd.read_parquet(
        args.xgb_predictions
    )

    xgb_predictions[
        "target_date"
    ] = pd.to_datetime(
        xgb_predictions[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    # ----------------------------------------------------------
    # Build naive forecasts
    # ----------------------------------------------------------

    persistence_predictions = (
        build_naive_predictions(
            target_panel=
                target_panel,

            xgb_predictions=
                xgb_predictions,

            lag=1,

            model_name=
                "persistence_1d",
        )
    )

    weekly_predictions = (
        build_naive_predictions(
            target_panel=
                target_panel,

            xgb_predictions=
                xgb_predictions,

            lag=7,

            model_name=
                "seasonal_naive_7d",
        )
    )

    # ----------------------------------------------------------
    # Evaluate all three using exactly the same evaluator
    # ----------------------------------------------------------

    results_by_model = {
        "xgboost":
            evaluate_top10_predictions(
                predictions=
                    xgb_predictions,

                top_k=
                    args.top_k,
            ),

        "persistence_1d":
            evaluate_top10_predictions(
                predictions=
                    persistence_predictions,

                top_k=
                    args.top_k,
            ),

        "seasonal_naive_7d":
            evaluate_top10_predictions(
                predictions=
                    weekly_predictions,

                top_k=
                    args.top_k,
            ),
    }

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------

    summary = pd.concat(
        [
            summary_to_row(
                results,
                model_name,
            )
            for (
                model_name,
                results,
            ) in results_by_model.items()
        ],
        ignore_index=True,
    )

    display_columns = [
        "model",
        "mean_correct_top10",
        "median_correct_top10",
        "mean_top10_accuracy_pct",
        "pct_days_10_of_10",
        "pct_days_at_least_9",
        "pct_days_at_least_8",
        "mean_top10_rank_error",
        "mean_top10_volume_capture_pct",
        "mean_overall_rank_correlation",
    ]

    print(
        "\nTop-10 model comparison\n"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Daily paired comparison
    # ----------------------------------------------------------

    daily_comparison = (
        build_daily_comparison(
            results_by_model
        )
    )

    print_pairwise_wins(
        daily_comparison=
            daily_comparison,

        candidate="xgboost",

        baseline="persistence_1d",
    )

    print_pairwise_wins(
        daily_comparison=
            daily_comparison,

        candidate="xgboost",

        baseline="seasonal_naive_7d",
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_dir
        / "comparison_summary.csv",
        index=False,
    )

    daily_comparison.to_parquet(
        output_dir
        / "daily_comparison.parquet",
        index=False,
    )

    for (
        model_name,
        results,
    ) in results_by_model.items():

        results.to_parquet(
            output_dir
            / f"{model_name}_daily_results.parquet",
            index=False,
        )


if __name__ == "__main__":
    main()
