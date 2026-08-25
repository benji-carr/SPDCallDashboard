# run_calendar_tests.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecasting.features.calendar import (
    FEATURE_SETS,
    save_calendar_features,
)

from forecasting.backtests.calendar_sarimax import (
    run_calendar_backtests,
    compare_to_baseline,
)
from forecasting.paths import (
    BACKTEST_DIR,
    CALENDAR_FEATURES_PATH,
    SARIMA_BACKTEST_DIR,
    SARIMAX_CALENDAR_BACKTEST_DIR,
    TARGET_PANEL_5Y_PATH,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

TARGET_PANEL_PATH = (
    TARGET_PANEL_5Y_PATH
)

BASELINE_DIR = (
    SARIMA_BACKTEST_DIR
)

BASELINE_ORDERS_PATH = (
    BASELINE_DIR
    / "baseline_sarima_v2_orders.parquet"
)

BASELINE_METRICS_PATH = (
    BASELINE_DIR
    / "baseline_sarima_v2_metrics.parquet"
)

CALENDAR_PATH = (
    CALENDAR_FEATURES_PATH
)

CALENDAR_OUTPUT_ROOT = (
    SARIMAX_CALENDAR_BACKTEST_DIR
)


# ---------------------------------------------------------------------
# Exact backtest design used by baseline
# ---------------------------------------------------------------------

TEST_DAYS = 365
VALIDATION_DAYS = 90
N_FOLDS = 4


def build_backtest_folds(
    target_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Reconstruct the exact deterministic split used
    for the SARIMA baseline.

    Last 365 days:
        untouched final test set.

    Earlier development period:
        four contiguous 90-day validation folds.

    Training window:
        expanding from the first development date.
    """

    panel = target_panel.copy()

    panel["target_date"] = pd.to_datetime(
        panel["target_date"]
    )

    max_target_date = (
        panel["target_date"]
        .max()
        .normalize()
    )

    test_start = (
        max_target_date
        - pd.Timedelta(
            days=TEST_DAYS - 1
        )
    )

    development = panel.loc[
        panel["target_date"]
        < test_start
    ].copy()

    train_start = (
        development["target_date"]
        .min()
        .normalize()
    )

    development_end = (
        development["target_date"]
        .max()
        .normalize()
    )

    folds = []

    for fold_index in range(N_FOLDS):

        val_end = (
            development_end
            - pd.Timedelta(
                days=(
                    VALIDATION_DAYS
                    * (
                        N_FOLDS
                        - fold_index
                        - 1
                    )
                )
            )
        )

        val_start = (
            val_end
            - pd.Timedelta(
                days=VALIDATION_DAYS - 1
            )
        )

        fold_train_end = (
            val_start
            - pd.Timedelta(days=1)
        )

        folds.append(
            {
                "fold": fold_index + 1,
                "train_start": train_start,
                "train_end": fold_train_end,
                "val_start": val_start,
                "val_end": val_end,
            }
        )

    backtest_folds = pd.DataFrame(
        folds
    )

    # --------------------------------------------------------------
    # QA
    # --------------------------------------------------------------

    for _, fold in (
        backtest_folds.iterrows()
    ):

        n_validation_days = (
            fold["val_end"]
            - fold["val_start"]
        ).days + 1

        assert (
            n_validation_days
            == VALIDATION_DAYS
        )

        assert (
            fold["train_end"]
            + pd.Timedelta(days=1)
            == fold["val_start"]
        )

    assert (
        backtest_folds.iloc[-1][
            "val_end"
        ]
        == development_end
    )

    assert (
        development_end
        + pd.Timedelta(days=1)
        == test_start
    )

    return (
        backtest_folds,
        test_start,
    )


# ---------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------

def normalize_baseline_data(
    baseline_orders: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
):
    baseline_orders = (
        baseline_orders.copy()
    )

    baseline_metrics = (
        baseline_metrics.copy()
    )

    baseline_orders["fold"] = (
        pd.to_numeric(
            baseline_orders["fold"]
        )
        .astype(int)
    )

    baseline_metrics["fold"] = (
        pd.to_numeric(
            baseline_metrics["fold"]
        )
        .astype(int)
    )

    def parse_bool(value):

        if isinstance(value, str):
            return (
                value.strip().lower()
                in {
                    "true",
                    "1",
                    "yes",
                }
            )

        return bool(value)

    baseline_orders[
        "with_intercept"
    ] = (
        baseline_orders[
            "with_intercept"
        ]
        .map(parse_bool)
    )

    return (
        baseline_orders,
        baseline_metrics,
    )


# ---------------------------------------------------------------------
# Experiment subsets
# ---------------------------------------------------------------------

def prepare_mode_data(
    target_panel: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    baseline_orders: pd.DataFrame,
    mode: str,
):

    all_neighborhoods = sorted(
        target_panel[
            "neighborhood"
        ].unique()
    )

    if mode == "sanity":

        selected_neighborhoods = (
            all_neighborhoods[:1]
        )

        selected_folds = (
            backtest_folds.iloc[
                [0]
            ].copy()
        )

    elif mode == "smoke":

        selected_neighborhoods = (
            all_neighborhoods[:7]
        )

        selected_folds = (
            backtest_folds.iloc[
                [0]
            ].copy()
        )

    elif mode == "full":

        selected_neighborhoods = (
            all_neighborhoods
        )

        selected_folds = (
            backtest_folds.copy()
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    selected_panel = (
        target_panel.loc[
            target_panel[
                "neighborhood"
            ].isin(
                selected_neighborhoods
            )
        ]
        .copy()
    )

    selected_orders = (
        baseline_orders.loc[
            baseline_orders[
                "neighborhood"
            ].isin(
                selected_neighborhoods
            )
            &
            baseline_orders[
                "fold"
            ].isin(
                selected_folds[
                    "fold"
                ]
            )
        ]
        .copy()
    )

    return (
        selected_panel,
        selected_folds,
        selected_orders,
    )


# ---------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------

def build_summary(
    feature_set_name: str,
    mode: str,
    results: dict,
    comparison: pd.DataFrame,
) -> pd.DataFrame:

    metrics = results["metrics"]
    diagnostics = results[
        "diagnostics"
    ]
    failures = results[
        "failures"
    ]

    row = {
        "mode": mode,
        "feature_set": (
            feature_set_name
        ),

        "successful_jobs": len(
            metrics
        ),

        "failed_jobs": len(
            failures
        ),

        # Raw SARIMAX performance
        "mean_mae": (
            metrics["mae"].mean()
        ),

        "median_mae": (
            metrics["mae"].median()
        ),

        "mean_rmse": (
            metrics["rmse"].mean()
        ),

        "median_rmse": (
            metrics["rmse"].median()
        ),

        "mean_mase": (
            metrics["mase"].mean()
        ),

        "median_mase": (
            metrics["mase"].median()
        ),

        "mean_smape": (
            metrics["smape"].mean()
        ),

        "median_smape": (
            metrics["smape"].median()
        ),

        "mean_bias": (
            metrics["bias"].mean()
        ),

        "median_bias": (
            metrics["bias"].median()
        ),

        # Relative to SARIMA baseline
        "mean_delta_mae": (
            comparison[
                "delta_mae"
            ].mean()
        ),

        "median_delta_mae": (
            comparison[
                "delta_mae"
            ].median()
        ),

        "mean_delta_mase": (
            comparison[
                "delta_mase"
            ].mean()
        ),

        "median_delta_mase": (
            comparison[
                "delta_mase"
            ].median()
        ),

        "mean_delta_smape": (
            comparison[
                "delta_smape"
            ].mean()
        ),

        "median_delta_smape": (
            comparison[
                "delta_smape"
            ].median()
        ),

        "pct_jobs_improved_mase": (
            comparison[
                "improved_mase"
            ].mean()
        ),

        "pct_jobs_mase_below_1": (
            (
                metrics["mase"]
                < 1
            ).mean()
        ),
    }

    if not diagnostics.empty:

        row[
            "mean_negative_prediction_rate"
        ] = (
            diagnostics[
                "negative_prediction_rate"
            ].mean()
        )

        row[
            "pct_residuals_pass_lb7"
        ] = (
            (
                diagnostics[
                    "ljung_box_p_7"
                ]
                > 0.05
            ).mean()
        )

        row[
            "pct_residuals_pass_lb14"
        ] = (
            (
                diagnostics[
                    "ljung_box_p_14"
                ]
                > 0.05
            ).mean()
        )

        row[
            "median_condition_number"
        ] = (
            diagnostics[
                "condition_number"
            ].median()
        )

        row[
            "max_prediction"
        ] = (
            diagnostics[
                "max_prediction"
            ].max()
        )

    return pd.DataFrame(
        [row]
    )


# ---------------------------------------------------------------------
# Run one feature set
# ---------------------------------------------------------------------

def run_feature_set(
    feature_set_name: str,
    mode: str,
    n_jobs: int,
    target_panel: pd.DataFrame,
    calendar_features: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    baseline_orders: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
):

    feature_columns = (
        FEATURE_SETS[
            feature_set_name
        ]
    )

    (
        selected_panel,
        selected_folds,
        selected_orders,
    ) = prepare_mode_data(
        target_panel=target_panel,
        backtest_folds=(
            backtest_folds
        ),
        baseline_orders=(
            baseline_orders
        ),
        mode=mode,
    )

    # Don't launch more workers than jobs.
    n_expected_jobs = (
        selected_panel[
            "neighborhood"
        ].nunique()
        * len(selected_folds)
    )

    effective_n_jobs = min(
        n_jobs,
        n_expected_jobs,
    )

    if mode == "sanity":
        effective_n_jobs = 1

    output_dir = (
        CALENDAR_OUTPUT_ROOT
        / mode
    )

    print()
    print("=" * 70)
    print(
        f"Feature set: "
        f"{feature_set_name}"
    )
    print(
        f"Mode: {mode}"
    )
    print(
        f"Features: "
        f"{len(feature_columns)}"
    )
    print(
        f"Expected jobs: "
        f"{n_expected_jobs}"
    )
    print(
        f"Workers: "
        f"{effective_n_jobs}"
    )
    print("=" * 70)

    results = run_calendar_backtests(
        target_panel=(
            selected_panel
        ),

        calendar_features=(
            calendar_features
        ),

        backtest_folds=(
            selected_folds
        ),

        baseline_orders=(
            selected_orders
        ),

        feature_set_name=(
            feature_set_name
        ),

        feature_columns=(
            feature_columns
        ),

        output_dir=output_dir,

        n_jobs=effective_n_jobs,
    )

    comparison = compare_to_baseline(
        calendar_metrics=(
            results["metrics"]
        ),

        baseline_metrics=(
            baseline_metrics
        ),
    )

    experiment_dir = (
        output_dir
        / feature_set_name
    )

    comparison.to_parquet(
        experiment_dir
        / "comparison_to_baseline.parquet",
        index=False,
    )

    summary = build_summary(
        feature_set_name=(
            feature_set_name
        ),
        mode=mode,
        results=results,
        comparison=comparison,
    )

    summary.to_csv(
        experiment_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(summary.T)

    if not results[
        "failures"
    ].empty:

        print()
        print("FAILURES:")
        print(
            results[
                "failures"
            ]
        )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run calendar SARIMAX "
            "validation experiments."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "sanity",
            "smoke",
            "full",
        ],
        default="sanity",
    )

    parser.add_argument(
        "--feature-set",
        choices=[
            *FEATURE_SETS.keys(),
            "all",
        ],
        default="holiday_core",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=28,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    print(
        f"Loading target panel from "
        f"{TARGET_PANEL_PATH}"
    )

    target_panel = (
        pd.read_parquet(
            TARGET_PANEL_PATH
        )
    )

    target_panel[
        "target_date"
    ] = pd.to_datetime(
        target_panel[
            "target_date"
        ]
    )

    print(
        f"Loading baseline orders from "
        f"{BASELINE_ORDERS_PATH}"
    )

    baseline_orders = (
        pd.read_parquet(
            BASELINE_ORDERS_PATH
        )
    )

    baseline_metrics = (
        pd.read_parquet(
            BASELINE_METRICS_PATH
        )
    )

    (
        baseline_orders,
        baseline_metrics,
    ) = normalize_baseline_data(
        baseline_orders,
        baseline_metrics,
    )

    # --------------------------------------------------------------
    # Rebuild exact folds
    # --------------------------------------------------------------

    (
        backtest_folds,
        test_start,
    ) = build_backtest_folds(
        target_panel
    )

    print()
    print("Backtest folds:")
    print(backtest_folds)

    print()
    print(
        "Final untouched test starts:",
        test_start.date(),
    )

    # Save only for audit/reproducibility.
    audit_fold_path = (
        BACKTEST_DIR
        / "generated_backtest_folds.parquet"
    )

    audit_fold_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest_folds.to_parquet(
        audit_fold_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------------

    if CALENDAR_PATH.exists():

        print()
        print(
            f"Loading existing calendar "
            f"features from "
            f"{CALENDAR_PATH}"
        )

        calendar_features = (
            pd.read_parquet(
                CALENDAR_PATH
            )
        )

    else:

        print()
        print(
            "Building calendar "
            "feature dataset..."
        )

        calendar_features = (
            save_calendar_features(
                target_panel=(
                    target_panel
                ),
                output_path=(
                    CALENDAR_PATH
                ),
            )
        )

    calendar_features[
        "target_date"
    ] = pd.to_datetime(
        calendar_features[
            "target_date"
        ]
    )

    # --------------------------------------------------------------
    # Feature-set selection
    # --------------------------------------------------------------

    if args.feature_set == "all":

        selected_feature_sets = [
            "holiday_core",
            "special_days",
            "seasonal_cycle",
            "calendar_compact",
            "full_19",
        ]

    else:

        selected_feature_sets = [
            args.feature_set
        ]

    summaries = []

    for feature_set_name in (
        selected_feature_sets
    ):

        summary = run_feature_set(
            feature_set_name=(
                feature_set_name
            ),

            mode=args.mode,

            n_jobs=args.n_jobs,

            target_panel=(
                target_panel
            ),

            calendar_features=(
                calendar_features
            ),

            backtest_folds=(
                backtest_folds
            ),

            baseline_orders=(
                baseline_orders
            ),

            baseline_metrics=(
                baseline_metrics
            ),
        )

        summaries.append(
            summary
        )

    combined_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    summary_dir = (
        CALENDAR_OUTPUT_ROOT
        / args.mode
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_summary.to_csv(
        summary_dir
        / "calendar_experiment_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(
        combined_summary[
            [
                "feature_set",
                "successful_jobs",
                "failed_jobs",
                "median_mase",
                "median_smape",
                "median_delta_mase",
                "pct_jobs_improved_mase",
            ]
        ]
    )


if __name__ == "__main__":
    main()
