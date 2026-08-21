from pathlib import Path
import argparse

import pandas as pd

from sarima_backtest import (
    run_all_baseline_backtests,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_DAYS = 365
VALIDATION_DAYS = 90
N_FOLDS = 4


def build_backtest_folds(
    target_panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Recreate the deterministic development/test split
    and four expanding-window validation folds used
    during notebook development.
    """

    target_panel = target_panel.copy()

    target_panel["target_date"] = pd.to_datetime(
        target_panel["target_date"]
    )

    # ----------------------------------------------
    # Final untouched 365-day test period
    # ----------------------------------------------

    max_target_date = (
        target_panel["target_date"].max()
    )

    test_start = (
        max_target_date
        - pd.Timedelta(
            days=TEST_DAYS - 1
        )
    )

    train_panel = target_panel.loc[
        target_panel["target_date"]
        < test_start
    ].copy()

    test_panel = target_panel.loc[
        target_panel["target_date"]
        >= test_start
    ].copy()

    # ----------------------------------------------
    # Four 90-day validation folds
    # ----------------------------------------------

    train_start = (
        train_panel["target_date"].min()
    )

    train_end = (
        train_panel["target_date"].max()
    )

    folds = []

    for fold in range(N_FOLDS):

        val_end = (
            train_end
            - pd.Timedelta(
                days=(
                    VALIDATION_DAYS
                    * (
                        N_FOLDS
                        - fold
                        - 1
                    )
                )
            )
        )

        val_start = (
            val_end
            - pd.Timedelta(
                days=(
                    VALIDATION_DAYS
                    - 1
                )
            )
        )

        fold_train_end = (
            val_start
            - pd.Timedelta(days=1)
        )

        folds.append(
            {
                "fold": fold + 1,
                "train_start": train_start,
                "train_end": (
                    fold_train_end
                ),
                "val_start": val_start,
                "val_end": val_end,
            }
        )

    backtest_folds = pd.DataFrame(
        folds
    )

    # ----------------------------------------------
    # QA
    # ----------------------------------------------

    assert (
        test_panel[
            "target_date"
        ].nunique()
        == TEST_DAYS
    )

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
        == train_panel[
            "target_date"
        ].max()
    )

    assert (
        backtest_folds.iloc[-1][
            "val_end"
        ]
        < test_panel[
            "target_date"
        ].min()
    )

    return (
        backtest_folds,
        train_panel,
        test_panel,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "data/backtest/azure_v2"
        ),
    )

    args = parser.parse_args()

    # ----------------------------------------------
    # Load target panel
    # ----------------------------------------------

    target_path = Path(
        "data/target_panel_5y.parquet"
    )

    target_panel = pd.read_parquet(
        target_path
    )

    target_panel[
        "target_date"
    ] = pd.to_datetime(
        target_panel["target_date"]
    )

    print(
        "Target panel:",
        target_panel.shape,
    )

    print(
        "Neighborhoods:",
        target_panel[
            "neighborhood"
        ].nunique(),
    )

    print(
        "Date range:",
        target_panel[
            "target_date"
        ].min(),
        "→",
        target_panel[
            "target_date"
        ].max(),
    )

    # ----------------------------------------------
    # Rebuild folds deterministically
    # ----------------------------------------------

    (
        backtest_folds,
        train_panel,
        test_panel,
    ) = build_backtest_folds(
        target_panel
    )

    print("\nBacktest folds:")
    print(
        backtest_folds.to_string(
            index=False
        )
    )

    print(
        "\nDevelopment period:",
        train_panel[
            "target_date"
        ].min(),
        "→",
        train_panel[
            "target_date"
        ].max(),
    )

    print(
        "Final untouched test:",
        test_panel[
            "target_date"
        ].min(),
        "→",
        test_panel[
            "target_date"
        ].max(),
    )

    # ----------------------------------------------
    # Run baseline
    # ----------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    print(
        "\nOutput directory:",
        output_dir,
    )

    print(
        "Workers:",
        args.n_jobs,
    )

    print(
        "Max jobs:",
        args.max_jobs,
    )

    results = (
        run_all_baseline_backtests(
            target_panel=target_panel,

            backtest_folds=(
                backtest_folds
            ),

            output_dir=output_dir,

            n_jobs=args.n_jobs,

            checkpoint_every=(
                args.checkpoint_every
            ),

            max_jobs=args.max_jobs,
        )
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n=== FINAL SUMMARY ===")

    print(
        "Successful jobs:",
        len(results["metrics"]),
    )

    print(
        "Failed jobs:",
        len(results["failures"]),
    )

    print(
        "Predictions:",
        len(
            results["predictions"]
        ),
    )

    if not results["metrics"].empty:

        metric_cols = [
            "mae",
            "rmse",
            "mase",
            "smape",
            "bias",
        ]

        print(
            "\nMetric summary:"
        )

        print(
            results["metrics"][
                metric_cols
            ]
            .agg(
                ["mean", "median"]
            )
        )

    if not results["failures"].empty:

        print(
            "\nFailures:"
        )

        print(
            results["failures"][
                [
                    "fold",
                    "neighborhood",
                    "stage",
                    "failed_date",
                    "error_type",
                ]
            ]
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()