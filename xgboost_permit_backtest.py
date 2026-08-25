from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xgboost_backtest import (
    run_feature_set_backtest,
    save_backtest_outputs,
    validate_folds,
)
from xgboost_features import (
    XGB_FEATURE_SETS,
    merge_external_features,
    prepare_target_panel,
    validate_xgboost_feature_panel,
)


TARGET_PANEL_PATH = (
    "data/target_panel_5y.parquet"
)

BASE_FEATURE_PANEL_PATH = (
    "data/xgboost/"
    "xgboost_feature_panel.parquet"
)

PERMIT_PANEL_PATH = (
    "data/permitted_events/"
    "special_events_feature_panel.parquet"
)

PERMIT_FOLDS_PATH = (
    "data/backtest/permitted_events/"
    "folds.parquet"
)

OUTPUT_DIR = (
    "data/backtest/"
    "xgboost_permitted_events"
)

BASELINE_EXPERIMENT_NAME = (
    "baseline"
)

BASELINE_NUMERIC_FEATURES = [
    *XGB_FEATURE_SETS[
        "lags_rolling_calendar"
    ]
]

PERMIT_XGB_FEATURE_SETS = {
    "permit_count": [
        "se_permit_count",
    ],
    "permit_log_attendance": [
        "se_log_split_attendance_sum",
    ],
    "permit_count_log_attendance": [
        "se_permit_count",
        "se_log_split_attendance_sum",
    ],
    "permit_full": [
        "se_permit_count",
        "se_attendance_known_count",
        "se_attendance_missing_count",
        "se_split_attendance_sum",
        "se_max_attendance_known",
        "se_log_split_attendance_sum",
    ],
}


def build_permit_feature_specs() -> dict[str, list[str]]:
    specs = {
        BASELINE_EXPERIMENT_NAME: list(
            BASELINE_NUMERIC_FEATURES
        )
    }

    for (
        candidate,
        permit_features,
    ) in PERMIT_XGB_FEATURE_SETS.items():
        specs[candidate] = [
            *BASELINE_NUMERIC_FEATURES,
            *permit_features,
        ]

    return specs


def restrict_panel_to_folds(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    start = pd.Timestamp(
        folds["train_start"].min()
    ).normalize()

    end = pd.Timestamp(
        folds["val_end"].max()
    ).normalize()

    df = panel.copy()
    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    ).dt.normalize()

    return (
        df.loc[
            df["target_date"].between(
                start,
                end,
            )
        ]
        .copy()
        .reset_index(drop=True)
    )


def compare_candidate_to_baseline(
    candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    candidate_name: str,
) -> pd.DataFrame:
    keys = [
        "fold",
        "neighborhood",
    ]

    metric_columns = [
        "mae",
        "rmse",
        "mase",
        "smape",
        "bias",
    ]

    for frame_name, frame in [
        ("candidate_metrics", candidate_metrics),
        ("baseline_metrics", baseline_metrics),
    ]:
        duplicate_count = (
            frame
            .duplicated(subset=keys)
            .sum()
        )

        if duplicate_count:
            raise ValueError(
                f"{frame_name} contains duplicate "
                "fold/neighborhood rows."
            )

    candidate_keys = set(
        map(
            tuple,
            candidate_metrics[keys]
            .itertuples(
                index=False,
                name=None,
            ),
        )
    )

    baseline_keys = set(
        map(
            tuple,
            baseline_metrics[keys]
            .itertuples(
                index=False,
                name=None,
            ),
        )
    )

    if candidate_keys != baseline_keys:
        raise ValueError(
            "Candidate and baseline metrics must "
            "contain the exact same fold/neighborhood keys."
        )

    baseline = (
        baseline_metrics[
            [
                *keys,
                *metric_columns,
            ]
        ]
        .rename(
            columns={
                column:
                    f"baseline_{column}"
                for column
                in metric_columns
            }
        )
    )

    candidate = (
        candidate_metrics[
            [
                *keys,
                *metric_columns,
            ]
        ]
        .rename(
            columns={
                column:
                    f"candidate_{column}"
                for column
                in metric_columns
            }
        )
    )

    comparison = candidate.merge(
        baseline,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    comparison["candidate"] = (
        candidate_name
    )

    for metric in metric_columns:
        comparison[
            f"delta_{metric}"
        ] = (
            comparison[
                f"candidate_{metric}"
            ]
            - comparison[
                f"baseline_{metric}"
            ]
        )

    comparison[
        "improved_mase"
    ] = (
        comparison["delta_mase"]
        < 0
    )

    return (
        comparison.sort_values(keys)
        .reset_index(drop=True)
    )


def summarize_paired_comparisons(
    paired_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    if paired_comparisons.empty:
        return pd.DataFrame()

    rows = []

    for (
        candidate,
        group,
    ) in paired_comparisons.groupby(
        "candidate"
    ):
        neighborhood_average = (
            group.groupby(
                "neighborhood"
            )["delta_mase"]
            .mean()
        )

        rows.append(
            {
                "candidate":
                    candidate,
                "mean_delta_mase":
                    group[
                        "delta_mase"
                    ].mean(),
                "median_delta_mase":
                    group[
                        "delta_mase"
                    ].median(),
                "pct_jobs_improved":
                    100.0
                    * group[
                        "improved_mase"
                    ].mean(),
                "pct_neighborhoods_improved_on_average":
                    100.0
                    * (
                        neighborhood_average
                        < 0
                    ).mean(),
                "mean_delta_smape":
                    group[
                        "delta_smape"
                    ].mean(),
                "median_delta_smape":
                    group[
                        "delta_smape"
                    ].median(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("candidate")
        .reset_index(drop=True)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run permit-specific XGBoost backtests "
            "on the dedicated permitted-events folds."
        )
    )

    parser.add_argument(
        "--target-panel",
        default=TARGET_PANEL_PATH,
    )

    parser.add_argument(
        "--feature-panel",
        default=BASE_FEATURE_PANEL_PATH,
    )

    parser.add_argument(
        "--permit-panel",
        default=PERMIT_PANEL_PATH,
    )

    parser.add_argument(
        "--folds",
        default=PERMIT_FOLDS_PATH,
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_panel = prepare_target_panel(
        pd.read_parquet(
            args.target_panel
        )
    )

    base_feature_panel = (
        pd.read_parquet(
            args.feature_panel
        )
    )

    validate_xgboost_feature_panel(
        base_feature_panel
    )

    permit_panel = pd.read_parquet(
        args.permit_panel
    )

    folds = pd.read_parquet(
        args.folds
    )

    validate_folds(folds)

    restricted_base_panel = (
        restrict_panel_to_folds(
            base_feature_panel,
            folds,
        )
    )

    feature_specs = (
        build_permit_feature_specs()
    )

    permit_feature_columns = sorted(
        {
            feature
            for features in
            PERMIT_XGB_FEATURE_SETS.values()
            for feature in features
        }
    )

    experiment_panel = (
        merge_external_features(
            feature_panel=
                restricted_base_panel,
            external_panel=
                permit_panel,
            feature_columns=
                permit_feature_columns,
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds.to_parquet(
        output_dir
        / "folds.parquet",
        index=False,
    )

    experiment_panel.to_parquet(
        output_dir
        / "xgboost_permit_feature_panel.parquet",
        index=False,
    )

    all_predictions = []
    all_metrics = []
    all_diagnostics = []
    results_by_name = {}

    for (
        experiment_name,
        numeric_features,
    ) in feature_specs.items():
        print(
            f"Running {experiment_name}..."
        )

        result = (
            run_feature_set_backtest(
                feature_panel=
                    experiment_panel,
                target_panel=
                    target_panel,
                folds=folds,
                feature_set_name=
                    experiment_name,
                numeric_features=
                    numeric_features,
            )
        )

        experiment_dir = (
            output_dir
            / experiment_name
        )

        save_backtest_outputs(
            predictions=
                result[
                    "predictions"
                ],
            metrics=
                result["metrics"],
            diagnostics=
                result[
                    "diagnostics"
                ],
            output_dir=
                experiment_dir,
        )

        all_predictions.append(
            result["predictions"]
        )
        all_metrics.append(
            result["metrics"]
        )
        all_diagnostics.append(
            result["diagnostics"]
        )
        results_by_name[
            experiment_name
        ] = result

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    combined_diagnostics = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    combined_predictions.to_parquet(
        output_dir
        / "all_predictions.parquet",
        index=False,
    )

    combined_metrics.to_parquet(
        output_dir
        / "all_metrics.parquet",
        index=False,
    )

    combined_diagnostics.to_parquet(
        output_dir
        / "all_diagnostics.parquet",
        index=False,
    )

    baseline_metrics = (
        results_by_name[
            BASELINE_EXPERIMENT_NAME
        ]["metrics"]
    )

    paired_frames = []

    for candidate_name in (
        PERMIT_XGB_FEATURE_SETS.keys()
    ):
        paired = compare_candidate_to_baseline(
            candidate_metrics=
                results_by_name[
                    candidate_name
                ]["metrics"],
            baseline_metrics=
                baseline_metrics,
            candidate_name=
                candidate_name,
        )

        paired_frames.append(paired)

    paired_comparisons = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    paired_summary = (
        summarize_paired_comparisons(
            paired_comparisons
        )
    )

    paired_comparisons.to_parquet(
        output_dir
        / "paired_comparisons.parquet",
        index=False,
    )

    paired_summary.to_csv(
        output_dir
        / "paired_summary.csv",
        index=False,
    )

    if not paired_summary.empty:
        print()
        print(
            paired_summary.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
