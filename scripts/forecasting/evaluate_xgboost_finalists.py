from __future__ import annotations

import argparse
import json

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.backtests.xgboost import (
    TEST_DAYS,
    build_xgboost_pipeline,
    load_or_build_feature_panel,
    run_sequential_fold,
)

from forecasting.evaluation.top10 import (
    evaluate_top10_predictions,
    summarize_top10,
)

from forecasting.features.xgboost import (
    prepare_target_panel,
)

from forecasting.paths import (
    TARGET_PANEL_5Y_PATH,
    XGBOOST_FEATURE_PANEL_PATH,
)


DEFAULT_FEATURE_SET = (
    "lags_rolling_calendar"
)

DEFAULT_TOP_N = 10

# IMPORTANT:
#
# This candidate was selected BEFORE inspecting final-test results.
#
# Other finalists are evaluated for benchmarking/robustness only.
# Their test performance must not be used to replace the locked
# candidate without treating the final test period as development data.
DEFAULT_LOCKED_CONFIG_ID = (
    "be7924a5110a"
)

MANIFEST_TARGET_PANEL_KEYS = (
    "input_target_panel_path",
    "target_panel",
    "target_panel_path",
)

MANIFEST_FEATURE_PANEL_KEYS = (
    "input_feature_panel_path",
    "feature_panel",
    "feature_panel_path",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_json(
    data: dict,
    path: str | Path,
) -> None:
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )


def parse_params_json(
    value: str,
) -> dict:
    params = json.loads(
        value
    )

    if not isinstance(
        params,
        dict,
    ):
        raise ValueError(
            "params_json must decode "
            "to a dictionary."
        )

    return params


def _path_attempts(
    value: str | Path | None,
    manifest_path: Path,
) -> list[Path]:
    if value is None or not str(value).strip():
        return []

    path = Path(value)
    attempts = [path]

    if not path.is_absolute():
        attempts.append(manifest_path.parent / path)

    return list(dict.fromkeys(attempts))


def _manifest_paths(
    manifest: dict,
    keys: tuple[str, ...],
    manifest_path: Path,
) -> list[Path]:
    attempts = []

    for key in keys:
        attempts.extend(
            _path_attempts(
                manifest.get(key),
                manifest_path,
            )
        )

    return list(dict.fromkeys(attempts))


def resolve_data_paths(
    tuning_results_path: str | Path,
    target_panel_override: str | Path | None = None,
    feature_panel_override: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the panels recorded by a tuning run before using defaults."""
    tuning_path = Path(tuning_results_path)
    manifest_path = tuning_path.parent / "manifest.json"
    manifest = {}

    if manifest_path.exists():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tuning manifest is not valid JSON: {manifest_path}"
            ) from exc

        if not isinstance(manifest, dict):
            raise ValueError(
                f"Tuning manifest must contain a JSON object: {manifest_path}"
            )

    target_attempts = [
        *_path_attempts(target_panel_override, manifest_path),
        *_manifest_paths(
            manifest,
            MANIFEST_TARGET_PANEL_KEYS,
            manifest_path,
        ),
        *_path_attempts(TARGET_PANEL_5Y_PATH, manifest_path),
    ]
    feature_attempts = [
        *_path_attempts(feature_panel_override, manifest_path),
        *_manifest_paths(
            manifest,
            MANIFEST_FEATURE_PANEL_KEYS,
            manifest_path,
        ),
        *_path_attempts(XGBOOST_FEATURE_PANEL_PATH, manifest_path),
    ]
    target_attempts = list(dict.fromkeys(target_attempts))
    feature_attempts = list(dict.fromkeys(feature_attempts))

    target_path = next(
        (path for path in target_attempts if path.is_file()),
        None,
    )
    feature_path = next(
        (path for path in feature_attempts if path.is_file()),
        None,
    )

    if target_path is None:
        formatted_targets = "\n  ".join(
            str(path) for path in target_attempts
        ) or "(none)"
        formatted_features = "\n  ".join(
            str(path) for path in feature_attempts
        ) or "(none)"
        raise FileNotFoundError(
            "Unable to locate the target panel associated with the tuning "
            "run. The evaluator will not substitute unrelated data.\n"
            f"Attempted manifest: {manifest_path}\n"
            f"Attempted target-panel paths:\n  {formatted_targets}\n"
            f"Attempted feature-panel paths:\n  {formatted_features}"
        )

    # A missing feature panel is rebuilt from the resolved target panel at
    # the highest-precedence intended location.
    if feature_path is None:
        feature_path = feature_attempts[0]

    return target_path, feature_path, manifest_path


def build_final_test_fold(
    target_panel: pd.DataFrame,
) -> pd.Series:
    """
    Build the single final holdout interval.

    Training contains all target dates before the reserved
    final TEST_DAYS period.

    Validation corresponds exactly to the reserved final
    test period.
    """

    max_target_date = (
        target_panel[
            "target_date"
        ].max()
    )

    test_start = (
        max_target_date
        - pd.Timedelta(
            days=TEST_DAYS - 1
        )
    )

    development = (
        target_panel.loc[
            target_panel[
                "target_date"
            ]
            < test_start
        ]
    )

    test = (
        target_panel.loc[
            target_panel[
                "target_date"
            ]
            >= test_start
        ]
    )

    if development.empty:
        raise ValueError(
            "Development period is empty."
        )

    if test.empty:
        raise ValueError(
            "Final test period is empty."
        )

    if (
        test["target_date"].nunique()
        != TEST_DAYS
    ):
        raise ValueError(
            "Final test period does not "
            f"contain exactly {TEST_DAYS} days."
        )

    train_start = (
        development[
            "target_date"
        ].min()
    )

    train_end = (
        development[
            "target_date"
        ].max()
    )

    val_start = (
        test[
            "target_date"
        ].min()
    )

    val_end = (
        test[
            "target_date"
        ].max()
    )

    if (
        train_end
        >= val_start
    ):
        raise ValueError(
            "Development and final test "
            "periods overlap."
        )

    return pd.Series(
        {
            # run_sequential_fold expects a fold value.
            # Zero identifies the single final-test fold.
            "fold": 0,

            "train_start":
                train_start,

            "train_end":
                train_end,

            "val_start":
                val_start,

            "val_end":
                val_end,
        }
    )


def summarize_test_metrics(
    metrics: pd.DataFrame,
) -> dict:
    """
    Summarize final-test metrics across neighborhoods.

    Each neighborhood contributes equally to the summary.
    """

    if metrics.empty:
        raise ValueError(
            "Final-test metric table is empty."
        )

    return {
        "n_neighborhoods":
            int(
                metrics[
                    "neighborhood"
                ].nunique()
            ),

        "mean_mase":
            float(
                metrics[
                    "mase"
                ].mean()
            ),

        "median_mase":
            float(
                metrics[
                    "mase"
                ].median()
            ),

        "mean_mae":
            float(
                metrics[
                    "mae"
                ].mean()
            ),

        "median_mae":
            float(
                metrics[
                    "mae"
                ].median()
            ),

        "mean_rmse":
            float(
                metrics[
                    "rmse"
                ].mean()
            ),

        "mean_smape":
            float(
                metrics[
                    "smape"
                ].mean()
            ),

        "mean_bias":
            float(
                metrics[
                    "bias"
                ].mean()
            ),

        "mean_abs_bias":
            float(
                metrics[
                    "bias"
                ]
                .abs()
                .mean()
            ),

        "pct_neighborhoods_mase_below_1":
            float(
                100.0
                * (
                    metrics[
                        "mase"
                    ]
                    < 1.0
                ).mean()
            ),
    }


# ---------------------------------------------------------------------
# Single finalist
# ---------------------------------------------------------------------

def evaluate_finalist(
    *,
    config_row: pd.Series,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    fold_row: pd.Series,
    feature_set_name: str,
    locked_config_id: str,
) -> dict:
    """
    Evaluate one already-selected tuning configuration on the
    reserved final test year.
    """

    config_id = str(
        config_row[
            "config_id"
        ]
    )

    tuning_rank = int(
        config_row[
            "rank"
        ]
    )

    params = parse_params_json(
        config_row[
            "params_json"
        ]
    )

    # XGBoost parallelism remains one thread per model.
    params[
        "n_jobs"
    ] = 1

    from forecasting.backtests.xgboost import (
        resolve_numeric_features,
    )

    numeric_features = (
        resolve_numeric_features(
            feature_set_name=
                feature_set_name
        )
    )

    model = (
        build_xgboost_pipeline(
            numeric_features=
                numeric_features,

            model_params=
                params,
        )
    )

    print()
    print(
        f"Evaluating tuning rank "
        f"{tuning_rank}: "
        f"{config_id}"
    )

    result = (
        run_sequential_fold(
            feature_panel=
                feature_panel,

            target_panel=
                target_panel,

            fold_row=
                fold_row,

            feature_set_name=
                feature_set_name,

            numeric_features=
                numeric_features,

            model=
                model,
        )
    )

    predictions = (
        result[
            "predictions"
        ]
        .copy()
    )

    metrics = (
        result[
            "metrics"
        ]
        .copy()
    )

    diagnostics = (
        result[
            "diagnostics"
        ]
        .copy()
    )

    is_locked = (
        config_id
        == locked_config_id
    )

    for frame in [
        predictions,
        metrics,
        diagnostics,
    ]:
        frame[
            "config_id"
        ] = config_id

        frame[
            "tuning_rank"
        ] = tuning_rank

        frame[
            "locked_candidate"
        ] = is_locked

    summary = (
        summarize_test_metrics(
            metrics
        )
    )

    summary.update(
        {
            "config_id":
                config_id,

            "tuning_rank":
                tuning_rank,

            "locked_candidate":
                is_locked,

            "development_mean_fold_mase":
                float(
                    config_row[
                        "mean_fold_mase"
                    ]
                ),

            "development_median_mase":
                float(
                    config_row[
                        "median_mase"
                    ]
                ),

            "params_json":
                config_row[
                    "params_json"
                ],
        }
    )

    # --------------------------------------------------------------
    # Ranking evaluation
    # --------------------------------------------------------------

    daily_top10 = (
        evaluate_top10_predictions(
            predictions=
                predictions,

            top_k=10,
        )
    )

    daily_top10[
        "config_id"
    ] = config_id

    daily_top10[
        "tuning_rank"
    ] = tuning_rank

    daily_top10[
        "locked_candidate"
    ] = is_locked

    top10_summary = (
        summarize_top10(
            daily_top10
        )
    )

    top10_summary[
        "config_id"
    ] = config_id

    top10_summary[
        "tuning_rank"
    ] = tuning_rank

    top10_summary[
        "locked_candidate"
    ] = is_locked

    return {
        "predictions":
            predictions,

        "metrics":
            metrics,

        "diagnostics":
            diagnostics,

        "summary":
            summary,

        "daily_top10":
            daily_top10,

        "top10_summary":
            top10_summary,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the top tuned XGBRegressor configurations "
            "on the reserved final 365-day test set."
        )
    )

    parser.add_argument(
        "--tuning-results",
        required=True,
        help=(
            "Path to configuration_results.csv "
            "from the completed tuning run."
        ),
    )

    parser.add_argument(
        "--target-panel",
        default=None,
        help=(
            "Optional target panel override. Defaults to the tuning "
            "manifest, then the repository default."
        ),
    )

    parser.add_argument(
        "--feature-panel",
        default=None,
        help=(
            "Optional feature panel override. Defaults to the tuning "
            "manifest, then the repository default."
        ),
    )

    parser.add_argument(
        "--feature-set",
        default=
            DEFAULT_FEATURE_SET,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=
            DEFAULT_TOP_N,
    )

    parser.add_argument(
        "--locked-config-id",
        default=
            DEFAULT_LOCKED_CONFIG_ID,
        help=(
            "Candidate selected BEFORE final-test evaluation. "
            "Other candidates are benchmark-only."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/backtest/xgboost/"
            "finalist_test_benchmark"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.top_n < 1:
        raise ValueError(
            "--top-n must be >= 1."
        )

    tuning_path = Path(
        args.tuning_results
    )

    tuning_results = (
        pd.read_csv(
            tuning_path
        )
    )

    required_columns = {
        "rank",
        "config_id",
        "feature_set",
        "mean_fold_mase",
        "median_mase",
        "params_json",
    }

    missing = (
        required_columns
        - set(
            tuning_results.columns
        )
    )

    if missing:
        raise ValueError(
            "Tuning results are missing columns: "
            f"{sorted(missing)}"
        )

    finalists = (
        tuning_results.loc[
            tuning_results[
                "feature_set"
            ]
            == args.feature_set
        ]
        .sort_values(
            "rank"
        )
        .head(
            args.top_n
        )
        .copy()
    )

    if finalists.empty:
        raise ValueError(
            "No finalists were found."
        )

    finalist_ids = set(
        finalists[
            "config_id"
        ].astype(str)
    )

    if (
        args.locked_config_id
        not in finalist_ids
    ):
        raise ValueError(
            "The locked production candidate must "
            "be included among the evaluated finalists."
        )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    target_path, feature_path, tuning_manifest_path = (
        resolve_data_paths(
            tuning_results_path=tuning_path,
            target_panel_override=args.target_panel,
            feature_panel_override=args.feature_panel,
        )
    )

    print(f"Tuning manifest: {tuning_manifest_path}")
    print(f"Target panel: {target_path}")
    print(f"Feature panel: {feature_path}")

    target_panel = (
        prepare_target_panel(
            pd.read_parquet(
                target_path
            )
        )
    )

    feature_panel = (
        load_or_build_feature_panel(
            target_panel=
                target_panel,

            feature_panel_path=
                feature_path,

            rebuild=False,
        )
    )

    final_fold = (
        build_final_test_fold(
            target_panel
        )
    )

    # --------------------------------------------------------------
    # Create immutable-looking run directory
    # --------------------------------------------------------------

    timestamp = (
        datetime
        .now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------------
    # Record the locked candidate BEFORE evaluation
    # --------------------------------------------------------------

    locked_row = (
        finalists.loc[
            finalists[
                "config_id"
            ]
            .astype(str)
            == args.locked_config_id
        ]
        .iloc[0]
    )

    locked_manifest = {
        "locked_before_test":
            True,

        "locked_config_id":
            args.locked_config_id,

        "locked_tuning_rank":
            int(
                locked_row[
                    "rank"
                ]
            ),

        "feature_set":
            args.feature_set,

        "params":
            parse_params_json(
                locked_row[
                    "params_json"
                ]
            ),

        "policy": (
            "Other finalist configurations are evaluated "
            "for benchmarking and robustness only. "
            "Final-test performance must not be used to "
            "replace the locked production candidate while "
            "claiming this period remains an untouched holdout."
        ),
    }

    write_json(
        locked_manifest,
        output_dir
        / "selected_before_test.json",
    )

    print()
    print(
        "LOCKED BEFORE TEST:"
    )

    print(
        f"Config: "
        f"{args.locked_config_id}"
    )

    print(
        f"Tuning rank: "
        f"{int(locked_row['rank'])}"
    )

    print()
    print(
        "Final test period:"
    )

    print(
        f"{final_fold['val_start'].date()} "
        f"-> "
        f"{final_fold['val_end'].date()}"
    )

    print()
    print(
        "NOTE: Other finalists are benchmark-only."
    )

    # --------------------------------------------------------------
    # Evaluate finalists
    # --------------------------------------------------------------

    prediction_frames = []
    metric_frames = []
    diagnostic_frames = []
    summary_rows = []
    daily_top10_frames = []
    top10_summary_frames = []

    for _, config_row in (
        finalists
        .sort_values("rank")
        .iterrows()
    ):
        result = (
            evaluate_finalist(
                config_row=
                    config_row,

                feature_panel=
                    feature_panel,

                target_panel=
                    target_panel,

                fold_row=
                    final_fold,

                feature_set_name=
                    args.feature_set,

                locked_config_id=
                    args.locked_config_id,
            )
        )

        prediction_frames.append(
            result[
                "predictions"
            ]
        )

        metric_frames.append(
            result[
                "metrics"
            ]
        )

        diagnostic_frames.append(
            result[
                "diagnostics"
            ]
        )

        summary_rows.append(
            result[
                "summary"
            ]
        )

        daily_top10_frames.append(
            result[
                "daily_top10"
            ]
        )

        top10_summary_frames.append(
            result[
                "top10_summary"
            ]
        )

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "tuning_rank",
                "target_date",
                "neighborhood",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    metrics = (
        pd.concat(
            metric_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "tuning_rank",
                "neighborhood",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    diagnostics = (
        pd.concat(
            diagnostic_frames,
            ignore_index=True,
        )
        .sort_values(
            "tuning_rank"
        )
        .reset_index(
            drop=True
        )
    )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "tuning_rank"
        )
        .reset_index(
            drop=True
        )
    )

    daily_top10 = (
        pd.concat(
            daily_top10_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "tuning_rank",
                "target_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    top10_summary = (
        pd.concat(
            top10_summary_frames,
            ignore_index=True,
        )
        .sort_values(
            "tuning_rank"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    predictions.to_parquet(
        output_dir
        / "predictions.parquet",
        index=False,
    )

    metrics.to_parquet(
        output_dir
        / "neighborhood_metrics.parquet",
        index=False,
    )

    diagnostics.to_parquet(
        output_dir
        / "diagnostics.parquet",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "finalist_summary.csv",
        index=False,
    )

    daily_top10.to_parquet(
        output_dir
        / "daily_top10_results.parquet",
        index=False,
    )

    top10_summary.to_csv(
        output_dir
        / "top10_summary.csv",
        index=False,
    )

    manifest = {
        "created_at_utc":
            timestamp,

        "test_days":
            TEST_DAYS,

        "test_start":
            final_fold[
                "val_start"
            ],

        "test_end":
            final_fold[
                "val_end"
            ],

        "feature_set":
            args.feature_set,

        "top_n":
            args.top_n,

        "locked_config_id":
            args.locked_config_id,

        "tuning_results":
            str(
                tuning_path
            ),

        "tuning_manifest":
            str(
                tuning_manifest_path
            ),

        "target_panel":
            str(
                target_path
            ),

        "feature_panel":
            str(
                feature_path
            ),

        "benchmark_only_config_ids":
            [
                config_id
                for config_id
                in finalists[
                    "config_id"
                ].astype(str)
                if (
                    config_id
                    != args.locked_config_id
                )
            ],
    }

    write_json(
        manifest,
        output_dir
        / "manifest.json",
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    display_columns = [
        "tuning_rank",
        "config_id",
        "locked_candidate",
        "development_mean_fold_mase",
        "mean_mase",
        "median_mase",
        "mean_mae",
        "mean_rmse",
        "mean_smape",
        "mean_bias",
        "mean_abs_bias",
        "pct_neighborhoods_mase_below_1",
    ]

    print()
    print(
        "Final-test regression performance:"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Ranking performance:"
    )

    print(
        top10_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Outputs saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
