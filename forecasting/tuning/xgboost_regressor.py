from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

from joblib import Parallel, delayed

from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
)

from forecasting.backtests.xgboost import (
    DEFAULT_XGB_REGRESSOR_PARAMS,
    TEST_DAYS,
    build_standard_backtest_folds,
    build_xgboost_pipeline,
    load_or_build_feature_panel,
    run_sequential_fold,
    validate_folds,
)
from forecasting.features.xgboost import (
    XGB_FEATURE_SETS,
    prepare_target_panel,
)
from forecasting.paths import (
    TARGET_PANEL_5Y_PATH,
    XGBOOST_FEATURE_PANEL_PATH,
    XGBOOST_TUNING_DIR,
)


DEFAULT_FEATURE_SET = "lags_rolling_calendar"
DEFAULT_SEARCH_MODE = "random"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ITER = 60

RANDOM_SEARCH_SPACE = {
    "n_estimators": [
        300,
        500,
        700,
        1000,
        1400,
    ],
    "learning_rate": [
        0.015,
        0.025,
        0.03,
        0.05,
        0.075,
        0.10,
    ],
    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        8,
    ],
    "min_child_weight": [
        1,
        3,
        5,
        8,
        12,
        20,
    ],
    "subsample": [
        0.65,
        0.75,
        0.85,
        0.95,
        1.0,
    ],
    "colsample_bytree": [
        0.65,
        0.75,
        0.85,
        0.95,
        1.0,
    ],
    "reg_lambda": [
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
    ],
    "reg_alpha": [
        0.0,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
    ],
    "gamma": [
        0.0,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
    ],
}

GRID_SEARCH_SPACE = {
    "n_estimators": [
        300,
        500,
        700,
    ],
    "learning_rate": [
        0.025,
        0.03,
        0.05,
    ],
    "max_depth": [
        3,
        4,
        5,
    ],
    "min_child_weight": [
        3,
        5,
        8,
    ],
    "subsample": [
        0.75,
        0.85,
        1.0,
    ],
    "colsample_bytree": [
        0.75,
        0.85,
        1.0,
    ],
    "reg_lambda": [
        0.5,
        1.0,
        2.0,
    ],
    "reg_alpha": [
        0.0,
        0.05,
        0.1,
    ],
    "gamma": [
        0.0,
        0.05,
        0.1,
    ],
}

PRIMARY_SELECTION_METRIC = "mean_fold_mase"


def resolve_worker_count(
    n_workers: int | None = None,
    detected_cpu_count: int | None = None,
) -> tuple[int, int]:
    detected = (
        os.cpu_count()
        if detected_cpu_count is None
        else detected_cpu_count
    )
    detected = detected or 2

    if n_workers is not None and n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    return (
        n_workers
        if n_workers is not None
        else max(1, detected - 1),
        detected,
    )


def canonicalize_params(
    model_params: dict | None,
) -> dict:
    resolved_params = {
        **DEFAULT_XGB_REGRESSOR_PARAMS,
        **(
            dict(model_params)
            if model_params is not None
            else {}
        ),
    }

    # Candidate-level processes provide concurrency; keeping each model
    # single-threaded prevents workers from oversubscribing the CPU.
    resolved_params["n_jobs"] = 1
    return resolved_params


def make_config_id(
    model_params: dict,
) -> str:
    canonical_json = json.dumps(
        canonicalize_params(
            model_params
        ),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()[:12]


def _baseline_candidate() -> dict:
    params = canonicalize_params(
        DEFAULT_XGB_REGRESSOR_PARAMS
    )

    return {
        "config_id": make_config_id(
            params
        ),
        "model_params": params,
        "is_baseline": True,
        "search_source": "baseline",
    }


def generate_parameter_candidates(
    search_mode: str = DEFAULT_SEARCH_MODE,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    random_search_space: dict | None = None,
    grid_search_space: dict | None = None,
) -> list[dict]:
    if search_mode not in {
        "random",
        "grid",
    }:
        raise ValueError(
            f"Unknown search mode: {search_mode}"
        )

    baseline = _baseline_candidate()

    if search_mode == "random":
        sampled_params = ParameterSampler(
            param_distributions=(
                random_search_space
                or RANDOM_SEARCH_SPACE
            ),
            n_iter=n_iter,
            random_state=random_state,
        )

        source_label = "random"

    else:
        sampled_params = ParameterGrid(
            grid_search_space
            or GRID_SEARCH_SPACE
        )
        source_label = "grid"

    candidates_by_id = {
        baseline["config_id"]: baseline
    }

    for sampled in sampled_params:
        params = canonicalize_params(
            sampled
        )
        config_id = make_config_id(
            params
        )

        candidates_by_id.setdefault(
            config_id,
            {
                "config_id": config_id,
                "model_params": params,
                "is_baseline": (
                    params
                    == DEFAULT_XGB_REGRESSOR_PARAMS
                ),
                "search_source": source_label,
            },
        )

    return list(
        candidates_by_id.values()
    )


def compute_file_sha256(
    path: str | Path,
) -> str | None:
    file_path = Path(path)

    if not file_path.exists():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip() or None


def get_reserved_test_start(
    target_panel: pd.DataFrame,
    test_days: int = TEST_DAYS,
) -> pd.Timestamp:
    max_target_date = pd.to_datetime(
        target_panel["target_date"],
        errors="raise",
    ).max()

    return (
        pd.Timestamp(max_target_date)
        - pd.Timedelta(days=test_days - 1)
    ).normalize()


def validate_folds_exclude_final_test_period(
    folds: pd.DataFrame,
    target_panel: pd.DataFrame,
    test_days: int = TEST_DAYS,
) -> pd.Timestamp:
    reserved_test_start = get_reserved_test_start(
        target_panel=target_panel,
        test_days=test_days,
    )

    offending = folds.loc[
        pd.to_datetime(
            folds["val_end"],
            errors="raise",
        )
        .dt.normalize()
        >= reserved_test_start
    ]

    if not offending.empty:
        raise ValueError(
            "Validation folds must end before the "
            "reserved final test period begins."
        )

    return reserved_test_start


def build_run_output_dir(
    base_dir: str | Path,
    feature_set_name: str,
    search_mode: str,
    fold_limit: int | None = None,
) -> Path:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_name = (
        f"{timestamp}_{feature_set_name}_{search_mode}"
    )

    if fold_limit is not None:
        run_name += (
            f"_foldlimit{fold_limit}"
        )

    output_dir = Path(base_dir) / run_name

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def summarize_candidate_metrics(
    metrics: pd.DataFrame,
) -> dict:
    fold_means = (
        metrics.groupby("fold", sort=True)
        .agg(
            fold_mean_mae=("mae", "mean"),
            fold_mean_rmse=("rmse", "mean"),
            fold_mean_mase=("mase", "mean"),
            fold_mean_smape=("smape", "mean"),
            fold_mean_bias=("bias", "mean"),
            fold_mean_abs_bias=(
                "bias",
                lambda series: np.abs(
                    series.to_numpy(
                        dtype=float
                    )
                ).mean(),
            ),
        )
        .reset_index()
    )

    return {
        "mean_fold_mase": float(
            fold_means[
                "fold_mean_mase"
            ].mean()
        ),
        "median_mase": float(
            metrics["mase"].median()
        ),
        "std_fold_mase": float(
            fold_means[
                "fold_mean_mase"
            ].std(ddof=0)
        ),
        "worst_fold_mean_mase": float(
            fold_means[
                "fold_mean_mase"
            ].max()
        ),
        "mean_fold_mae": float(
            fold_means[
                "fold_mean_mae"
            ].mean()
        ),
        "mean_fold_rmse": float(
            fold_means[
                "fold_mean_rmse"
            ].mean()
        ),
        "mean_fold_smape": float(
            fold_means[
                "fold_mean_smape"
            ].mean()
        ),
        "mean_fold_bias": float(
            fold_means[
                "fold_mean_bias"
            ].mean()
        ),
        "mean_fold_abs_bias": float(
            fold_means[
                "fold_mean_abs_bias"
            ].mean()
        ),
        "pct_jobs_mase_below_1": float(
            100.0
            * (
                metrics["mase"]
                < 1
            ).mean()
        ),
        "n_folds": int(
            fold_means["fold"].nunique()
        ),
        "n_neighborhoods": int(
            metrics["neighborhood"].nunique()
        ),
        "n_fold_neighborhood_jobs": int(
            len(metrics)
        ),
    }


def evaluate_parameter_configuration(
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    folds: pd.DataFrame,
    feature_set_name: str,
    model_params: dict,
    pipeline_builder=build_xgboost_pipeline,
) -> dict:
    metrics_frames = []
    prediction_frames = []
    diagnostic_frames = []

    numeric_features = list(
        XGB_FEATURE_SETS[
            feature_set_name
        ]
    )

    started_at = time.perf_counter()

    for _, fold_row in (
        folds.sort_values("fold")
        .iterrows()
    ):
        # No early stopping here: these outer validation folds are
        # reserved for unbiased model-selection scoring only.
        model = pipeline_builder(
            numeric_features=
                numeric_features,
            model_params=model_params,
        )

        fold_result = run_sequential_fold(
            feature_panel=
                feature_panel,
            target_panel=
                target_panel,
            fold_row=
                fold_row,
            feature_set_name=
                feature_set_name,
            model=
                model,
            numeric_features=
                numeric_features,
        )

        metrics_frames.append(
            fold_result["metrics"]
        )
        prediction_frames.append(
            fold_result["predictions"]
        )
        diagnostic_frames.append(
            fold_result["diagnostics"]
        )

    metrics = pd.concat(
        metrics_frames,
        ignore_index=True,
    )
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    diagnostics = pd.concat(
        diagnostic_frames,
        ignore_index=True,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    summary = summarize_candidate_metrics(
        metrics
    )
    summary["elapsed_seconds"] = float(
        elapsed_seconds
    )
    summary["n_model_fits"] = int(
        folds["fold"].nunique()
    )

    return {
        "summary": summary,
        "metrics": metrics,
        "predictions": predictions,
        "diagnostics": diagnostics,
    }


def evaluate_candidate(
    candidate: dict,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    folds: pd.DataFrame,
    feature_set_name: str,
) -> dict:
    """Return ordinary candidate failures to the parent instead of raising."""
    try:
        evaluation = evaluate_parameter_configuration(
            feature_panel=feature_panel,
            target_panel=target_panel,
            folds=folds,
            feature_set_name=feature_set_name,
            model_params=candidate["model_params"],
        )
    except Exception as exc:
        return {
            "candidate": candidate,
            "error": {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        }

    return {
        "candidate": candidate,
        "evaluation": evaluation,
    }


def build_configuration_result_row(
    candidate: dict,
    summary: dict,
    feature_set_name: str,
) -> dict:
    params = candidate[
        "model_params"
    ]

    return {
        "config_id": candidate["config_id"],
        "feature_set": feature_set_name,
        "search_source": candidate[
            "search_source"
        ],
        "is_baseline": candidate[
            "is_baseline"
        ],
        **summary,
        "params_json": json.dumps(
            params,
            sort_keys=True,
        ),
        **params,
    }


def rank_configuration_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    ranked = (
        results.sort_values(
            [
                "mean_fold_mase",
                "mean_fold_smape",
                "mean_fold_rmse",
                "mean_fold_abs_bias",
                "config_id",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
        .copy()
    )

    ranked.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranked) + 1,
            dtype=int,
        ),
    )

    return ranked


def write_checkpoint_artifacts(
    output_dir: Path,
    configuration_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    best_params: dict | None,
    failures: pd.DataFrame,
) -> None:
    ranked_results = rank_configuration_results(
        configuration_results
    )

    ranked_results.to_csv(
        output_dir
        / "configuration_results.csv",
        index=False,
    )

    ordered_metrics = (
        fold_metrics.sort_values(
            ["config_id", "fold", "neighborhood"]
        ).reset_index(drop=True)
        if not fold_metrics.empty
        else fold_metrics.copy()
    )

    ordered_metrics.to_parquet(
        output_dir
        / "fold_neighborhood_metrics.parquet",
        index=False,
    )

    if best_params is not None:
        (
            output_dir
            / "best_params.json"
        ).write_text(
            json.dumps(
                best_params,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    failures_path = (
        output_dir / "failures.csv"
    )

    if failures.empty:
        if failures_path.exists():
            failures_path.unlink()
    else:
        failures.sort_values("config_id").to_csv(
            failures_path,
            index=False,
        )


def build_manifest(
    target_panel_path: str | Path,
    feature_panel_path: str | Path,
    folds_path: str | Path | None,
    feature_set_name: str,
    search_mode: str,
    n_iter: int,
    fold_limit: int | None,
    reserved_test_start: pd.Timestamp,
    n_workers: int,
    detected_cpu_count: int,
) -> dict:
    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "feature_set": feature_set_name,
        "random_seed": DEFAULT_RANDOM_STATE,
        "search_mode": search_mode,
        "n_iter": n_iter,
        "fold_limit": fold_limit,
        "n_workers": n_workers,
        "detected_cpu_count": detected_cpu_count,
        "primary_selection_metric": (
            PRIMARY_SELECTION_METRIC
        ),
        "final_reserved_test_start_date": (
            reserved_test_start
            .date()
            .isoformat()
        ),
        "number_of_reserved_final_test_days": (
            TEST_DAYS
        ),
        "input_target_panel_path": str(
            Path(target_panel_path)
        ),
        "input_feature_panel_path": str(
            Path(feature_panel_path)
        ),
        "input_folds_path": (
            str(Path(folds_path))
            if folds_path is not None
            else None
        ),
        "input_target_panel_sha256": (
            compute_file_sha256(
                target_panel_path
            )
        ),
        "input_feature_panel_sha256": (
            compute_file_sha256(
                feature_panel_path
            )
        ),
        "python_version": (
            platform.python_version()
        ),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": (
            sklearn.__version__
        ),
        "xgboost_version": (
            xgboost.__version__
        ),
        "baseline_parameter_configuration": (
            deepcopy(
                DEFAULT_XGB_REGRESSOR_PARAMS
            )
        ),
        "parameter_search_space": {
            "random": deepcopy(
                RANDOM_SEARCH_SPACE
            ),
            "grid": deepcopy(
                GRID_SEARCH_SPACE
            ),
        },
    }


def run_tuning(
    target_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    folds: pd.DataFrame,
    feature_set_name: str,
    search_mode: str,
    n_iter: int,
    output_dir: str | Path,
    fold_limit: int | None = None,
    n_workers: int | None = None,
) -> dict:
    if feature_set_name not in XGB_FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: "
            f"{feature_set_name}"
        )

    ordered_folds = (
        folds.sort_values("fold")
        .reset_index(drop=True)
        .copy()
    )

    if fold_limit is not None:
        if fold_limit <= 0:
            raise ValueError(
                "fold_limit must be positive."
            )

        ordered_folds = ordered_folds.head(
            fold_limit
        ).copy()

    candidates = generate_parameter_candidates(
        search_mode=search_mode,
        n_iter=n_iter,
    )
    resolved_workers, _ = resolve_worker_count(n_workers)

    configuration_rows = []
    metric_frames = []
    failure_rows = []

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The generator yields completed candidates to this parent process, which
    # alone owns checkpoint files and avoids concurrent artifact writes.
    completed_results = Parallel(
        n_jobs=resolved_workers,
        return_as="generator_unordered",
    )(
        delayed(evaluate_candidate)(
            candidate=candidate,
            feature_panel=feature_panel,
            target_panel=target_panel,
            folds=ordered_folds,
            feature_set_name=feature_set_name,
        )
        for candidate in candidates
    )

    for candidate_index, completed in enumerate(
        completed_results,
        start=1,
    ):
        candidate = completed["candidate"]
        config_id = candidate["config_id"]
        print(
            f"[{candidate_index}/{len(candidates)}] "
            f"Completed config {config_id}."
        )

        if "error" in completed:
            error = completed["error"]
            failure_rows.append(
                {
                    "config_id": config_id,
                    "feature_set": feature_set_name,
                    "search_source": candidate[
                        "search_source"
                    ],
                    "is_baseline": candidate[
                        "is_baseline"
                    ],
                    "error_type": error["error_type"],
                    "error_message": error["error_message"],
                }
            )
        else:
            evaluation = completed["evaluation"]
            metrics = evaluation["metrics"].copy()
            metrics.insert(
                0,
                "config_id",
                config_id,
            )
            metrics.insert(
                1,
                "search_source",
                candidate["search_source"],
            )
            metrics.insert(
                2,
                "is_baseline",
                candidate["is_baseline"],
            )

            metric_frames.append(
                metrics
            )

            configuration_rows.append(
                build_configuration_result_row(
                    candidate=
                        candidate,
                    summary=
                        evaluation[
                            "summary"
                        ],
                    feature_set_name=
                        feature_set_name,
                )
            )

        configuration_results = pd.DataFrame(
            configuration_rows
        )
        fold_metrics = (
            pd.concat(
                metric_frames,
                ignore_index=True,
            )
            if metric_frames
            else pd.DataFrame()
        )
        failures = pd.DataFrame(
            failure_rows
        )

        best_params = None

        if not configuration_results.empty:
            best_row = (
                rank_configuration_results(
                    configuration_results
                )
                .iloc[0]
            )
            best_params = json.loads(
                best_row["params_json"]
            )

        write_checkpoint_artifacts(
            output_dir=output_path,
            configuration_results=
                configuration_results,
            fold_metrics=
                fold_metrics,
            best_params=best_params,
            failures=failures,
        )

    ranked_results = rank_configuration_results(
        pd.DataFrame(
            configuration_rows
        )
    )
    fold_metrics = (
        pd.concat(
            metric_frames,
            ignore_index=True,
        )
        if metric_frames
        else pd.DataFrame()
    )
    failures = pd.DataFrame(
        failure_rows
    )

    if not fold_metrics.empty:
        fold_metrics = fold_metrics.sort_values(
            ["config_id", "fold", "neighborhood"]
        ).reset_index(drop=True)

    if not failures.empty:
        failures = failures.sort_values(
            "config_id"
        ).reset_index(drop=True)

    return {
        "configuration_results": ranked_results,
        "fold_metrics": fold_metrics,
        "failures": failures,
        "candidates": candidates,
        "folds_used": ordered_folds,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune the leakage-safe XGBRegressor on "
            "the locked rolling-origin development "
            "folds while keeping the final 365-day "
            "test period untouched."
        )
    )

    parser.add_argument(
        "--target-panel",
        default=str(
            TARGET_PANEL_5Y_PATH
        ),
    )
    parser.add_argument(
        "--feature-panel",
        default=str(
            XGBOOST_FEATURE_PANEL_PATH
        ),
    )
    parser.add_argument(
        "--folds",
        default=None,
        help=(
            "Optional Parquet fold table. Folds must "
            "still end before the reserved final "
            "365-day test period."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=list(
            XGB_FEATURE_SETS.keys()
        ),
        default=DEFAULT_FEATURE_SET,
    )
    parser.add_argument(
        "--search-mode",
        choices=[
            "random",
            "grid",
        ],
        default=DEFAULT_SEARCH_MODE,
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=(
            "Number of random candidates to sample "
            "before baseline deduplication. Ignored "
            "for grid search."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            XGBOOST_TUNING_DIR
        ),
        help=(
            "Base directory for timestamped tuning "
            "run outputs."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Parallel hyperparameter configurations. "
            "Defaults to all but one logical CPU."
        ),
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
        help=(
            "Optional debug-only cap on the earliest "
            "chronological folds to evaluate. Not "
            "valid for final model selection."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_workers, detected_cpu_count = resolve_worker_count(
        args.n_workers
    )

    print(f"Detected CPU count: {detected_cpu_count}")
    print(f"Selected worker count: {n_workers}")

    target_panel = pd.read_parquet(
        args.target_panel
    )
    target_panel = prepare_target_panel(
        target_panel
    )

    feature_panel = load_or_build_feature_panel(
        target_panel=target_panel,
        feature_panel_path=args.feature_panel,
        rebuild=False,
    )

    if args.folds:
        folds = pd.read_parquet(
            args.folds
        )
    else:
        folds = build_standard_backtest_folds(
            target_panel
        )

    validate_folds(folds)

    reserved_test_start = (
        validate_folds_exclude_final_test_period(
            folds=folds,
            target_panel=target_panel,
        )
    )

    output_dir = build_run_output_dir(
        base_dir=args.output_dir,
        feature_set_name=args.feature_set,
        search_mode=args.search_mode,
        fold_limit=args.fold_limit,
    )

    folds_used = (
        folds.sort_values("fold")
        .head(args.fold_limit)
        .copy()
        if args.fold_limit is not None
        else folds.sort_values("fold")
        .copy()
    )

    if args.fold_limit is not None:
        print(
            "WARNING: --fold-limit is for smoke "
            "testing/debugging only and is not valid "
            "for final model selection."
        )

    folds_used.to_parquet(
        output_dir / "folds.parquet",
        index=False,
    )

    manifest = build_manifest(
        target_panel_path=args.target_panel,
        feature_panel_path=args.feature_panel,
        folds_path=args.folds,
        feature_set_name=args.feature_set,
        search_mode=args.search_mode,
        n_iter=args.n_iter,
        fold_limit=args.fold_limit,
        reserved_test_start=reserved_test_start,
        n_workers=n_workers,
        detected_cpu_count=detected_cpu_count,
    )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    started_at = time.perf_counter()

    run_result = run_tuning(
        target_panel=target_panel,
        feature_panel=feature_panel,
        folds=folds,
        feature_set_name=args.feature_set,
        search_mode=args.search_mode,
        n_iter=args.n_iter,
        output_dir=output_dir,
        fold_limit=args.fold_limit,
        n_workers=n_workers,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    results = run_result[
        "configuration_results"
    ]

    print(
        "\nTuning complete."
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Elapsed seconds: "
        f"{elapsed_seconds:.2f}"
    )

    if not results.empty:
        print(
            "\nTop configurations:\n"
        )
        print(
            results.head(10).to_string(
                index=False
            )
        )
    else:
        print(
            "No successful configurations were "
            "completed."
        )

    if not run_result["failures"].empty:
        print(
            "\nSome configurations failed. See "
            f"{output_dir / 'failures.csv'}"
        )


if __name__ == "__main__":
    main()
