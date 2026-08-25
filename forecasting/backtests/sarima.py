# sarima_backtest.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm

from joblib import Parallel, delayed, parallel_config
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_VERSION = "v2"


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class BacktestError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        target_date: pd.Timestamp | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.target_date = target_date


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def get_neighborhood_series(
    target_panel: pd.DataFrame,
    neighborhood: str,
) -> pd.Series:

    series = (
        target_panel.loc[
            target_panel["neighborhood"].eq(neighborhood),
            ["target_date", "calls"],
        ]
        .sort_values("target_date")
        .set_index("target_date")["calls"]
    )

    series.index = pd.DatetimeIndex(series.index)

    # Force true daily frequency.
    series = series.asfreq("D")

    if series.empty:
        raise ValueError(
            f"No observations found for {neighborhood!r}."
        )

    if series.isna().any():
        raise ValueError(
            f"{neighborhood!r} contains "
            f"{int(series.isna().sum())} missing days."
        )

    if not series.index.is_unique:
        raise ValueError(
            f"{neighborhood!r} contains duplicate dates."
        )

    return series.astype(float)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def smape(y_true, y_pred) -> float:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    valid = denominator != 0

    if not np.any(valid):
        return np.nan

    return float(
        200
        * np.mean(
            np.abs(
                y_true[valid] - y_pred[valid]
            )
            / denominator[valid]
        )
    )


def mase(
    y_true,
    y_pred,
    y_train,
    seasonality: int = 7,
) -> float:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        return np.nan

    scale = np.mean(
        np.abs(
            y_train[seasonality:]
            - y_train[:-seasonality]
        )
    )

    if not np.isfinite(scale) or scale == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
        / scale
    )


def evaluate_forecast(
    y_true,
    y_pred,
    y_train,
    seasonality: int = 7,
) -> dict:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true

    return {
        "mae": float(
            np.mean(np.abs(error))
        ),
        "rmse": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "mase": mase(
            y_true,
            y_pred,
            y_train,
            seasonality=seasonality,
        ),
        "smape": smape(
            y_true,
            y_pred,
        ),
        "bias": float(
            np.mean(error)
        ),
    }


# ---------------------------------------------------------------------
# Auto-SARIMA order selection
# ---------------------------------------------------------------------

def select_sarima_order(
    y: pd.Series,
) -> dict:

    try:
        model = pm.auto_arima(
            y.astype(float),

            seasonal=True,
            m=7,

            d=None,
            D=None,

            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,

            start_P=0,
            start_Q=0,
            max_P=2,
            max_Q=2,

            max_d=1,
            max_D=1,

            information_criterion="aicc",

            stepwise=True,

            # Outer joblib layer owns parallelism.
            n_jobs=1,

            with_intercept="auto",

            maxiter=200,

            suppress_warnings=True,
            error_action="ignore",
            trace=False,

            enforce_stationarity=True,
            enforce_invertibility=True,
        )

    except Exception as exc:

        raise BacktestError(
            stage="order_selection",
            message=(
                f"Auto-ARIMA failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    return {
        "order": tuple(model.order),

        "seasonal_order": tuple(
            model.seasonal_order
        ),

        "with_intercept": bool(
            model.with_intercept
        ),

        "aic": float(model.aic()),
        "aicc": float(model.aicc()),
        "bic": float(model.bic()),
    }


# ---------------------------------------------------------------------
# SARIMA fitting with numerical retries
# ---------------------------------------------------------------------

def _fit_sarimax_with_retries(
    history: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    with_intercept: bool,
    start_params=None,
) -> dict:

    trend = (
        "c"
        if with_intercept
        else "n"
    )

    model = SARIMAX(
        history.astype(float),

        order=order,
        seasonal_order=seasonal_order,

        trend=trend,

        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    attempts = []

    # First try yesterday's parameters.
    if start_params is not None:
        attempts.append(
            (
                "lbfgs",
                np.asarray(
                    start_params,
                    dtype=float,
                ),
                200,
            )
        )

    # Technical fallbacks.
    attempts.extend(
        [
            ("lbfgs", None, 200),
            ("powell", None, 300),
        ]
    )

    messages = []

    for attempt_number, (
        method,
        params,
        maxiter,
    ) in enumerate(
        attempts,
        start=1,
    ):

        try:
            fit_kwargs = {
                "disp": False,
                "method": method,
                "maxiter": maxiter,
            }

            if (
                params is not None
                and np.isfinite(params).all()
            ):
                fit_kwargs[
                    "start_params"
                ] = params

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    **fit_kwargs
                )

            mle_retvals = (
                getattr(
                    result,
                    "mle_retvals",
                    {},
                )
                or {}
            )

            converged = bool(
                mle_retvals.get(
                    "converged",
                    False,
                )
            )

            finite_params = (
                np.isfinite(
                    np.asarray(
                        result.params,
                        dtype=float,
                    )
                )
                .all()
            )

            if (
                converged
                and finite_params
            ):
                return {
                    "result": result,
                    "fit_method": method,
                    "fit_attempts": (
                        attempt_number
                    ),
                    "warning_count": (
                        len(caught)
                    ),
                }

            messages.append(
                f"{method}: "
                f"converged={converged}, "
                f"finite_params={finite_params}"
            )

        except Exception as exc:

            messages.append(
                f"{method}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    raise BacktestError(
        stage="model_fit",
        message=(
            "All SARIMA fitting attempts "
            "failed. "
            + " | ".join(messages)
        ),
    )


# ---------------------------------------------------------------------
# Daily-refit rolling-origin backtest
# ---------------------------------------------------------------------

def rolling_sarima_backtest(
    train: pd.Series,
    validation: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    with_intercept: bool,
) -> pd.DataFrame:

    if train.empty:
        raise BacktestError(
            "data_validation",
            "Training series is empty.",
        )

    if validation.empty:
        raise BacktestError(
            "data_validation",
            "Validation series is empty.",
        )

    expected_first_validation = (
        train.index.max()
        + pd.Timedelta(days=1)
    )

    if (
        validation.index.min()
        != expected_first_validation
    ):
        raise BacktestError(
            "data_validation",
            (
                "Validation must begin "
                "exactly one day after "
                "training ends."
            ),
        )

    history = (
        train
        .astype(float)
        .copy()
    )

    # Initial fit for first forecast.
    fitted = (
        _fit_sarimax_with_retries(
            history=history,
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=(
                with_intercept
            ),
        )
    )

    result = fitted["result"]

    current_method = (
        fitted["fit_method"]
    )

    current_attempts = (
        fitted["fit_attempts"]
    )

    current_warning_count = (
        fitted["warning_count"]
    )

    rows = []

    validation_items = list(
        validation
        .astype(float)
        .items()
    )

    for i, (
        target_date,
        actual,
    ) in enumerate(
        validation_items
    ):

        # ---------------------------------
        # Forecast t+1
        # ---------------------------------

        try:
            forecast = result.forecast(
                steps=1
            )

            prediction = float(
                forecast.iloc[0]
            )

        except Exception as exc:

            raise BacktestError(
                stage="forecast",

                message=(
                    f"Forecast failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                target_date=pd.Timestamp(
                    target_date
                ),
            ) from exc

        if not np.isfinite(
            prediction
        ):
            raise BacktestError(
                stage="forecast",

                message=(
                    "Forecast was "
                    f"non-finite: "
                    f"{prediction}"
                ),

                target_date=pd.Timestamp(
                    target_date
                ),
            )

        rows.append(
            {
                "target_date": (
                    pd.Timestamp(
                        target_date
                    )
                ),

                "actual": float(
                    actual
                ),

                "prediction": (
                    prediction
                ),

                # residual = actual - forecast
                "residual": float(
                    actual
                    - prediction
                ),

                "fit_method": (
                    current_method
                ),

                "fit_attempts": (
                    current_attempts
                ),

                "fit_warning_count": (
                    current_warning_count
                ),
            }
        )

        # Last forecast doesn't need
        # another refit afterward.
        if (
            i
            == len(
                validation_items
            ) - 1
        ):
            continue

        # ---------------------------------
        # Reveal today's actual
        # ---------------------------------

        new_observation = pd.Series(
            [float(actual)],

            index=pd.DatetimeIndex(
                [target_date]
            ),

            name=history.name,
        )

        history = pd.concat(
            [
                history,
                new_observation,
            ]
        )

        # Warm-start tomorrow's refit.
        previous_params = np.asarray(
            result.params,
            dtype=float,
        )

        # ---------------------------------
        # Daily parameter refit
        # ---------------------------------

        try:
            fitted = (
                _fit_sarimax_with_retries(
                    history=history,

                    order=order,

                    seasonal_order=(
                        seasonal_order
                    ),

                    with_intercept=(
                        with_intercept
                    ),

                    start_params=(
                        previous_params
                    ),
                )
            )

        except BacktestError as exc:

            raise BacktestError(
                stage="daily_refit",

                message=str(exc),

                target_date=pd.Timestamp(
                    target_date
                ),
            ) from exc

        result = fitted["result"]

        current_method = (
            fitted["fit_method"]
        )

        current_attempts = (
            fitted["fit_attempts"]
        )

        current_warning_count = (
            fitted["warning_count"]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# One independent parallel job
# ---------------------------------------------------------------------

def run_baseline_job(
    series: pd.Series,
    neighborhood: str,
    fold: dict,
) -> dict:

    fold_number = int(
        fold["fold"]
    )

    selected = None

    try:
        train_start = pd.Timestamp(
            fold["train_start"]
        )

        train_end = pd.Timestamp(
            fold["train_end"]
        )

        val_start = pd.Timestamp(
            fold["val_start"]
        )

        val_end = pd.Timestamp(
            fold["val_end"]
        )

        initial_train = series.loc[
            train_start:train_end
        ].copy()

        validation = series.loc[
            val_start:val_end
        ].copy()

        if (
            initial_train.empty
            or validation.empty
        ):
            raise BacktestError(
                "data_validation",
                (
                    "Train or validation "
                    "slice is empty."
                ),
            )

        # ---------------------------------
        # Hyperparameter selection
        # ---------------------------------

        selected = (
            select_sarima_order(
                initial_train
            )
        )

        # ---------------------------------
        # Rolling backtest
        # ---------------------------------

        predictions = (
            rolling_sarima_backtest(
                train=initial_train,

                validation=validation,

                order=(
                    selected["order"]
                ),

                seasonal_order=(
                    selected[
                        "seasonal_order"
                    ]
                ),

                # Fixes the error from
                # the previous runner.
                with_intercept=(
                    selected[
                        "with_intercept"
                    ]
                ),
            )
        )

        metrics = evaluate_forecast(
            y_true=(
                predictions["actual"]
            ),

            y_pred=(
                predictions["prediction"]
            ),

            y_train=initial_train,

            seasonality=7,
        )

        predictions.insert(
            0,
            "neighborhood",
            neighborhood,
        )

        predictions.insert(
            0,
            "fold",
            fold_number,
        )

        predictions["order"] = str(
            selected["order"]
        )

        predictions[
            "seasonal_order"
        ] = str(
            selected["seasonal_order"]
        )

        predictions[
            "with_intercept"
        ] = selected[
            "with_intercept"
        ]

        return {
            "status": "ok",
            "fold": fold_number,
            "neighborhood": neighborhood,
            "selected": selected,
            "predictions": predictions,
            "metrics": metrics,
            "error": None,
        }

    except BacktestError as exc:

        return {
            "status": "failed",
            "fold": fold_number,
            "neighborhood": neighborhood,
            "selected": selected,
            "predictions": None,
            "metrics": None,

            "error": {
                "stage": exc.stage,

                "failed_date": (
                    exc.target_date
                ),

                "error_type": (
                    type(exc).__name__
                ),

                "error_message": (
                    str(exc)
                ),
            },
        }

    except Exception as exc:

        return {
            "status": "failed",
            "fold": fold_number,
            "neighborhood": neighborhood,
            "selected": selected,
            "predictions": None,
            "metrics": None,

            "error": {
                "stage": "unexpected",
                "failed_date": None,

                "error_type": (
                    type(exc).__name__
                ),

                "error_message": (
                    str(exc)
                ),
            },
        }


# ---------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------

def _checkpoint_paths(
    output_dir: Path,
) -> dict:

    prefix = (
        f"baseline_sarima_"
        f"{MODEL_VERSION}"
    )

    return {
        "orders": (
            output_dir
            / f"{prefix}_orders.parquet"
        ),

        "predictions": (
            output_dir
            / f"{prefix}_predictions.parquet"
        ),

        "metrics": (
            output_dir
            / f"{prefix}_metrics.parquet"
        ),

        "failures": (
            output_dir
            / f"{prefix}_failures.parquet"
        ),
    }


def _empty_frames() -> dict:

    return {
        "orders": pd.DataFrame(
            columns=[
                "fold",
                "neighborhood",
                "order",
                "seasonal_order",
                "with_intercept",
                "aic",
                "aicc",
                "bic",
            ]
        ),

        "predictions": pd.DataFrame(
            columns=[
                "fold",
                "neighborhood",
                "target_date",
                "actual",
                "prediction",
                "residual",
                "fit_method",
                "fit_attempts",
                "fit_warning_count",
                "order",
                "seasonal_order",
                "with_intercept",
            ]
        ),

        "metrics": pd.DataFrame(
            columns=[
                "fold",
                "neighborhood",
                "mae",
                "rmse",
                "mase",
                "smape",
                "bias",
            ]
        ),

        "failures": pd.DataFrame(
            columns=[
                "fold",
                "neighborhood",
                "stage",
                "failed_date",
                "error_type",
                "error_message",
                "order",
                "seasonal_order",
                "with_intercept",
            ]
        ),
    }


def _load_checkpoints(
    output_dir: Path,
) -> dict:

    frames = _empty_frames()

    paths = _checkpoint_paths(
        output_dir
    )

    for name, path in paths.items():

        if path.exists():
            frames[name] = (
                pd.read_parquet(path)
            )

    return frames


def _drop_job_rows(
    df: pd.DataFrame,
    fold: int,
    neighborhood: str,
) -> pd.DataFrame:

    if df.empty:
        return df

    mask = (
        df["fold"]
        .astype(int)
        .eq(int(fold))
        &
        df["neighborhood"]
        .eq(neighborhood)
    )

    return df.loc[
        ~mask
    ].copy()


def _record_result(
    frames: dict,
    job_result: dict,
) -> None:

    fold = int(
        job_result["fold"]
    )

    neighborhood = (
        job_result[
            "neighborhood"
        ]
    )

    # Remove any prior record
    # for this exact job.
    for name in frames:

        frames[name] = (
            _drop_job_rows(
                frames[name],
                fold,
                neighborhood,
            )
        )

    selected = (
        job_result["selected"]
    )

    # ---------------------------------
    # Save selected order even when
    # later forecasting fails.
    # ---------------------------------

    if selected is not None:

        order_row = pd.DataFrame(
            [
                {
                    "fold": fold,

                    "neighborhood": (
                        neighborhood
                    ),

                    "order": str(
                        selected["order"]
                    ),

                    "seasonal_order": str(
                        selected[
                            "seasonal_order"
                        ]
                    ),

                    "with_intercept": (
                        selected[
                            "with_intercept"
                        ]
                    ),

                    "aic": (
                        selected["aic"]
                    ),

                    "aicc": (
                        selected["aicc"]
                    ),

                    "bic": (
                        selected["bic"]
                    ),
                }
            ]
        )

        frames["orders"] = (
            pd.concat(
                [
                    frames["orders"],
                    order_row,
                ],
                ignore_index=True,
            )
        )

    # ---------------------------------
    # Successful job
    # ---------------------------------

    if (
        job_result["status"]
        == "ok"
    ):

        frames["predictions"] = (
            pd.concat(
                [
                    frames[
                        "predictions"
                    ],

                    job_result[
                        "predictions"
                    ],
                ],

                ignore_index=True,
            )
        )

        metric_row = {
            "fold": fold,
            "neighborhood": (
                neighborhood
            ),
            **job_result["metrics"],
        }

        frames["metrics"] = (
            pd.concat(
                [
                    frames["metrics"],
                    pd.DataFrame(
                        [metric_row]
                    ),
                ],

                ignore_index=True,
            )
        )

    # ---------------------------------
    # Failed job
    # ---------------------------------

    else:

        error = (
            job_result["error"]
        )

        failure_row = {
            "fold": fold,

            "neighborhood": (
                neighborhood
            ),

            "stage": (
                error["stage"]
            ),

            "failed_date": (
                error["failed_date"]
            ),

            "error_type": (
                error["error_type"]
            ),

            "error_message": (
                error["error_message"]
            ),

            "order": (
                str(selected["order"])
                if selected is not None
                else None
            ),

            "seasonal_order": (
                str(
                    selected[
                        "seasonal_order"
                    ]
                )
                if selected is not None
                else None
            ),

            "with_intercept": (
                selected[
                    "with_intercept"
                ]
                if selected is not None
                else None
            ),
        }

        frames["failures"] = (
            pd.concat(
                [
                    frames["failures"],

                    pd.DataFrame(
                        [failure_row]
                    ),
                ],

                ignore_index=True,
            )
        )

    _normalize_checkpoint_dtypes(
        frames
    )


def _normalize_checkpoint_dtypes(
    frames: dict,
) -> None:

    # Predictions
    prediction_numeric = [
        "actual",
        "prediction",
        "residual",
        "fit_attempts",
        "fit_warning_count",
    ]

    for col in prediction_numeric:
        if col in frames["predictions"].columns:
            frames["predictions"][col] = pd.to_numeric(
                frames["predictions"][col],
                errors="coerce",
            )

    if "fold" in frames["predictions"].columns:
        frames["predictions"]["fold"] = pd.to_numeric(
            frames["predictions"]["fold"],
            errors="coerce",
        ).astype("Int64")

    if "target_date" in frames["predictions"].columns:
        frames["predictions"]["target_date"] = pd.to_datetime(
            frames["predictions"]["target_date"],
            errors="coerce",
        )

    # Metrics
    metric_numeric = [
        "mae",
        "rmse",
        "mase",
        "smape",
        "bias",
    ]

    for col in metric_numeric:
        if col in frames["metrics"].columns:
            frames["metrics"][col] = pd.to_numeric(
                frames["metrics"][col],
                errors="coerce",
            )

    if "fold" in frames["metrics"].columns:
        frames["metrics"]["fold"] = pd.to_numeric(
            frames["metrics"]["fold"],
            errors="coerce",
        ).astype("Int64")

    # Auto-SARIMA information criteria
    for col in ["aic", "aicc", "bic"]:
        if col in frames["orders"].columns:
            frames["orders"][col] = pd.to_numeric(
                frames["orders"][col],
                errors="coerce",
            )

    if "fold" in frames["orders"].columns:
        frames["orders"]["fold"] = pd.to_numeric(
            frames["orders"]["fold"],
            errors="coerce",
        ).astype("Int64")

    # Failures
    if "fold" in frames["failures"].columns:
        frames["failures"]["fold"] = pd.to_numeric(
            frames["failures"]["fold"],
            errors="coerce",
        ).astype("Int64")

    if "failed_date" in frames["failures"].columns:
        frames["failures"]["failed_date"] = pd.to_datetime(
            frames["failures"]["failed_date"],
            errors="coerce",
        )

def _save_checkpoints(
    output_dir: Path,
    frames: dict,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _normalize_checkpoint_dtypes(
        frames
    )

    paths = _checkpoint_paths(
        output_dir
    )

    for name, df in frames.items():

        df.to_parquet(
            paths[name],
            index=False,
        )


# ---------------------------------------------------------------------
# Main parallel runner
# ---------------------------------------------------------------------

def run_all_baseline_backtests(
    target_panel: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    output_dir,
    n_jobs: int = 7,
    checkpoint_every: int = 5,
    neighborhoods: Iterable[str] | None = None,
    rerun_failed: bool = False,
    max_jobs: int | None = None,
    verbose: int = 10,
) -> dict:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = _load_checkpoints(
        output_dir
    )

    if neighborhoods is None:

        neighborhood_list = sorted(
            target_panel[
                "neighborhood"
            ]
            .dropna()
            .unique()
        )

    else:
        neighborhood_list = list(
            neighborhoods
        )

    # Build each series once in parent.
    series_by_neighborhood = {
        neighborhood:
        get_neighborhood_series(
            target_panel,
            neighborhood,
        )

        for neighborhood
        in neighborhood_list
    }

    successful_keys = set(
        zip(
            frames["metrics"][
                "fold"
            ].astype(int),

            frames["metrics"][
                "neighborhood"
            ],
        )
    )

    failed_keys = set(
        zip(
            frames["failures"][
                "fold"
            ].astype(int),

            frames["failures"][
                "neighborhood"
            ],
        )
    )

    if rerun_failed:
        completed_keys = (
            successful_keys
        )

    else:
        completed_keys = (
            successful_keys
            | failed_keys
        )

    jobs = []

    for fold in (
        backtest_folds
        .to_dict(
            orient="records"
        )
    ):

        fold_number = int(
            fold["fold"]
        )

        for neighborhood in (
            neighborhood_list
        ):

            key = (
                fold_number,
                neighborhood,
            )

            if key in completed_keys:
                continue

            jobs.append(
                (
                    series_by_neighborhood[
                        neighborhood
                    ],

                    neighborhood,

                    fold,
                )
            )

    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    print(
        f"Successful jobs already saved: "
        f"{len(successful_keys)}"
    )

    print(
        f"Failed jobs already saved: "
        f"{len(failed_keys)}"
    )

    print(
        f"Jobs to run now: "
        f"{len(jobs)}"
    )

    print(
        f"Parallel workers: "
        f"{n_jobs}"
    )

    if not jobs:
        return frames

    completed_this_run = 0

    # 7 independent Python workers.
    # Each native math library gets only
    # one internal thread.
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):

        result_generator = Parallel(
            n_jobs=n_jobs,

            verbose=verbose,

            # Parent gets completed jobs
            # as soon as workers finish.
            return_as=(
                "generator_unordered"
            ),
        )(
            delayed(
                run_baseline_job
            )(
                series,
                neighborhood,
                fold,
            )

            for (
                series,
                neighborhood,
                fold,
            )
            in jobs
        )

        for job_result in (
            result_generator
        ):

            _record_result(
                frames,
                job_result,
            )

            completed_this_run += 1

            print(
                f"[{completed_this_run}/"
                f"{len(jobs)}] "
                f"{job_result['status'].upper()} "
                f"| fold="
                f"{job_result['fold']} "
                f"| neighborhood="
                f"{job_result['neighborhood']}"
            )

            if (
                checkpoint_every > 0
                and
                completed_this_run
                % checkpoint_every
                == 0
            ):

                _save_checkpoints(
                    output_dir,
                    frames,
                )

                print(
                    "Checkpoint saved."
                )

    # Final save.
    _save_checkpoints(
        output_dir,
        frames,
    )

    print("\nRun complete.")

    print(
        "Successful jobs:",
        len(frames["metrics"]),
    )

    print(
        "Failed jobs:",
        len(frames["failures"]),
    )

    print(
        "Prediction rows:",
        len(
            frames[
                "predictions"
            ]
        ),
    )

    return frames