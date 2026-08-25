from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_config
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from forecasting.backtests.sarima import (
    evaluate_forecast,
)



# ============================================================
# Candidate feature sets
# ============================================================

PERMIT_FEATURE_SETS = {
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

    "permit_count_attendance": [
        "se_permit_count",
        "se_split_attendance_sum",
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


# ============================================================
# Exceptions / parsing
# ============================================================

class BacktestError(RuntimeError):
    def __init__(
        self,
        message: str,
        stage: str,
        failed_date=None,
    ):
        super().__init__(message)
        self.stage = stage
        self.failed_date = failed_date


def _parse_tuple(value):
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, np.ndarray):
        return tuple(value.tolist())

    if isinstance(value, str):
        return tuple(
            ast.literal_eval(value)
        )

    raise ValueError(
        f"Cannot parse tuple from {value!r}"
    )


def _parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "1", "yes"}:
            return True

        if text in {"false", "0", "no"}:
            return False

    return bool(value)


# ============================================================
# Metrics
# ============================================================

def smape(
    actual,
    prediction,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    denominator = (
        np.abs(actual)
        + np.abs(prediction)
    )

    valid = denominator > 0

    if not valid.any():
        return 0.0

    return float(
        100
        * np.mean(
            (
                2
                * np.abs(
                    actual[valid]
                    - prediction[valid]
                )
            )
            / denominator[valid]
        )
    )


def seasonal_naive_scale(
    train,
    seasonal_period=7,
):
    train = np.asarray(
        train,
        dtype=float,
    )

    if len(train) <= seasonal_period:
        return np.nan

    scale = np.mean(
        np.abs(
            train[seasonal_period:]
            - train[:-seasonal_period]
        )
    )

    if scale == 0:
        return np.nan

    return float(scale)


def evaluate_forecast(
    actual,
    prediction,
    train,
    seasonal_period=7,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    errors = actual - prediction

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    scale = seasonal_naive_scale(
        train,
        seasonal_period=seasonal_period,
    )

    mase = (
        float(mae / scale)
        if np.isfinite(scale)
        else np.nan
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "smape": smape(
            actual,
            prediction,
        ),

        # Positive bias means actual > prediction:
        # systematic underprediction.
        "bias": float(
            np.mean(errors)
        ),
    }


# ============================================================
# SARIMAX fitting
# ============================================================

def _fit_sarimax_with_retries(
    y,
    exog,
    order,
    seasonal_order,
    with_intercept,
    start_params=None,
):
    trend = (
        "c"
        if with_intercept
        else "n"
    )

    model = SARIMAX(
        endog=y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    attempts = [
        {
            "method": "lbfgs",
            "start_params": start_params,
            "maxiter": 200,
            "label": "warm_lbfgs",
        },
        {
            "method": "lbfgs",
            "start_params": None,
            "maxiter": 200,
            "label": "fresh_lbfgs",
        },
        {
            "method": "powell",
            "start_params": start_params,
            "maxiter": 400,
            "label": "warm_powell",
        },
        {
            "method": "powell",
            "start_params": None,
            "maxiter": 400,
            "label": "fresh_powell",
        },
    ]

    errors = []

    for attempt in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                result = model.fit(
                    method=attempt["method"],
                    start_params=attempt[
                        "start_params"
                    ],
                    maxiter=attempt["maxiter"],
                    disp=False,
                )

            params = np.asarray(
                result.params,
                dtype=float,
            )

            if not np.isfinite(
                params
            ).all():
                raise ValueError(
                    "Non-finite fitted parameters."
                )

            converged = (
                result.mle_retvals.get(
                    "converged",
                    True,
                )
            )

            if not converged:
                raise ValueError(
                    "Optimizer did not converge."
                )

            return (
                result,
                attempt["label"],
            )

        except Exception as exc:
            errors.append(
                f"{attempt['label']}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    raise RuntimeError(
        "All SARIMAX fitting attempts failed. "
        + " | ".join(errors)
    )


# ============================================================
# Diagnostics
# ============================================================

def _design_diagnostics(
    train_exog: pd.DataFrame,
):
    matrix = train_exog.to_numpy(
        dtype=float
    )

    rank = int(
        np.linalg.matrix_rank(
            matrix
        )
    )

    try:
        condition_number = float(
            np.linalg.cond(
                matrix
            )
        )
    except Exception:
        condition_number = np.nan

    if train_exog.shape[1] <= 1:
        max_abs_corr = np.nan

    else:
        corr = (
            train_exog
            .corr()
            .abs()
        )

        mask = np.triu(
            np.ones(
                corr.shape,
                dtype=bool,
            ),
            k=1,
        )

        values = corr.where(
            mask
        ).stack()

        max_abs_corr = (
            float(values.max())
            if len(values)
            else np.nan
        )

    return {
        "matrix_rank": rank,
        "condition_number":
            condition_number,
        "max_abs_correlation":
            max_abs_corr,
    }


def _coefficient_table(
    result,
    used_features,
    fold,
    neighborhood,
):
    rows = []

    for feature in used_features:
        if feature not in result.params.index:
            continue

        rows.append(
            {
                "fold": fold,
                "neighborhood":
                    neighborhood,
                "feature": feature,
                "coefficient": float(
                    result.params[
                        feature
                    ]
                ),
                "std_error": float(
                    result.bse[
                        feature
                    ]
                ),
                "p_value": float(
                    result.pvalues[
                        feature
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# One rolling backtest
# ============================================================

def rolling_permitted_events_backtest(
    target_panel: pd.DataFrame,
    permit_panel: pd.DataFrame,
    neighborhood: str,
    fold_row,
    order,
    seasonal_order,
    with_intercept,
    requested_features,
):
    fold = int(
        fold_row["fold"]
    )

    train_start = pd.Timestamp(
        fold_row["train_start"]
    )

    train_end = pd.Timestamp(
        fold_row["train_end"]
    )

    val_start = pd.Timestamp(
        fold_row["val_start"]
    )

    val_end = pd.Timestamp(
        fold_row["val_end"]
    )

    # --------------------------------------------------------
    # Target series
    # --------------------------------------------------------

    y = (
        target_panel.loc[
            target_panel[
                "neighborhood"
            ].eq(neighborhood),
            [
                "target_date",
                "calls",
            ],
        ]
        .set_index("target_date")
        .sort_index()["calls"]
        .astype(float)
    )

    y_train = y.loc[
        train_start:train_end
    ].copy()

    y_val = y.loc[
        val_start:val_end
    ].copy()

    if y_train.empty:
        raise BacktestError(
            "Training series is empty.",
            stage="prepare_target",
        )

    if len(y_val) != (
        val_end - val_start
    ).days + 1:
        raise BacktestError(
            "Validation series is incomplete.",
            stage="prepare_target",
        )

    # --------------------------------------------------------
    # Exogenous panel
    # --------------------------------------------------------

    x = (
        permit_panel.loc[
            permit_panel[
                "neighborhood"
            ].eq(neighborhood),
            [
                "target_date",
                *requested_features,
            ],
        ]
        .set_index("target_date")
        .sort_index()
    )

    required_dates = pd.date_range(
        train_start,
        val_end,
        freq="D",
    )

    x = x.reindex(
        required_dates
    )

    if x.isna().any().any():
        missing_dates = x.index[
            x.isna().any(axis=1)
        ]

        raise BacktestError(
            "Missing permitted-event exogenous "
            f"data. First missing date: "
            f"{missing_dates[0]}",
            stage="prepare_exog",
            failed_date=missing_dates[0],
        )

    x_train_raw = x.loc[
        train_start:train_end,
        requested_features,
    ].copy()

    # Remove regressors with no variation in the
    # initial training set.
    zero_variance = [
        col
        for col in requested_features
        if (
            x_train_raw[col]
            .nunique(
                dropna=False
            )
            <= 1
        )
    ]

    used_features = [
        col
        for col in requested_features
        if col not in zero_variance
    ]

    if not used_features:
        return {
            "status": "skipped",
            "reason":
                "no_training_variation",
            "fold": fold,
            "neighborhood":
                neighborhood,
            "zero_variance_features":
                zero_variance,
        }

    # --------------------------------------------------------
    # Fit scaler ONLY on the initial training period.
    #
    # Keeping the scaler fixed throughout the fold:
    # - prevents leakage
    # - keeps parameter scale fixed
    # - makes warm-starting sensible
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        x_train_raw[
            used_features
        ]
    )

    x_scaled = pd.DataFrame(
        scaler.transform(
            x[
                used_features
            ]
        ),
        index=x.index,
        columns=used_features,
    )

    x_train = x_scaled.loc[
        train_start:train_end
    ].copy()

    diagnostics = (
        _design_diagnostics(
            x_train
        )
    )

    # --------------------------------------------------------
    # Initial fit
    # --------------------------------------------------------

    try:
        result, initial_method = (
            _fit_sarimax_with_retries(
                y=y_train,
                exog=x_train,
                order=order,
                seasonal_order=
                    seasonal_order,
                with_intercept=
                    with_intercept,
            )
        )

    except Exception as exc:
        raise BacktestError(
            str(exc),
            stage="initial_fit",
        ) from exc

    coefficients = (
        _coefficient_table(
            result=result,
            used_features=
                used_features,
            fold=fold,
            neighborhood=
                neighborhood,
        )
    )

    # --------------------------------------------------------
    # Walk-forward validation
    # --------------------------------------------------------

    history_y = y_train.copy()
    history_x = x_train.copy()

    prediction_rows = []

    fit_methods = [
        initial_method
    ]

    validation_dates = list(
        y_val.index
    )

    for i, target_date in enumerate(
        validation_dates
    ):
        x_next = x_scaled.loc[
            [target_date],
            used_features,
        ]

        try:
            forecast = (
                result
                .get_forecast(
                    steps=1,
                    exog=x_next,
                )
                .predicted_mean
            )

            prediction = float(
                forecast.iloc[0]
            )

        except Exception as exc:
            raise BacktestError(
                str(exc),
                stage="forecast",
                failed_date=
                    target_date,
            ) from exc

        if not np.isfinite(
            prediction
        ):
            raise BacktestError(
                "Non-finite forecast.",
                stage="forecast",
                failed_date=
                    target_date,
            )

        actual = float(
            y_val.loc[
                target_date
            ]
        )

        prediction_rows.append(
            {
                "fold": fold,
                "neighborhood":
                    neighborhood,
                "target_date":
                    target_date,
                "actual": actual,
                "prediction":
                    prediction,
                "residual":
                    actual
                    - prediction,
            }
        )

        # No need to refit after the final
        # validation forecast.
        if i == len(
            validation_dates
        ) - 1:
            continue

        history_y = pd.concat(
            [
                history_y,
                pd.Series(
                    [actual],
                    index=[
                        target_date
                    ],
                    name="calls",
                ),
            ]
        )

        history_x = pd.concat(
            [
                history_x,
                x_next,
            ]
        )

        try:
            result, method = (
                _fit_sarimax_with_retries(
                    y=history_y,
                    exog=history_x,
                    order=order,
                    seasonal_order=
                        seasonal_order,
                    with_intercept=
                        with_intercept,
                    start_params=
                        result.params,
                )
            )

            fit_methods.append(
                method
            )

        except Exception as exc:
            raise BacktestError(
                str(exc),
                stage="daily_refit",
                failed_date=
                    target_date,
            ) from exc

    predictions = pd.DataFrame(
        prediction_rows
    )

    metrics = evaluate_forecast(
        actual=predictions[
            "actual"
        ],
        prediction=predictions[
            "prediction"
        ],
        train=y_train,
        seasonal_period=7,
    )

    residuals = predictions[
        "residual"
    ].to_numpy()

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[7, 14],
        return_df=True,
    )

    diagnostics.update(
        {
            "fold": fold,
            "neighborhood":
                neighborhood,

            "features_requested":
                json.dumps(
                    requested_features
                ),

            "features_used":
                json.dumps(
                    used_features
                ),

            "zero_variance_features":
                json.dumps(
                    zero_variance
                ),

            "n_features_requested":
                len(
                    requested_features
                ),

            "n_features_used":
                len(
                    used_features
                ),

            "negative_prediction_rate":
                float(
                    (
                        predictions[
                            "prediction"
                        ]
                        < 0
                    ).mean()
                ),

            "max_prediction":
                float(
                    predictions[
                        "prediction"
                    ].max()
                ),

            "min_prediction":
                float(
                    predictions[
                        "prediction"
                    ].min()
                ),

            "residual_mean":
                float(
                    residuals.mean()
                ),

            "residual_std":
                float(
                    residuals.std(
                        ddof=1
                    )
                ),

            "ljung_box_p_7":
                float(
                    ljung_box.loc[
                        7,
                        "lb_pvalue",
                    ]
                ),

            "ljung_box_p_14":
                float(
                    ljung_box.loc[
                        14,
                        "lb_pvalue",
                    ]
                ),

            "initial_fit_method":
                initial_method,

            "n_powell_fits":
                sum(
                    "powell"
                    in method
                    for method
                    in fit_methods
                ),

            "scaler_mean":
                json.dumps(
                    dict(
                        zip(
                            used_features,
                            scaler.mean_
                            .tolist(),
                        )
                    )
                ),

            "scaler_scale":
                json.dumps(
                    dict(
                        zip(
                            used_features,
                            scaler.scale_
                            .tolist(),
                        )
                    )
                ),
        }
    )

    metrics_row = {
        "fold": fold,
        "neighborhood":
            neighborhood,

        "order":
            str(order),

        "seasonal_order":
            str(
                seasonal_order
            ),

        "with_intercept":
            with_intercept,

        **metrics,
    }

    return {
        "status": "ok",
        "predictions":
            predictions,
        "metrics":
            pd.DataFrame(
                [metrics_row]
            ),
        "diagnostics":
            pd.DataFrame(
                [diagnostics]
            ),
        "coefficients":
            coefficients,
    }


# ============================================================
# One fold/neighborhood job
# ============================================================

def run_permitted_events_job(
    target_panel,
    permit_panel,
    fold_row,
    neighborhood,
    baseline_orders,
    feature_set,
):
    fold = int(
        fold_row["fold"]
    )

    try:
        order_match = (
            baseline_orders.loc[
                baseline_orders[
                    "fold"
                ].eq(fold)
                & baseline_orders[
                    "neighborhood"
                ].eq(
                    neighborhood
                )
            ]
        )

        if len(order_match) != 1:
            raise BacktestError(
                "Expected exactly one baseline "
                "order row.",
                stage=
                    "baseline_order_lookup",
            )

        order_row = (
            order_match.iloc[0]
        )

        order = _parse_tuple(
            order_row["order"]
        )

        seasonal_order = (
            _parse_tuple(
                order_row[
                    "seasonal_order"
                ]
            )
        )

        with_intercept = (
            _parse_bool(
                order_row[
                    "with_intercept"
                ]
            )
        )

        return (
            rolling_permitted_events_backtest(
                target_panel=
                    target_panel,
                permit_panel=
                    permit_panel,
                neighborhood=
                    neighborhood,
                fold_row=
                    fold_row,
                order=order,
                seasonal_order=
                    seasonal_order,
                with_intercept=
                    with_intercept,
                requested_features=
                    feature_set,
            )
        )

    except BacktestError as exc:
        return {
            "status": "failed",
            "failure":
                pd.DataFrame(
                    [
                        {
                            "fold": fold,
                            "neighborhood":
                                neighborhood,
                            "stage":
                                exc.stage,
                            "failed_date":
                                exc.failed_date,
                            "error_type":
                                type(
                                    exc
                                ).__name__,
                            "error_message":
                                str(exc),
                        }
                    ]
                ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "failure":
                pd.DataFrame(
                    [
                        {
                            "fold": fold,
                            "neighborhood":
                                neighborhood,
                            "stage":
                                "unexpected",
                            "failed_date":
                                None,
                            "error_type":
                                type(
                                    exc
                                ).__name__,
                            "error_message":
                                str(exc),
                        }
                    ]
                ),
        }


# ============================================================
# Checkpoint helpers
# ============================================================

def _load_or_empty(
    path: Path,
):
    if path.exists():
        return pd.read_parquet(
            path
        )

    return pd.DataFrame()


def _save_frame(
    frame,
    path,
):
    if frame is None:
        return

    if len(frame) == 0:
        return

    frame.to_parquet(
        path,
        index=False,
    )


# ============================================================
# Full parallel experiment
# ============================================================

def run_permitted_events_backtests(
    target_panel,
    permit_panel,
    folds,
    baseline_orders,
    feature_set_name,
    output_dir,
    neighborhoods=None,
    n_jobs=1,
    batch_size=None,
    resume=True,
):
    if (
        feature_set_name
        not in PERMIT_FEATURE_SETS
    ):
        raise ValueError(
            "Unknown feature set: "
            f"{feature_set_name}"
        )

    requested_features = (
        PERMIT_FEATURE_SETS[
            feature_set_name
        ]
    )

    missing_features = (
        set(requested_features)
        - set(
            permit_panel.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Permit panel is missing features: "
            f"{sorted(missing_features)}"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "predictions":
            output_dir
            / "predictions.parquet",

        "metrics":
            output_dir
            / "metrics.parquet",

        "diagnostics":
            output_dir
            / "diagnostics.parquet",

        "coefficients":
            output_dir
            / "coefficients.parquet",

        "failures":
            output_dir
            / "failures.parquet",

        "skipped":
            output_dir
            / "skipped.parquet",
    }

    frames = {
        name: (
            _load_or_empty(path)
            if resume
            else pd.DataFrame()
        )
        for name, path
        in paths.items()
    }

    if neighborhoods is None:
        neighborhoods = sorted(
            baseline_orders[
                "neighborhood"
            ].unique()
        )

    completed = set()

    if not frames[
        "metrics"
    ].empty:
        completed |= set(
            zip(
                frames[
                    "metrics"
                ]["fold"],
                frames[
                    "metrics"
                ]["neighborhood"],
            )
        )

    if not frames[
        "skipped"
    ].empty:
        completed |= set(
            zip(
                frames[
                    "skipped"
                ]["fold"],
                frames[
                    "skipped"
                ]["neighborhood"],
            )
        )

    jobs = []

    for _, fold_row in (
        folds.iterrows()
    ):
        fold = int(
            fold_row["fold"]
        )

        for neighborhood in neighborhoods:
            key = (
                fold,
                neighborhood,
            )

            if (
                resume
                and key
                in completed
            ):
                continue

            jobs.append(
                (
                    fold_row.copy(),
                    neighborhood,
                )
            )

    if not jobs:
        print(
            "No unfinished jobs."
        )

        return frames

    if batch_size is None:
        batch_size = max(
            n_jobs,
            1,
        )

    total = len(jobs)

    for start in range(
        0,
        total,
        batch_size,
    ):
        batch = jobs[
            start:
            start + batch_size
        ]

        with parallel_config(
            backend="loky",
            inner_max_num_threads=1,
        ):
            results = Parallel(
                n_jobs=n_jobs
            )(
                delayed(
                    run_permitted_events_job
                )(
                    target_panel=
                        target_panel,
                    permit_panel=
                        permit_panel,
                    fold_row=
                        fold_row,
                    neighborhood=
                        neighborhood,
                    baseline_orders=
                        baseline_orders,
                    feature_set=
                        requested_features,
                )
                for (
                    fold_row,
                    neighborhood,
                )
                in batch
            )

        new_frames = {
            "predictions": [],
            "metrics": [],
            "diagnostics": [],
            "coefficients": [],
            "failures": [],
            "skipped": [],
        }

        for result in results:
            status = result[
                "status"
            ]

            if status == "ok":
                for name in [
                    "predictions",
                    "metrics",
                    "diagnostics",
                    "coefficients",
                ]:
                    if (
                        name in result
                        and not result[
                            name
                        ].empty
                    ):
                        new_frames[
                            name
                        ].append(
                            result[
                                name
                            ]
                        )

            elif status == "failed":
                new_frames[
                    "failures"
                ].append(
                    result[
                        "failure"
                    ]
                )

            elif status == "skipped":
                new_frames[
                    "skipped"
                ].append(
                    pd.DataFrame(
                        [
                            {
                                "fold":
                                    result[
                                        "fold"
                                    ],
                                "neighborhood":
                                    result[
                                        "neighborhood"
                                    ],
                                "reason":
                                    result[
                                        "reason"
                                    ],
                                "zero_variance_features":
                                    json.dumps(
                                        result[
                                            "zero_variance_features"
                                        ]
                                    ),
                            }
                        ]
                    )
                )

        for name, pieces in (
            new_frames.items()
        ):
            if not pieces:
                continue

            new = pd.concat(
                pieces,
                ignore_index=True,
            )

            if frames[
                name
            ].empty:
                frames[
                    name
                ] = new

            else:
                frames[
                    name
                ] = pd.concat(
                    [
                        frames[
                            name
                        ],
                        new,
                    ],
                    ignore_index=True,
                )

            _save_frame(
                frames[name],
                paths[name],
            )

        finished = min(
            start + batch_size,
            total,
        )

        print(
            f"Completed "
            f"{finished}/{total} "
            f"jobs"
        )

    return frames


# ============================================================
# Baseline comparison
# ============================================================

def compare_to_baseline(
    candidate_metrics,
    baseline_metrics,
):
    metric_cols = [
        "mae",
        "rmse",
        "mase",
        "smape",
        "bias",
    ]

    baseline = (
        baseline_metrics[
            [
                "fold",
                "neighborhood",
                *metric_cols,
            ]
        ]
        .rename(
            columns={
                metric:
                    f"baseline_"
                    f"{metric}"
                for metric
                in metric_cols
            }
        )
    )

    comparison = (
        candidate_metrics
        .merge(
            baseline,
            on=[
                "fold",
                "neighborhood",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    for metric in [
        "mae",
        "rmse",
        "mase",
        "smape",
    ]:
        comparison[
            f"delta_{metric}"
        ] = (
            comparison[metric]
            - comparison[
                f"baseline_{metric}"
            ]
        )

    comparison[
        "improved_mase"
    ] = (
        comparison[
            "delta_mase"
        ]
        < 0
    )

    return comparison
