# calendar_sarimax_backtest.py

from __future__ import annotations

import ast
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from joblib import (
    Parallel,
    delayed,
    parallel_config,
)

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)

from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
)

from forecasting.backtests.sarima import (
    BacktestError,
    get_neighborhood_series,
    evaluate_forecast,
)


MODEL_VERSION = "v1"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_tuple(value):
    if isinstance(value, tuple):
        return value

    return tuple(
        ast.literal_eval(str(value))
    )


def _design_diagnostics(
    X: pd.DataFrame,
) -> dict:

    X = X.astype(float)

    zero_variance = [
        col
        for col in X.columns
        if X[col].nunique() <= 1
    ]

    active_columns = [
        col
        for col in X.columns
        if col not in zero_variance
    ]

    if not active_columns:
        return {
            "n_features_requested": len(X.columns),
            "n_features_used": 0,
            "zero_variance_features": ",".join(
                zero_variance
            ),
            "matrix_rank": 0,
            "condition_number": np.nan,
            "max_abs_correlation": np.nan,
        }

    active = X[
        active_columns
    ].copy()

    matrix_rank = np.linalg.matrix_rank(
        active.to_numpy()
    )

    # Standardize ONLY for condition-number diagnostics.
    std = active.std(ddof=0)

    standardized = (
        active - active.mean()
    ) / std

    condition_number = np.linalg.cond(
        standardized.to_numpy()
    )

    if len(active_columns) > 1:
        corr = (
            active.corr()
            .abs()
        )

        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1,
            ).astype(bool)
        )

        max_abs_correlation = (
            upper.max().max()
        )

    else:
        max_abs_correlation = np.nan

    return {
        "n_features_requested": len(
            X.columns
        ),
        "n_features_used": len(
            active_columns
        ),
        "zero_variance_features": ",".join(
            zero_variance
        ),
        "matrix_rank": int(
            matrix_rank
        ),
        "condition_number": float(
            condition_number
        ),
        "max_abs_correlation": float(
            max_abs_correlation
        )
        if np.isfinite(
            max_abs_correlation
        )
        else np.nan,
    }


def _fit_sarimax_with_retries(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    with_intercept: bool,
    start_params=None,
):

    trend = (
        "c"
        if with_intercept
        else "n"
    )

    model = SARIMAX(
        y.astype(float),

        exog=X.astype(float),

        order=order,
        seasonal_order=seasonal_order,

        trend=trend,

        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    attempts = []

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
            kwargs = {
                "disp": False,
                "method": method,
                "maxiter": maxiter,
            }

            if (
                params is not None
                and np.isfinite(params).all()
            ):
                kwargs["start_params"] = params

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    **kwargs
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

            finite_params = np.isfinite(
                np.asarray(
                    result.params,
                    dtype=float,
                )
            ).all()

            if (
                converged
                and finite_params
            ):
                return {
                    "result": result,
                    "fit_method": method,
                    "fit_attempts": attempt_number,
                    "warning_count": len(
                        caught
                    ),
                }

            messages.append(
                f"{method}: "
                f"converged={converged}, "
                f"finite={finite_params}"
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
            "All SARIMAX fitting attempts failed. "
            + " | ".join(messages)
        ),
    )


def _coefficient_table(
    result,
    feature_columns,
) -> pd.DataFrame:

    rows = []

    for feature in feature_columns:

        if feature not in result.params.index:
            continue

        rows.append(
            {
                "feature": feature,
                "coefficient": float(
                    result.params[feature]
                ),
                "std_error": float(
                    result.bse[feature]
                ),
                "p_value": float(
                    result.pvalues[feature]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Rolling SARIMAX
# ---------------------------------------------------------------------

def rolling_sarimax_backtest(
    train_y: pd.Series,
    validation_y: pd.Series,
    train_X: pd.DataFrame,
    validation_X: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    with_intercept: bool,
):

    # Remove features that are constant in training.
    usable_features = [
        col
        for col in train_X.columns
        if train_X[col].nunique() > 1
    ]

    if not usable_features:
        raise BacktestError(
            stage="design_matrix",
            message=(
                "No usable exogenous features "
                "after removing constants."
            ),
        )

    train_X = train_X[
        usable_features
    ].astype(float)

    validation_X = validation_X[
        usable_features
    ].astype(float)

    history_y = (
        train_y
        .astype(float)
        .copy()
    )

    history_X = (
        train_X
        .copy()
    )

    fitted = _fit_sarimax_with_retries(
        y=history_y,
        X=history_X,
        order=order,
        seasonal_order=seasonal_order,
        with_intercept=with_intercept,
    )

    result = fitted["result"]

    # Inference from TRAINING DATA ONLY.
    coefficients = _coefficient_table(
        result,
        usable_features,
    )

    rows = []

    validation_items = list(
        validation_y
        .astype(float)
        .items()
    )

    for i, (
        target_date,
        actual,
    ) in enumerate(
        validation_items
    ):

        target_date = pd.Timestamp(
            target_date
        )

        next_X = validation_X.loc[
            [target_date]
        ]

        forecast = result.forecast(
            steps=1,
            exog=next_X,
        )

        prediction = float(
            np.asarray(
                forecast
            ).ravel()[0]
        )

        if not np.isfinite(
            prediction
        ):
            raise BacktestError(
                stage="forecast",
                message=(
                    f"Non-finite forecast: "
                    f"{prediction}"
                ),
                target_date=target_date,
            )

        rows.append(
            {
                "target_date": target_date,
                "actual": float(actual),
                "prediction": prediction,
                "residual": float(
                    actual - prediction
                ),
                "fit_method": fitted[
                    "fit_method"
                ],
                "fit_attempts": fitted[
                    "fit_attempts"
                ],
                "fit_warning_count": fitted[
                    "warning_count"
                ],
            }
        )

        if (
            i
            == len(validation_items) - 1
        ):
            continue

        new_y = pd.Series(
            [float(actual)],
            index=pd.DatetimeIndex(
                [target_date]
            ),
            name=history_y.name,
        )

        history_y = pd.concat(
            [
                history_y,
                new_y,
            ]
        )

        history_X = pd.concat(
            [
                history_X,
                next_X,
            ]
        )

        previous_params = np.asarray(
            result.params,
            dtype=float,
        )

        try:
            fitted = (
                _fit_sarimax_with_retries(
                    y=history_y,
                    X=history_X,
                    order=order,
                    seasonal_order=seasonal_order,
                    with_intercept=with_intercept,
                    start_params=previous_params,
                )
            )

        except BacktestError as exc:
            raise BacktestError(
                stage="daily_refit",
                message=str(exc),
                target_date=target_date,
            ) from exc

        result = fitted["result"]

    predictions = pd.DataFrame(
        rows
    )

    residuals = predictions[
        "residual"
    ].to_numpy()

    lb = acorr_ljungbox(
        residuals,
        lags=[7, 14],
        return_df=True,
    )

    diagnostics = {
        "negative_prediction_rate": float(
            (
                predictions[
                    "prediction"
                ] < 0
            ).mean()
        ),

        "max_prediction": float(
            predictions[
                "prediction"
            ].max()
        ),

        "min_prediction": float(
            predictions[
                "prediction"
            ].min()
        ),

        "residual_mean": float(
            predictions[
                "residual"
            ].mean()
        ),

        "residual_std": float(
            predictions[
                "residual"
            ].std()
        ),

        "ljung_box_p_7": float(
            lb.loc[
                7,
                "lb_pvalue",
            ]
        ),

        "ljung_box_p_14": float(
            lb.loc[
                14,
                "lb_pvalue",
            ]
        ),
    }

    return (
        predictions,
        coefficients,
        diagnostics,
        usable_features,
    )


# ---------------------------------------------------------------------
# One fold × neighborhood job
# ---------------------------------------------------------------------

def run_calendar_job(
    series,
    calendar_features,
    baseline_order_row,
    neighborhood,
    fold,
    feature_set_name,
    feature_columns,
):

    fold_number = int(
        fold["fold"]
    )

    try:
        train_y = series.loc[
            pd.Timestamp(
                fold["train_start"]
            ):
            pd.Timestamp(
                fold["train_end"]
            )
        ]

        validation_y = series.loc[
            pd.Timestamp(
                fold["val_start"]
            ):
            pd.Timestamp(
                fold["val_end"]
            )
        ]

        calendar = (
            calendar_features
            .set_index(
                "target_date"
            )
            .sort_index()
        )

        train_X = calendar.loc[
            train_y.index,
            feature_columns,
        ]

        validation_X = calendar.loc[
            validation_y.index,
            feature_columns,
        ]

        if (
            train_X.isna().any().any()
            or validation_X.isna().any().any()
        ):
            raise BacktestError(
                stage="feature_alignment",
                message=(
                    "Missing calendar features."
                ),
            )

        design_diag = (
            _design_diagnostics(
                train_X
            )
        )

        order = _parse_tuple(
            baseline_order_row[
                "order"
            ]
        )

        seasonal_order = _parse_tuple(
            baseline_order_row[
                "seasonal_order"
            ]
        )

        with_intercept = bool(
            baseline_order_row[
                "with_intercept"
            ]
        )

        (
            predictions,
            coefficients,
            residual_diag,
            usable_features,
        ) = rolling_sarimax_backtest(
            train_y=train_y,
            validation_y=validation_y,
            train_X=train_X,
            validation_X=validation_X,
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=with_intercept,
        )

        metrics = evaluate_forecast(
            y_true=predictions[
                "actual"
            ],
            y_pred=predictions[
                "prediction"
            ],
            y_train=train_y,
            seasonality=7,
        )

        predictions.insert(
            0,
            "feature_set",
            feature_set_name,
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

        coefficients.insert(
            0,
            "feature_set",
            feature_set_name,
        )

        coefficients.insert(
            0,
            "neighborhood",
            neighborhood,
        )

        coefficients.insert(
            0,
            "fold",
            fold_number,
        )

        diagnostics = {
            "fold": fold_number,
            "neighborhood": neighborhood,
            "feature_set": feature_set_name,
            **design_diag,
            **residual_diag,
            "features_used": ",".join(
                usable_features
            ),
        }

        metric_row = {
            "fold": fold_number,
            "neighborhood": neighborhood,
            "feature_set": feature_set_name,
            **metrics,
        }

        return {
            "status": "ok",
            "predictions": predictions,
            "coefficients": coefficients,
            "diagnostics": diagnostics,
            "metrics": metric_row,
            "failure": None,
        }

    except Exception as exc:

        return {
            "status": "failed",
            "predictions": None,
            "coefficients": None,
            "diagnostics": None,
            "metrics": None,
            "failure": {
                "fold": fold_number,
                "neighborhood": neighborhood,
                "feature_set": feature_set_name,
                "error_type": type(
                    exc
                ).__name__,
                "error_message": str(
                    exc
                ),
            },
        }


# ---------------------------------------------------------------------
# Run one complete feature-set experiment
# ---------------------------------------------------------------------

def run_calendar_backtests(
    target_panel: pd.DataFrame,
    calendar_features: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    baseline_orders: pd.DataFrame,
    feature_set_name: str,
    feature_columns: list[str],
    output_dir,
    n_jobs: int = 7,
):

    output_dir = (
        Path(output_dir)
        / feature_set_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    neighborhoods = sorted(
        target_panel[
            "neighborhood"
        ].unique()
    )

    series_by_neighborhood = {
        neighborhood:
        get_neighborhood_series(
            target_panel,
            neighborhood,
        )
        for neighborhood
        in neighborhoods
    }

    order_lookup = (
        baseline_orders
        .set_index(
            [
                "fold",
                "neighborhood",
            ]
        )
    )

    jobs = []

    for fold in backtest_folds.to_dict(
        orient="records"
    ):

        fold_number = int(
            fold["fold"]
        )

        for neighborhood in neighborhoods:

            key = (
                fold_number,
                neighborhood,
            )

            if key not in order_lookup.index:
                continue

            jobs.append(
                (
                    series_by_neighborhood[
                        neighborhood
                    ],
                    neighborhood,
                    fold,
                    order_lookup.loc[
                        key
                    ],
                )
            )

    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):

        results = Parallel(
            n_jobs=n_jobs,
            verbose=10,
        )(
            delayed(
                run_calendar_job
            )(
                series=series,
                calendar_features=(
                    calendar_features
                ),
                baseline_order_row=(
                    order_row
                ),
                neighborhood=(
                    neighborhood
                ),
                fold=fold,
                feature_set_name=(
                    feature_set_name
                ),
                feature_columns=(
                    feature_columns
                ),
            )

            for (
                series,
                neighborhood,
                fold,
                order_row,
            ) in jobs
        )

    successful = [
        result
        for result in results
        if result["status"] == "ok"
    ]

    failed = [
        result
        for result in results
        if result["status"] != "ok"
    ]

    predictions = pd.concat(
        [
            result["predictions"]
            for result in successful
        ],
        ignore_index=True,
    )

    metrics = pd.DataFrame(
        [
            result["metrics"]
            for result in successful
        ]
    )

    diagnostics = pd.DataFrame(
        [
            result["diagnostics"]
            for result in successful
        ]
    )

    coefficient_frames = [
        result["coefficients"]
        for result in successful
        if (
            result["coefficients"]
            is not None
            and not result[
                "coefficients"
            ].empty
        )
    ]

    if coefficient_frames:
        coefficients = pd.concat(
            coefficient_frames,
            ignore_index=True,
        )
    else:
        coefficients = pd.DataFrame()

    failures = pd.DataFrame(
        [
            result["failure"]
            for result in failed
        ]
    )

    predictions.to_parquet(
        output_dir
        / "predictions.parquet",
        index=False,
    )

    metrics.to_parquet(
        output_dir
        / "metrics.parquet",
        index=False,
    )

    diagnostics.to_parquet(
        output_dir
        / "diagnostics.parquet",
        index=False,
    )

    coefficients.to_parquet(
        output_dir
        / "coefficients.parquet",
        index=False,
    )

    failures.to_parquet(
        output_dir
        / "failures.parquet",
        index=False,
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "coefficients": coefficients,
        "failures": failures,
    }


# ---------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------

def compare_to_baseline(
    calendar_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:

    baseline = baseline_metrics.rename(
        columns={
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "mase": "baseline_mase",
            "smape": "baseline_smape",
            "bias": "baseline_bias",
        }
    )

    comparison = calendar_metrics.merge(
        baseline[
            [
                "fold",
                "neighborhood",
                "baseline_mae",
                "baseline_rmse",
                "baseline_mase",
                "baseline_smape",
                "baseline_bias",
            ]
        ],
        on=[
            "fold",
            "neighborhood",
        ],
        how="left",
        validate="one_to_one",
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
        comparison["mase"]
        < comparison[
            "baseline_mase"
        ]
    )

    return comparison
