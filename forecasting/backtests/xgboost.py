from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from forecasting.backtests.sarima import (
    evaluate_forecast,
)
from forecasting.features.xgboost import (
    XGB_FEATURE_SETS,
    build_xgboost_feature_panel,
    prepare_target_panel,
    validate_xgboost_feature_panel,
)
from forecasting.paths import (
    TARGET_PANEL_5Y_PATH,
    XGBOOST_FEATURE_PANEL_PATH,
    XGBOOST_REGRESSION_BACKTEST_DIR,
)


TEST_DAYS = 365
VALIDATION_DAYS = 90
N_FOLDS = 4

DEFAULT_XGB_REGRESSOR_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 1,
}


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class XGBoostBacktestError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------

def build_standard_backtest_folds(
    target_panel: pd.DataFrame,
    test_days: int = TEST_DAYS,
    validation_days: int = VALIDATION_DAYS,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """
    Reconstruct the locked rolling-origin development folds.

    The final 365 days remain untouched as the final test period.
    """

    if target_panel.empty:
        raise ValueError(
            "Target panel is empty."
        )

    dates = pd.to_datetime(
        target_panel["target_date"],
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError(
            "Target panel contains invalid target dates."
        )

    max_target_date = dates.max()

    test_start = (
        max_target_date
        - pd.Timedelta(
            days=test_days - 1
        )
    )

    development_dates = dates.loc[
        dates < test_start
    ]

    if development_dates.empty:
        raise ValueError(
            "No development data remain after "
            "reserving the final test period."
        )

    train_start = (
        development_dates.min()
    )

    development_end = (
        development_dates.max()
    )

    fold_rows = []

    for fold_index in range(
        n_folds
    ):
        val_end = (
            development_end
            - pd.Timedelta(
                days=(
                    validation_days
                    * (
                        n_folds
                        - fold_index
                        - 1
                    )
                )
            )
        )

        val_start = (
            val_end
            - pd.Timedelta(
                days=validation_days - 1
            )
        )

        train_end = (
            val_start
            - pd.Timedelta(
                days=1
            )
        )

        fold_rows.append(
            {
                "fold":
                    fold_index + 1,

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

    return pd.DataFrame(
        fold_rows
    )


def validate_folds(
    folds: pd.DataFrame,
) -> None:

    required = {
        "fold",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
    }

    missing = (
        required
        - set(folds.columns)
    )

    if missing:
        raise ValueError(
            "Fold table is missing columns: "
            f"{sorted(missing)}"
        )

    if folds.empty:
        raise ValueError(
            "Fold table is empty."
        )

    for column in [
        "train_start",
        "train_end",
        "val_start",
        "val_end",
    ]:
        folds[column] = pd.to_datetime(
            folds[column],
            errors="raise",
        ).dt.normalize()

    if folds[
        "fold"
    ].duplicated().any():
        raise ValueError(
            "Fold numbers must be unique."
        )

    for _, row in folds.iterrows():

        if (
            row["train_end"]
            >= row["val_start"]
        ):
            raise ValueError(
                f"Fold {row['fold']} "
                "training overlaps validation."
            )

        if (
            row["val_start"]
            > row["val_end"]
        ):
            raise ValueError(
                f"Fold {row['fold']} "
                "has invalid validation dates."
            )


# ---------------------------------------------------------------------
# Fold data
# ---------------------------------------------------------------------

def get_fold_data(
    feature_panel: pd.DataFrame,
    fold_row,
    feature_set_name: str | None = None,
    numeric_features:
        list[str] | None = None,
) -> dict:
    resolved_numeric_features = (
        resolve_numeric_features(
            feature_set_name=
                feature_set_name,
            numeric_features=
                numeric_features,
        )
    )

    train_start = pd.Timestamp(
        fold_row["train_start"]
    ).normalize()

    train_end = pd.Timestamp(
        fold_row["train_end"]
    ).normalize()

    val_start = pd.Timestamp(
        fold_row["val_start"]
    ).normalize()

    val_end = pd.Timestamp(
        fold_row["val_end"]
    ).normalize()

    train = feature_panel.loc[
        feature_panel[
            "target_date"
        ].between(
            train_start,
            train_end,
        )
    ].copy()

    validation = feature_panel.loc[
        feature_panel[
            "target_date"
        ].between(
            val_start,
            val_end,
        )
    ].copy()

    if train.empty:
        raise XGBoostBacktestError(
            "Training data is empty."
        )

    if validation.empty:
        raise XGBoostBacktestError(
            "Validation data is empty."
        )

    expected_validation_dates = (
        pd.date_range(
            val_start,
            val_end,
            freq="D",
        )
    )

    actual_validation_dates = (
        pd.DatetimeIndex(
            validation[
                "target_date"
            ].unique()
        )
        .sort_values()
    )

    if not np.array_equal(
        actual_validation_dates.to_numpy(),
        expected_validation_dates.to_numpy(),
    ):
        raise XGBoostBacktestError(
            "Validation dates are incomplete."
        )

    missing_features = (
        set(
            resolved_numeric_features
        )
        - set(
            feature_panel.columns
        )
    )

    if missing_features:
        raise XGBoostBacktestError(
            "Feature panel is missing predictors: "
            f"{sorted(missing_features)}"
        )

    model_features = [
        "neighborhood",
        *resolved_numeric_features,
    ]

    return {
        "train":
            train,

        "validation":
            validation,

        "model_features":
            model_features,

        "numeric_features":
            resolved_numeric_features,

        "train_start":
            train_start,

        "train_end":
            train_end,

        "val_start":
            val_start,

        "val_end":
            val_end,
    }


def resolve_numeric_features(
    feature_set_name: str | None = None,
    numeric_features:
        list[str] | None = None,
) -> list[str]:

    if numeric_features is not None:
        return list(numeric_features)

    if feature_set_name is None:
        raise ValueError(
            "Either feature_set_name or "
            "numeric_features must be provided."
        )

    if (
        feature_set_name
        not in XGB_FEATURE_SETS
    ):
        raise ValueError(
            f"Unknown feature set: "
            f"{feature_set_name}"
        )

    return list(
        XGB_FEATURE_SETS[
            feature_set_name
        ]
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_xgboost_pipeline(
    numeric_features:
        list[str],
    model_params:
        dict | None = None,
) -> Pipeline:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "neighborhood",
                encoder,
                ["neighborhood"],
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    resolved_model_params = {
        **DEFAULT_XGB_REGRESSOR_PARAMS,
        **(
            dict(model_params)
            if model_params is not None
            else {}
        ),
    }

    regressor = XGBRegressor(
        **resolved_model_params
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                regressor,
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def summarize_fold_predictions(
    predictions:
        pd.DataFrame,
) -> dict:

    residuals = (
        predictions["actual"]
        - predictions["prediction"]
    )

    return {
        "n_predictions":
            len(predictions),

        "min_prediction":
            float(
                predictions[
                    "prediction"
                ].min()
            ),

        "max_prediction":
            float(
                predictions[
                    "prediction"
                ].max()
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

        # Explicitly named because evaluate_forecast()
        # uses prediction - actual for its bias.
        "residual_mean_actual_minus_pred":
            float(
                residuals.mean()
            ),

        "residual_std":
            float(
                residuals.std(
                    ddof=1
                )
            ),
    }


# ---------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------

def run_sequential_fold(
    feature_panel:
        pd.DataFrame,

    target_panel:
        pd.DataFrame,

    fold_row,

    feature_set_name:
        str,

    model,

    numeric_features:
        list[str] | None = None,
) -> dict:
    """
    Fit one global model on the initial training window and evaluate
    sequential one-day-ahead forecasts over one validation fold.

    The fitted XGBoost model stays fixed during the 90-day fold.

    Lag and rolling features for target date t use actual target
    observations available through t-1.
    """

    fold_data = get_fold_data(
        feature_panel=
            feature_panel,

        fold_row=
            fold_row,

        feature_set_name=
            feature_set_name,

        numeric_features=
            numeric_features,
    )

    train = fold_data[
        "train"
    ]

    validation = fold_data[
        "validation"
    ]

    model_features = fold_data[
        "model_features"
    ]

    # --------------------------------------------------------------
    # Fit global model
    # --------------------------------------------------------------

    X_train = train[
        model_features
    ]

    y_train = train[
        "calls"
    ]

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------------
    # Sequential one-day-ahead validation
    # --------------------------------------------------------------

    validation_dates = sorted(
        validation[
            "target_date"
        ].unique()
    )

    prediction_frames = []

    for target_date in validation_dates:

        target_date = pd.Timestamp(
            target_date
        )

        day = validation.loc[
            validation[
                "target_date"
            ]
            == target_date
        ].copy()

        X_day = day[
            model_features
        ]

        y_pred = model.predict(
            X_day
        )

        if (
            len(y_pred)
            != len(day)
        ):
            raise XGBoostBacktestError(
                "Prediction count does not match "
                "validation rows."
            )

        day_predictions = (
            pd.DataFrame(
                {
                    "fold":
                        fold_row[
                            "fold"
                        ],

                    "feature_set":
                        feature_set_name,

                    "target_date":
                        target_date,

                    "forecast_origin":
                        (
                            target_date
                            - pd.Timedelta(
                                days=1
                            )
                        ),

                    "neighborhood":
                        day[
                            "neighborhood"
                        ].to_numpy(),

                    "actual":
                        day[
                            "calls"
                        ].to_numpy(),

                    "prediction":
                        y_pred,
                }
            )
        )

        prediction_frames.append(
            day_predictions
        )

    if not prediction_frames:
        raise XGBoostBacktestError(
            "No validation predictions "
            "were produced."
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions = (
        predictions
        .sort_values(
            [
                "target_date",
                "neighborhood",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------------

    if (
        len(predictions)
        != len(validation)
    ):
        raise XGBoostBacktestError(
            "Final prediction count does not match "
            "validation row count."
        )

    if predictions.duplicated(
        subset=[
            "target_date",
            "neighborhood",
        ]
    ).any():
        raise XGBoostBacktestError(
            "Duplicate target-date/neighborhood "
            "predictions were produced."
        )

    if not np.isfinite(
        predictions[
            "prediction"
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise XGBoostBacktestError(
            "Model produced non-finite predictions."
        )

    # --------------------------------------------------------------
    # Full raw training history for MASE denominator
    # --------------------------------------------------------------

    raw_train = target_panel.loc[
        target_panel[
            "target_date"
        ].between(
            fold_data[
                "train_start"
            ],
            fold_data[
                "train_end"
            ],
        )
    ].copy()

    # --------------------------------------------------------------
    # Neighborhood-level metrics
    # --------------------------------------------------------------

    metric_rows = []

    for (
        neighborhood,
        group,
    ) in predictions.groupby(
        "neighborhood"
    ):

        group = (
            group
            .sort_values(
                "target_date"
            )
        )

        neighborhood_model_train = (
            train.loc[
                train[
                    "neighborhood"
                ]
                == neighborhood
            ]
            .sort_values(
                "target_date"
            )
        )

        neighborhood_raw_train = (
            raw_train.loc[
                raw_train[
                    "neighborhood"
                ]
                == neighborhood
            ]
            .sort_values(
                "target_date"
            )
        )

        if neighborhood_raw_train.empty:
            raise XGBoostBacktestError(
                "No raw training history found for "
                f"{neighborhood!r}."
            )

        metric_values = (
            evaluate_forecast(
                y_true=
                    group[
                        "actual"
                    ],

                y_pred=
                    group[
                        "prediction"
                    ],

                y_train=
                    neighborhood_raw_train[
                        "calls"
                    ],

                seasonality=7,
            )
        )

        metric_rows.append(
            {
                "fold":
                    fold_row[
                        "fold"
                    ],

                "feature_set":
                    feature_set_name,

                "neighborhood":
                    neighborhood,

                "n_model_train":
                    len(
                        neighborhood_model_train
                    ),

                "n_mase_train":
                    len(
                        neighborhood_raw_train
                    ),

                "n_validation":
                    len(
                        group
                    ),

                **metric_values,
            }
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    # --------------------------------------------------------------
    # Fold-level diagnostics
    # --------------------------------------------------------------

    diagnostic_values = (
        summarize_fold_predictions(
            predictions
        )
    )

    diagnostics = pd.DataFrame(
        [
            {
                "fold":
                    fold_row[
                        "fold"
                    ],

                "feature_set":
                    feature_set_name,

                "train_start":
                    fold_data[
                        "train_start"
                    ],

                "train_end":
                    fold_data[
                        "train_end"
                    ],

                "val_start":
                    fold_data[
                        "val_start"
                    ],

                "val_end":
                    fold_data[
                        "val_end"
                    ],

                "n_training_rows":
                    len(
                        train
                    ),

                **diagnostic_values,
            }
        ]
    )

    return {
        "predictions":
            predictions,

        "metrics":
            metrics,

        "diagnostics":
            diagnostics,
    }


# ---------------------------------------------------------------------
# Multi-fold runner
# ---------------------------------------------------------------------

def run_feature_set_backtest(
    feature_panel:
        pd.DataFrame,

    target_panel:
        pd.DataFrame,

    folds:
        pd.DataFrame,

    feature_set_name:
        str,

    numeric_features:
        list[str] | None = None,
) -> dict:
    """
    Run all validation folds for one feature specification.

    A brand-new Pipeline is created for each fold so fitted state
    cannot leak from one fold into another.
    """

    resolved_numeric_features = (
        resolve_numeric_features(
            feature_set_name=
                feature_set_name,
            numeric_features=
                numeric_features,
        )
    )

    prediction_frames = []
    metric_frames = []
    diagnostic_frames = []

    ordered_folds = (
        folds.sort_values(
            "fold"
        )
    )

    for _, fold_row in (
        ordered_folds.iterrows()
    ):

        fold_number = int(
            fold_row["fold"]
        )

        print(
            f"Running {feature_set_name} "
            f"fold {fold_number}..."
        )

        model = build_xgboost_pipeline(
            numeric_features=
                resolved_numeric_features
        )

        result = run_sequential_fold(
            feature_panel=
                feature_panel,

            target_panel=
                target_panel,

            fold_row=
                fold_row,

                feature_set_name=
                    feature_set_name,

                numeric_features=
                    resolved_numeric_features,

                model=
                    model,
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

    return {
        "predictions":
            pd.concat(
                prediction_frames,
                ignore_index=True,
            ),

        "metrics":
            pd.concat(
                metric_frames,
                ignore_index=True,
            ),

        "diagnostics":
            pd.concat(
                diagnostic_frames,
                ignore_index=True,
            ),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_metrics(
    metrics:
        pd.DataFrame,
) -> pd.DataFrame:

    if metrics.empty:
        return pd.DataFrame()

    rows = []

    for (
        feature_set,
        group,
    ) in metrics.groupby(
        "feature_set"
    ):

        rows.append(
            {
                "feature_set":
                    feature_set,

                "n_jobs":
                    len(group),

                "n_neighborhoods":
                    group[
                        "neighborhood"
                    ].nunique(),

                "median_mae":
                    group[
                        "mae"
                    ].median(),

                "mean_mae":
                    group[
                        "mae"
                    ].mean(),

                "median_rmse":
                    group[
                        "rmse"
                    ].median(),

                "mean_rmse":
                    group[
                        "rmse"
                    ].mean(),

                "median_mase":
                    group[
                        "mase"
                    ].median(),

                "mean_mase":
                    group[
                        "mase"
                    ].mean(),

                "median_smape":
                    group[
                        "smape"
                    ].median(),

                "mean_smape":
                    group[
                        "smape"
                    ].mean(),

                "median_bias":
                    group[
                        "bias"
                    ].median(),

                "mean_bias":
                    group[
                        "bias"
                    ].mean(),

                "pct_jobs_mase_below_1":
                    (
                        100.0
                        * (
                            group[
                                "mase"
                            ]
                            < 1
                        ).mean()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "feature_set"
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_backtest_outputs(
    predictions:
        pd.DataFrame,

    metrics:
        pd.DataFrame,

    diagnostics:
        pd.DataFrame,

    output_dir:
        str | Path,
) -> None:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    summary = summarize_metrics(
        metrics
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Feature panel
# ---------------------------------------------------------------------

def load_or_build_feature_panel(
    target_panel:
        pd.DataFrame,

    feature_panel_path:
        str | Path,

    rebuild:
        bool = False,
) -> pd.DataFrame:

    feature_panel_path = Path(
        feature_panel_path
    )

    if (
        feature_panel_path.exists()
        and not rebuild
    ):

        feature_panel = (
            pd.read_parquet(
                feature_panel_path
            )
        )

        feature_panel[
            "target_date"
        ] = pd.to_datetime(
            feature_panel[
                "target_date"
            ],
            errors="raise",
        ).dt.normalize()

        validate_xgboost_feature_panel(
            feature_panel
        )

        return feature_panel

    feature_panel = (
        build_xgboost_feature_panel(
            target_panel
        )
    )

    validate_xgboost_feature_panel(
        feature_panel
    )

    feature_panel_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_panel.to_parquet(
        feature_panel_path,
        index=False,
    )

    return feature_panel


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run global one-day-ahead XGBoost "
            "backtests across SPD neighborhoods."
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
            "Optional Parquet fold table. "
            "If omitted, the locked 4x90-day "
            "development folds are reconstructed "
            "while reserving the final 365 days."
        ),
    )

    parser.add_argument(
        "--feature-set",
        choices=[
            *XGB_FEATURE_SETS.keys(),
            "all",
        ],
        default="lags_only",
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            XGBOOST_REGRESSION_BACKTEST_DIR
        ),
    )

    parser.add_argument(
        "--rebuild-features",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------------
    # Target panel
    # --------------------------------------------------------------

    target_panel = pd.read_parquet(
        args.target_panel
    )

    target_panel = prepare_target_panel(
        target_panel
    )

    # --------------------------------------------------------------
    # Feature panel
    # --------------------------------------------------------------

    feature_panel = (
        load_or_build_feature_panel(
            target_panel=
                target_panel,

            feature_panel_path=
                args.feature_panel,

            rebuild=
                args.rebuild_features,
        )
    )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Folds
    # --------------------------------------------------------------

    if args.folds:

        folds = pd.read_parquet(
            args.folds
        )

    else:

        folds = (
            build_standard_backtest_folds(
                target_panel
            )
        )

    validate_folds(
        folds
    )

    folds.to_parquet(
        output_dir
        / "folds.parquet",
        index=False,
    )

    # --------------------------------------------------------------
    # Feature specifications
    # --------------------------------------------------------------

    if (
        args.feature_set
        == "all"
    ):

        feature_sets = list(
            XGB_FEATURE_SETS.keys()
        )

    else:

        feature_sets = [
            args.feature_set
        ]

    combined_predictions = []
    combined_metrics = []
    combined_diagnostics = []

    # --------------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------------

    for feature_set_name in (
        feature_sets
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"Feature set: "
            f"{feature_set_name}"
        )

        print(
            "========================================"
        )

        result = (
            run_feature_set_backtest(
                feature_panel=
                    feature_panel,

                target_panel=
                    target_panel,

                folds=
                    folds,

                feature_set_name=
                    feature_set_name,
            )
        )

        feature_output_dir = (
            output_dir
            / feature_set_name
        )

        save_backtest_outputs(
            predictions=
                result[
                    "predictions"
                ],

            metrics=
                result[
                    "metrics"
                ],

            diagnostics=
                result[
                    "diagnostics"
                ],

            output_dir=
                feature_output_dir,
        )

        combined_predictions.append(
            result[
                "predictions"
            ]
        )

        combined_metrics.append(
            result[
                "metrics"
            ]
        )

        combined_diagnostics.append(
            result[
                "diagnostics"
            ]
        )

    # --------------------------------------------------------------
    # Combined outputs
    # --------------------------------------------------------------

    predictions = pd.concat(
        combined_predictions,
        ignore_index=True,
    )

    metrics = pd.concat(
        combined_metrics,
        ignore_index=True,
    )

    diagnostics = pd.concat(
        combined_diagnostics,
        ignore_index=True,
    )

    predictions.to_parquet(
        output_dir
        / "all_predictions.parquet",
        index=False,
    )

    metrics.to_parquet(
        output_dir
        / "all_metrics.parquet",
        index=False,
    )

    diagnostics.to_parquet(
        output_dir
        / "all_diagnostics.parquet",
        index=False,
    )

    summary = summarize_metrics(
        metrics
    )

    summary.to_csv(
        output_dir
        / "all_summary.csv",
        index=False,
    )

    print(
        "\nBacktest complete.\n"
    )

    if not summary.empty:

        print(
            summary.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
