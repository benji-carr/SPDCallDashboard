from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRanker

from forecasting.evaluation.top10 import (
    evaluate_top10_predictions,
    summarize_top10,
)
from forecasting.backtests.xgboost import (
    build_standard_backtest_folds,
    get_fold_data,
    load_or_build_feature_panel,
    validate_folds,
)
from forecasting.features.xgboost import (
    XGB_FEATURE_SETS,
    prepare_target_panel,
)
from forecasting.paths import (
    TARGET_PANEL_5Y_PATH,
    XGBOOST_FEATURE_PANEL_PATH,
    XGBOOST_RANKING_BACKTEST_DIR,
    XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_PREDICTIONS_PATH,
)


FEATURE_SET_NAME = "lags_rolling_calendar"

RANKER_SPECS = {
    "full_ndcg": {
        "objective": "rank:ndcg",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 58,
        "eval_metric": "ndcg",
    },
    "full_pairwise": {
        "objective": "rank:pairwise",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 58,
        "eval_metric": "ndcg",
    },
}


class XGBoostRankerBacktestError(
    RuntimeError
):
    pass


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def build_ranker_preprocessor(
    numeric_features: list[str],
) -> ColumnTransformer:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    return ColumnTransformer(
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


# ---------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------

def build_xgboost_ranker(
    ranker_spec: str,
) -> XGBRanker:
    if ranker_spec not in RANKER_SPECS:
        raise ValueError(
            f"Unknown ranker spec: "
            f"{ranker_spec}"
        )

    spec = RANKER_SPECS[
        ranker_spec
    ]

    return XGBRanker(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,

        objective=
            spec[
                "objective"
            ],

        lambdarank_pair_method=
            spec[
                "lambdarank_pair_method"
            ],
        lambdarank_num_pair_per_sample=
            spec[
                "lambdarank_num_pair_per_sample"
            ],

        # Raw call counts can exceed 31.
        # Use relevance directly rather than:
        # 2 ** relevance - 1.
        ndcg_exp_gain=False,

        eval_metric=
            spec[
                "eval_metric"
            ],

        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )


# ---------------------------------------------------------------------
# Query IDs
# ---------------------------------------------------------------------

def build_qid(
    dates: pd.Series,
) -> np.ndarray:
    """
    Assign one query ID to each target date.

    All neighborhoods from the same date belong to
    the same ranking problem.
    """

    qid, _ = pd.factorize(
        dates,
        sort=True,
    )

    qid = qid.astype(
        np.int32
    )

    if (
        np.diff(qid) < 0
    ).any():
        raise XGBoostRankerBacktestError(
            "qid must be sorted in non-decreasing order."
        )

    return qid


# ---------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------

def run_ranker_fold(
    feature_panel: pd.DataFrame,
    fold_row,
    ranker_spec: str,
    top_k: int = 10,
) -> pd.DataFrame:

    fold_data = get_fold_data(
        feature_panel=
            feature_panel,

        fold_row=
            fold_row,

        feature_set_name=
            FEATURE_SET_NAME,
    )

    train = (
        fold_data[
            "train"
        ]
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

    validation = (
        fold_data[
            "validation"
        ]
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

    model_features = (
        fold_data[
            "model_features"
        ]
    )

    numeric_features = (
        fold_data[
            "numeric_features"
        ]
    )

    # --------------------------------------------------------------
    # Ranking labels
    # --------------------------------------------------------------

    y_train = train[
        "calls"
    ].to_numpy(
        dtype=float
    )

    if (
        y_train < 0
    ).any():
        raise XGBoostRankerBacktestError(
            "Ranking labels cannot contain "
            "negative call counts."
        )

    # Calls are counts. Ensure we haven't somehow
    # introduced fractional target values.
    if not np.allclose(
        y_train,
        np.round(
            y_train
        ),
    ):
        raise XGBoostRankerBacktestError(
            "Call-count relevance labels "
            "must be integer-valued."
        )

    y_train = np.round(
        y_train
    ).astype(
        np.int32
    )

    # --------------------------------------------------------------
    # One query = one calendar date
    # --------------------------------------------------------------

    qid_train = build_qid(
        train[
            "target_date"
        ]
    )

    # --------------------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------------

    preprocessor = (
        build_ranker_preprocessor(
            numeric_features=
                numeric_features
        )
    )

    X_train = (
        preprocessor.fit_transform(
            train[
                model_features
            ]
        )
    )

    # --------------------------------------------------------------
    # Fit ranker
    # --------------------------------------------------------------

    ranker = build_xgboost_ranker(
        ranker_spec=ranker_spec
    )

    ranker.fit(
        X_train,
        y_train,
        qid=qid_train,
        verbose=False,
    )

    # --------------------------------------------------------------
    # Sequential daily ranking
    # --------------------------------------------------------------

    prediction_frames = []

    validation_dates = sorted(
        validation[
            "target_date"
        ].unique()
    )

    for target_date in (
        validation_dates
    ):

        target_date = pd.Timestamp(
            target_date
        )

        day = (
            validation.loc[
                validation[
                    "target_date"
                ]
                == target_date
            ]
            .sort_values(
                "neighborhood"
            )
            .copy()
        )

        X_day = (
            preprocessor.transform(
                day[
                    model_features
                ]
            )
        )

        ranking_scores = (
            ranker.predict(
                X_day
            )
        )

        if (
            len(ranking_scores)
            != len(day)
        ):
            raise XGBoostRankerBacktestError(
                "Prediction count does not "
                "match daily row count."
            )

        if not np.isfinite(
            ranking_scores
        ).all():
            raise XGBoostRankerBacktestError(
                "Ranker produced non-finite scores."
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold":
                        int(
                            fold_row[
                                "fold"
                            ]
                        ),

                    "model":
                        (
                            "xgboost_ranker_"
                            f"{ranker_spec}"
                        ),

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

                    # These are relevance/ranking scores,
                    # NOT predicted call counts.
                    "prediction":
                        ranking_scores,
                }
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if (
        len(predictions)
        != len(validation)
    ):
        raise XGBoostRankerBacktestError(
            "Fold prediction count does not "
            "match validation row count."
        )

    if predictions.duplicated(
        subset=[
            "target_date",
            "neighborhood",
        ]
    ).any():
        raise XGBoostRankerBacktestError(
            "Duplicate daily neighborhood "
            "rankings were produced."
        )

    return predictions


# ---------------------------------------------------------------------
# NDCG diagnostics
# ---------------------------------------------------------------------

def calculate_daily_ndcg(
    predictions: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:

    rows = []

    for (
        target_date,
        day,
    ) in predictions.groupby(
        "target_date",
        sort=True,
    ):

        actual = day[
            "actual"
        ].to_numpy(
            dtype=float
        )

        scores = day[
            "prediction"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "fold":
                    int(
                        day[
                            "fold"
                        ].iloc[0]
                    ),

                "target_date":
                    target_date,

                f"ndcg_at_{top_k}":
                    float(
                        ndcg_score(
                            actual.reshape(
                                1,
                                -1,
                            ),
                            scores.reshape(
                                1,
                                -1,
                            ),
                            k=top_k,
                        )
                    ),

                "ndcg_full":
                    float(
                        ndcg_score(
                            actual.reshape(
                                1,
                                -1,
                            ),
                            scores.reshape(
                                1,
                                -1,
                            ),
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Fold summary
# ---------------------------------------------------------------------

def build_fold_summary(
    daily_top10: pd.DataFrame,
    daily_ndcg: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:

    combined = (
        daily_top10.merge(
            daily_ndcg,
            on=[
                "fold",
                "target_date",
            ],
            validate="one_to_one",
        )
    )

    return (
        combined
        .groupby(
            "fold"
        )
        .agg(
            days=(
                "target_date",
                "count",
            ),

            mean_correct=(
                "overlap_count",
                "mean",
            ),

            mean_accuracy=(
                "top_k_accuracy",
                "mean",
            ),

            mean_volume_capture=(
                "top10_volume_capture",
                "mean",
            ),

            mean_rank_error=(
                "mean_top10_rank_error",
                "mean",
            ),

            mean_rank_correlation=(
                "overall_rank_correlation",
                "mean",
            ),

            mean_ndcg_at_k=(
                f"ndcg_at_{top_k}",
                "mean",
            ),

            mean_ndcg_full=(
                "ndcg_full",
                "mean",
            ),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# Compare against existing regression XGBoost
# ---------------------------------------------------------------------

def compare_with_regressor(
    ranker_daily: pd.DataFrame,
    regressor_predictions: pd.DataFrame,
    top_k: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    regressor_daily = (
        evaluate_top10_predictions(
            predictions=
                regressor_predictions,

            top_k=
                top_k,
        )
    )

    ranker = ranker_daily[
        [
            "target_date",
            "overlap_count",
            "top10_volume_capture",
            "mean_top10_rank_error",
            "overall_rank_correlation",
        ]
    ].rename(
        columns={
            "overlap_count":
                "ranker_overlap",

            "top10_volume_capture":
                "ranker_volume_capture",

            "mean_top10_rank_error":
                "ranker_rank_error",

            "overall_rank_correlation":
                "ranker_rank_correlation",
        }
    )

    regressor = regressor_daily[
        [
            "target_date",
            "overlap_count",
            "top10_volume_capture",
            "mean_top10_rank_error",
            "overall_rank_correlation",
        ]
    ].rename(
        columns={
            "overlap_count":
                "regressor_overlap",

            "top10_volume_capture":
                "regressor_volume_capture",

            "mean_top10_rank_error":
                "regressor_rank_error",

            "overall_rank_correlation":
                "regressor_rank_correlation",
        }
    )

    paired = ranker.merge(
        regressor,
        on="target_date",
        validate="one_to_one",
    )

    summary = pd.DataFrame(
        [
            {
                "pct_days_ranker_more_top10":
                    (
                        100
                        * (
                            paired[
                                "ranker_overlap"
                            ]
                            > paired[
                                "regressor_overlap"
                            ]
                        ).mean()
                    ),

                "pct_days_top10_tied":
                    (
                        100
                        * (
                            paired[
                                "ranker_overlap"
                            ]
                            == paired[
                                "regressor_overlap"
                            ]
                        ).mean()
                    ),

                "pct_days_regressor_more_top10":
                    (
                        100
                        * (
                            paired[
                                "ranker_overlap"
                            ]
                            < paired[
                                "regressor_overlap"
                            ]
                        ).mean()
                    ),

                "mean_overlap_difference":
                    (
                        paired[
                            "ranker_overlap"
                        ]
                        - paired[
                            "regressor_overlap"
                        ]
                    ).mean(),

                "mean_volume_capture_difference_pct_points":
                    (
                        100
                        * (
                            paired[
                                "ranker_volume_capture"
                            ]
                            - paired[
                                "regressor_volume_capture"
                            ]
                        ).mean()
                    ),

                "mean_rank_error_difference":
                    (
                        paired[
                            "ranker_rank_error"
                        ]
                        - paired[
                            "regressor_rank_error"
                        ]
                    ).mean(),

                "mean_rank_correlation_difference":
                    (
                        paired[
                            "ranker_rank_correlation"
                        ]
                        - paired[
                            "regressor_rank_correlation"
                        ]
                    ).mean(),
            }
        ]
    )

    return paired, summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Backtest an XGBoost LambdaMART "
            "neighborhood ranking model."
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
    )

    parser.add_argument(
        "--regressor-predictions",
        default=str(
            XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_PREDICTIONS_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            XGBOOST_RANKING_BACKTEST_DIR
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ranker-spec",
        choices=[
            "full_ndcg",
            "full_pairwise",
        ],
        default="full_ndcg",
        help=(
            "Ranker experiment to run: "
            "full_ndcg or full_pairwise."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    target_panel = pd.read_parquet(
        args.target_panel
    )

    target_panel = prepare_target_panel(
        target_panel
    )

    feature_panel = (
        load_or_build_feature_panel(
            target_panel=
                target_panel,

            feature_panel_path=
                args.feature_panel,

            rebuild=False,
        )
    )

    # --------------------------------------------------------------
    # Same locked folds as regression XGBoost
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

    # --------------------------------------------------------------
    # Backtest
    # --------------------------------------------------------------

    prediction_frames = []

    for _, fold_row in (
        folds
        .sort_values(
            "fold"
        )
        .iterrows()
    ):

        fold_number = int(
            fold_row[
                "fold"
            ]
        )

        print(
            f"Running XGBRanker fold "
            f"{fold_number} "
            f"({args.ranker_spec})..."
        )

        fold_predictions = (
            run_ranker_fold(
                feature_panel=
                    feature_panel,

                fold_row=
                    fold_row,

                ranker_spec=
                    args.ranker_spec,

                top_k=
                    args.top_k,
            )
        )

        prediction_frames.append(
            fold_predictions
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Same top-10 evaluation as regression XGBoost
    # --------------------------------------------------------------

    daily_top10 = (
        evaluate_top10_predictions(
            predictions=
                predictions,

            top_k=
                args.top_k,
        )
    )

    top10_summary = (
        summarize_top10(
            daily_top10
        )
    )

    daily_ndcg = (
        calculate_daily_ndcg(
            predictions=
                predictions,

            top_k=
                args.top_k,
        )
    )

    fold_summary = (
        build_fold_summary(
            daily_top10=
                daily_top10,

            daily_ndcg=
                daily_ndcg,

            top_k=
                args.top_k,
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

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

    predictions.to_parquet(
        output_dir
        / "predictions.parquet",
        index=False,
    )

    daily_top10.to_parquet(
        output_dir
        / "daily_top10_results.parquet",
        index=False,
    )

    daily_ndcg.to_parquet(
        output_dir
        / "daily_ndcg.parquet",
        index=False,
    )

    top10_summary.to_csv(
        output_dir
        / "top10_summary.csv",
        index=False,
    )

    fold_summary.to_csv(
        output_dir
        / "fold_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Compare against existing XGBRegressor
    # --------------------------------------------------------------

    regressor_path = Path(
        args.regressor_predictions
    )

    comparison_summary = None

    if regressor_path.exists():

        regressor_predictions = (
            pd.read_parquet(
                regressor_path
            )
        )

        (
            paired_comparison,
            comparison_summary,
        ) = compare_with_regressor(
            ranker_daily=
                daily_top10,

            regressor_predictions=
                regressor_predictions,

            top_k=
                args.top_k,
        )

        paired_comparison.to_parquet(
            output_dir
            / "ranker_vs_regressor_daily.parquet",
            index=False,
        )

        comparison_summary.to_csv(
            output_dir
            / "ranker_vs_regressor_summary.csv",
            index=False,
        )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print(
        "\nXGBRanker top-10 results\n"
    )

    print(
        top10_summary.to_string(
            index=False
        )
    )

    print(
        "\nBy fold\n"
    )

    display_fold = (
        fold_summary.copy()
    )

    display_fold[
        "mean_accuracy"
    ] *= 100

    display_fold[
        "mean_volume_capture"
    ] *= 100

    print(
        display_fold.to_string(
            index=False
        )
    )

    if comparison_summary is not None:

        print(
            "\nXGBRanker vs "
            "XGBRegressor\n"
        )

        print(
            comparison_summary.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
