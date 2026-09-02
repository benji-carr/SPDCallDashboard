from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_SPD_DIR = RAW_DIR / "spd"
RAW_PERMITTED_EVENTS_DIR = (
    RAW_DIR / "permitted_events"
)
RAW_PREDICTHQ_DIR = (
    RAW_DIR / "predicthq"
)

PROCESSED_DIR = DATA_DIR / "processed"
TARGET_PANEL_5Y_PATH = DATA_DIR / "target_panel_5y.parquet"

FEATURES_DIR = DATA_DIR / "features"
XGBOOST_FEATURE_PANEL_PATH = (
    DATA_DIR / "xgboost" / "xgboost_feature_panel.parquet"
)
SPECIAL_EVENTS_FEATURE_PANEL_PATH = (
    FEATURES_DIR
    / "special_events_feature_panel.parquet"
)
CALENDAR_FEATURES_PATH = (
    FEATURES_DIR / "calendar_features.parquet"
)
PREDICTHQ_OUTPUT_DIR = (
    FEATURES_DIR / "predicthq"
)

BACKTEST_DIR = DATA_DIR / "backtest"
GENERATED_FOLDS_PATH = (
    BACKTEST_DIR
    / "generated_backtest_folds.parquet"
)

SARIMA_BACKTEST_DIR = (
    BACKTEST_DIR / "sarima"
)
SARIMAX_CALENDAR_BACKTEST_DIR = (
    BACKTEST_DIR / "sarimax_calendar"
)
SARIMAX_PERMITTED_EVENTS_BACKTEST_DIR = (
    BACKTEST_DIR
    / "sarimax_permitted_events"
)

XGBOOST_BACKTEST_DIR = (
    BACKTEST_DIR / "xgboost"
)
XGBOOST_REGRESSION_BACKTEST_DIR = (
    XGBOOST_BACKTEST_DIR / "regression"
)
XGBOOST_TUNING_DIR = (
    XGBOOST_BACKTEST_DIR / "tuning"
)
MODEL_ARTIFACTS_DIR = ROOT / "artifacts" / "models"
FORECASTS_DIR = DATA_DIR / "forecasts"
XGBOOST_PERMITTED_EVENTS_BACKTEST_DIR = (
    XGBOOST_BACKTEST_DIR
    / "permitted_events"
)
XGBOOST_RANKING_BACKTEST_DIR = (
    XGBOOST_BACKTEST_DIR / "ranking"
)

PREDICTHQ_CHUNK_DIR = (
    RAW_PREDICTHQ_DIR / "chunks"
)
PERMITTED_EVENTS_FOLDS_PATH = (
    SARIMAX_PERMITTED_EVENTS_BACKTEST_DIR
    / "folds.parquet"
)

XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_DIR = (
    XGBOOST_REGRESSION_BACKTEST_DIR
    / "lags_rolling_calendar"
)
XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_PREDICTIONS_PATH = (
    XGBOOST_REGRESSOR_LAGS_ROLLING_CALENDAR_DIR
    / "predictions.parquet"
)

XGBOOST_TOP10_EVALUATION_DIR = (
    XGBOOST_REGRESSION_BACKTEST_DIR
    / "top10_evaluation"
)
XGBOOST_TOP10_BASELINE_COMPARISON_DIR = (
    XGBOOST_REGRESSION_BACKTEST_DIR
    / "top10_baseline_comparison"
)
XGBOOST_FULL_RANKING_EVALUATION_DIR = (
    XGBOOST_REGRESSION_BACKTEST_DIR
    / "full_ranking"
)
