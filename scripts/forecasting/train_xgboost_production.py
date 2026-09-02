from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from forecasting.backtests.xgboost import load_or_build_feature_panel, resolve_numeric_features
from forecasting.features.xgboost import prepare_target_panel
from forecasting.paths import MODEL_ARTIFACTS_DIR, TARGET_PANEL_5Y_PATH, XGBOOST_FEATURE_PANEL_PATH
from forecasting.production.xgboost import FEATURE_SET_NAME, MODEL_VERSION, train_production_model, validate_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train the frozen XGBoost production model on all eligible history.")
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    parser.add_argument("--feature-panel", default=str(XGBOOST_FEATURE_PANEL_PATH))
    parser.add_argument("--output-root", default=str(MODEL_ARTIFACTS_DIR))
    parser.add_argument("--model-version", default=MODEL_VERSION)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    target_path, feature_path = Path(args.target_panel), Path(args.feature_panel)
    if not target_path.is_file():
        raise FileNotFoundError(f"Target panel was not found: {target_path}")
    logging.info("Resolving inputs")
    target_panel = prepare_target_panel(pd.read_parquet(target_path))
    feature_panel = load_or_build_feature_panel(target_panel, feature_path, rebuild=False)
    training_preview = validate_training_data(
        target_panel,
        feature_panel,
        resolve_numeric_features(feature_set_name=FEATURE_SET_NAME),
    )
    logging.info("Target panel: %s", target_path)
    logging.info("Feature panel: %s", feature_path)
    logging.info("Feature set: %s", FEATURE_SET_NAME)
    logging.info("First target date: %s", target_panel["target_date"].min().date())
    logging.info("First eligible training date: %s", training_preview["target_date"].min().date())
    logging.info("Last training date: %s", training_preview["target_date"].max().date())
    logging.info("Training rows: %s; dates: %s; neighborhoods: %s", len(training_preview), training_preview["target_date"].nunique(), training_preview["neighborhood"].nunique())
    logging.info("Target mean/std/min/max: %.6f / %.6f / %.6f / %.6f", training_preview["calls"].mean(), training_preview["calls"].std(), training_preview["calls"].min(), training_preview["calls"].max())
    result = train_production_model(
        target_panel=target_panel, feature_panel=feature_panel,
        target_panel_path=target_path, feature_panel_path=feature_path,
        output_root=Path(args.output_root), model_version=args.model_version,
    )
    summary = result["summary"]
    logging.info("Training range: %s to %s", summary["training_start_date"], summary["training_end_date"])
    logging.info("Training rows: %s; dates: %s; neighborhoods: %s", summary["n_training_rows"], summary["n_training_dates"], summary["n_neighborhoods"])
    logging.info("Artifacts: %s", result["artifact_dir"])


if __name__ == "__main__":
    main()
