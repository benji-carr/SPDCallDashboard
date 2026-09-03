from __future__ import annotations

import argparse
import logging

import pandas as pd

from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.monitoring import run_monitoring


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update production monitoring outputs for an explicit XGBoost artifact."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    parser.add_argument("--forecasts-root", default=str(FORECASTS_DIR))
    parser.add_argument("--monitoring-root", default=str(MONITORING_DIR))
    parser.add_argument("--allow-mismatched-artifact-run-id", action="store_true")
    parser.add_argument("--max-data-age-days", type=int)
    parser.add_argument("--no-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_monitoring(
        artifact_dir=args.artifact_dir,
        target_panel=pd.read_parquet(args.target_panel),
        forecasts_root=args.forecasts_root,
        monitoring_root=args.monitoring_root,
        allow_mismatched_artifact_run_id=args.allow_mismatched_artifact_run_id,
        max_data_age_days=args.max_data_age_days,
        update_latest=not args.no_latest,
    )
    summary = result["summary"]
    logging.info("Artifact run id: %s", summary["artifact_run_id"])
    logging.info("Forecasts discovered: %s", summary["n_forecasts"])
    logging.info("Forecasts awaiting actuals: %s", summary["n_awaiting_actuals"])
    logging.info("Forecasts matured: %s", summary["n_matured"])
    logging.info("Forecasts evaluated: %s", summary["n_evaluated"])
    logging.info("Latest observed target date: %s", summary["latest_observed_target_date"])
    logging.info("Latest forecast target date: %s", summary["latest_forecast_target_date"])
    logging.info("Latest realized target date: %s", summary["latest_realized_target_date"])
    logging.info("Source data age days: %s", summary["source_data_age_days"])
    logging.info("Report directory: %s", result["report_dir"])
    if result["latest_dir"] is not None:
        logging.info("Latest convenience directory: %s", result["latest_dir"])


if __name__ == "__main__":
    main()
