from __future__ import annotations

import argparse
import logging

from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, OPERATIONS_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.orchestration import run_daily_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily production XGBoost data, forecast, and monitoring pipeline.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    parser.add_argument("--forecast-output-root", default=str(FORECASTS_DIR))
    parser.add_argument("--monitoring-output-root", default=str(MONITORING_DIR))
    parser.add_argument("--operations-output-root", default=str(OPERATIONS_DIR))
    parser.add_argument("--max-source-age-days", type=int)
    parser.add_argument("--skip-source-refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_daily_pipeline(artifact_dir=args.artifact_dir, target_panel_path=args.target_panel, forecasts_root=args.forecast_output_root, monitoring_root=args.monitoring_output_root, operations_root=args.operations_output_root, max_source_age_days=args.max_source_age_days, skip_source_refresh=args.skip_source_refresh)
    logging.info("Production run %s (%s)", result["run_dir"], "idempotent" if result["idempotent"] else "created")


if __name__ == "__main__":
    main()
