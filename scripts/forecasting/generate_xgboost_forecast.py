from __future__ import annotations

import argparse
import logging

import pandas as pd

from forecasting.paths import FORECASTS_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.inference import generate_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a one-day-ahead forecast from an explicit production artifact.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    parser.add_argument("--forecast-origin")
    parser.add_argument("--forecast-output-root", default=str(FORECASTS_DIR))
    parser.add_argument("--max-data-age-days", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = generate_forecast(artifact_dir=args.artifact_dir, target_panel=pd.read_parquet(args.target_panel), forecast_origin=args.forecast_origin, output_root=args.forecast_output_root, max_data_age_days=args.max_data_age_days)
    logging.info("Forecast ID: %s", result["diagnostics"]["forecast_id"])
    logging.info("Snapshot: %s", result["snapshot"])
    logging.info("Idempotent existing snapshot: %s", result["idempotent"])


if __name__ == "__main__":
    main()
