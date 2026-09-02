from __future__ import annotations

import argparse
import json
import logging

from forecasting.paths import TARGET_PANEL_5Y_PATH
from forecasting.production.data_refresh import refresh_production_data
from forecasting.production.inference import load_verified_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the bounded XGBoost production target panel.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact = load_verified_artifact(args.artifact_dir)
    result = refresh_production_data(expected_neighborhoods=artifact["baseline"]["expected_neighborhoods"], target_panel_path=args.target_panel)
    logging.info("%s", json.dumps(result["summary"], sort_keys=True, default=str))


if __name__ == "__main__":
    main()
