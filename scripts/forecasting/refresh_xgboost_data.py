from __future__ import annotations

import argparse
import json
import logging

from forecasting.paths import TARGET_PANEL_5Y_PATH
from forecasting.production.data_refresh import (
    FULL_REFRESH_MODE,
    SMOKE_REFRESH_MODE,
    refresh_production_data,
    run_connectivity_check,
)
from forecasting.production.inference import load_verified_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the bounded XGBoost production target panel.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-panel", default=str(TARGET_PANEL_5Y_PATH))
    parser.add_argument("--check-connectivity", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--write-target-panel", action="store_true")
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("--read-timeout", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--retry-backoff-seconds", type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.check_connectivity and args.smoke_test:
        parser.error("--check-connectivity and --smoke-test are mutually exclusive")

    if args.check_connectivity:
        summary = run_connectivity_check(
            timeout=(
                args.connect_timeout if args.connect_timeout is not None else 5.0,
                args.read_timeout if args.read_timeout is not None else 10.0,
            ),
            max_retries=args.max_retries if args.max_retries is not None else 3,
            retry_backoff_seconds=args.retry_backoff_seconds if args.retry_backoff_seconds is not None else 1.0,
        )
        logging.info("%s", json.dumps(summary, sort_keys=True, default=str))
        return

    artifact = load_verified_artifact(args.artifact_dir)
    refresh_mode = SMOKE_REFRESH_MODE if args.smoke_test else FULL_REFRESH_MODE
    result = refresh_production_data(
        expected_neighborhoods=artifact["baseline"]["expected_neighborhoods"],
        target_panel_path=args.target_panel,
        refresh_mode=refresh_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        write_target_panel=args.write_target_panel if args.smoke_test or args.start_date or args.end_date else None,
        page_size=args.page_size or 50000,
        connect_timeout=args.connect_timeout if args.connect_timeout is not None else 5.0,
        read_timeout=args.read_timeout if args.read_timeout is not None else 60.0,
        max_retries=args.max_retries if args.max_retries is not None else 3,
        retry_backoff_seconds=args.retry_backoff_seconds if args.retry_backoff_seconds is not None else 1.0,
    )
    logging.info("%s", json.dumps(result["summary"], sort_keys=True, default=str))


if __name__ == "__main__":
    main()
