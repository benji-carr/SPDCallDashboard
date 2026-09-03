from __future__ import annotations

import argparse

from forecasting.production.status import format_production_status, load_production_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a read-only production XGBoost status report.")
    parser.add_argument("--artifact-dir", required=True, help="Explicit production artifact directory.")
    parser.add_argument("--verbose", action="store_true", help="Include optional report diagnostics.")
    args = parser.parse_args()
    print(format_production_status(load_production_status(args.artifact_dir), verbose=args.verbose))


if __name__ == "__main__":
    main()
