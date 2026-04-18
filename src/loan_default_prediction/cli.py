from __future__ import annotations

import argparse

from loan_default_prediction.logging_utils import configure_logging
from loan_default_prediction.paths import LOG_DIR, ensure_directories
from loan_default_prediction.pipeline import run_demo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Loan default prediction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", default="demo", choices=["demo"])
    args = parser.parse_args()

    ensure_directories()
    logger = configure_logging(LOG_DIR / "pipeline.log")

    if args.command == "run":
        run_demo_pipeline(logger)


if __name__ == "__main__":
    main()

