from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForecastConfig
from .pipeline import run_demo_pipeline, run_pipeline
from .progress import ConsoleRunProgress, NullRunProgress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast hourly San Francisco Bay Wheels departures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    common.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    common.add_argument("--holdout-hours", type=int, default=24 * 28)
    common.add_argument("--rolling-train-hours", type=int, default=24 * 180)
    common.add_argument("--rolling-test-hours", type=int, default=24 * 7)
    common.add_argument("--rolling-step-hours", type=int, default=24 * 7)
    common.add_argument("--demo-hours", type=int, default=24 * 365)
    common.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars.")

    real = subparsers.add_parser("run", parents=[common], help="Run the full pipeline on real Bay Wheels and NOAA files.")
    real.add_argument("--raw-baywheels-dir", type=Path, default=Path("data/raw/baywheels"))
    real.add_argument("--weather-path", type=Path, default=Path("data/raw/weather/ksfo_hourly.csv"))
    real.add_argument("--baywheels-url", action="append", default=[])
    real.add_argument("--weather-url", default=None)

    subparsers.add_parser("demo", parents=[common], help="Run the pipeline on synthetic data for a quick verification.")
    return parser


def _config_from_args(args: argparse.Namespace) -> ForecastConfig:
    return ForecastConfig(
        raw_baywheels_dir=getattr(args, "raw_baywheels_dir", Path("data/raw/baywheels")),
        weather_path=getattr(args, "weather_path", Path("data/raw/weather/ksfo_hourly.csv")),
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        baywheels_urls=getattr(args, "baywheels_url", []),
        weather_url=getattr(args, "weather_url", None),
        demo_hours=args.demo_hours,
        holdout_hours=args.holdout_hours,
        rolling_train_hours=args.rolling_train_hours,
        rolling_test_hours=args.rolling_test_hours,
        rolling_step_hours=args.rolling_step_hours,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _config_from_args(args)
    progress = NullRunProgress() if args.no_progress else ConsoleRunProgress(total_stages=11 if args.command == "run" else 7)

    outputs = run_demo_pipeline(config, progress=progress) if args.command == "demo" else run_pipeline(config, progress=progress)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
