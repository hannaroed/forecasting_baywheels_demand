from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import (
    aggregate_hourly_departures,
    filter_san_francisco_trips,
    generate_demo_dataset,
    load_trip_history,
    load_weather_data,
    maybe_download_sources,
)
from .evaluation import evaluate_models, fit_and_score_holdout, rolling_origin_slices
from .features import build_modeling_frame, select_feature_columns
from .progress import NullRunProgress, RunProgress


def summarize_spectrum(series: pd.Series, sample_spacing_hours: float = 1.0, top_k: int = 10) -> pd.DataFrame:
    centered = series.astype(float).to_numpy() - float(series.mean())
    frequencies = np.fft.rfftfreq(len(centered), d=sample_spacing_hours)
    power = np.abs(np.fft.rfft(centered)) ** 2 / len(centered)
    spectral = pd.DataFrame({"frequency_per_hour": frequencies, "power": power})
    spectral = spectral.loc[spectral["frequency_per_hour"] > 0].copy()
    spectral["period_hours"] = 1.0 / spectral["frequency_per_hour"]
    spectral = spectral.sort_values("power", ascending=False).head(top_k).reset_index(drop=True)
    return spectral


def _split_train_holdout(frame: pd.DataFrame, holdout_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) <= holdout_hours:
        raise ValueError("Holdout period is larger than the available modeling frame.")
    return frame.iloc[:-holdout_hours].copy(), frame.iloc[-holdout_hours:].copy()


def _save_plot(predictions: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=False)

    holdout = predictions.loc[predictions["split"] == "holdout"]
    if not holdout.empty:
        for model_name, group in holdout.groupby("model"):
            axes[0].plot(group["timestamp"], group["predicted"], label=model_name)
        axes[0].plot(holdout["timestamp"].unique(), holdout.groupby("timestamp")["actual"].first(), color="black", linewidth=1.5, label="actual")
        axes[0].set_title("Holdout Forecasts")
        axes[0].legend(loc="upper right")

    rolling = predictions.loc[predictions["split"] != "holdout"].copy()
    if not rolling.empty:
        rolling["absolute_error"] = (rolling["actual"] - rolling["predicted"]).abs()
        error_summary = rolling.groupby(["model", "timestamp"], as_index=False)["absolute_error"].mean()
        for model_name, group in error_summary.groupby("model"):
            axes[1].plot(group["timestamp"], group["absolute_error"], label=model_name)
        axes[1].set_title("Mean Absolute Error Across Rolling Splits")
        axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _write_outputs(
    config: ForecastConfig,
    modeling_frame: pd.DataFrame,
    spectral_summary: pd.DataFrame,
    rolling_results,
    holdout_results,
) -> dict[str, Path]:
    config.ensure_directories()

    processed_path = config.processed_dir / "modeling_frame.csv"
    spectral_path = config.artifacts_dir / "spectral_summary.csv"
    metrics_path = config.artifacts_dir / "metrics.csv"
    predictions_path = config.artifacts_dir / "predictions.csv"
    coefficients_path = config.artifacts_dir / "elastic_net_coefficients.csv"
    summary_path = config.artifacts_dir / "run_summary.json"
    plot_path = config.artifacts_dir / "forecast_diagnostics.png"

    modeling_frame.to_csv(processed_path)
    spectral_summary.to_csv(spectral_path, index=False)

    metrics = pd.concat([rolling_results.metrics, holdout_results.metrics], ignore_index=True)
    predictions = pd.concat([rolling_results.predictions, holdout_results.predictions], ignore_index=True)
    coefficients = pd.concat([rolling_results.coefficients, holdout_results.coefficients], ignore_index=True)

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    _save_plot(predictions, plot_path)

    summary = {
        "rows_in_modeling_frame": int(len(modeling_frame)),
        "rolling_splits": int(rolling_results.metrics["split"].nunique()),
        "holdout_hours": int(config.holdout_hours),
        "best_holdout_model_by_rmse": holdout_results.metrics.sort_values("rmse").iloc[0]["model"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "modeling_frame": processed_path,
        "spectral_summary": spectral_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "coefficients": coefficients_path,
        "summary": summary_path,
        "plot": plot_path,
    }


def run_pipeline(config: ForecastConfig, progress: RunProgress | None = None) -> dict[str, Path]:
    progress = progress or NullRunProgress()
    progress.stage("Preparing directories")
    config.ensure_directories()
    progress.stage("Checking source downloads")
    maybe_download_sources(config)

    progress.stage("Loading Bay Wheels trip files")
    trips = load_trip_history(config.raw_baywheels_dir, timezone=config.timezone)
    progress.stage("Filtering San Francisco trips")
    sf_trips = filter_san_francisco_trips(trips, config)
    progress.stage("Aggregating hourly departures")
    departures = aggregate_hourly_departures(sf_trips, config)
    progress.stage("Loading KSFO weather")
    weather = load_weather_data(config.weather_path, timezone=config.timezone)

    progress.stage("Building modeling frame")
    modeling_frame = build_modeling_frame(departures, weather, config)
    train_frame, holdout_frame = _split_train_holdout(modeling_frame, config.holdout_hours)
    feature_sets = select_feature_columns(modeling_frame)
    rolling_splits = rolling_origin_slices(
        len(train_frame),
        config.rolling_train_hours,
        config.rolling_test_hours,
        config.rolling_step_hours,
    )

    progress.stage("Computing spectral summary")
    spectral_summary = summarize_spectrum(train_frame[config.target_column])
    progress.stage("Running rolling evaluation")
    rolling_progress = progress.task("Rolling evaluation", total=len(feature_sets) * len(rolling_splits))
    rolling_results = evaluate_models(train_frame, feature_sets, config, progress=rolling_progress)
    rolling_progress.close()
    progress.stage("Scoring holdout period")
    holdout_progress = progress.task("Holdout evaluation", total=len(feature_sets))
    holdout_results = fit_and_score_holdout(train_frame, holdout_frame, feature_sets, config, progress=holdout_progress)
    holdout_progress.close()

    progress.stage("Writing outputs")
    outputs = _write_outputs(config, modeling_frame, spectral_summary, rolling_results, holdout_results)
    progress.close()
    return outputs


def run_demo_pipeline(config: ForecastConfig, progress: RunProgress | None = None) -> dict[str, Path]:
    progress = progress or NullRunProgress()
    progress.stage("Preparing directories")
    config.ensure_directories()
    progress.stage("Generating demo dataset")
    departures, weather = generate_demo_dataset(config)
    progress.stage("Building modeling frame")
    modeling_frame = build_modeling_frame(departures, weather, config)
    train_frame, holdout_frame = _split_train_holdout(modeling_frame, config.holdout_hours)
    feature_sets = select_feature_columns(modeling_frame)
    rolling_splits = rolling_origin_slices(
        len(train_frame),
        config.rolling_train_hours,
        config.rolling_test_hours,
        config.rolling_step_hours,
    )

    progress.stage("Computing spectral summary")
    spectral_summary = summarize_spectrum(train_frame[config.target_column])
    progress.stage("Running rolling evaluation")
    rolling_progress = progress.task("Rolling evaluation", total=len(feature_sets) * len(rolling_splits))
    rolling_results = evaluate_models(train_frame, feature_sets, config, progress=rolling_progress)
    rolling_progress.close()
    progress.stage("Scoring holdout period")
    holdout_progress = progress.task("Holdout evaluation", total=len(feature_sets))
    holdout_results = fit_and_score_holdout(train_frame, holdout_frame, feature_sets, config, progress=holdout_progress)
    holdout_progress.close()

    progress.stage("Writing outputs")
    outputs = _write_outputs(config, modeling_frame, spectral_summary, rolling_results, holdout_results)
    progress.close()
    return outputs
