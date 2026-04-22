from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .features import inverse_transform
from .models import ElasticNetForecaster, SarimaxForecaster
from .progress import NullTaskProgress, TaskProgress


@dataclass
class ForecastOutputs:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def rolling_origin_slices(length: int, train_size: int, test_size: int, step_size: int) -> list[tuple[slice, slice]]:
    splits: list[tuple[slice, slice]] = []
    train_end = train_size
    while train_end + test_size <= length:
        splits.append((slice(train_end - train_size, train_end), slice(train_end, train_end + test_size)))
        train_end += step_size
    return splits


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _predict_model(name: str, train_frame: pd.DataFrame, test_frame: pd.DataFrame, feature_columns: list[str], config: ForecastConfig) -> tuple[np.ndarray, pd.Series | None]:
    if name.startswith("sarimax"):
        model = SarimaxForecaster(config=config).fit(train_frame, feature_columns, "target_transformed")
        return model.predict(test_frame), None

    model = ElasticNetForecaster(config=config).fit(train_frame, feature_columns, "target_transformed")
    return model.predict(test_frame), model.coefficients()


def evaluate_models(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: ForecastConfig,
    progress: TaskProgress | None = None,
) -> ForecastOutputs:
    progress = progress or NullTaskProgress()
    splits = rolling_origin_slices(
        length=len(frame),
        train_size=config.rolling_train_hours,
        test_size=config.rolling_test_hours,
        step_size=config.rolling_step_hours,
    )
    if not splits:
        raise ValueError("Not enough rows to create rolling-origin splits. Reduce the window sizes or provide more data.")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []

    for split_id, (train_slice, test_slice) in enumerate(splits, start=1):
        train_frame = frame.iloc[train_slice].copy()
        test_frame = frame.iloc[test_slice].copy()
        actual = test_frame["departures"].to_numpy()

        for model_name, feature_columns in feature_sets.items():
            transformed_pred, coefficients = _predict_model(model_name, train_frame, test_frame, feature_columns, config)
            predicted = inverse_transform(transformed_pred, config.transform)
            progress.update(description=f"Split {split_id}/{len(splits)}: {model_name}")

            metric_rows.append(
                {
                    "split": split_id,
                    "model": model_name,
                    "rmse": _rmse(actual, predicted),
                    "mae": _mae(actual, predicted),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "split": split_id,
                        "model": model_name,
                        "timestamp": test_frame.index,
                        "actual": actual,
                        "predicted": predicted,
                    }
                )
            )

            if coefficients is not None:
                coefficient_frames.append(
                    coefficients.rename_axis("feature")
                    .reset_index()
                    .assign(split=split_id, model=model_name)
                )

    return ForecastOutputs(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        coefficients=pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame(),
    )


def fit_and_score_holdout(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: ForecastConfig,
    progress: TaskProgress | None = None,
) -> ForecastOutputs:
    progress = progress or NullTaskProgress()
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    actual = holdout_frame["departures"].to_numpy()

    for model_name, feature_columns in feature_sets.items():
        transformed_pred, coefficients = _predict_model(model_name, train_frame, holdout_frame, feature_columns, config)
        predicted = inverse_transform(transformed_pred, config.transform)
        progress.update(description=f"Holdout: {model_name}")

        metric_rows.append(
            {
                "split": "holdout",
                "model": model_name,
                "rmse": _rmse(actual, predicted),
                "mae": _mae(actual, predicted),
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": "holdout",
                    "model": model_name,
                    "timestamp": holdout_frame.index,
                    "actual": actual,
                    "predicted": predicted,
                }
            )
        )

        if coefficients is not None:
            coefficient_frames.append(
                coefficients.rename_axis("feature")
                .reset_index()
                .assign(split="holdout", model=model_name)
            )

    return ForecastOutputs(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        coefficients=pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame(),
    )
