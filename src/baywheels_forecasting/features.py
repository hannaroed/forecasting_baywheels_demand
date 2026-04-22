from __future__ import annotations

import math

import holidays
import numpy as np
import pandas as pd

from .config import ForecastConfig


def transform_target(series: pd.Series, transform: str) -> pd.Series:
    if transform == "log1p":
        return np.log1p(series.astype(float))
    raise ValueError(f"Unsupported transform: {transform}")


def inverse_transform(values: pd.Series | np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        safe_values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=8.0, neginf=-20.0)
        return np.clip(np.expm1(np.clip(safe_values, a_min=-20.0, a_max=8.0)), a_min=0.0, a_max=None)
    raise ValueError(f"Unsupported transform: {transform}")


def build_calendar_features(index: pd.DatetimeIndex, config: ForecastConfig) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["hour"] = index.hour
    frame["day_of_week"] = index.dayofweek
    frame["month"] = index.month
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["is_peak_commute"] = (((index.hour >= 7) & (index.hour <= 9)) | ((index.hour >= 16) & (index.hour <= 18))).astype(int)

    years = range(index.min().year, index.max().year + 1)
    us_holidays = holidays.US(years=years)
    frame["is_holiday"] = index.normalize().map(lambda ts: int(ts in us_holidays))
    frame["time_idx"] = np.arange(len(index), dtype=float)

    for prefix, period, order in (
        ("daily", config.daily_period, config.daily_fourier_order),
        ("weekly", config.weekly_period, config.weekly_fourier_order),
    ):
        for harmonic in range(1, order + 1):
            angle = 2.0 * math.pi * harmonic * np.arange(len(index)) / period
            frame[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
            frame[f"{prefix}_cos_{harmonic}"] = np.cos(angle)

    return frame


def add_lagged_features(frame: pd.DataFrame, columns: list[str], lags: tuple[int, ...], prefix: str) -> pd.DataFrame:
    lagged = pd.DataFrame(index=frame.index)
    for column in columns:
        for lag in lags:
            lagged[f"{prefix}_{column}_lag_{lag}"] = frame[column].shift(lag)
    return lagged


def build_modeling_frame(
    hourly_departures: pd.DataFrame,
    weather: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    data = hourly_departures.join(weather, how="left").sort_index()
    weather_columns = [column for column in data.columns if column.startswith("weather_")]
    if weather_columns:
        data[weather_columns] = data[weather_columns].ffill().bfill()

    data["target_transformed"] = transform_target(data[config.target_column], config.transform)
    calendar = build_calendar_features(data.index, config)
    lagged_target = add_lagged_features(
        data[["target_transformed"]],
        columns=["target_transformed"],
        lags=config.demand_lags,
        prefix="lag",
    )
    lagged_weather = add_lagged_features(data[weather_columns], weather_columns, config.weather_lags, prefix="lag") if weather_columns else pd.DataFrame(index=data.index)

    interactions = pd.DataFrame(index=data.index)
    if "weather_precipitation" in data:
        interactions["interaction_precip_peak"] = data["weather_precipitation"].shift(1) * calendar["is_peak_commute"]
    if "weather_temperature" in data:
        interactions["interaction_temp_peak"] = data["weather_temperature"].shift(1) * calendar["is_peak_commute"]

    modeling_frame = pd.concat([data, calendar, lagged_target, lagged_weather, interactions], axis=1)
    return modeling_frame.dropna().copy()


def select_feature_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    calendar_columns = [
        column
        for column in frame.columns
        if column
        in {
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_peak_commute",
            "is_holiday",
            "time_idx",
        }
        or column.startswith("daily_")
        or column.startswith("weekly_")
    ]
    weather_columns = [column for column in frame.columns if column.startswith("lag_weather_") or column.startswith("interaction_")]
    autoregressive_columns = [column for column in frame.columns if column.startswith("lag_target_transformed_")]

    return {
        "sarimax_baseline": calendar_columns,
        "sarimax_weather": calendar_columns + weather_columns,
        "elastic_net_baseline": autoregressive_columns + calendar_columns,
        "elastic_net_full": autoregressive_columns + calendar_columns + weather_columns,
    }
