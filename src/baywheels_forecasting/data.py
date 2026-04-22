from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ForecastConfig

TRIP_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "started_at": ("started_at", "start_time", "starttime"),
    "start_station_id": ("start_station_id", "from_station_id"),
    "start_station_name": ("start_station_name", "from_station_name"),
    "start_station_latitude": ("start_lat", "start_station_latitude", "start station latitude"),
    "start_station_longitude": ("start_lng", "start_lon", "start_station_longitude", "start station longitude"),
    "start_city": ("start_station_city", "start_city", "start station city"),
}

WEATHER_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "timestamp": ("DATE", "date", "timestamp"),
    "temperature": ("HourlyDryBulbTemperature", "temperature", "dry_bulb_temperature"),
    "dew_point": ("HourlyDewPointTemperature", "dew_point"),
    "humidity": ("HourlyRelativeHumidity", "relative_humidity"),
    "precipitation": ("HourlyPrecipitation", "precipitation"),
    "wind_speed": ("HourlyWindSpeed", "wind_speed"),
    "visibility": ("HourlyVisibility", "visibility"),
    "pressure": ("HourlyStationPressure", "pressure"),
}


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def maybe_download_sources(config: ForecastConfig) -> None:
    for url in config.baywheels_urls:
        destination = config.raw_baywheels_dir / url.rsplit("/", 1)[-1]
        if not destination.exists():
            download_file(url, destination)
    if config.weather_url and not config.weather_path.exists():
        download_file(config.weather_url, config.weather_path)


def discover_trip_files(raw_baywheels_dir: Path) -> list[Path]:
    patterns = ("*.csv", "*.csv.gz", "*.zip", "*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(raw_baywheels_dir.rglob(pattern)))
    return files


def _normalize_columns(columns: pd.Index) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for column in columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        normalized[key] = str(column)
    return normalized


def _match_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = _normalize_columns(df.columns)
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "_", candidate.strip().lower()).strip("_")
        if key in normalized:
            return normalized[key]
    return None


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if not csv_names:
            raise ValueError(f"No CSV files found in archive: {path}")
        with archive.open(csv_names[0]) as handle:
            return pd.read_csv(io.BytesIO(handle.read()), low_memory=False)


def _read_trip_file(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        return _read_zip_csv(path)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def canonicalize_trip_history(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    selected: dict[str, str] = {}
    for canonical_name, candidates in TRIP_COLUMN_CANDIDATES.items():
        source = _match_column(df, candidates)
        if source:
            selected[source] = canonical_name
    if "started_at" not in selected.values():
        raise ValueError("Trip data must include a trip start timestamp column.")

    trips = df[list(selected)].rename(columns=selected).copy()
    trips["started_at"] = pd.to_datetime(trips["started_at"], errors="coerce")
    if trips["started_at"].dt.tz is None:
        trips["started_at"] = trips["started_at"].dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        trips["started_at"] = trips["started_at"].dt.tz_convert(timezone)
    trips["started_at"] = trips["started_at"].dt.tz_localize(None)

    for column in ("start_station_latitude", "start_station_longitude"):
        if column in trips:
            trips[column] = pd.to_numeric(trips[column], errors="coerce")
    if "start_city" in trips:
        trips["start_city"] = trips["start_city"].astype(str).str.strip().str.lower()
    return trips.dropna(subset=["started_at"])


def load_trip_history(raw_baywheels_dir: Path, timezone: str) -> pd.DataFrame:
    files = discover_trip_files(raw_baywheels_dir)
    if not files:
        raise FileNotFoundError(
            f"No trip files found in {raw_baywheels_dir}. Add Bay Wheels CSV/ZIP/Parquet files or pass download URLs."
        )
    frames = [canonicalize_trip_history(_read_trip_file(path), timezone=timezone) for path in files]
    return pd.concat(frames, ignore_index=True)


def filter_san_francisco_trips(trips: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    city_mask = pd.Series(False, index=trips.index)
    if "start_city" in trips:
        city_mask = trips["start_city"].fillna("").str.contains(config.target_city, regex=False)

    geo_mask = pd.Series(False, index=trips.index)
    if {"start_station_latitude", "start_station_longitude"}.issubset(trips.columns):
        geo_mask = trips["start_station_latitude"].between(*config.sf_latitude_bounds) & trips[
            "start_station_longitude"
        ].between(*config.sf_longitude_bounds)

    filtered = trips.loc[city_mask | geo_mask].copy()
    if filtered.empty:
        raise ValueError("No San Francisco trips were found after filtering. Check the schema or geographic bounds.")
    return filtered


def aggregate_hourly_departures(trips: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    series = (
        trips.set_index("started_at")
        .sort_index()
        .groupby(pd.Grouper(freq="h"))
        .size()
        .rename(config.target_column)
        .to_frame()
    )
    hourly_index = pd.date_range(
        start=config.analysis_start,
        end=config.analysis_end,
        freq="h",
    )
    series = series.reindex(hourly_index, fill_value=0)
    series.index.name = "timestamp"
    return series


def _coerce_numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def load_weather_data(weather_path: Path, timezone: str) -> pd.DataFrame:
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {weather_path}. Add the NOAA LCD export or pass a weather download URL."
        )
    raw = pd.read_csv(weather_path, low_memory=False)
    timestamp_column = _match_column(raw, WEATHER_COLUMN_CANDIDATES["timestamp"])
    if not timestamp_column:
        raise ValueError("Weather data must include a DATE or timestamp column.")

    weather = pd.DataFrame(index=raw.index)
    weather["timestamp"] = pd.to_datetime(raw[timestamp_column], errors="coerce")
    if weather["timestamp"].dt.tz is None:
        weather["timestamp"] = weather["timestamp"].dt.tz_localize("UTC").dt.tz_convert(timezone)
    else:
        weather["timestamp"] = weather["timestamp"].dt.tz_convert(timezone)
    weather["timestamp"] = weather["timestamp"].dt.tz_localize(None).dt.floor("h")

    for canonical_name, candidates in WEATHER_COLUMN_CANDIDATES.items():
        if canonical_name == "timestamp":
            continue
        source = _match_column(raw, candidates)
        if source:
            weather[f"weather_{canonical_name}"] = _coerce_numeric(raw[source])

    weather = weather.dropna(subset=["timestamp"]).groupby("timestamp", as_index=True).mean(numeric_only=True)
    weather = weather.sort_index()
    weather.index.name = "timestamp"
    return weather


def generate_demo_dataset(config: ForecastConfig, seed: int = 248) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=config.demo_hours, freq="h")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    annual_phase = np.linspace(0.0, 4.0 * np.pi, len(index))

    temperature = 58 + 8 * np.sin(annual_phase) + 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.5, len(index))
    precipitation = np.maximum(0, rng.gamma(shape=0.5, scale=0.4, size=len(index)) - 0.15)
    precipitation[rng.random(len(index)) < 0.9] = 0
    wind_speed = 7 + 3 * np.sin(2 * np.pi * (hour - 3) / 24) + rng.normal(0, 0.8, len(index))

    commute = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18))
    weekday = dow < 5
    baseline = 10 + 25 * commute * weekday + 8 * (hour >= 11) * (hour <= 14)
    weekly = 4 * np.sin(2 * np.pi * np.arange(len(index)) / config.weekly_period)
    weather_effect = -9 * precipitation + 0.25 * (temperature - 58) - 0.3 * np.maximum(wind_speed - 10, 0)
    intensity = np.clip(baseline + weekly + weather_effect + rng.normal(0, 2, len(index)), 1, None)
    departures = rng.poisson(intensity)

    demand = pd.DataFrame({config.target_column: departures}, index=index)
    demand.index.name = "timestamp"
    weather = pd.DataFrame(
        {
            "weather_temperature": temperature,
            "weather_precipitation": precipitation,
            "weather_wind_speed": wind_speed,
        },
        index=index,
    )
    weather.index.name = "timestamp"
    return demand, weather
