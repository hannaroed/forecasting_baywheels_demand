from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ForecastConfig:
    raw_baywheels_dir: Path = Path("data/raw/baywheels")
    weather_path: Path = Path("data/raw/weather/ksfo_hourly.csv")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    baywheels_urls: list[str] = field(default_factory=list)
    weather_url: str | None = None
    demo_hours: int = 24 * 365
    timezone: str = "America/Los_Angeles"
    analysis_start: str = "2022-01-01"
    analysis_end: str = "2025-12-31 23:00:00"
    target_column: str = "departures"
    transform: str = "log1p"
    daily_period: int = 24
    weekly_period: int = 168
    demand_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 25, 48, 72, 168, 169)
    weather_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    holdout_hours: int = 24 * 28
    rolling_train_hours: int = 24 * 180
    rolling_test_hours: int = 24 * 7
    rolling_step_hours: int = 24 * 7
    daily_fourier_order: int = 3
    weekly_fourier_order: int = 2
    arima_order: tuple[int, int, int] = (1, 0, 1)
    arima_seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 24)
    elastic_net_l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9, 1.0)
    elastic_net_cv_splits: int = 5
    sf_latitude_bounds: tuple[float, float] = (37.70, 37.83)
    sf_longitude_bounds: tuple[float, float] = (-122.53, -122.35)
    target_city: str = "san francisco"

    def ensure_directories(self) -> None:
        for path in (self.raw_baywheels_dir, self.weather_path.parent, self.processed_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
