# Forecasting Hourly Bay Wheels Demand in San Francisco

Forecast hourly Bay Wheels departures in San Francisco using Bay Wheels trip history and KSFO weather data.

## Setup

```bash
uv sync --extra dev
```

## Data

Expected files:

- `data/raw/baywheels/` with monthly Bay Wheels trip files from `202201` through `202512`
- `data/raw/weather/ksfo_hourly.csv`

## Run

Run the full pipeline on the real data:

```bash
uv run baywheels-forecast run
```

Run the synthetic demo:

```bash
uv run baywheels-forecast demo
```

Run a smaller demo for a quick smoke test:

```bash
uv run baywheels-forecast demo \
  --demo-hours 2880 \
  --holdout-hours 336 \
  --rolling-train-hours 1440 \
  --rolling-test-hours 168 \
  --rolling-step-hours 168
```

Disable progress bars if needed:

```bash
uv run baywheels-forecast run --no-progress
```

## Outputs

The pipeline writes:

- `data/processed/modeling_frame.csv`
- `artifacts/spectral_summary.csv`
- `artifacts/metrics.csv`
- `artifacts/predictions.csv`
- `artifacts/elastic_net_coefficients.csv`
- `artifacts/forecast_diagnostics.png`
- `artifacts/run_summary.json`

## What It Does

- filters Bay Wheels trips to San Francisco
- aggregates trips to hourly departures
- merges hourly KSFO weather data
- creates lagged demand, weather, and calendar features
- fits spectral, SARIMAX, and elastic-net forecasting components
- evaluates models with rolling-origin splits and a held-out test period
