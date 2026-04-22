from __future__ import annotations

import json

from baywheels_forecasting.config import ForecastConfig
from baywheels_forecasting.pipeline import run_demo_pipeline


def test_demo_pipeline_creates_artifacts(tmp_path):
    config = ForecastConfig(
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        demo_hours=24 * 120,
        holdout_hours=24 * 14,
        rolling_train_hours=24 * 60,
        rolling_test_hours=24 * 7,
        rolling_step_hours=24 * 7,
    )

    outputs = run_demo_pipeline(config)

    for path in outputs.values():
        assert path.exists()

    summary = json.loads(outputs["summary"].read_text())
    assert summary["rows_in_modeling_frame"] > 0
    assert summary["best_holdout_model_by_rmse"] in {
        "sarimax_baseline",
        "sarimax_weather",
        "elastic_net_baseline",
        "elastic_net_full",
    }
