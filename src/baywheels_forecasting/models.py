from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import ForecastConfig


@dataclass
class ElasticNetForecaster:
    config: ForecastConfig
    model: Pipeline | None = None
    feature_columns: list[str] | None = None

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> "ElasticNetForecaster":
        self.feature_columns = feature_columns
        n_samples = len(frame)
        n_splits = max(2, min(self.config.elastic_net_cv_splits, n_samples // 48))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        estimator = ElasticNetCV(
            l1_ratio=self.config.elastic_net_l1_ratios,
            cv=splitter,
            alphas=40,
            max_iter=20000,
        )
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", estimator),
            ]
        )
        self.model.fit(frame[feature_columns], frame[target_column])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("ElasticNetForecaster must be fit before prediction.")
        return self.model.predict(frame[self.feature_columns])

    def coefficients(self) -> pd.Series:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("ElasticNetForecaster must be fit before requesting coefficients.")
        regressor = self.model.named_steps["regressor"]
        return pd.Series(regressor.coef_, index=self.feature_columns, name="coefficient").sort_values(key=np.abs, ascending=False)


@dataclass
class SarimaxForecaster:
    config: ForecastConfig
    results: object | None = None
    feature_columns: list[str] | None = None

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> "SarimaxForecaster":
        self.feature_columns = feature_columns
        exog = frame[feature_columns] if feature_columns else None
        model = SARIMAX(
            endog=frame[target_column],
            exog=exog,
            order=self.config.arima_order,
            seasonal_order=self.config.arima_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("SarimaxForecaster must be fit before prediction.")
        exog = frame[self.feature_columns] if self.feature_columns else None
        prediction = self.results.get_forecast(steps=len(frame), exog=exog)
        return np.asarray(prediction.predicted_mean)
