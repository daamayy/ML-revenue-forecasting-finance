"""Tier 3 — Tree-based ML models, formation-date / exogenous path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TabularModel, register


@register("lightgbm")
class LightGBMModel(TabularModel):
    name = "LightGBM"

    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb

        params = dict(
            objective="regression_l1",        # L1 loss matches our MAE target
            num_leaves=self.params.get("num_leaves", 63),
            max_depth=self.params.get("max_depth", 7),
            learning_rate=self.params.get("learning_rate", 0.05),
            n_estimators=self.params.get("n_estimators", 500),
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.params.get("random_state", 42),
            n_jobs=-1, verbose=-1,
        )
        self.model = lgb.LGBMRegressor(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(50, verbose=False)] if eval_set else None
        self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)
        self._cols = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        # Gain-based importance (secondary diagnostic in the plan).
        return pd.Series(self.model.booster_.feature_importance("gain"),
                         index=self._cols)


@register("xgboost")
class XGBoostModel(TabularModel):
    name = "XGBoost"

    def fit(self, X, y, X_val=None, y_val=None):
        import xgboost as xgb

        params = dict(
            objective="reg:absoluteerror",
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.05),
            n_estimators=self.params.get("n_estimators", 500),
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.params.get("random_state", 42),
            n_jobs=-1, early_stopping_rounds=50 if X_val is not None else None,
        )
        self.model = xgb.XGBRegressor(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self._cols = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return pd.Series(self.model.feature_importances_, index=self._cols)


@register("random_forest")
class RandomForestModel(TabularModel):
    name = "RandomForest"

    def fit(self, X, y, X_val=None, y_val=None):
        import os

        from sklearn.ensemble import RandomForestRegressor

        # n_jobs is env-configurable so the GPU host can leave a couple of cores
        # for the LSTM/TFT data pipeline when RF runs in parallel on CPU, e.g.
        # `AS_FIN_RF_N_JOBS=-2` (all but one core). Default -1 = all cores.
        n_jobs = int(os.environ.get("AS_FIN_RF_N_JOBS", "-1"))
        self.model = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 500),
            max_depth=self.params.get("max_depth", 20),
            min_samples_leaf=self.params.get("min_samples_leaf", 20),
            max_samples=self.params.get("max_samples", None),   # row subsampling (Option B)
            criterion="absolute_error" if self.params.get("mae_split") else "squared_error",
            n_jobs=n_jobs, random_state=self.params.get("random_state", 42),
        )
        self.model.fit(X, y)
        self._cols = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return pd.Series(self.model.feature_importances_, index=self._cols)
