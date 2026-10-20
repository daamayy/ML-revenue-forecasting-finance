"""Tier 2 — Elastic Net linear ML benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TabularModel, register


@register("elastic_net")
class ElasticNetModel(TabularModel):
    name = "ElasticNet"
    needs_scaling = True              # standardize features (preprocess.standardize)

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.linear_model import ElasticNet

        self.model = ElasticNet(
            alpha=self.params.get("alpha", 0.01),
            l1_ratio=self.params.get("l1_ratio", 0.5),
            max_iter=self.params.get("max_iter", 5000),
            random_state=self.params.get("random_state", 42),
        )
        self.model.fit(X.values, y.values)
        self._cols = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X.values)

    def feature_importance(self):
        # Absolute standardized coefficients as importance.
        return pd.Series(np.abs(self.model.coef_), index=self._cols)
