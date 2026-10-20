"""Tier 5 — Simple-average ensembles."""

from __future__ import annotations

import pandas as pd

# Ensemble definitions: name -> component model names (must match the ``model``
# column produced by the runners: LightGBM/XGBoost/RandomForest/LSTM/TFT — NOT the
# lowercase registry keys, which silently matched nothing). ML_Ensemble (trees
# averaged) parallels Cao & You's main model (RF+GBR+ANN average).
ENSEMBLES = {
    "ML_Ensemble": ["LightGBM", "XGBoost", "RandomForest"],
    "DL_Ensemble": ["LSTM", "TFT"],
    "Full_Ensemble": ["LightGBM", "XGBoost", "RandomForest", "LSTM", "TFT"],
}


class SimpleAverageEnsemble:
    """Average predictions from several models given a long-format frame.

    Expects a tidy frame with columns ['gvkey', 'qid', 'horizon', 'model', 'y_pred'].
    """

    def __init__(self, components: list[str], name: str = "Ensemble"):
        self.components = components
        self.name = name

    def combine(self, preds_long: pd.DataFrame) -> pd.DataFrame:
        sub = preds_long[preds_long["model"].isin(self.components)]
        keys = ["gvkey", "qid", "horizon"]
        agg = sub.groupby(keys, as_index=False)["y_pred"].mean()
        # Only keep cells where every component produced a prediction.
        counts = sub.groupby(keys)["model"].nunique()
        full = counts[counts == len(self.components)].index
        agg = agg.set_index(keys).loc[agg.set_index(keys).index.isin(full)].reset_index()
        agg["model"] = self.name
        return agg


def build_all_ensembles(preds_long: pd.DataFrame,
                        available: set[str] | None = None) -> pd.DataFrame:
    """Build every ensemble whose components are all present in ``preds_long``."""
    have = available or set(preds_long["model"].unique())
    out = []
    for name, comps in ENSEMBLES.items():
        if set(comps).issubset(have):
            out.append(SimpleAverageEnsemble(comps, name).combine(preds_long))
    return pd.concat(out, ignore_index=True) if out else preds_long.iloc[0:0]
