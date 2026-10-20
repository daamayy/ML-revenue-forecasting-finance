"""Q2 ablation study and variable importance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger
from .metrics import mae

log = get_logger(__name__)

# Cumulative specification -> blocks included.
CUMULATIVE_SPECS = {
    "M0_A": ["A"],
    "M1_AB": ["A", "B"],
    "M2_ABC": ["A", "B", "C"],
    "M3_ABCD": ["A", "B", "C", "D"],
    "M4_ABCDE": ["A", "B", "C", "D", "E"],
}
# Leave-one-block-out on the full set.
LOBO_SPECS = {
    "Full": ["A", "B", "C", "D", "E"],
    "Full_minus_B": ["A", "C", "D", "E"],
    "Full_minus_C": ["A", "B", "D", "E"],
    "Full_minus_D": ["A", "B", "C", "E"],
    "Full_minus_E": ["A", "B", "C", "D"],
}


def cumulative_improvement(spec_errors: dict[str, float]) -> pd.DataFrame:
    """Turn {spec_name -> MAE} into the cumulative %-improvement table (Table 4).

    %Improvement_k = (Error_{k-1} - Error_k) / Error_{k-1}.
    """
    order = list(CUMULATIVE_SPECS)
    rows = []
    prev = None
    for name in order:
        if name not in spec_errors:
            continue
        err = spec_errors[name]
        imp = np.nan if prev is None else 100.0 * (prev - err) / prev
        rows.append({"spec": name, "MAE": err, "pct_improvement": imp})
        prev = err
    return pd.DataFrame(rows)


def lobo_contribution(spec_errors: dict[str, float]) -> pd.DataFrame:
    """Leave-one-block-out: each block's conditional contribution vs Full."""
    full = spec_errors.get("Full")
    rows = []
    for name in LOBO_SPECS:
        if name == "Full" or name not in spec_errors:
            continue
        err = spec_errors[name]
        # Removing an informative block should *raise* error: contribution > 0.
        contrib = 100.0 * (err - full) / full if full else np.nan
        rows.append({"spec": name, "MAE": err, "contribution_pct": contrib})
    return pd.DataFrame(rows)


def permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 5, random_state: int = 42) -> pd.Series:
    """Out-of-sample permutation importance: rise in MAE after shuffling a feature.

    Model-agnostic; uses the fitted model's ``predict``. Reported alongside SHAP
    for Figure 1 (plan 3.5).
    """
    rng = np.random.default_rng(random_state)
    base = mae(y, model.predict(X))
    importances = {}
    for col in X.columns:
        scores = []
        saved = X[col].copy()
        for _ in range(n_repeats):
            X[col] = rng.permutation(saved.values)
            scores.append(mae(y, model.predict(X)) - base)
        X[col] = saved
        importances[col] = float(np.mean(scores))
    return pd.Series(importances).sort_values(ascending=False)


def shap_analysis(model, X: pd.DataFrame, max_rows: int = 5000):
    """Full SHAP result for a fitted tree model, for both Figure 1a and the beeswarm.

    Returns ``(mean_abs, shap_df, x_sample)``:
      * ``mean_abs`` — mean |SHAP| per feature (global magnitude, sorted);
      * ``shap_df``  — the raw per-row SHAP matrix (rows x features);
      * ``x_sample`` — the matching feature VALUES (same rows), needed to colour
                       the beeswarm by high/low feature value.
    The two frames are row-aligned so the beeswarm can be redrawn from cache.
    """
    import shap

    sample = X.sample(min(len(X), max_rows), random_state=0).reset_index(drop=True)
    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(sample)
    shap_df = pd.DataFrame(values, columns=list(X.columns))
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    return mean_abs, shap_df, sample


def shap_importance(model, X: pd.DataFrame, max_rows: int = 5000) -> pd.Series:
    """Mean |SHAP| per feature for a fitted tree model (thin wrapper)."""
    return shap_analysis(model, X, max_rows)[0]
