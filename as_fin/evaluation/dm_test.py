"""Diebold-Mariano / Newey-West tests on quarterly loss differences (plan 3.5)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def _loss(err: np.ndarray, kind: str) -> np.ndarray:
    return np.abs(err) if kind == "abs" else err ** 2


def diebold_mariano(df_a: pd.DataFrame, df_b: pd.DataFrame, *,
                    time_col="qid", true_col="y_true", pred_col="y_pred",
                    loss="abs", lag: int | None = None) -> dict:
    """DM test comparing model A vs B on a shared panel.

    Each frame holds columns [time_col, true_col, pred_col] on the same index of
    firm-quarters. Returns mean loss differential, HAC t-stat and p-value.
    """
    a = df_a[[time_col, true_col, pred_col]].rename(columns={pred_col: "pa"})
    b = df_b[[time_col, pred_col]].rename(columns={pred_col: "pb"})
    m = a.join(b.set_index(b.index)["pb"]) if a.index.equals(df_b.index) else \
        a.reset_index(drop=True).assign(pb=b["pb"].reset_index(drop=True))

    m["la"] = _loss((m[true_col] - m["pa"]).values, loss)
    m["lb"] = _loss((m[true_col] - m["pb"]).values, loss)
    m["d"] = m["la"] - m["lb"]
    dt = m.groupby(time_col)["d"].mean().dropna()        # quarterly mean diff
    if len(dt) < 3:
        return dict(mean_diff=np.nan, t_stat=np.nan, p_value=np.nan, n_quarters=len(dt))

    d = dt.values
    T = len(d)
    dbar = d.mean()
    # Newey-West long-run variance with automatic lag if not specified.
    L = lag if lag is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * w * cov
    se = np.sqrt(var / T)
    t_stat = dbar / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1)) if np.isfinite(t_stat) else np.nan
    return dict(mean_diff=float(dbar), t_stat=float(t_stat), p_value=float(p),
                n_quarters=T, lag=L)


def dm_pairwise_table(preds_long: pd.DataFrame, *, horizon: int = 1,
                      time_col="qid", loss="abs") -> pd.DataFrame:
    """All-pairs DM table for a given horizon (Table 2).

    ``preds_long`` columns: ['model','gvkey','qid','horizon','y_true','y_pred'].
    Returns a tidy frame of (model_a, model_b, mean_diff, t_stat, p_value).
    """
    sub = preds_long[preds_long["horizon"] == horizon]
    models = sorted(sub["model"].unique())
    frames = {m: sub[sub["model"] == m].set_index(["gvkey", "qid"]) for m in models}

    rows = []
    for a, b in itertools.combinations(models, 2):
        idx = frames[a].index.intersection(frames[b].index)
        fa = frames[a].loc[idx].reset_index()
        fb = frames[b].loc[idx].reset_index()
        res = diebold_mariano(fa, fb, time_col=time_col, loss=loss)
        rows.append(dict(model_a=a, model_b=b, **res))
    return pd.DataFrame(rows)
