"""Forecast-accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_pred):
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m], m


def mae(y_true, y_pred) -> float:
    yt, yp, _ = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else np.nan


def rmse(y_true, y_pred) -> float:
    yt, yp, _ = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else np.nan


def mase(y_true, y_pred, y_pred_naive) -> float:
    """MASE = MAE(model) / MAE(seasonal naive). M5 standard, scale-free."""
    num = mae(y_true, y_pred)
    den = mae(y_true, y_pred_naive)
    return float(num / den) if den and np.isfinite(den) and den > 0 else np.nan


def smape_level(saleq_true, saleq_pred) -> float:
    """Symmetric MAPE on reconstructed revenue levels (percent, 0-200)."""
    yt, yp, _ = _clean(saleq_true, saleq_pred)
    denom = np.abs(yt) + np.abs(yp)
    ok = denom > 0
    return float(np.mean(2 * np.abs(yt[ok] - yp[ok]) / denom[ok]) * 100) if ok.any() else np.nan


def scaled_level_mae(saleq_true, saleq_pred, saleq_lag4) -> float:
    """Mean |SALEQ - SALEQ_hat| / SALEQ_{t-4}: analyst-comparable level error."""
    yt = np.asarray(saleq_true, "float64")
    yp = np.asarray(saleq_pred, "float64")
    base = np.asarray(saleq_lag4, "float64")
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(base) & (base > 0)
    return float(np.mean(np.abs(yt[m] - yp[m]) / base[m])) if m.any() else np.nan


def matched_accuracy_table(preds: pd.DataFrame, models, horizons=(1,),
                           key=("gvkey", "qid")) -> pd.DataFrame:
    """MAE + MASE per model on the **common** sample present in ALL ``models``.

    For each horizon, restrict to firm-quarters (``key``) that every model in
    ``models`` forecasts, then score each model on that shared set — the fair,
    apples-to-apples ranking used for Q1/Q3 (statistical baselines cover far fewer
    obs than the trees, so the raw table is not comparable). Deep models can carry
    duplicate key rows, so we de-duplicate per model first. Reproduces
    ``table1b_matched_sample_*`` and ``table1c_dl_subsample_*``.
    """
    key = list(key)
    rows = []
    for h in horizons:
        sub = preds[(preds["horizon"] == h) & (preds["model"].isin(models))]
        sub = sub.drop_duplicates(key + ["model"])
        cnt = sub.groupby(key)["model"].nunique()
        common = cnt[cnt == len(set(models))].index
        if len(common) == 0:
            continue
        sub = sub.set_index(key).loc[common].reset_index()
        for m, g in sub.groupby("model"):
            rows.append({"horizon": h, "model": m, "n_matched": len(g),
                         "MAE": mae(g["y_true"], g["y_pred"]),
                         "MASE": mase(g["y_true"], g["y_pred"], g["y_pred_naive"])})
    out = pd.DataFrame(rows)
    return out.sort_values(["horizon", "MASE"]).reset_index(drop=True) if len(out) else out


def accuracy_table(df: pd.DataFrame, *, growth_true="y_true", growth_pred="y_pred",
                   naive_pred="y_pred_naive", level_true="saleq_future",
                   level_pred="saleq_pred", level_base="saleq_realbase",
                   weight=None, group=("model", "horizon")) -> pd.DataFrame:
    """Compute all accuracy metrics per (model, horizon).

    ``weight`` optionally names a column (e.g. market cap) for the value-weighted
    variants; when given, both EW and VW MAE columns are returned.
    """
    rows = []
    for keys, g in df.groupby(list(group)):
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec["n"] = len(g)
        rec["MAE"] = mae(g[growth_true], g[growth_pred])
        rec["RMSE"] = rmse(g[growth_true], g[growth_pred])
        if naive_pred in g:
            rec["MASE"] = mase(g[growth_true], g[growth_pred], g[naive_pred])
        if level_true in g and level_pred in g:
            rec["sMAPE"] = smape_level(g[level_true], g[level_pred])
            rec["scaled_level_MAE"] = scaled_level_mae(
                g[level_true], g[level_pred], g[level_base])
        if weight and weight in g:
            w = g[weight].clip(lower=0).fillna(0)
            err = (g[growth_true] - g[growth_pred]).abs()
            m = np.isfinite(err) & (w > 0)
            rec["MAE_vw"] = float(np.average(err[m], weights=w[m])) if m.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(list(group)).reset_index(drop=True)
