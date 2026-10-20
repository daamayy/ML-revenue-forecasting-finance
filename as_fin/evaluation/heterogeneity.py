"""Q4 heterogeneity analysis: where does ML add the most value? (plan 3.1 Q4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae


def _improvement(g: pd.DataFrame, model_col="model", ml="LightGBM",
                 naive_pred="y_pred_naive", true="y_true", pred="y_pred") -> float:
    """% MAE reduction of ``ml`` vs the per-row seasonal-naive scaling reference.

    The seasonal-naive forecast is zero seasonal-log-growth, carried on every row
    as ``y_pred_naive`` (not a separate competitor model), so the naive MAE is the
    error of ``ml``'s own rows against that zero baseline.
    """
    gm = g[g[model_col] == ml]
    m_ml = mae(gm[true], gm[pred])
    m_nv = mae(gm[true], gm[naive_pred])
    if not (np.isfinite(m_ml) and np.isfinite(m_nv) and m_nv > 0):
        return np.nan
    return 100.0 * (m_nv - m_ml) / m_nv


def improvement_by_group(preds_long: pd.DataFrame, group_col: str, *,
                         horizon: int = 1, **kw) -> pd.DataFrame:
    """ML improvement per category of ``group_col`` (Tables 3 / 6)."""
    sub = preds_long[preds_long["horizon"] == horizon]
    rows = []
    for key, g in sub.groupby(group_col):
        rows.append({group_col: key, "ml_improvement_pct": _improvement(g, **kw),
                     "n": len(g)})
    return pd.DataFrame(rows).sort_values("ml_improvement_pct", ascending=False)


def improvement_vs_baseline(preds_long: pd.DataFrame, group_col: str, *,
                            ml: str = "LightGBM", baseline: str = "naive",
                            horizon: int = 1) -> pd.DataFrame:
    """% MAE reduction of ``ml`` over ``baseline``, per category of ``group_col``.

    * ``baseline="naive"`` — vs the per-row seasonal-naive reference (`y_pred_naive`),
      on ``ml``'s full sample.
    * ``baseline=<model name>`` (e.g. ``"AutoARIMA"``) — vs that competitor model on
      the **matched** (common gvkey-qid) sample, so the comparison is apples-to-apples.

    ``group_col`` must be present on the ``ml`` rows (e.g. ``sector`` or a quintile
    column merged on beforehand). Returns one row per group with improvement_pct + n.
    """
    sub = preds_long[preds_long["horizon"] == horizon]
    ml_df = sub[sub["model"] == ml]
    if baseline == "naive":
        work = ml_df.assign(base=ml_df["y_pred_naive"])
    else:
        base = (sub[sub["model"] == baseline][["gvkey", "qid", "y_pred"]]
                .rename(columns={"y_pred": "base"}))
        work = ml_df.merge(base, on=["gvkey", "qid"], how="inner")
    rows = []
    for key, g in work.dropna(subset=[group_col]).groupby(group_col):
        m_ml = mae(g["y_true"], g["y_pred"])
        m_b = mae(g["y_true"], g["base"])
        imp = 100.0 * (m_b - m_ml) / m_b if (np.isfinite(m_b) and m_b > 0) else np.nan
        rows.append({group_col: key, "improvement_pct": round(imp, 1),
                     "mae_ml": round(m_ml, 4), "mae_baseline": round(m_b, 4),
                     "n": len(g)})
    return pd.DataFrame(rows).sort_values("improvement_pct", ascending=False)


# Firm-characteristic dimensions for the Q4 quintile splits: column -> quintile name.
CHARACTERISTICS = {
    "market_cap": "size_q",          # small -> large
    "yoy_growth": "growth_q",        # decline -> high-growth
    "book_to_market": "bm_q",        # low B/M (glamour) -> high B/M (value)
    "rolling_std_4q": "vol_q",       # low -> high volatility
}


def add_characteristic_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    """Tag each firm-quarter with within-quarter quintiles of firm characteristics.

    Quintiles are formed within each formation quarter to keep them comparable over
    time: size (market_cap), growth (yoy_growth), book-to-market and volatility
    (rolling_std_4q). Analyst-coverage buckets follow the plan's thresholds.
    """
    out = panel.copy()
    # book-to-market = book equity / market equity (low = glamour/growth, high = value).
    if "book_to_market" not in out:
        if {"ceqq", "market_cap"}.issubset(out.columns):
            out["book_to_market"] = out["ceqq"] / out["market_cap"].replace(0, np.nan)
        elif "market_to_book" in out:
            out["book_to_market"] = 1.0 / out["market_to_book"].replace(0, np.nan)

    def q5(s):
        return pd.qcut(s, 5, labels=False, duplicates="drop")

    for col, qname in CHARACTERISTICS.items():
        if col in out:
            out[qname] = out.groupby("qid")[col].transform(q5)
    if "analyst_coverage" in out:
        cov = out["analyst_coverage"]
        out["coverage_bucket"] = np.select(
            [cov.isna(), cov < 3, cov <= 10, cov > 10],
            ["none", "low", "mid", "high"], default="none")
    return out


def regress_improvement(panel_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Pooled OLS: per-firm ML improvement ~ volatility + size + coverage + growth + industry FE.

    ``panel_with_errors`` must hold a per-firm ``ml_improvement`` plus the
    characteristic columns. Returns a coefficient table with HAC-robust SEs.
    """
    import statsmodels.formula.api as smf

    cols = ["ml_improvement", "rolling_std_4q", "log_market_cap",
            "analyst_coverage", "yoy_growth", "sic2"]
    d = panel_with_errors[[c for c in cols if c in panel_with_errors]].dropna()
    formula = "ml_improvement ~ rolling_std_4q + log_market_cap + yoy_growth + C(sic2)"
    if "analyst_coverage" in d:
        formula += " + analyst_coverage"
    res = smf.ols(formula, data=d).fit(cov_type="HC1")
    return res.summary2().tables[1]
