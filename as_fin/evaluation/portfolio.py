"""Q5 portfolio strategy on ML revenue-forecast surprises (plan 3.1 Q5 / 3.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils import get_logger

log = get_logger(__name__)


def attach_forward_returns(preds: pd.DataFrame, crsp: pd.DataFrame, *,
                           date_col: str = "formation_date", permno_col: str = "permno",
                           k_months: int = 3, out_col: str = "fwd_ret") -> pd.DataFrame:
    """Attach the forward k-month compounded CRSP return after the formation date.

    For each permno we precompute, at every month-end t, the compounded return
    over the *next* ``k_months`` (t+1..t+k). Each prediction row is then matched
    by ``merge_asof`` to the last CRSP month at or before its formation date, so
    the holding period is strictly after the information is acted on (no
    look-ahead). With ``k_months=3`` this gives a one-quarter forward return,
    consistent with the quarterly rebalance used in performance_stats.
    """
    c = crsp[[permno_col, "date", "ret"]].dropna(subset=["date"]).copy()
    c["ret"] = c["ret"].astype("float64").fillna(0.0)
    c = c.sort_values([permno_col, "date"])

    # Forward compounded return over the next k months (shifted so it uses t+1..t+k).
    g = c.groupby(permno_col, group_keys=False)
    fwd = pd.Series(1.0, index=c.index)
    for j in range(1, k_months + 1):
        fwd = fwd * (1 + g["ret"].shift(-j))
    c[out_col] = fwd - 1.0

    # merge_asof needs globally sorted keys and matching 'by' dtypes (CRSP permno
    # is a nullable Int64). Cast both sides to plain int64.
    left = preds.dropna(subset=[date_col, permno_col]).sort_values(date_col).copy()
    left[permno_col] = left[permno_col].astype("int64")
    right = c.dropna(subset=[out_col, permno_col]).sort_values("date").copy()
    right[permno_col] = right[permno_col].astype("int64")
    merged = pd.merge_asof(left, right[[permno_col, "date", out_col]],
                           left_on=date_col, right_on="date", by=permno_col,
                           direction="backward")
    log.info("Forward returns: matched %d/%d prediction rows",
             merged[out_col].notna().sum(), len(merged))
    return merged


def load_ff_factors(path) -> pd.DataFrame | None:
    """Load a Fama-French factor file (qid-indexed) if present, else None.

    Expected columns: ``qid`` plus any of {mkt_rf, smb, hml, rmw, cma, rf} in
    decimal monthly/quarterly units. Returns None (alphas skipped) when absent.
    """
    import os
    if not os.path.exists(path):
        log.warning("FF factor file %s not found; CAPM/FF3/FF5 alphas skipped.", path)
        return None
    ff = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return ff.set_index("qid") if "qid" in ff else ff


def assign_surprise_quintiles(df: pd.DataFrame, *, surprise_col="ml_surprise",
                              industry_col="sic3", time_col="qid",
                              n_quintiles: int = 5) -> pd.DataFrame:
    """Assign within-industry, within-quarter quintiles of the ML surprise."""
    out = df.dropna(subset=[surprise_col]).copy()

    def _q(s):
        if s.nunique() < n_quintiles:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s, n_quintiles, labels=False, duplicates="drop")

    out["quintile"] = (out.groupby([industry_col, time_col])[surprise_col]
                          .transform(_q))
    return out.dropna(subset=["quintile"])


def quintile_returns(df: pd.DataFrame, *, ret_col="fwd_ret", time_col="qid",
                     weight_col: str | None = None) -> pd.DataFrame:
    """Average forward return per quintile per period; plus the Q5-Q1 hedge leg.

    ``ret_col`` is the realized post-formation return used to evaluate the sort.
    Equal-weighted by default; pass ``weight_col`` (e.g. market_cap) for VW.
    """
    def _avg(g):
        if weight_col and weight_col in g:
            w = g[weight_col].clip(lower=0)
            return np.average(g[ret_col], weights=w) if w.sum() > 0 else np.nan
        return g[ret_col].mean()

    pr = (df.groupby([time_col, "quintile"]).apply(_avg)
            .rename("ret").reset_index())
    wide = pr.pivot(index=time_col, columns="quintile", values="ret")
    qmax = wide.columns.max()
    wide["hedge"] = wide[qmax] - wide[0]                  # long top, short bottom
    return wide


def performance_stats(returns: pd.Series, periods_per_year: int = 4,
                      factors: pd.DataFrame | None = None) -> dict:
    """Mean return, annualized Sharpe and (optionally) CAPM/FF3/FF5 alphas.

    ``factors`` is a frame aligned to ``returns`` with columns among
    {mkt_rf, smb, hml, rmw, cma, rf}. Alphas are reported per period; annualize
    externally if needed.
    """
    r = returns.dropna()
    stats = {
        "mean_return": float(r.mean()),
        "volatility": float(r.std()),
        "sharpe_annual": float(r.mean() / r.std() * np.sqrt(periods_per_year))
        if r.std() > 0 else np.nan,
        "n_periods": int(len(r)),
    }
    if factors is None:
        return stats

    f = factors.reindex(r.index)
    excess = r - f.get("rf", 0)
    specs = {
        "CAPM_alpha": ["mkt_rf"],
        "FF3_alpha": ["mkt_rf", "smb", "hml"],
        "FF5_alpha": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    }
    for name, cols in specs.items():
        use = [c for c in cols if c in f]
        if not use:
            continue
        X = sm.add_constant(f[use].dropna())
        y = excess.reindex(X.index)
        model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        stats[name] = float(model.params["const"])
        stats[name + "_t"] = float(model.tvalues["const"])
    return stats


def _leg_weights(g: pd.DataFrame, weight_col: str | None, id_col: str) -> pd.Series:
    """Normalized portfolio weights for one leg in one period, indexed by id."""
    if weight_col and weight_col in g and g[weight_col].clip(lower=0).sum() > 0:
        w = g[weight_col].clip(lower=0)
        w = w / w.sum()
    else:
        w = pd.Series(1.0 / len(g), index=g.index)
    # A firm can appear twice in one calendar quarter (e.g. a fiscal-year change);
    # consolidate to one position per id so the index is unique (weights still sum to 1).
    return pd.Series(w.values, index=g[id_col].values).groupby(level=0).sum()


def portfolio_turnover(sorted_df: pd.DataFrame, *, weight_col: str | None = None,
                       time_col: str = "qid", id_col: str = "permno") -> dict:
    """One-sided turnover ½·Σ|w_t − w_{t−1}| and avg # names, per leg (instructor Q4).

    Computed on the long (top quintile) and short (bottom quintile) legs separately,
    treating a name absent in a period as weight 0, then averaged over rebalances.
    Turnover is what converts gross alpha into net-of-cost alpha, so it bounds how
    tradable the hedge is (see plan §3.1 Q5 construction details).
    """
    q = sorted_df["quintile"]
    qmax = q.max()
    out = {}
    for leg_name, leg in (("long", sorted_df[q == qmax]), ("short", sorted_df[q == 0])):
        periods = sorted(leg[time_col].unique())
        wmaps = {t: _leg_weights(leg[leg[time_col] == t], weight_col, id_col)
                 for t in periods}
        turns, ncounts = [], []
        prev = None
        for t in periods:
            wt = wmaps[t]
            ncounts.append(len(wt))
            if prev is not None:
                ids = prev.index.union(wt.index)
                a = wt.reindex(ids, fill_value=0.0)
                b = prev.reindex(ids, fill_value=0.0)
                turns.append(0.5 * np.abs(a - b).sum())
            prev = wt
        out[f"{leg_name}_turnover"] = float(np.mean(turns)) if turns else np.nan
        out[f"{leg_name}_avg_names"] = float(np.mean(ncounts)) if ncounts else np.nan
    out["avg_turnover"] = np.nanmean([out["long_turnover"], out["short_turnover"]])
    return out


def vw_leg_diagnostics(sorted_df: pd.DataFrame, *, weight_col: str = "market_cap",
                       time_col: str = "qid", id_col: str = "permno") -> pd.DataFrame:
    """Per-period name count + weight concentration (HHI, effective N) for each leg.

    HHI = Σ wᵢ²; effective number of names = 1/HHI. Under value weighting a leg
    dominated by a few mega-caps has low effective N — the diagnostic behind the
    VW hedge's concentrated drawdowns (instructor's 2000 example).
    """
    q = sorted_df["quintile"]
    qmax = q.max()
    rows = []
    for t, g in sorted_df.groupby(time_col):
        rec = {time_col: t}
        for leg, sub in (("long", g[g["quintile"] == qmax]), ("short", g[g["quintile"] == 0])):
            if len(sub) == 0:
                rec[f"n_{leg}"], rec[f"hhi_{leg}"], rec[f"eff_n_{leg}"] = 0, np.nan, np.nan
                continue
            w = _leg_weights(sub, weight_col, id_col)
            hhi = float((w ** 2).sum())
            rec[f"n_{leg}"] = int(len(w))          # unique names in the leg
            rec[f"hhi_{leg}"] = hhi
            rec[f"eff_n_{leg}"] = (1.0 / hhi) if hhi > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)


def hedge_contributions(sorted_df: pd.DataFrame, *, weight_col: str | None = None,
                        time_col: str = "qid", id_col: str = "permno",
                        ret_col: str = "fwd_ret", periods=None) -> pd.DataFrame:
    """Per-name signed contribution to the hedge return, summed over ``periods``.

    Contribution of name i in period t = ±wᵢ,ₜ·retᵢ,ₜ (+ for the long/top leg, − for
    the short/bottom leg). Summed over the requested periods, the most negative names
    are the drivers of a drawdown (used to attribute the VW hedge's 2000–01 fall).
    """
    q = sorted_df["quintile"]
    qmax = q.max()
    d = sorted_df if periods is None else sorted_df[sorted_df[time_col].isin(list(periods))]
    recs = []
    for t, g in d.groupby(time_col):
        for sign, sub in ((+1, g[g["quintile"] == qmax]), (-1, g[g["quintile"] == 0])):
            if len(sub) == 0:
                continue
            w = _leg_weights(sub, weight_col, id_col)          # unique-id index
            ret = sub.groupby(id_col)[ret_col].mean()          # unique-id index
            contrib = sign * w * ret.reindex(w.index)
            for pid, c in contrib.items():
                recs.append({id_col: pid, time_col: t, "leg": "long" if sign > 0 else "short",
                             "contribution": float(c) if np.isfinite(c) else np.nan})
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    agg = (out.groupby(id_col)["contribution"].sum()
              .sort_values().rename("total_contribution").reset_index())
    return agg


def ml_surprise_portfolio(df: pd.DataFrame, *, factors: pd.DataFrame | None = None,
                          weight_col: str | None = None,
                          time_col: str = "qid") -> dict:
    """End-to-end Q5 pipeline: sort -> quintile returns -> hedge performance.

    ``time_col`` is the rebalance period for both the within-period sort and the
    return series (e.g. a calendar-quarter index for clean factor alignment).
    Returns {'quintile_returns': DataFrame, 'stats': {...}} (Table 8 / Figure 4).
    """
    sorted_df = assign_surprise_quintiles(df, time_col=time_col)
    qret = quintile_returns(sorted_df, time_col=time_col, weight_col=weight_col)
    stats = performance_stats(qret["hedge"], factors=factors)
    log.info("Hedge Sharpe (annual): %.2f over %d periods",
             stats.get("sharpe_annual", np.nan), stats.get("n_periods", 0))
    return {"quintile_returns": qret, "stats": stats}
