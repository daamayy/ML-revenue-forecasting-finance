"""Q1 / Q5 analyst benchmark — *formation-aligned* ML vs analyst comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import load_ibes_sal
from ..features import build_merged_panel, build_modeling_panel
from ..evaluation.metrics import mae, smape_level
from ..utils import get_logger

log = get_logger(__name__)


def _cqtr_int(period: pd.Series) -> pd.Series:
    """Map a quarterly Period series to a sortable/join-friendly integer key."""
    return period.dt.year * 4 + (period.dt.quarter - 1)


def _load_panel_keys(cfg) -> pd.DataFrame:
    """Panel (gvkey, qid) -> formation date, 8-digit CUSIP, market cap, sic."""
    raw = cfg.path("data_raw")
    base = build_merged_panel(
        pd.read_parquet(raw / "compustat_fundq.parquet"),
        pd.read_parquet(raw / "crsp_msf.parquet"),
        pd.read_parquet(raw / "ccm_link.parquet"),
        save=False,
    )
    macro = pd.read_parquet(raw / "fred_macro.parquet")
    panel = build_modeling_panel(base, macro_raw=macro, with_analyst=True, save=False)
    keys = panel[["gvkey", "qid", "datadate", "avail_date", "cusip",
                  "market_cap", "sic"]].copy()
    keys["cusip8"] = keys["cusip"].astype(str).str.zfill(9).str[:8]
    return keys


def _ibes_snapshots(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All IBES revenue snapshots + one realized actual per (cusip8, calendar qtr)."""
    ib = load_ibes_sal()
    ib["cusip8"] = ib["cusip"].astype(str).str.zfill(8).str[:8]
    ib["target_cq"] = _cqtr_int(ib["fpedats"].dt.to_period("Q"))
    snaps = (ib.dropna(subset=["medest", "statpers"])
               [["cusip8", "target_cq", "statpers", "medest"]]
               .sort_values("statpers")
               .reset_index(drop=True))
    actuals = (ib.dropna(subset=["actual"])
                 .sort_values("statpers")
                 .drop_duplicates(["cusip8", "target_cq"], keep="last")
                 [["cusip8", "target_cq", "actual"]]
                 .reset_index(drop=True))
    return snaps, actuals


def build_formation_aligned_analyst(cfg, ml_preds: pd.DataFrame,
                                    horizon: int = 1, *, keys=None,
                                    snaps=None, actuals=None) -> pd.DataFrame:
    """Analyst-consensus benchmark rows aligned to the ML formation date.

    For each ML row at (gvkey, formation qid ``s``) the analyst forecast is the
    latest IBES median revenue estimate for target quarter ``s+h`` with
    ``statpers <= avail_date(s)``. The cutoff is the availability date (``rdq`` or
    quarter-end+60d) — the same instant the ML model's quarter-``s`` features
    become public — so **both forecasters stand at the identical information
    cutoff** (removes the ~45-day datadate-vs-rdq asymmetry that previously flattered
    ML). The target quarter ``s+h`` is still defined by the fiscal-quarter calendar
    (``datadate``). Returns tidy rows (model='AnalystConsensus') in the same
    growth-space schema as the ML predictions, scored against IBES actuals with an
    IBES seasonal base.

    ``keys`` / ``snaps`` / ``actuals`` may be pre-built and passed in to avoid
    rebuilding the panel and reloading IBES once per horizon (see
    :func:`ml_vs_analyst_horizons`).
    """
    if keys is None:
        keys = _load_panel_keys(cfg)
    if snaps is None or actuals is None:
        snaps, actuals = _ibes_snapshots(cfg)

    # Formation rows: one per (gvkey, s) with the ML horizon-h target present.
    ml = ml_preds[ml_preds["horizon"] == horizon][["gvkey", "qid"]].drop_duplicates()
    ml = ml.merge(keys[["gvkey", "qid", "datadate", "avail_date", "cusip8"]],
                  on=["gvkey", "qid"], how="inner").dropna(
                      subset=["datadate", "avail_date", "cusip8"])
    fq = ml["datadate"].dt.to_period("Q")            # target quarter = fiscal calendar
    ml["target_cq"] = _cqtr_int(fq) + horizon       # target quarter s+h
    ml["base_cq"] = ml["target_cq"] - 4             # IBES seasonal base s+h-4
    ml["cutoff"] = ml["avail_date"]                  # info cutoff = when s-features are public

    # As-of join: latest analyst snapshot with statpers <= formation cutoff.
    left = ml.sort_values("cutoff").reset_index(drop=True)
    asof = pd.merge_asof(left, snaps.rename(columns={"statpers": "snap_date"}),
                         left_on="cutoff", right_on="snap_date",
                         by=["cusip8", "target_cq"], direction="backward")

    # Attach IBES realized actual for target and its seasonal base (s+h-4).
    asof = asof.merge(actuals.rename(columns={"actual": "ibes_actual"}),
                      on=["cusip8", "target_cq"], how="left")
    asof = asof.merge(
        actuals.rename(columns={"target_cq": "base_cq", "actual": "ibes_base"}),
        on=["cusip8", "base_cq"], how="left")

    ok = (asof["medest"].gt(0) & asof["ibes_actual"].gt(0) & asof["ibes_base"].gt(0)
          & np.isfinite(asof["medest"]) & np.isfinite(asof["ibes_actual"])
          & np.isfinite(asof["ibes_base"]))
    a = asof[ok].copy()
    if a.empty:
        log.warning("Formation-aligned analyst: no matched rows for h=%d", horizon)
        return pd.DataFrame()

    a = a.merge(keys[["gvkey", "qid", "market_cap", "sic"]].drop_duplicates(["gvkey", "qid"]),
                on=["gvkey", "qid"], how="left")
    out = pd.DataFrame({
        "model": "AnalystConsensus", "gvkey": a["gvkey"].values,
        "qid": a["qid"].values, "horizon": horizon,
        # Source-specific seasonal growth: base cancels in (y_true - y_pred).
        "y_true": np.log(a["ibes_actual"].values / a["ibes_base"].values),
        "y_pred": np.log(a["medest"].values / a["ibes_base"].values),
        "y_pred_naive": 0.0,
        "saleq_future": a["ibes_actual"].values,
        "saleq_pred": a["medest"].values,
        "saleq_realbase": a["ibes_base"].values,
        "market_cap": a["market_cap"].values, "sic": a["sic"].values,
    })
    log.info("Formation-aligned analyst: %d matched rows (h=%d)", len(out), horizon)
    return out


def _abs_log_error(df: pd.DataFrame) -> pd.Series:
    return (df["y_true"] - df["y_pred"]).abs()


def ml_vs_analyst(cfg, ml_preds: pd.DataFrame, ml_models=("LightGBM", "ElasticNet"),
                  horizon: int = 1, *, keys=None, snaps=None,
                  actuals=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired ML-vs-analyst accuracy + Diebold-Mariano on the *same* firm-quarters.

    Returns ``(accuracy_table, dm_table)``. Accuracy uses MAE / sMAPE only — the
    only cross-source-comparable metrics. DM aggregates the per-obs absolute-log-
    error differential to target-quarter means and tests it with a HAC (Newey-West)
    standard error.
    """
    analyst = build_formation_aligned_analyst(cfg, ml_preds, horizon=horizon,
                                              keys=keys, snaps=snaps, actuals=actuals)
    if analyst.empty:
        return pd.DataFrame(), pd.DataFrame()

    ml = ml_preds[(ml_preds["horizon"] == horizon)
                  & (ml_preds["model"].isin(ml_models))].copy()
    # Restrict everything to firm-quarters the analyst also covers (paired set).
    pair_keys = set(map(tuple, analyst[["gvkey", "qid"]].values))
    ml = ml[ml.set_index(["gvkey", "qid"]).index.isin(pair_keys)]

    rows, dm_rows = [], []
    for name, g in list(ml.groupby("model")) + [("AnalystConsensus", analyst)]:
        rows.append({"model": name, "horizon": horizon, "n": len(g),
                     "MAE": mae(g["y_true"], g["y_pred"]),
                     "sMAPE": smape_level(g["saleq_future"], g["saleq_pred"]),
                     "actual_source": ("IBES actual SAL" if name == "AnalystConsensus"
                                       else "Compustat SALEQ")})
    acc = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    # Paired DM: analyst vs each ML model on identical (gvkey, qid).
    a_err = analyst.assign(ae=_abs_log_error(analyst))[["gvkey", "qid", "ae"]]
    a_err = a_err.rename(columns={"ae": "ae_analyst"})
    for name, g in ml.groupby("model"):
        merged = (g.assign(ae_ml=_abs_log_error(g))[["gvkey", "qid", "ae_ml"]]
                    .merge(a_err, on=["gvkey", "qid"], how="inner"))
        if merged.empty:
            continue
        merged["diff"] = merged["ae_ml"] - merged["ae_analyst"]   # >0 => analyst better
        d_t = merged.groupby("qid")["diff"].mean()                 # per-quarter loss diff
        t, p = _hac_tstat(d_t.values, lags=4)
        dm_rows.append({"horizon": horizon, "model_a": name,
                        "model_b": "AnalystConsensus",
                        # Quarter-equal-weighted mean, matching what t_stat/p_value
                        # actually test (see dm_test.py's dbar) -- NOT the pooled,
                        # observation-weighted mean, which can disagree in sign when
                        # per-quarter sample sizes are unbalanced.
                        "mean_abs_err_diff": float(d_t.mean()),
                        "t_stat": t, "p_value": p, "n_pairs": len(merged),
                        "n_periods": int(d_t.size)})
    dm = pd.DataFrame(dm_rows)
    return acc, dm


def ml_vs_analyst_horizons(cfg, ml_preds: pd.DataFrame,
                           ml_models=("LightGBM", "ElasticNet"),
                           horizons=(1, 2, 3, 4)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`ml_vs_analyst` for several horizons and stack the results.

    Only horizons actually present in ``ml_preds`` are evaluated, so this is safe
    whether the IBES subsample was run at h=1 only or h=1..4.
    """
    have = set(ml_preds["horizon"].unique())
    # Build the panel keys + IBES snapshots once and reuse across horizons.
    keys = _load_panel_keys(cfg)
    snaps, actuals = _ibes_snapshots(cfg)
    accs, dms = [], []
    for h in horizons:
        if h not in have:
            continue
        acc, dm = ml_vs_analyst(cfg, ml_preds, ml_models=ml_models, horizon=h,
                                keys=keys, snaps=snaps, actuals=actuals)
        if len(acc):
            accs.append(acc)
        if len(dm):
            dms.append(dm)
    acc = pd.concat(accs, ignore_index=True) if accs else pd.DataFrame()
    dm = pd.concat(dms, ignore_index=True) if dms else pd.DataFrame()
    return acc, dm


def _hac_tstat(x: np.ndarray, lags: int = 2) -> tuple[float, float]:
    """Newey-West t-stat that the mean of ``x`` is zero (and two-sided p-value)."""
    from scipy import stats

    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan, np.nan
    xc = x - x.mean()
    gamma0 = float(np.mean(xc * xc))
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = float(np.mean(xc[k:] * xc[:-k]))
        var += 2 * w * cov
    se = np.sqrt(var / n)
    if se <= 0:
        return np.nan, np.nan
    t = float(x.mean() / se)
    p = float(2 * (1 - stats.norm.cdf(abs(t))))
    return t, p
