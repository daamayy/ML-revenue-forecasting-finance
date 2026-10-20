"""Tier 3 — Tree-based ML via Nixtla ``mlforecast``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regressor factories. mlforecast accepts any sklearn-API estimator instance,
# so the tree tier is defined as a dict of name -> estimator.
# ---------------------------------------------------------------------------
def make_regressors(which=("lightgbm", "xgboost", "random_forest"),
                    params: dict | None = None, seed: int = 42) -> dict:
    """Build the {model_name: estimator} dict consumed by MLForecast.

    ``params`` optionally maps a model key to a hyper-parameter dict (e.g. the
    one selected by forward validation). Objectives are set to L1 so the trees
    optimize the same MAE loss as the project's primary metric.
    """
    params = params or {}
    out: dict = {}
    if "lightgbm" in which:
        from lightgbm import LGBMRegressor
        p = dict(objective="regression_l1", num_leaves=63, max_depth=7,
                 learning_rate=0.05, n_estimators=500, subsample=0.8,
                 colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=seed)
        p.update(params.get("lightgbm", {}))
        out["LightGBM"] = LGBMRegressor(**p)
    if "xgboost" in which:
        from xgboost import XGBRegressor
        p = dict(objective="reg:absoluteerror", max_depth=5, learning_rate=0.05,
                 n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                 n_jobs=-1, random_state=seed)
        p.update(params.get("xgboost", {}))
        out["XGBoost"] = XGBRegressor(**p)
    if "random_forest" in which:
        from sklearn.ensemble import RandomForestRegressor
        p = dict(n_estimators=500, max_depth=20, min_samples_leaf=20,
                 n_jobs=-1, random_state=seed)
        p.update(params.get("random_forest", {}))
        out["RandomForest"] = RandomForestRegressor(**p)
    return out


def build_mlforecast(regressors: dict, season_length: int = 4):
    """Configure an MLForecast for quarterly revenue-growth trees.

    Lags and rolling transforms are chosen to mirror Set A's revenue-history
    block (lag-1/4/8 growth, 4-quarter rolling mean/std, seasonal rolling mean)
    while remaining origin-safe. ``date_features=['quarter']`` supplies the
    seasonal dummy that Set D lists explicitly.
    """
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean, RollingStd, SeasonalRollingMean

    return MLForecast(
        models=regressors,
        freq="QE",
        lags=[1, 2, 4, 8],                       # g_{t-1}, g_{t-2}, g_{t-4}, g_{t-8}
        lag_transforms={
            1: [RollingMean(window_size=4), RollingStd(window_size=4)],
            4: [SeasonalRollingMean(season_length=season_length, window_size=2)],
        },
        date_features=["quarter"],
    )


def _qid_to_ds(qid):
    """Map the integer quarter index to a canonical quarter-end timestamp.

    qid = fyearq*4 + (fqtr-1) -> (year, quarter) -> calendar quarter-end. This puts
    every firm on a single regular ``QE`` grid regardless of its fiscal calendar,
    so gaps in qid become real gaps for mlforecast and the mapping is invertible.
    """
    qid = np.asarray(qid, dtype="int64")
    year = qid // 4
    month = (qid % 4 + 1) * 3
    return pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)


def _ds_to_qid(ds: pd.Series) -> pd.Series:
    """Inverse of _qid_to_ds: recover the integer quarter index from a QE date."""
    return ds.dt.year * 4 + (ds.dt.quarter - 1)


_UID_SEP = "__"


def panel_to_runs_long(panel: pd.DataFrame, value_col: str, exog: list[str] | None = None,
                       log_value: bool = False) -> pd.DataFrame:
    """Panel -> Nixtla long format with gapless **contiguous-run** series.

    Shared by the mlforecast tree tier (``value_col='seasonal_log_growth'``) and
    the statsforecast baselines (``value_col='saleq', log_value=True``).

    Real Compustat panels have gaps (firms skip quarters) and off-calendar fiscal
    years, but both Nixtla engines need a *gapless* regular ``QE`` grid per series
    and reject null targets. So we (1) build ``ds`` from the integer quarter index
    ``qid`` (canonical, invertible), and (2) split each firm into contiguous-qid
    runs: a new ``unique_id`` (``gvkey__run``) starts at every gap. Series are then
    gapless without inventing data and lags never bridge a gap; ``gvkey`` and
    ``qid`` are recovered downstream by splitting ``unique_id`` and inverting ``ds``.
    """
    cols = ["gvkey", "qid", value_col] + (exog or [])
    out = panel[cols].copy()
    if log_value:
        out = out[out[value_col] > 0]
        out["y"] = np.log(out[value_col])
    else:
        out = out.dropna(subset=[value_col]).rename(columns={value_col: "y"})
    out = (out.dropna(subset=["y"]).sort_values(["gvkey", "qid"])
              .drop_duplicates(["gvkey", "qid"], keep="last"))
    # Run id increments at the first row of each firm and at every qid gap (>1).
    new_run = out.groupby("gvkey")["qid"].diff().ne(1)        # NaN (first) -> True
    out["_run"] = new_run.groupby(out["gvkey"]).cumsum().astype(int)
    out["unique_id"] = out["gvkey"].astype(str) + _UID_SEP + out["_run"].astype(str)
    out["ds"] = _qid_to_ds(out["qid"].values)
    keep = ["unique_id", "ds", "y"] + (exog or [])
    return out[keep].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def to_long(panel: pd.DataFrame, target: str = "seasonal_log_growth",
            exog: list[str] | None = None) -> pd.DataFrame:
    """mlforecast long format for the growth target (see ``panel_to_runs_long``)."""
    return panel_to_runs_long(panel, target, exog=exog, log_value=False)


def tune_hyperparams_once(panel: pd.DataFrame, model_key: str, grid: dict,
                          tuning_quantile: float = 0.7, cfg=None) -> dict:
    """Select hyper-parameters once on a representative rolling window (plan §3.5).

    Full per-window/per-horizon grid search is computationally explosive, so — as
    the plan permits when "computational constraints bind" — we tune **once** on a
    single representative refit window and fix the winner for the whole rolling
    backtest. Tree hyper-parameters are stable across windows, so this is a sound
    first-pass approximation of forward validation.

    Procedure:
      1. Pick the window at ``tuning_quantile`` of the rolling schedule (default
         70% through the sample — late enough to resemble the test era, but still
         chosen on validation data only, never on test).
      2. Build the mlforecast design matrix once (target lags + rolling/seasonal +
         quarter), join back the integer ``qid`` to slice train vs. validation.
      3. For every grid combination, fit the underlying regressor on the training
         window and score validation MAE; return the lowest-MAE parameters.
    """
    from itertools import product
    from ..evaluation.metrics import mae
    from ..pipeline.rolling import generate_windows

    cfg = cfg or load_config()
    windows = generate_windows(cfg)
    w = windows[min(int(len(windows) * tuning_quantile), len(windows) - 1)]
    log.info("Tuning %s on window test_year=%d (val %d..%d)", model_key,
             w.test_year, w.val_qids.start, w.val_qids.stop - 1)

    long = to_long(panel)
    mlf = build_mlforecast(make_regressors(which=(model_key,)))
    design = mlf.preprocess(long, dropna=True)          # lags/rolling features + y
    # Recover qid from the canonical QE grid (no fragile date join needed).
    design["qid"] = _ds_to_qid(design["ds"])
    feat_cols = [c for c in design.columns
                 if c not in ("unique_id", "ds", "y", "qid")]

    tr = design[design["qid"].between(w.train_qids.start, w.train_qids.stop - 1)]
    va = design[design["qid"].between(w.val_qids.start, w.val_qids.stop - 1)]
    if len(tr) < 1000 or len(va) < 200:
        log.warning("Tuning window too small (tr=%d, va=%d); using defaults", len(tr), len(va))
        return {}

    keys = list(grid)
    best_params, best_mae = {}, float("inf")
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    log.info("  grid: %d combinations on %d train / %d val rows", n_combos, len(tr), len(va))
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        reg = make_regressors(which=(model_key,), params={model_key: params})
        model = next(iter(reg.values()))
        model.fit(tr[feat_cols], tr["y"])
        score = mae(va["y"], model.predict(va[feat_cols]))
        if score < best_mae:
            best_mae, best_params = score, params
    log.info("  best %s params: %s (val MAE=%.4f)", model_key, best_params, best_mae)
    return best_params


def run_cross_validation(panel: pd.DataFrame, regressors: dict | None = None,
                         horizons=(1, 2, 3, 4), exog: list[str] | None = None,
                         static_features: list[str] | None = None) -> pd.DataFrame:
    """Rolling-origin OOS backtest of the tree tier via MLForecast.cross_validation.

    Returns tidy predictions (one row per model x firm x cutoff x horizon) in the
    project's schema: ['model','gvkey','qid','horizon','y_true','y_pred', ...].
    Annual refit is achieved with ``step_size=4`` and the Direct multi-step
    strategy with ``max_horizon=max(horizons)``.
    """
    cfg = load_config()
    regressors = regressors or make_regressors()
    H = max(horizons)

    long = to_long(panel, exog=exog)
    mlf = build_mlforecast(regressors)

    # Number of annual refit windows that fit in the post-warmup sample.
    n_quarters = long["ds"].dt.to_period("Q").nunique()
    n_windows = max(1, (n_quarters - cfg.rolling["train_quarters"]) // cfg.rolling["refit_every"])

    log.info("mlforecast CV: %d models, %d windows, max_horizon=%d",
             len(regressors), n_windows, H)
    cv = mlf.cross_validation(
        df=long, n_windows=n_windows, h=H, step_size=cfg.rolling["refit_every"],
        max_horizon=H, refit=True,
        static_features=static_features or [],
    )

    return _tidy_cv(cv, panel, list(regressors), horizons)


def _tidy_cv(cv: pd.DataFrame, panel: pd.DataFrame, model_names: list[str],
             horizons) -> pd.DataFrame:
    """Reshape mlforecast CV output into the project's long prediction schema.

    Derives the integer horizon from (cutoff, ds), maps ds back to the firm's
    ``qid``, attaches the realized future level and the reconstruction base, and
    melts the per-model columns into rows.
    """
    df = cv.copy()
    # Recover gvkey (strip the contiguous-run suffix) and the integer quarter
    # index from the canonical QE grid; horizon = qid distance from cutoff origin.
    df["gvkey"] = df["unique_id"].str.split(_UID_SEP).str[0]
    df["qid"] = _ds_to_qid(df["ds"])
    df["horizon"] = df["qid"] - _ds_to_qid(df["cutoff"])
    df = df[df["horizon"].isin(list(horizons))]

    # Bring the level columns from the panel by (gvkey, qid) — robust to fiscal
    # calendars since both sides use the same integer quarter index.
    ref_cols = ["gvkey", "qid", "saleq", "saleq_lag4", "market_cap", "sic"]
    ref = panel[[c for c in ref_cols if c in panel]].copy()
    ref["gvkey"] = ref["gvkey"].astype(str)
    df = df.merge(ref, on=["gvkey", "qid"], how="left")

    frames = []
    for name in model_names:
        if name not in df:
            continue
        block = pd.DataFrame({
            "model": name,
            "gvkey": df["gvkey"].values,
            "qid": df["qid"].values,
            "horizon": df["horizon"].values,
            "y_true": df["y"].values,
            "y_pred": df[name].values,
            "y_pred_naive": 0.0,                          # seasonal naive = zero growth
            "saleq_future": df["saleq"].values,           # realized level at target
            "saleq_pred": df["saleq_lag4"].values * np.exp(df[name].values),
            "saleq_realbase": df["saleq_lag4"].values,
            "market_cap": df.get("market_cap", np.nan),
            "sic": df.get("sic", np.nan),
        })
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["qid"]).reset_index(drop=True)
