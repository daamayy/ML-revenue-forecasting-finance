"""Drive a model across all rolling windows and horizons, returning tidy predictions."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger, reconstruct_level
from ..features.preprocess import Preprocessor
from ..models.base import build_model
from .rolling import generate_windows

log = get_logger(__name__)


def _select_hyperparams(model_key, grid, X_tr, y_tr, X_val, y_val, needs_scaling):
    """Forward-validation grid search: pick params with lowest validation MAE.

    Iterates the Cartesian product of ``grid``. For large grids this is the slow
    step; the plan allows a reduced grid as long as the selection rule is fixed.
    """
    from ..evaluation.metrics import mae

    keys = list(grid)
    best_params, best_mae = None, np.inf
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        m = build_model(model_key, **params)
        m.fit(X_tr, y_tr, X_val, y_val)
        score = mae(y_val, m.predict(X_val))
        if score < best_mae:
            best_mae, best_params = score, params
    log.info("  best %s params: %s (val MAE=%.4f)", model_key, best_params, best_mae)
    return best_params


def _tune_once(panel, model_key, grid, feature_cols, cfg, needs_scaling,
               tuning_quantile=0.7, horizon=1):
    """Select hyper-parameters once on a representative rolling window (plan §3.5).

    Per-window grid search over the full ~100-feature design is explosive, so —
    as the plan permits when compute binds — we tune once on a window ~70% through
    the sample (validation data only, never test) and reuse the winner for the
    whole backtest. Mirrors the tree-tier ``mlf.tune_hyperparams_once`` policy.
    """
    windows = generate_windows(cfg)
    w = windows[min(int(len(windows) * tuning_quantile), len(windows) - 1)]
    target = f"target_h{horizon}"
    cols = list(dict.fromkeys(feature_cols + [target, "qid"]))   # dedupe
    d = panel[[c for c in cols if c in panel]].copy()
    # Horizon-aware trim: a training pair (formation t, target t+h) is only usable
    # if its target is realized by the refit origin s, i.e. t <= s - h. Without it,
    # the last (h-1) validation quarters carry targets that postdate the origin
    # (a mild multi-step look-ahead). See the project notes(11).
    usable = d["qid"] <= (w.test_qids.start - horizon)
    tr = d[w.mask(d["qid"], "train") & usable].dropna(subset=[target])
    va = d[w.mask(d["qid"], "val") & usable].dropna(subset=[target])
    if len(tr) < 500 or len(va) < 100:
        return {}
    pre = Preprocessor(feature_cols, target, standardize=needs_scaling,
                       winsor_lower=cfg.sample["winsor_lower"],
                       winsor_upper=cfg.sample["winsor_upper"]).fit(tr)
    trp, vap = pre.transform(tr), pre.transform(va)
    log.info("Tuning %s once on window test_year=%d (%d combos)", model_key,
             w.test_year, int(np.prod([len(v) for v in grid.values()])))
    return _select_hyperparams(model_key, grid, trp[feature_cols], trp[target],
                               vap[feature_cols], vap[target], needs_scaling) or {}


def run_tabular_model(panel: pd.DataFrame, model_key: str, feature_cols: list[str],
                      horizons=(1,), grid: dict | None = None,
                      model_name: str | None = None, tune: str = "once",
                      min_train: int = 500) -> pd.DataFrame:
    """Run one tabular model over all rolling windows / horizons -> tidy predictions.

    ``tune`` controls hyper-parameter selection when ``grid`` is given:
      * ``"once"`` (default) — tune once on a representative window, reuse (fast).
      * ``"per_window"`` — full forward validation each window (rigorous, slow).
      * ``"none"`` / empty grid — use model defaults.
    ``min_train`` is the minimum training rows a window needs to be used (lower it
    for small smoke tests).
    """
    cfg = load_config()
    windows = generate_windows(cfg)
    grid = grid or {}
    model_name = model_name or build_model(model_key).name
    needs_scaling = build_model(model_key).needs_scaling

    # Tune-once: fix hyper-parameters before the rolling loop.
    fixed_params: dict = {}
    if grid and tune == "once":
        fixed_params = _tune_once(panel, model_key, grid, feature_cols, cfg, needs_scaling)

    records = []
    for w in windows:
        for h in horizons:
            target = f"target_h{h}"
            base_col = f"saleq_realbase_h{h}"
            future_col = f"saleq_future_h{h}"
            cols = list(dict.fromkeys(feature_cols + [target, base_col, future_col,
                                      "gvkey", "qid", "market_cap", "sic"]))   # dedupe
            d = panel[[c for c in cols if c in panel]].copy()

            # Horizon-aware trim (look-ahead fix, the project notes(11)):
            # keep only formation quarters whose horizon-h target is realized by the
            # refit origin s, i.e. qid <= s - h. Drops the last (h-1) validation
            # quarters whose targets would otherwise postdate the origin. No effect
            # at h=1; removes the mild multi-step leak at h>=2.
            usable = d["qid"] <= (w.test_qids.start - h)
            tr = d[w.mask(d["qid"], "train") & usable].dropna(subset=[target])
            va = d[w.mask(d["qid"], "val") & usable].dropna(subset=[target])
            te = d[w.mask(d["qid"], "test")].dropna(subset=[target])
            if len(tr) < min_train or len(te) == 0:
                continue

            pre = Preprocessor(feature_cols, target, standardize=needs_scaling,
                               winsor_lower=cfg.sample["winsor_lower"],
                               winsor_upper=cfg.sample["winsor_upper"]).fit(tr)
            trp, vap, tep = (pre.transform(x) for x in (tr, va, te))

            # Hyper-parameter selection (see ``tune``).
            if grid and tune == "per_window":
                params = _select_hyperparams(model_key, grid, trp[feature_cols],
                                             trp[target], vap[feature_cols],
                                             vap[target], needs_scaling)
            else:
                params = fixed_params

            # Refit on train+val with the chosen params, then predict the test block.
            fit_df = pd.concat([trp, vap])
            final = build_model(model_key, **params)
            final.fit(fit_df[feature_cols], fit_df[target],
                      vap[feature_cols], vap[target])
            yhat = final.predict(tep[feature_cols])

            saleq_pred = reconstruct_level(te[base_col].values, yhat)

            block = pd.DataFrame({
                "model": model_name, "gvkey": te["gvkey"].values,
                "qid": te["qid"].values, "horizon": h,
                "y_true": te[target].values, "y_pred": yhat,
                # MASE scaling reference: zero seasonal-log-growth = random-walk /
                # seasonal-naive baseline (Cao & You). This is the scale denominator,
                # NOT a reported competitor model.
                "y_pred_naive": 0.0,
                "saleq_future": te[future_col].values,
                "saleq_pred": saleq_pred,
                "saleq_realbase": te[base_col].values,
                "market_cap": te["market_cap"].values,
                "sic": te["sic"].values,
            })
            records.append(block)
        log.info("[%s] window %d done", model_name, w.test_year)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
