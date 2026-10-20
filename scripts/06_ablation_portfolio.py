"""Q2 ablation + Q2 importance + Q5 portfolio."""

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.features.build import FEATURE_BLOCKS, all_feature_columns, build_modeling_panel
from as_fin.features import build_merged_panel
from as_fin.features.preprocess import Preprocessor
from as_fin.pipeline import run_tabular_model
from as_fin.pipeline.rolling import generate_windows
from as_fin.models.base import build_model
from as_fin.evaluation.metrics import mae
from as_fin.evaluation.ablation import (CUMULATIVE_SPECS, LOBO_SPECS,
                                        cumulative_improvement, lobo_contribution,
                                        permutation_importance, shap_analysis)
from as_fin.evaluation.portfolio import (ml_surprise_portfolio,
                                          attach_forward_returns, load_ff_factors,
                                          assign_surprise_quintiles, portfolio_turnover,
                                          vw_leg_diagnostics, hedge_contributions)
from as_fin.evaluation.analyst import build_formation_aligned_analyst
from as_fin.utils import get_logger

log = get_logger("ablation")


def _panel(cfg, with_analyst):
    raw = cfg.path("data_raw")
    panel = build_merged_panel(
        pd.read_parquet(raw / "compustat_fundq.parquet"),
        pd.read_parquet(raw / "crsp_msf.parquet"),
        pd.read_parquet(raw / "ccm_link.parquet"), save=False)
    macro = pd.read_parquet(raw / "fred_macro.parquet")
    return build_modeling_panel(panel, macro_raw=macro, with_analyst=with_analyst, save=False)


def run_ablation(cfg, full_grid: bool = False, cumulative_only: bool = False):
    """Q2 ablation: % MAE improvement as feature blocks A->B->C->D->E are added.

    By default uses **fixed** LightGBM hyper-parameters across all specs: this is
    far faster (~30-45 min vs. hours) and gives a fair block comparison under one
    parameterization (plan §3.5 permits a reduced grid for the first pass). Pass
    ``full_grid=True`` for the rigorous per-spec, per-window tuning later.
    """
    panel = _panel(cfg, with_analyst=True)
    grid = cfg.models["lightgbm"] if full_grid else {}     # {} -> fixed defaults
    specs = dict(CUMULATIVE_SPECS) if cumulative_only else {**CUMULATIVE_SPECS, **LOBO_SPECS}
    log.info("Ablation: %d specs, %s hyper-parameters", len(specs),
             "FULL grid (slow)" if full_grid else "fixed defaults (fast)")
    errors = {}
    for name, blocks in specs.items():
        feats = all_feature_columns(blocks)
        preds = run_tabular_model(panel, "lightgbm", feats, horizons=(1,),
                                  grid=grid, model_name=f"LGBM_{name}")
        errors[name] = mae(preds["y_true"], preds["y_pred"])
        log.info("ablation %-14s MAE=%.4f (%d feats)", name, errors[name], len(feats))

    cum = cumulative_improvement(errors)
    cum.to_csv(cfg.path("tables") / "table4a_cumulative_ablation.csv", index=False)
    log.info("Table 4a (cumulative):\n%s", cum.round(4).to_string(index=False))
    if not cumulative_only:
        lobo = lobo_contribution(errors)
        lobo.to_csv(cfg.path("tables") / "table4b_leave_one_block_out.csv", index=False)
        log.info("Table 4b (leave-one-block-out):\n%s", lobo.round(4).to_string(index=False))


def run_importance(cfg, top_n: int = 20):
    """Q2 variable-level importance (Figure 1): Gain + Permutation + SHAP.

    Fits a single LightGBM (fixed defaults) on the most recent rolling window's
    train+val, then scores the held-out **test block** three complementary ways:
      * Gain      — in-model split-gain importance (nearly free, but biased to
                    high-cardinality features and split among correlated ones);
      * Permutation — out-of-sample MAE rise when a feature is shuffled (model-
                    agnostic, honest OOS signal);
      * SHAP      — mean |SHAP| contribution per feature (adds sign/direction).
    Writes table4c_importance.csv and fig_q2_importance.png. This is the micro
    (variable) complement to the macro (block) ablation in run_ablation; see
    the project notes
    """
    panel = _panel(cfg, with_analyst=True)
    feats = all_feature_columns(["A", "B", "C", "D", "E"])
    needs_scaling = build_model("lightgbm").needs_scaling
    horizon = 1

    # Most recent rolling window with enough data (mirrors runner's slicing/trim).
    windows = generate_windows(cfg)
    target = f"target_h{horizon}"
    chosen = None
    for w in reversed(windows):
        cols = list(dict.fromkeys(feats + [target, "qid"]))
        d = panel[[c for c in cols if c in panel]].copy()
        usable = d["qid"] <= (w.test_qids.start - horizon)
        tr = d[w.mask(d["qid"], "train") & usable].dropna(subset=[target])
        va = d[w.mask(d["qid"], "val") & usable].dropna(subset=[target])
        te = d[w.mask(d["qid"], "test")].dropna(subset=[target])
        if len(tr) >= 500 and len(te) > 0:
            chosen = (w, tr, va, te)
            break
    if chosen is None:
        log.warning("No usable window for importance; skipped.")
        return
    w, tr, va, te = chosen
    log.info("Importance on window test_year=%d (train=%d, test=%d)",
             w.test_year, len(tr), len(te))

    pre = Preprocessor(feats, target, standardize=needs_scaling,
                       winsor_lower=cfg.sample["winsor_lower"],
                       winsor_upper=cfg.sample["winsor_upper"]).fit(tr)
    trp, vap, tep = (pre.transform(x) for x in (tr, va, te))
    model = build_model("lightgbm")
    fit_df = pd.concat([trp, vap])
    model.fit(fit_df[feats], fit_df[target], vap[feats], vap[target])

    X_te, y_te = tep[feats].copy(), tep[target]
    gain = model.feature_importance()
    perm = permutation_importance(model, X_te, y_te, n_repeats=5)
    shap_imp, shap_df, x_sample = shap_analysis(model, X_te)

    imp = pd.DataFrame({"gain": gain, "permutation": perm, "shap_mean_abs": shap_imp})
    imp = imp.sort_values("shap_mean_abs", ascending=False)
    imp.index.name = "feature"
    imp.to_csv(cfg.path("tables") / "table4c_importance.csv")
    # Cache the raw SHAP matrix + matching feature values so the beeswarm can be
    # (re)drawn from disk without a re-fit (keeps `make figures` recolor fast).
    shap_df.to_parquet(cfg.path("tables") / "table4c_shap_values.parquet")
    x_sample.to_parquet(cfg.path("tables") / "table4c_shap_xsample.parquet")
    log.info("Table 4c (importance) top 10 by |SHAP|:\n%s",
             imp.head(10).round(4).to_string())
    log.info("Cached SHAP matrix %s + feature sample for the beeswarm.", shap_df.shape)


def run_portfolio(cfg):
    """Build the ML-surprise hedge from IBES-subsample predictions.

    Signal (fixed): the surprise is the **growth-space, formation-aligned**
    disagreement ``ML_growth - analyst_growth`` (both unitless seasonal log
    growth), so it is free of the Compustat-vs-IBES level-basis mismatch and of
    firm-size scaling, and the analyst leg uses the same-information consensus as
    of the formation date (see evaluation/analyst.py). Firms are sorted into
    within-industry quintiles each **calendar quarter**; the hedge is long Q5
    (ML more bullish than analysts) minus short Q1, held the next quarter. Factor
    alphas align the holding quarter (cqid+1) to the formation calendar quarter.
    """
    preds = pd.read_parquet(cfg.path("predictions") / "predictions_ibes.parquet")
    keys = ["gvkey", "qid"]
    ml = (preds[(preds["model"] == "LightGBM") & (preds["horizon"] == 1)]
          [keys + ["y_pred"]].rename(columns={"y_pred": "ml_growth"}))

    # Formation-aligned analyst growth (correct timing + source-specific base).
    analyst = build_formation_aligned_analyst(cfg, preds, horizon=1)
    if analyst.empty:
        log.warning("No formation-aligned analyst rows; portfolio skipped.")
        return
    analyst = analyst[keys + ["y_pred"]].rename(columns={"y_pred": "analyst_growth"})

    df = ml.merge(analyst, on=keys, how="inner")           # paired firm-quarters
    df["ml_surprise"] = df["ml_growth"] - df["analyst_growth"]   # unitless, scale-free

    panel = _panel(cfg, with_analyst=True)
    info = (panel[keys + ["sic", "market_cap", "permno", "avail_date"]]
            .drop_duplicates(keys))
    df = df.merge(info, on=keys, how="left")
    df["sic3"] = (pd.to_numeric(df["sic"], errors="coerce") // 10)
    df = df.rename(columns={"avail_date": "formation_date"})
    df = df.dropna(subset=["formation_date", "permno"])
    # Calendar quarter of formation -> clean factor alignment (vs fiscal qid).
    fq = pd.to_datetime(df["formation_date"]).dt.to_period("Q")
    df["cqid"] = fq.dt.year * 4 + (fq.dt.quarter - 1)

    # Forward one-quarter CRSP returns after the formation date.
    crsp = pd.read_parquet(cfg.path("data_raw") / "crsp_msf.parquet")
    df = attach_forward_returns(df, crsp, date_col="formation_date", k_months=3)
    if df["fwd_ret"].notna().sum() == 0:
        log.warning("No forward returns matched; check permno coverage.")
        return

    # Factors: the hedge return at formation calendar quarter t is earned over the
    # *next* quarter, so align factors of quarter t+1 to index t (shift index -1).
    factors = load_ff_factors(cfg.path("data_raw") / "ff_factors.parquet")
    if factors is not None:
        factors = factors.copy()
        factors.index = factors.index - 1

    # Report both equal-weighted (EW, small caps carry more drift) and
    # value-weighted (VW, investable) variants of the same sort.
    stats = {}
    for wname, wcol in [("EW", None), ("VW", "market_cap")]:
        result = ml_surprise_portfolio(df, factors=factors, weight_col=wcol,
                                       time_col="cqid")
        result["quintile_returns"].to_csv(
            cfg.path("tables") / f"table8_portfolio_{wname.lower()}.csv")
        stats[wname] = result["stats"]
        log.info("Q5 portfolio (%s): %s", wname, result["stats"])
    pd.DataFrame(stats).to_csv(cfg.path("tables") / "table8_stats.csv")

    # --- Instructor feedback 4: turnover + names per leg (EW and VW) ---
    sorted_df = assign_surprise_quintiles(df, time_col="cqid")
    turn = {w: portfolio_turnover(sorted_df, weight_col=c, time_col="cqid", id_col="permno")
            for w, c in [("EW", None), ("VW", "market_cap")]}
    pd.DataFrame(turn).to_csv(cfg.path("tables") / "table8_turnover.csv")
    log.info("Q5 turnover / names per leg:\n%s", pd.DataFrame(turn).round(3).to_string())

    # --- Instructor feedback 3: VW concentration + 2000-01 drawdown drivers ---
    diag = vw_leg_diagnostics(sorted_df, weight_col="market_cap", time_col="cqid",
                              id_col="permno")
    diag.to_csv(cfg.path("tables") / "table8_vw_diagnostic.csv", index=False)
    # cqid = year*4 + (quarter-1); dot-com window = 2000Q1..2001Q4.
    dot_com = range(2000 * 4, 2002 * 4)
    contrib = hedge_contributions(sorted_df, weight_col="market_cap", time_col="cqid",
                                  id_col="permno", ret_col="fwd_ret", periods=dot_com)
    if not contrib.empty:
        contrib.head(15).to_csv(cfg.path("tables") / "table8_vw_drawdown_contributors.csv",
                                index=False)
    eff = diag[["eff_n_long", "eff_n_short"]].mean()
    log.info("VW effective #names — long %.1f, short %.1f (1/HHI, whole sample)",
             eff["eff_n_long"], eff["eff_n_short"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--importance", action="store_true",
                    help="Q2 variable-level Gain/Permutation/SHAP -> Figure 1")
    ap.add_argument("--portfolio", action="store_true")
    ap.add_argument("--full-grid", action="store_true",
                    help="ablation: per-spec per-window grid tuning (slow, rigorous)")
    ap.add_argument("--quick", action="store_true",
                    help="ablation: cumulative specs only (skip leave-one-block-out)")
    args = ap.parse_args()
    cfg = load_config()
    ensure_dirs()
    any_flag = args.ablation or args.importance or args.portfolio
    if args.ablation or not any_flag:
        run_ablation(cfg, full_grid=args.full_grid, cumulative_only=args.quick)
    if args.importance:
        run_importance(cfg)
    if args.portfolio:
        run_portfolio(cfg)


if __name__ == "__main__":
    main()
