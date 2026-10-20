"""Block 5 — Build the result tables and figures from tidy predictions."""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.models.ensemble import build_all_ensembles
from as_fin.evaluation.metrics import accuracy_table, matched_accuracy_table
from as_fin.evaluation.dm_test import dm_pairwise_table
from as_fin.evaluation.heterogeneity import (improvement_vs_baseline,
                                             add_characteristic_quintiles)
from as_fin.evaluation.analyst import ml_vs_analyst_horizons
from as_fin.viz_style import (apply_style, color_for, marker_for, save_fig,
                              CYCLE, PALETTE, EW_VW, IMPORTANCE, Q4_BASELINE,
                              GREY_REFERENCE, NEGATIVE)
from as_fin.utils import get_logger

log = get_logger("tables")


def sic_sector(sic):
    """Coarse SIC division label for the by-sector table (Table 3)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if 1000 <= s < 1500:
        return "Mining"
    if 1500 <= s < 1800:
        return "Construction"
    if 2000 <= s < 4000:
        return "Manufacturing"
    if 4000 <= s < 4900:
        return "Transportation"
    if 5000 <= s < 5200:
        return "Wholesale"
    if 5200 <= s < 6000:
        return "Retail"
    if 7000 <= s < 9000:
        return "Services"
    return "Other"


def _plot_all_from_csv(tables_dir, fig_dir):
    """Draw EVERY figure from saved table CSVs. This is the ONLY plotting path.

    Tables and figures are fully decoupled: the compute path (`main`) only
    computes and writes CSVs, then calls this; `--plot-only` calls this alone.
    So drawing a figure never triggers a table recompute, and a palette change
    in viz_style.py repaints all figures in seconds.
    """
    def rd(name):
        p = tables_dir / name
        return pd.read_csv(p) if p.exists() else None

    t = rd("table1b_matched_sample_h1.csv")
    if t is not None:
        _plot_q1_matched(t, fig_dir / "fig_q1_matched_mase.png")
    t = rd("table1b_matched_sample_by_horizon.csv")
    if t is not None:
        _plot_horizon_mase(t, fig_dir / "fig_q3_horizon_mase.png", _Q1_MODELS)
    t = rd("table1c_dl_subsample_by_horizon.csv")
    if t is not None:
        _plot_horizon_mase(t, fig_dir / "fig_dl_vs_trees.png", _DL_MODELS,
                           title="Tier-4: LSTM/TFT vs trees (DL subsample)")
    t = rd("table7_ml_vs_analyst_h1.csv")
    if t is not None:
        _plot_q1b_analyst(t, fig_dir / "fig_q1b_analyst.png")
    t = rd("table3_by_sector.csv")
    if t is not None:
        _plot_q4_sector(t, fig_dir / "fig_q4_sector.png")
    t = rd("table6_by_characteristic.csv")
    if t is not None:
        _plot_q4_characteristics(t, fig_dir / "fig_q4_characteristics.png")
    _plot_q2_ablation(tables_dir, fig_dir)
    _plot_q5(tables_dir, fig_dir)
    _plot_q2_importance(tables_dir, fig_dir)
    _plot_q2_shap_beeswarm(tables_dir, fig_dir)
    _plot_q5_vw(tables_dir, fig_dir)
    log.info("Redrew all figures from cached CSVs (plot-only, no compute).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw figures from cached table CSVs; skip all compute (fast recolor)")
    args = ap.parse_args()
    cfg = load_config()
    ensure_dirs()
    apply_style()   # one shared look (fonts/grid/colors) for every figure
    tables_dir, fig_dir = cfg.path("tables"), cfg.path("figures")
    if args.plot_only:
        _plot_all_from_csv(tables_dir, fig_dir)
        return

    preds = pd.read_parquet(cfg.path("predictions") / "predictions_full.parquet")
    # Append simple-average ensembles built from component predictions.
    ens = build_all_ensembles(preds)
    if len(ens):
        preds = pd.concat([preds, _enrich(ens, preds)], ignore_index=True)

    # ---- Table 1: full-sample accuracy at h=1 (EW and VW) ----
    t1 = accuracy_table(preds[preds["horizon"] == 1], weight="market_cap")
    t1.to_csv(tables_dir / "table1_accuracy_h1.csv", index=False)
    log.info("Table 1 written (%d models)", t1["model"].nunique())

    # ---- Table 2: Diebold-Mariano pairwise tests (h=1) ----
    t2 = dm_pairwise_table(preds, horizon=1)
    t2.to_csv(tables_dir / "table2_dm_tests.csv", index=False)
    # h=2/3 pairwise DM (Q3 significance of the widening tree edge).
    t2h = pd.concat([dm_pairwise_table(preds, horizon=h).assign(horizon=h)
                     for h in (1, 2, 3)], ignore_index=True)
    t2h.to_csv(tables_dir / "table2_dm_tests_by_horizon.csv", index=False)

    # ---- Q4 heterogeneity: Table 3 (sector) + Table 6 (firm characteristic),
    #      both vs AutoARIMA only (matched sample); naive is a near-tautological
    #      benchmark for heterogeneity splits, see the project notes. ----
    _q4_heterogeneity(cfg, preds, tables_dir)

    # ---- Table 5: multi-step accuracy by horizon (full sample, all models) ----
    t5 = accuracy_table(preds)
    t5.to_csv(tables_dir / "table5_by_horizon.csv", index=False)

    # ---- Q1/Q3: matched-sample ranking (fair, common firm-quarters) + DL subsample ----
    _q1_q3_matched(preds, tables_dir)

    # ---- Table 7: ML vs analyst consensus, FORMATION-ALIGNED (same info set) ----
    # The analyst consensus is frozen at the ML formation date s (latest snapshot
    # with statpers <= datadate(s)), so both forecasters predict target s+h from the
    # same cutoff. Scored in source-specific seasonal-growth space; paired DM test.
    ibes_path = cfg.path("predictions") / "predictions_ibes.parquet"
    if ibes_path.exists():
        ibes_preds = pd.read_parquet(ibes_path)
        acc7, dm7 = ml_vs_analyst_horizons(cfg, ibes_preds,
                                           ml_models=("LightGBM", "ElasticNet"),
                                           horizons=(1, 2, 3, 4))
        if len(acc7):
            acc7.to_csv(tables_dir / "table7_ml_vs_analyst_h1.csv", index=False)
            dm7.to_csv(tables_dir / "table7_ml_vs_analyst_dm_tests.csv", index=False)
            log.info("Table 7 (formation-aligned) written for horizons %s",
                     sorted(acc7["horizon"].unique()))

    # ---- Figures: drawn ONLY here, from the CSVs just written (same path as
    #      --plot-only). Tables and figures are fully decoupled. ----
    _plot_all_from_csv(tables_dir, fig_dir)
    log.info("Tables written; figures drawn from CSVs -> %s", fig_dir)


# Model sets for the matched-sample rankings (must all forecast a firm-quarter to
# enter the common set). Q1/Q3 = trees + linear + statistical baselines; the DL
# subsample additionally requires LSTM/TFT, which cover far fewer firm-quarters.
_Q1_MODELS = ["AutoARIMA", "AutoETS", "ElasticNet", "LightGBM", "RandomForest", "XGBoost"]
_DL_MODELS = ["LSTM", "TFT", "LightGBM", "XGBoost", "RandomForest"]


def _q1_q3_matched(preds, tables_dir):
    """Fair matched-sample MASE ranking for Q1 (h=1) and Q3 (h=1..4) + DL subsample.

    Writes table1b_matched_sample_h1.csv, table1b_matched_sample_by_horizon.csv and
    (if DL present) table1c_dl_subsample_by_horizon.csv. Figures are drawn later by
    _plot_all_from_csv from these CSVs. All derived from predictions_full (no re-run).
    """
    t1b = matched_accuracy_table(preds, _Q1_MODELS, horizons=(1, 2, 3, 4))
    t1b.to_csv(tables_dir / "table1b_matched_sample_by_horizon.csv", index=False)
    t1b_h1 = t1b[t1b["horizon"] == 1].drop(columns="horizon")
    t1b_h1.to_csv(tables_dir / "table1b_matched_sample_h1.csv", index=False)

    if preds["model"].isin(["LSTM", "TFT"]).any():
        t1c = matched_accuracy_table(preds, _DL_MODELS, horizons=(1, 2, 3, 4))
        t1c.to_csv(tables_dir / "table1c_dl_subsample_by_horizon.csv", index=False)
        log.info("Q1/Q3 matched + DL subsample tables written.")
    else:
        log.info("Q1/Q3 matched tables written (no DL rows in predictions).")


def _plot_q1_matched(t1b_h1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = t1b_h1.sort_values("MASE")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(d["model"], d["MASE"], color=[color_for(m) for m in d["model"]])
    ax.axvline(1.0, ls="--", color=GREY_REFERENCE, label="seasonal naive (MASE=1)")
    ax.invert_yaxis(); ax.set_xlabel("MASE (h=1, matched sample) ↓")
    ax.set_title("Matched-sample accuracy (n=%d)" % int(d["n_matched"].iloc[0]))
    ax.legend(); ax.grid(axis="x")
    fig.tight_layout(); save_fig(fig, path)


def _plot_horizon_mase(t, path, models, title="Matched MASE by horizon"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        g = t[t["model"] == m].sort_values("horizon")
        if len(g):
            ax.plot(g["horizon"], g["MASE"], marker=marker_for(m),
                    color=color_for(m), label=m)
    ax.axhline(1.0, ls="--", color=GREY_REFERENCE, label="seasonal naive (MASE=1)")
    ax.set_xlabel("Forecast horizon (quarters)"); ax.set_ylabel("MASE ↓")
    ax.set_title(title); ax.set_xticks([1, 2, 3, 4]); ax.legend()
    ax.grid(True)
    fig.tight_layout(); save_fig(fig, path)


def _plot_q1b_analyst(acc7, path):
    """ML vs formation-aligned analyst consensus: MAE by horizon (Table 7)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m in ("AnalystConsensus", "LightGBM", "ElasticNet"):
        g = acc7[acc7["model"] == m].sort_values("horizon")
        if len(g):
            ax.plot(g["horizon"], g["MAE"], marker=marker_for(m),
                    color=color_for(m), label=m)
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_ylabel("MAE of seasonal log growth ↓")
    ax.set_title("ML vs analyst consensus (formation-aligned, same info set)")
    ax.set_xticks([1, 2, 3, 4]); ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig(fig, path)


def _plot_q2_ablation(tables_dir, fig_dir):
    """Q2 ablation bars from table4a (cumulative) + table4b (leave-one-block-out)."""
    a, b = tables_dir / "table4a_cumulative_ablation.csv", tables_dir / "table4b_leave_one_block_out.csv"
    if not (a.exists() and b.exists()):
        log.info("Q2 ablation CSVs absent (run 06 --ablation); skipping fig_q2_ablation.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t4a, t4b = pd.read_csv(a), pd.read_csv(b)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(range(len(t4a)), t4a["MAE"], marker="o", color=IMPORTANCE["gain"])
    a1.set_xticks(range(len(t4a))); a1.set_xticklabels(t4a["spec"], rotation=30, ha="right")
    a1.set_ylabel("MAE ↓"); a1.set_title("Cumulative: A → A+B+C+D+E"); a1.grid(True)
    a2.barh(t4b["spec"], t4b["contribution_pct"], color=IMPORTANCE["permutation"])
    a2.invert_yaxis(); a2.set_xlabel("% MAE rise if block removed")
    a2.set_title("Leave-one-block-out contribution"); a2.grid(axis="x")
    fig.suptitle("Ablation — which data blocks add value", y=1.02)
    fig.tight_layout(); save_fig(fig, fig_dir / "fig_q2_ablation")


def _plot_q5(tables_dir, fig_dir):
    """Q5 quintile-return bars + cumulative hedge growth-of-$1 from table8 EW/VW CSVs."""
    ew_p, vw_p = tables_dir / "table8_portfolio_ew.csv", tables_dir / "table8_portfolio_vw.csv"
    if not (ew_p.exists() and vw_p.exists()):
        log.info("Q5 portfolio CSVs absent (run 06 --portfolio); skipping Q5 figures.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ew, vw = pd.read_csv(ew_p), pd.read_csv(vw_p)
    qcols = ["0.0", "1.0", "2.0", "3.0", "4.0"]

    # Quintile mean returns (EW vs VW).
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = range(5); w = 0.38
    ax.bar([i - w / 2 for i in x], ew[qcols].mean() * 100, w, label="EW", color=EW_VW["EW"])
    ax.bar([i + w / 2 for i in x], vw[qcols].mean() * 100, w, label="VW", color=EW_VW["VW"])
    ax.set_xticks(list(x)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_ylabel("Mean quarterly return (%)")
    ax.set_title("ML-surprise quintile returns (Q1 bearish → Q5 bullish)")
    ax.legend(); ax.grid(axis="y")
    fig.tight_layout(); save_fig(fig, fig_dir / "fig_q5_quintiles")

    # Cumulative growth of $1 in the hedge, EW vs VW, with a t0=1.0 baseline point.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for d, nm in [(ew, "EW"), (vw, "VW")]:
        d = d.sort_values("cqid")
        yr = d["cqid"] / 4.0
        cum = (1.0 + d["hedge"]).cumprod()
        yr = pd.concat([pd.Series([yr.iloc[0] - 0.25]), yr], ignore_index=True)
        cum = pd.concat([pd.Series([1.0]), cum], ignore_index=True)
        ax.plot(yr, cum, label=nm, color=EW_VW[nm])
    ax.axhline(1.0, ls=":", color=GREY_REFERENCE)
    ax.set_xlabel("Year"); ax.set_ylabel("Growth of $1 in the Q5−Q1 hedge")
    ax.set_title("Cumulative hedge return (EW vs VW)")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig(fig, fig_dir / "fig_q5_cumulative")


def _plot_q2_importance(tables_dir, fig_dir, top_n=20):
    """Figure 1a — magnitude: Gain (in-model) vs Permutation (OOS), from table4c CSV.

    SHAP magnitude is left to the beeswarm (Figure 1b, which also shows direction).
    Each method is normalized to its own max, so this compares RANKINGS, not levels.
    """
    p = tables_dir / "table4c_importance.csv"
    if not p.exists():
        log.info("table4c_importance.csv absent (run 06 --importance); skip fig_q2_importance.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imp = pd.read_csv(p, index_col=0)
    top = imp.head(top_n).iloc[::-1]

    def _norm(s):
        s = s.clip(lower=0); m = s.max()
        return s / m if m and np.isfinite(m) and m > 0 else s
    g, pm = _norm(top["gain"]), _norm(top["permutation"])
    y = np.arange(len(top)); h = 0.4
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.barh(y + h / 2, pm, h, label="Permutation (out-of-sample MAE rise)",
            color=IMPORTANCE["permutation"])
    ax.barh(y - h / 2, g, h, label="Gain (in-model split gain)", color=IMPORTANCE["gain"])
    ax.set_yticks(y); ax.set_yticklabels(top.index)
    ax.set_xlabel("Relative importance (each method normalized to its own max)")
    ax.set_title(f"Top {top_n} feature importances: in-model vs out-of-sample")
    ax.legend(loc="lower right"); ax.grid(axis="x")
    fig.tight_layout(); save_fig(fig, fig_dir / "fig_q2_importance")


def _plot_q2_shap_beeswarm(tables_dir, fig_dir, top_n=15):
    """Figure 1b — direction: SHAP beeswarm from the cached SHAP matrix + feature values.

    Each dot is one firm-quarter; x = signed SHAP (left = pushes the forecast down,
    right = up; the centre line is 0); colour = the feature's value (high vs low).
    Uses a brand orange<->deep-blue diverging map instead of SHAP's default red/blue.
    """
    sv_p = tables_dir / "table4c_shap_values.parquet"
    xs_p = tables_dir / "table4c_shap_xsample.parquet"
    if not (sv_p.exists() and xs_p.exists()):
        log.info("SHAP cache absent (run 06 --importance); skip fig_q2_shap_beeswarm.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import shap
    sv = pd.read_parquet(sv_p)
    xs = pd.read_parquet(xs_p)[sv.columns]      # align feature-value columns to SHAP
    cmap = LinearSegmentedColormap.from_list("brand", [PALETTE["deepblue"],
                                                       "#c9c9c4", PALETTE["orange"]])
    fig = plt.figure(figsize=(8, 7.5))
    shap.summary_plot(sv.values, xs, feature_names=list(sv.columns),
                      max_display=top_n, show=False, cmap=cmap, plot_size=None, sort=True)
    fig = plt.gcf()
    fig.suptitle("SHAP beeswarm: how the top drivers move the forecast",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig_q2_shap_beeswarm")


def _permno_company_labels(permnos):
    """Map CRSP PERMNOs to readable company labels via the IBES linking table.

    Falls back to the bare PERMNO string for any code the table does not cover, so
    the figure still renders if the reference file is missing.
    """
    labels = {int(p): str(int(p)) for p in permnos}
    try:
        lk_path = Path(load_config().paths["data_ibes"]) / "Linking_Table.csv"
        lk = pd.read_csv(lk_path, usecols=["permno", "comnam"])
        name = lk.dropna(subset=["comnam"]).groupby("permno")["comnam"].last()
        drop = {"CORP", "CORPORATION", "INC", "CO", "PLC", "LTD", "GROUP",
                "HLDGS", "HOLDINGS", "THE", "&"}
        for p in labels:
            raw = name.get(p)
            if isinstance(raw, str) and raw.strip():
                toks = [t for t in raw.split() if t.upper() not in drop]
                labels[p] = " ".join(toks).title() or raw.title()
    except Exception as exc:  # pragma: no cover - reference file optional
        log.info("permno->company lookup skipped (%s); using raw PERMNOs.", exc)
    return labels


def _plot_q5_vw(tables_dir, fig_dir):
    """Redraw the Q5 VW-concentration figure from cached table8_vw CSVs (no rebuild)."""
    d = tables_dir / "table8_vw_diagnostic.csv"
    if not d.exists():
        log.info("table8_vw_diagnostic.csv absent (run 06 --portfolio); skip fig_q5_vw_concentration.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    diag = pd.read_csv(d)
    cp = tables_dir / "table8_vw_drawdown_contributors.csv"
    contrib = pd.read_csv(cp) if cp.exists() else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    yr = diag["cqid"] / 4.0
    a1.plot(yr, diag["eff_n_long"], label="long (Q5)", color=EW_VW["EW"])
    a1.plot(yr, diag["eff_n_short"], label="short (Q1)", color=EW_VW["VW"])
    a1.axvspan(2000, 2002, color=GREY_REFERENCE, alpha=0.2, label="2000-01 dot-com")
    a1.set_xlabel("Year"); a1.set_ylabel("Effective # names (1/HHI, VW)")
    a1.set_title("VW leg concentration over time"); a1.legend(); a1.grid(True)
    if contrib is not None and not contrib.empty:
        worst = contrib.head(10).iloc[::-1]
        labels = _permno_company_labels(worst["permno"])
        names = [labels[int(p)] for p in worst["permno"]]
        a2.barh(names, worst["total_contribution"], color=NEGATIVE)
        a2.set_xlabel("Summed hedge contribution, 2000-01 (VW)")
        a2.set_title("Top-10 drawdown drivers (most negative)"); a2.grid(axis="x")
    fig.tight_layout(); save_fig(fig, fig_dir / "fig_q5_vw_concentration")


def _enrich(ens: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Attach y_true / level columns to ensemble rows by (gvkey, qid, horizon)."""
    cols = ["gvkey", "qid", "horizon", "y_true", "y_pred_naive", "saleq_future",
            "saleq_realbase", "market_cap", "sic"]
    ref = preds.drop_duplicates(["gvkey", "qid", "horizon"])[cols]
    merged = ens.merge(ref, on=["gvkey", "qid", "horizon"], how="left")
    # Reconstruct ensemble revenue level from its averaged growth forecast.
    import numpy as np
    merged["saleq_pred"] = merged["saleq_realbase"] * np.exp(merged["y_pred"])
    return merged


def _q4_heterogeneity(cfg, preds, tables_dir):
    """Q4: LightGBM improvement by sector + firm characteristic, vs AutoARIMA and AutoETS.

    Writes table3_by_sector.csv and table6_by_characteristic.csv; figures are drawn
    later by _plot_all_from_csv. Characteristic quintiles are formed within-quarter on
    the LightGBM prediction sample; each classical baseline uses its own matched
    (common gvkey-qid) set.
    """
    p = preds.copy()
    p["gvkey"] = p["gvkey"].astype(str)

    # ---- Table 3: by sector ----
    p["sector"] = p["sic"].map(sic_sector)
    # vs-naive is not shown (near-tautological benchmark for heterogeneity splits,
    # same rationale as the characteristic table below, see the project notes);
    # both classical models are reported vs.\ AutoARIMA and vs.\ AutoETS, each on
    # its own matched sample (LightGBM's own MAE is identical across the two, since
    # both classical models have the same h=1 coverage).
    t3a = improvement_vs_baseline(p, "sector", baseline="AutoARIMA").rename(
        columns={"improvement_pct": "vs_autoarima_pct", "mae_baseline": "mae_autoarima",
                 "n": "n_matched"})
    t3e = improvement_vs_baseline(p, "sector", baseline="AutoETS").rename(
        columns={"improvement_pct": "vs_autoets_pct", "mae_baseline": "mae_autoets"}
    )[["sector", "mae_autoets", "vs_autoets_pct"]]
    t3 = (t3a.merge(t3e, on="sector", how="left")
             .sort_values("vs_autoarima_pct", ascending=False))
    t3.to_csv(tables_dir / "table3_by_sector.csv", index=False)

    # ---- Table 6: by firm characteristic ----
    panel = pd.read_parquet(cfg.path("data_processed") / "modeling_panel_full.parquet")
    panel["gvkey"] = panel["gvkey"].astype(str)
    raw = panel[["gvkey", "qid", "yoy_growth", "ceqq", "rolling_std_4q"]]
    # Quintile on the LightGBM sample (market_cap already on the prediction rows).
    lg = (p[p["model"] == "LightGBM"][["gvkey", "qid", "market_cap"]]
          .merge(raw, on=["gvkey", "qid"], how="left"))
    lg = add_characteristic_quintiles(lg)
    qcols = ["size_q", "growth_q", "bm_q", "vol_q"]
    pretty = {"size_q": "Size", "growth_q": "Growth",
              "bm_q": "Book/Market", "vol_q": "Volatility"}
    pq = p.merge(lg[["gvkey", "qid"] + qcols].drop_duplicates(["gvkey", "qid"]),
                 on=["gvkey", "qid"], how="left")
    # Characteristic table is vs-AutoARIMA and vs-AutoETS (each on its own matched
    # sample): vs-naive is not comparable here since it runs on LightGBM's full,
    # unmatched sample (see the project notes) and is a near-tautological
    # benchmark for heterogeneity splits. Raw MAE for all three models is kept
    # alongside the % improvement so the underlying accuracy, not just the ratio,
    # is visible.
    recs = []
    for qc in qcols:
        a_ = improvement_vs_baseline(pq, qc, baseline="AutoARIMA").set_index(qc)
        e_ = improvement_vs_baseline(pq, qc, baseline="AutoETS").set_index(qc)
        for q in sorted(x for x in pq[qc].dropna().unique()):
            r = a_.loc[q] if q in a_.index else None
            re_ = e_.loc[q] if q in e_.index else None
            recs.append({"characteristic": pretty[qc], "quintile": f"Q{int(q)+1}",
                         "vs_autoarima_pct": r["improvement_pct"] if r is not None else np.nan,
                         "mae_lightgbm": r["mae_ml"] if r is not None else np.nan,
                         "mae_autoarima": r["mae_baseline"] if r is not None else np.nan,
                         "vs_autoets_pct": re_["improvement_pct"] if re_ is not None else np.nan,
                         "mae_autoets": re_["mae_baseline"] if re_ is not None else np.nan,
                         "n_matched": r["n"] if r is not None else np.nan})
    t6 = pd.DataFrame(recs)
    t6.to_csv(tables_dir / "table6_by_characteristic.csv", index=False)
    log.info("Q4 heterogeneity tables written (sector + characteristics: vs AutoARIMA and vs AutoETS)")


def _plot_q4_sector(t3: pd.DataFrame, path):
    """Two panels, side by side, both on the sector order set by vs-AutoARIMA
    (left) so the same row means the same sector in both panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sec = t3["sector"].tolist()
    y = np.arange(len(sec))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    a1.barh(y, t3["vs_autoarima_pct"], color=Q4_BASELINE["arima"])
    a1.set_yticks(y); a1.set_yticklabels(sec); a1.invert_yaxis()
    a1.set_xlabel("vs AutoARIMA (%)"); a1.set_title("vs AutoARIMA (matched)")
    a1.grid(axis="x")
    a2.barh(y, t3["vs_autoets_pct"], color=Q4_BASELINE["ets"])
    a2.axvline(0, ls=":", color=GREY_REFERENCE)
    a2.set_xlabel("vs AutoETS (%)"); a2.set_title("vs AutoETS (matched)")
    a2.grid(axis="x")
    fig.suptitle("LightGBM MAE improvement by sector", y=1.02)
    fig.tight_layout(); save_fig(fig, path)


def _plot_q4_characteristics(t6: pd.DataFrame, path):
    """Two panels, side by side: vs-AutoARIMA (left) and vs-AutoETS (right), each
    on its own matched sample -- vs-naive was dropped, see the project notes
    §9: it runs on a different, unmatched sample and is a near-tautological
    benchmark for heterogeneity splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    x = [1, 2, 3, 4, 5]
    for i, nm in enumerate(t6["characteristic"].unique()):
        s = t6[t6["characteristic"] == nm].sort_values("quintile")
        col = CYCLE[i % len(CYCLE)]
        a1.plot(x[:len(s)], s["vs_autoarima_pct"].values, marker="o", color=col, label=nm)
        a2.plot(x[:len(s)], s["vs_autoets_pct"].values, marker="o", color=col, label=nm)
    a1.set_title("vs AutoARIMA (matched)"); a1.axhline(0, ls=":", color=GREY_REFERENCE)
    a1.set_xlabel("Within-quarter quintile (Q1 low -> Q5 high)")
    a1.set_ylabel("% MAE reduction"); a1.legend(); a1.grid(True); a1.set_xticks(x)
    a2.set_title("vs AutoETS (matched)"); a2.axhline(0, ls=":", color=GREY_REFERENCE)
    a2.set_xlabel("Within-quarter quintile (Q1 low -> Q5 high)")
    a2.grid(True); a2.set_xticks(x)
    fig.suptitle("LightGBM MAE improvement by firm characteristic", y=1.02)
    fig.tight_layout(); save_fig(fig, path)


def _plot_horizon_curves(t5: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, g in t5.groupby("model"):
        ax.plot(g["horizon"], g["MAE"], marker=marker_for(model),
                color=color_for(model), label=model)
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_ylabel("MAE of seasonal log growth")
    ax.set_title("Accuracy degradation by horizon")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


if __name__ == "__main__":
    main()
