"""Instructor-feedback checks."""

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.utils import get_logger

log = get_logger("feedback")

PANEL = "modeling_panel_full.parquet"


def _panel(cfg):
    return pd.read_parquet(cfg.path("data_processed") / PANEL)


# --------------------------------------------------------------------------- #
# Feedback 2: target metric verification (g vs r)                             #
# --------------------------------------------------------------------------- #
def run_metric(cfg):
    p = _panel(cfg)
    g = p["seasonal_log_growth"].to_numpy("float64")            # log(S_t/S_{t-4})
    r = p["yoy_growth"].to_numpy("float64")                     # S_t/S_{t-4} - 1
    m = np.isfinite(g) & np.isfinite(r)
    g, r = g[m], r[m]

    def _desc(x):
        from scipy import stats
        return {
            "n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "skew": float(stats.skew(x)), "kurtosis_excess": float(stats.kurtosis(x)),
            "min": float(np.min(x)), "p1": float(np.percentile(x, 1)),
            "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
        }

    tbl = pd.DataFrame({"r = SALEQ_t/SALEQ_t-4 - 1": _desc(r),
                        "g = log-growth (used)": _desc(g)}).T
    tbl.index.name = "target"
    tbl.to_csv(cfg.path("tables") / "table_metric_r_vs_g.csv")
    log.info("Metric verification (r vs g):\n%s", tbl.round(3).to_string())

    # Negative-reconstruction danger zone: in r-space, level = base*(1+r_hat) goes
    # negative once r_hat < -1; firms whose realized r is already deep-negative are
    # where an MAE-scale forecast error easily crosses that floor. g-space exp() cannot.
    mae_scale = float(np.mean(np.abs(g)))            # typical target magnitude
    danger = int(np.sum(r < -0.5))
    near_floor = int(np.sum(r < -0.9))
    big_right = int(np.sum(r > 1.0))                 # doubled+ (right tail)
    log.info("Danger zone: %d rows (%.2f%%) with r<-0.5; %d with r<-0.9 (near the -1 floor); "
             "%d with r>1.0 (>100%%). Typical |g| (MAE scale) = %.3f.",
             danger, 100 * danger / len(r), near_floor, big_right, mae_scale)
    pd.DataFrame([{"typical_abs_g": mae_scale, "n": len(r),
                   "n_r_below_-0.5": danger, "pct_r_below_-0.5": 100 * danger / len(r),
                   "n_r_below_-0.9": near_floor, "n_r_above_1.0": big_right}]
                 ).to_csv(cfg.path("tables") / "table_metric_danger_zone.csv", index=False)

    _plot_metric(r, g, cfg.path("figures") / "fig_metric_r_vs_g.png")
    log.info("Wrote table_metric_r_vs_g.csv, table_metric_danger_zone.csv, fig_metric_r_vs_g.png")


def _plot_metric(r, g, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.hist(np.clip(r, -1, 3), bins=120, color="#e6550d", alpha=0.85)
    a1.axvline(0, ls=":", color="k"); a1.axvline(-1, ls="--", color="grey")
    a1.set_title(f"r = SALEQ_t/SALEQ_t-4 - 1  (skew={_skew(r):.1f})")
    a1.set_xlabel("r  (clipped to [-1, 3];  floored at -1, unbounded right)")
    a2.hist(np.clip(g, -1.5, 1.5), bins=120, color="#3182bd", alpha=0.85)
    a2.axvline(0, ls=":", color="k")
    a2.set_title(f"g = log-growth (used)  (skew={_skew(g):.1f})")
    a2.set_xlabel("g  (clipped to [-1.5, 1.5];  near-symmetric)")
    for a in (a1, a2):
        a.set_ylabel("firm-quarters"); a.grid(alpha=0.25)
    fig.suptitle("Feedback 2 — why the log-growth target, not the simple ratio", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _skew(x):
    from scipy import stats
    return float(stats.skew(x))


# --------------------------------------------------------------------------- #
# Feedback 1: feature-timing worksheet for ONE real firm-quarter              #
# --------------------------------------------------------------------------- #
def run_timing(cfg):
    p = _panel(cfg)
    need = ["gvkey", "qid", "datadate", "rdq", "avail_date", "market_cap"]
    d = p.dropna(subset=[c for c in need if c in p]).copy()

    # Pick one large, mid-sample firm-quarter whose s-4..s+1 own-history all exists,
    # so every lag/target source quarter has its own rdq to display.
    d = d[(d["qid"] >= 2005 * 4) & (d["qid"] < 2012 * 4)]
    chosen = None
    for _, row in d.sort_values("market_cap", ascending=False).head(400).iterrows():
        gv, s = row["gvkey"], int(row["qid"])
        own = p[(p["gvkey"] == gv) & (p["qid"].between(s - 4, s + 1))]
        if set(own["qid"]) >= set(range(s - 4, s + 2)):
            chosen = (gv, s, own.set_index("qid"))
            break
    if chosen is None:
        log.warning("No firm-quarter with full s-4..s+1 history; timing worksheet skipped.")
        return
    gv, s, own = chosen

    def _av(qid):
        return own.loc[qid, "avail_date"] if qid in own.index else pd.NaT

    s_datadate = own.loc[s, "datadate"]
    s_rdq = own.loc[s, "rdq"]
    s_avail = own.loc[s, "avail_date"]

    rows = [
        ("A", "saleq_lag1 (revenue history)", f"fiscal q s-1 (qid {s-1})", _av(s - 1)),
        ("A", "saleq_lag4 (seasonal base)", f"fiscal q s-4 (qid {s-4})", _av(s - 4)),
        ("B", "cogsq / accounting @ s", f"fiscal q s (qid {s})", s_avail),
        ("C", "ret_12m (12-2 momentum)", "market data <= formation month", s_avail),
        ("D", "macro (lagged +1q)", "prior quarter, already released", s_avail),
        ("E", "analyst consensus", "statpers <= datadate(s)", s_datadate),
        ("TARGET", "g_{s+1} (what we predict)", f"fiscal q s+1 (qid {s+1})", _av(s + 1)),
    ]
    ws = pd.DataFrame(rows, columns=["block", "feature", "source", "source_public_date"])
    ws["formation_avail_date"] = s_avail
    ws["ok_before_formation"] = np.where(
        ws["block"] == "TARGET",
        ws["source_public_date"] > s_avail,          # target must be strictly AFTER
        ws["source_public_date"] <= s_avail,         # features must be <= formation
    )
    ws.to_csv(cfg.path("tables") / "audit_feature_timing.csv", index=False)

    log.info("Feature-timing worksheet — gvkey=%s, formation qid=%d "
             "(datadate=%s, rdq=%s, avail_date=%s):\n%s",
             gv, s, pd.Timestamp(s_datadate).date(), pd.Timestamp(s_rdq).date(),
             pd.Timestamp(s_avail).date(),
             ws.assign(source_public_date=ws["source_public_date"].apply(
                 lambda x: pd.Timestamp(x).date() if pd.notna(x) else None))
               .to_string(index=False))
    all_ok = bool(ws["ok_before_formation"].all())
    log.info("All feature timestamps <= formation and target strictly after? %s", all_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", action="store_true", help="feedback 2: g vs r verification")
    ap.add_argument("--timing", action="store_true", help="feedback 1: timing worksheet")
    args = ap.parse_args()
    cfg = load_config()
    ensure_dirs()
    if args.metric or not (args.metric or args.timing):
        run_metric(cfg)
    if args.timing:
        run_timing(cfg)


if __name__ == "__main__":
    main()
