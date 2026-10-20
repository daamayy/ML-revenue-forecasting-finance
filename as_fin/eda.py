"""Exploratory data analysis & descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .utils import get_logger


def _data_not_bundled(*_a, **_k):
    raise NotImplementedError(
        "Raw WRDS / I/B/E/S data loaders are not part of this public repo "
        "(licensed data) -- see as_fin/data/README.md."
    )


load_ibes_sal = load_ibes_linking = _data_not_bundled

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Descriptive tables
# ---------------------------------------------------------------------------
def sample_construction(raw_compustat: pd.DataFrame, merged: pd.DataFrame,
                        modeling: pd.DataFrame) -> pd.DataFrame:
    """Attrition table: rows and unique firms at each construction stage."""
    def row(stage, df):
        return {"stage": stage, "firm_quarters": len(df),
                "firms": df["gvkey"].nunique()}
    return pd.DataFrame([
        row("1. Raw Compustat fundq", raw_compustat),
        row("2. Merged w/ CRSP+CCM, SIC filters", merged),
        row("3. Modeling panel (history+clean)", modeling),
    ])


def coverage_over_time(modeling: pd.DataFrame) -> pd.DataFrame:
    """Per-year firm count and CRSP / IBES coverage rates."""
    df = modeling.copy()
    df["year"] = df["qid"] // 4
    out = df.groupby("year").agg(
        firm_quarters=("gvkey", "size"),
        firms=("gvkey", "nunique"),
        crsp_match=("market_cap", lambda s: s.notna().mean()),
    )
    if "analyst_consensus" in df:
        out["ibes_match"] = df.groupby("year")["analyst_consensus"].apply(lambda s: s.notna().mean())
    return out.reset_index()


def target_description(modeling: pd.DataFrame, col: str = "seasonal_log_growth") -> pd.DataFrame:
    """Moments and quantiles of the target (overall and by horizon)."""
    cols = [col] + [f"target_h{h}" for h in (1, 2, 3, 4) if f"target_h{h}" in modeling]
    rows = []
    for c in cols:
        s = modeling[c].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"variable": c, "n": len(s), "mean": s.mean(), "std": s.std(),
                     "skew": s.skew(), "kurt": s.kurt(),
                     "p1": s.quantile(.01), "p50": s.median(), "p99": s.quantile(.99),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows)


def feature_missingness(modeling: pd.DataFrame, feature_blocks: dict) -> pd.DataFrame:
    """Mean NaN rate per feature, tagged by block (A–E)."""
    rows = []
    for block, cols in feature_blocks.items():
        for c in cols:
            if c in modeling:
                rows.append({"block": block, "feature": c,
                             "na_rate": modeling[c].isna().mean()})
    return pd.DataFrame(rows).sort_values("na_rate", ascending=False)


def by_sector(modeling: pd.DataFrame) -> pd.DataFrame:
    """Firm-quarter counts and median seasonal growth per SIC division."""
    df = modeling.copy()
    sic = pd.to_numeric(df["sic"], errors="coerce")
    df["division"] = pd.cut(
        sic, [0, 1000, 1500, 1800, 2000, 4000, 5000, 5200, 6000, 7000, 9000, 10000],
        labels=["Agri", "Mining", "Construction", "Food/Tobacco", "Manufacturing",
                "Transport/Comm", "Wholesale", "Retail", "FIRE", "Services", "Public"],
        right=False)
    return (df.groupby("division", observed=True)
              .agg(firm_quarters=("gvkey", "size"), firms=("gvkey", "nunique"),
                   median_growth=("seasonal_log_growth", "median"))
              .reset_index())


# ---------------------------------------------------------------------------
# Instructor-required Compustat vs IBES consistency check (plan 3.5)
# ---------------------------------------------------------------------------
def compustat_ibes_check(modeling: pd.DataFrame) -> pd.DataFrame:
    """Compare Compustat SALEQ against the I/B/E/S reported actual revenue.

    Matches on **CUSIP (8-digit) + exact fiscal-period-end date** — more reliable
    than the fuzzy name-based permno link, and exact dates avoid pairing
    different fiscal quarters that fall in the same calendar quarter.

    Reports correlation, the relative-difference distribution, and the
    actual/SALEQ ratio spread. This determines whether a unified Compustat
    benchmark is acceptable (corr > 0.99 and tight ratio) or whether ML and
    analyst errors must each be measured against their own actuals (plan 3.5
    Step 2, "Approach a"). NB: I/B/E/S "sales" actuals frequently differ from
    Compustat SALEQ by genuine definition (e.g. excise taxes excluded for energy
    firms, net vs gross revenue), so a wide spread here is an economic finding,
    not a matching error.
    """
    ibes = load_ibes_sal()
    ibes["cusip8"] = ibes["cusip"].astype(str).str.zfill(8).str[:8]
    act = ibes.dropna(subset=["actual", "fpedats"]).copy()
    # One actual per (cusip, fiscal-period-end), latest by announcement date.
    act = (act.sort_values(["cusip8", "fpedats", "anndats_act"])
              .drop_duplicates(["cusip8", "fpedats"], keep="last")[["cusip8", "fpedats", "actual"]])

    m = modeling.copy()
    m["cusip8"] = m["cusip"].astype(str).str.zfill(9).str[:8]
    j = m.merge(act, left_on=["cusip8", "datadate"], right_on=["cusip8", "fpedats"],
                how="inner").dropna(subset=["saleq", "actual"])
    j = j[(j["saleq"] > 0) & (j["actual"] > 0)]

    corr = j["saleq"].corr(j["actual"])
    rel = (j["saleq"] - j["actual"]).abs() / j["saleq"]
    ratio = j["actual"] / j["saleq"]
    out = pd.DataFrame([{
        "matched_obs": len(j),
        "corr_saleq_actual": corr,
        "mean_abs_rel_diff": rel.mean(),
        "median_abs_rel_diff": rel.median(),
        "share_within_1pct": (rel < 0.01).mean(),
        "ratio_p25": ratio.quantile(.25),
        "ratio_median": ratio.median(),
        "ratio_p75": ratio.quantile(.75),
        "recommendation": "unified Compustat benchmark" if corr and corr > 0.99
                          and rel.median() < 0.01
                          else "use source-specific actuals (Approach a)",
    }])
    log.info("Compustat vs IBES: corr=%.4f, median|rel diff|=%.3f, ratio median=%.2f on %d obs",
             corr if corr else np.nan, rel.median(), ratio.median(), len(j))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_target_hist(modeling: pd.DataFrame, path, col: str = "seasonal_log_growth"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = modeling[col].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s.abs() < 3]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=120, color="steelblue", alpha=0.8)
    ax.set_title("Seasonal log revenue growth (cleaned)")
    ax.set_xlabel("g = log(SALEQ_t) − log(SALEQ_{t−4})")
    ax.axvline(s.median(), color="crimson", lw=1, label=f"median={s.median():.3f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_coverage(cov: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(cov["year"], cov["firms"], color="lightgray", label="firms")
    ax1.set_ylabel("# firms")
    ax2 = ax1.twinx()
    ax2.plot(cov["year"], cov["crsp_match"], color="navy", label="CRSP match")
    if "ibes_match" in cov:
        ax2.plot(cov["year"], cov["ibes_match"], color="darkorange", label="IBES match")
    ax2.set_ylabel("match rate"); ax2.set_ylim(0, 1)
    ax1.set_title("Sample coverage over time")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.92), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
