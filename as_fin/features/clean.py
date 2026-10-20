"""Source-level data cleaning for the modeling panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger

log = get_logger(__name__)

# Ratio / growth features prone to small-denominator blow-ups. Winsorized
# cross-sectionally each quarter. (Levels and logs are left untouched here.)
RATIO_FEATURES = [
    "gross_margin", "oper_margin", "ni_margin", "roa", "roe", "cfo_ratio",
    "capex_ratio", "rd_ratio", "dso", "cash_ratio", "debt_ratio",
    "asset_turnover", "market_to_book", "yoy_growth", "qoq_growth",
    "growth_accel", "cogsq_growth", "xsgaq_growth", "atq_growth",
    "rectq_to_atq", "invtq_to_atq", "apq_to_atq", "cheq_to_atq", "ppentq_to_atq",
]
# Growth targets winsorized cross-sectionally (also re-winsorized in-window at fit).
TARGET_FEATURES = ["seasonal_log_growth", "slg_lag1", "slg_lag4",
                   "target_h1", "target_h2", "target_h3", "target_h4"]
# Macro series with partial historical coverage (get missingness indicators).
MACRO_FLAG = ["usd_index_change", "oil_price_growth", "retail_sales_growth",
              "durable_goods_growth", "term_spread", "consumer_sent"]


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop firm-quarters that are almost certainly data errors.

    Filters (all configurable):
    * ``min_saleq`` — minimum quarterly revenue in $mn (removes ~0 stub rows).
    * ``max_abs_log_growth`` — cap on |seasonal log growth|; e.g. 3.0 ≈ a 20x
      year-over-year change, beyond which the observation is treated as an error.
    """
    cfg = load_config()
    clean = cfg.get("cleaning", {}) if hasattr(cfg, "get") else cfg["cleaning"]
    min_saleq = clean.get("min_saleq", 0.1)
    max_g = clean.get("max_abs_log_growth", 3.0)

    n0 = len(df)
    out = df.copy()
    out = out[(out["saleq"] >= min_saleq) & (out["saleq_lag4"] >= min_saleq)]
    n_rev = n0 - len(out)
    out = out[out["seasonal_log_growth"].abs() <= max_g]
    n_g = n0 - n_rev - len(out)
    log.info("Quality filters: dropped %d tiny-revenue (<$%.2fmn) and %d extreme-growth "
             "(|g|>%.1f) rows (%.2f%% of %d)", n_rev, min_saleq, n_g, max_g,
             100 * (n_rev + n_g) / max(n0, 1), n0)
    return out.reset_index(drop=True)


def winsorize_cross_section(df: pd.DataFrame, cols, group_col: str = "qid",
                            lower: float | None = None, upper: float | None = None) -> pd.DataFrame:
    """Clip ``cols`` to per-quarter [lower, upper] quantiles (within-quarter only)."""
    cfg = load_config()
    lower = lower if lower is not None else cfg.sample["winsor_lower"]
    upper = upper if upper is not None else cfg.sample["winsor_upper"]
    out = df.copy()
    present = [c for c in cols if c in out]
    for c in present:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        lo = out.groupby(group_col)[c].transform(lambda s: s.quantile(lower))
        hi = out.groupby(group_col)[c].transform(lambda s: s.quantile(upper))
        out[c] = out[c].clip(lower=lo, upper=hi)
    log.info("Winsorized %d columns cross-sectionally at [%.0f, %.0f] pct",
             len(present), 100 * lower, 100 * upper)
    return out


def add_macro_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_isna`` indicators for partially-covered macro series before any fill."""
    out = df.copy()
    added = 0
    for c in MACRO_FLAG:
        if c in out:
            out[f"{c}_isna"] = out[c].isna().astype("int8")
            added += 1
    if added:
        log.info("Added %d macro missingness flags", added)
    return out


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Full source-level cleaning pass used by ``build_modeling_panel``."""
    out = apply_quality_filters(df)
    out = add_macro_missing_flags(out)
    out = winsorize_cross_section(out, TARGET_FEATURES + RATIO_FEATURES)
    return out
