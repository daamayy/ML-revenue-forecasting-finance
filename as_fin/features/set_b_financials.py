"""Set B — Financial-statement features (~45)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import safe_log
from . import lags

# Accounting levels that get a log transform.
_LOG_LEVELS = {
    "log_atq": "atq", "log_rectq": "rectq", "log_invtq": "invtq",
    "log_apq": "apq", "log_cheq": "cheq", "log_ppentq": "ppentq",
    "log_ceqq": "ceqq",
}
# Items whose missingness we flag explicitly.
_MISSING_FLAG = ["xrdq", "drcq", "drltq", "oancfq", "capxq", "rectq", "invtq", "apq"]

COLUMNS: list[str] = []   # filled in build() so it always matches the output


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    produced: list[str] = []

    # --- Missingness indicators (before zero-fill) ---
    for c in _MISSING_FLAG:
        if c in out:
            flag = f"{c}_isna"
            out[flag] = out[c].isna().astype("int8")
            produced.append(flag)

    # --- Log levels ---
    for name, src in _LOG_LEVELS.items():
        out[name] = safe_log(out[src], log1p_nonneg=True)
        produced.append(name)

    # --- First differences (QoQ) ---
    for src in ["saleq", "rectq", "invtq", "apq", "cheq", "ppentq"]:
        name = f"d_{src}"
        out[name] = lags.diff(out, src, 1)
        produced.append(name)

    # --- Year-over-year differences ---
    for src in ["rectq", "invtq", "apq", "capxq"]:
        name = f"d4_{src}"
        out[name] = lags.diff(out, src, 4)
        produced.append(name)

    # --- Balance-sheet ratios (scaled by assets) ---
    atq = out["atq"].replace(0, np.nan)
    for src in ["rectq", "invtq", "apq", "cheq", "ppentq"]:
        name = f"{src}_to_atq"
        out[name] = out[src] / atq
        produced.append(name)

    # --- Margin / efficiency / leverage ratios ---
    saleq = out["saleq"].replace(0, np.nan)
    out["gross_margin"] = (out["saleq"] - out["cogsq"]) / saleq
    out["oper_margin"] = out["oiadpq"] / saleq
    out["dso"] = out["rectq"] / (saleq / 90.0)
    out["capex_ratio"] = out["capxq"] / saleq
    out["rd_ratio"] = out["xrdq"].fillna(0) / saleq
    out["deferred_rev"] = out["drcq"].fillna(0) + out["drltq"].fillna(0)
    out["cash_ratio"] = out["cheq"] / atq
    out["debt_ratio"] = (out["dlcq"].fillna(0) + out["dlttq"].fillna(0)) / atq
    out["roa"] = out["ibq"] / atq
    out["roe"] = out["ibq"] / out["ceqq"].replace(0, np.nan)
    out["cfo_ratio"] = out["oancfq"] / saleq
    out["ni_margin"] = out["ibq"] / saleq
    produced += ["gross_margin", "oper_margin", "dso", "capex_ratio", "rd_ratio",
                 "deferred_rev", "cash_ratio", "debt_ratio", "roa", "roe",
                 "cfo_ratio", "ni_margin"]

    # --- Growth of cost / asset items ---
    out["cogsq_growth"] = lags.yoy_growth(out, "cogsq")
    out["xsgaq_growth"] = lags.yoy_growth(out, "xsgaq")
    out["atq_growth"] = lags.yoy_growth(out, "atq")
    out["d_capex"] = lags.diff(out, "capxq", 1)
    out["d_cfo"] = lags.diff(out, "oancfq", 1)
    produced += ["cogsq_growth", "xsgaq_growth", "atq_growth", "d_capex", "d_cfo"]

    COLUMNS.clear()
    COLUMNS.extend(produced)
    return out
