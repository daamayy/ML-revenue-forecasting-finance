"""Set A — Revenue history features (~15)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import safe_log
from . import lags

# Column names produced by this builder (registered for the ablation study).
COLUMNS = [
    "saleq_lag1", "saleq_lag4", "saleq_lag8",
    "log_saleq_lag1", "log_saleq_lag4", "log_saleq_lag8",
    "slg_lag1", "slg_lag4",
    "yoy_growth", "qoq_growth", "growth_accel",
    "asset_turnover", "rolling_mean_4q", "rolling_std_4q",
    "revenue_trend", "firm_age",
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with Set A columns added.

    Assumes ``seasonal_log_growth`` and ``qid`` already exist (target.py).
    """
    out = df.copy()

    out["saleq_lag1"] = lags.lag(out, "saleq", 1)
    out["saleq_lag4"] = lags.lag(out, "saleq", 4)
    out["saleq_lag8"] = lags.lag(out, "saleq", 8)

    out["log_saleq_lag1"] = safe_log(out["saleq_lag1"], log1p_nonneg=True)
    out["log_saleq_lag4"] = safe_log(out["saleq_lag4"], log1p_nonneg=True)
    out["log_saleq_lag8"] = safe_log(out["saleq_lag8"], log1p_nonneg=True)

    out["slg_lag1"] = lags.lag(out, "seasonal_log_growth", 1)
    out["slg_lag4"] = lags.lag(out, "seasonal_log_growth", 4)

    out["yoy_growth"] = lags.yoy_growth(out, "saleq")
    out["qoq_growth"] = (out["saleq"] - out["saleq_lag1"]) / out["saleq_lag1"].abs()
    out["growth_accel"] = out["yoy_growth"] - lags.lag(out, "yoy_growth", 1)

    out["asset_turnover"] = out["saleq"] / out["atq"].replace(0, np.nan)

    # Rolling moments over the trailing 4 quarters (sorted by gvkey, qid).
    out = out.sort_values(["gvkey", "qid"])
    grp = out.groupby("gvkey")["saleq"]
    out["rolling_mean_4q"] = grp.transform(lambda s: s.rolling(4, min_periods=4).mean())
    out["rolling_std_4q"] = grp.transform(lambda s: s.rolling(4, min_periods=4).std())

    out["revenue_trend"] = _rolling_slope(out, "saleq", window=8)
    out["firm_age"] = out.groupby("gvkey").cumcount()    # quarters since first obs
    return out


def _rolling_slope(df: pd.DataFrame, col: str, window: int = 8) -> pd.Series:
    """OLS slope of ``col`` on a 0..window-1 time index over a trailing window."""
    x = np.arange(window)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()

    def slope(s: pd.Series) -> float:
        if s.notna().sum() < window:
            return np.nan
        y = s.values
        return float((x_centered * (y - y.mean())).sum() / denom)

    return df.groupby("gvkey")[col].transform(
        lambda s: s.rolling(window, min_periods=window).apply(slope, raw=False)
    )
