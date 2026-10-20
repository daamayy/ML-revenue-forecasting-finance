"""Forecast-formation timing / data-availability rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger

log = get_logger(__name__)


def add_availability_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``avail_date`` for accounting variables (quarter-aware fallback)."""
    cfg = load_config()
    lag_q13 = int(cfg.sample["availability_lag_days"])                  # 10-Q, Q1-3
    lag_q4 = int(cfg.sample.get("availability_lag_days_q4", 90))        # 10-K, Q4
    out = df.copy()
    lag_days = np.where(out["fqtr"].eq(4), lag_q4, lag_q13)
    fallback = out["datadate"] + pd.to_timedelta(lag_days, unit="D")
    out["avail_date"] = out["rdq"].where(out["rdq"].notna(), fallback)
    n_rdq = out["rdq"].notna().mean()
    log.info("avail_date set (rdq available for %.1f%% of rows; fallback lag "
             "%d/%d days for Q1-3/Q4)", 100 * n_rdq, lag_q13, lag_q4)
    return out


def lag_macro(macro: pd.DataFrame, value_cols, lag_quarters: int = 1) -> pd.DataFrame:
    """Lag macro series by ``lag_quarters`` to avoid using revised/unreleased data.

    Macro variables are matched to a firm's *formation* quarter only after the
    publication lag, so a value dated quarter q is attached to formation quarters
    >= q + lag_quarters.
    """
    out = macro.sort_values("qdate").copy()
    for col in value_cols:
        out[col] = out[col].shift(lag_quarters)
    return out
