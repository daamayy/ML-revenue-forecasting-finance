"""Prediction target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, seasonal_log_growth

log = get_logger(__name__)


def add_seasonal_log_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add the contemporaneous seasonal log growth ``g`` and robustness targets."""
    out = df.copy()
    out["seasonal_log_growth"] = seasonal_log_growth(out)

    # Robustness targets (plan 3.2): raw YoY growth and log level.
    saleq_lag4 = _lag_on_qid(out, "saleq", 4)
    out["yoy_growth"] = (out["saleq"] - saleq_lag4) / saleq_lag4.abs()
    out["log_saleq"] = np.log(out["saleq"].where(out["saleq"] > 0))
    out["saleq_lag4"] = saleq_lag4
    return out


def add_future_targets(df: pd.DataFrame, horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """Attach future targets ``target_h{h}`` = g_{i,s+h} to each formation row.

    Also stores ``saleq_realbase_h{h}`` = SALEQ_{s+h-4}, the base used to turn a
    growth forecast back into a dollar revenue forecast for horizon h.
    """
    out = df.copy()
    for h in horizons:
        # Future growth via a forward lag-(-h) join on qid.
        out[f"target_h{h}"] = _lag_on_qid(out, "seasonal_log_growth", -h)
        # Reconstruction base SALEQ_{s+h-4} = level lagged by (4 - h) quarters.
        out[f"saleq_realbase_h{h}"] = _lag_on_qid(out, "saleq", 4 - h)
        # Realized future level (for evaluation in dollar space).
        out[f"saleq_future_h{h}"] = _lag_on_qid(out, "saleq", -h)
    return out


def _lag_on_qid(df: pd.DataFrame, col: str, lag: int,
                group_col: str = "gvkey", qid_col: str = "qid") -> pd.Series:
    """Calendar-correct lag of ``col`` by ``lag`` quarters on the qid grid.

    ``lag`` may be negative to look forward (used for future targets). Uses an
    explicit join on (gvkey, qid) so missing quarters never alias an adjacent
    observation into the wrong slot.
    """
    src = df[[group_col, qid_col, col]].copy()
    src["_qid_tgt"] = src[qid_col] + lag
    merged = df[[group_col, qid_col]].merge(
        src[[group_col, "_qid_tgt", col]].rename(columns={col: "_val"}),
        left_on=[group_col, qid_col], right_on=[group_col, "_qid_tgt"], how="left",
    )
    return pd.Series(merged["_val"].values, index=df.index, name=f"{col}_lag{lag}")
