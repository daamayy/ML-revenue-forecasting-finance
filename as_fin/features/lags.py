"""Calendar-correct lag/diff helpers on the integer quarter grid (``qid``)."""

from __future__ import annotations

import pandas as pd


def lag(df: pd.DataFrame, col: str, k: int, group_col: str = "gvkey",
        qid_col: str = "qid") -> pd.Series:
    """Lag ``col`` by ``k`` quarters via an explicit (gvkey, qid) join.

    ``k`` > 0 is a backward lag; ``k`` < 0 looks forward.
    """
    src = df[[group_col, qid_col, col]].copy()
    src["_qid_tgt"] = src[qid_col] + k
    merged = df[[group_col, qid_col]].merge(
        src[[group_col, "_qid_tgt", col]].rename(columns={col: "_v"}),
        left_on=[group_col, qid_col], right_on=[group_col, "_qid_tgt"], how="left",
    )
    return pd.Series(merged["_v"].values, index=df.index)


def diff(df: pd.DataFrame, col: str, k: int = 1, **kw) -> pd.Series:
    """k-quarter difference: col_t - col_{t-k}."""
    return df[col] - lag(df, col, k, **kw)


def yoy_growth(df: pd.DataFrame, col: str, **kw) -> pd.Series:
    """Year-over-year growth (col_t - col_{t-4}) / |col_{t-4}|."""
    base = lag(df, col, 4, **kw)
    return (df[col] - base) / base.abs()
