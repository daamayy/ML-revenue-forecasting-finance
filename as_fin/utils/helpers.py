"""Small numerical / panel helpers shared across feature and model code."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_quarter_index(df: pd.DataFrame, year_col: str = "fyearq",
                      qtr_col: str = "fqtr") -> pd.DataFrame:
    """Add a contiguous integer quarter index ``qid``.

    qid = fyearq * 4 + (fqtr - 1). Two observations exactly one calendar quarter
    apart differ by 1; same quarter one year apart differ by 4. This lets us
    express seasonal (lag-4) operations as plain integer shifts after reindexing.
    """
    out = df.copy()
    out["qid"] = out[year_col].astype("int64") * 4 + (out[qtr_col].astype("int64") - 1)
    return out


def quarter_end_date(fyearq: int, fqtr: int) -> pd.Timestamp:
    """Calendar quarter-end Timestamp for a (fiscal year, fiscal quarter) pair.

    Used as a fallback availability anchor when Compustat ``rdq`` is missing
    (quarter-end + N days).
    """
    month = fqtr * 3
    return pd.Timestamp(year=int(fyearq), month=month, day=1) + pd.offsets.MonthEnd(0)


def safe_log(x: pd.Series | np.ndarray, *, log1p_nonneg: bool = False):
    """Numerically safe log.

    * Default: ``log(x)`` with non-positive values mapped to NaN (used for the
      seasonal-log-growth target, which requires SALEQ > 0).
    * ``log1p_nonneg=True``: ``log(1 + x)`` for non-negative accounting levels
      that may legitimately be zero (Set B log transforms).
    """
    arr = np.asarray(x, dtype="float64")
    if log1p_nonneg:
        out = np.where(arr >= 0, np.log1p(arr), np.nan)
    else:
        out = np.where(arr > 0, np.log(arr), np.nan)
    if isinstance(x, pd.Series):
        return pd.Series(out, index=x.index, name=x.name)
    return out


def seasonal_log_growth(df: pd.DataFrame, level_col: str = "saleq",
                        group_col: str = "gvkey", qid_col: str = "qid",
                        lag: int = 4, out_col: str = "seasonal_log_growth") -> pd.Series:
    """Year-over-year log growth g = log(SALEQ_t) - log(SALEQ_{t-4}).

    Implemented via a calendar-correct lag-4 join on ``qid`` (not a positional
    shift) so that gaps in a firm's history do not silently produce a wrong lag.
    """
    log_level = safe_log(df[level_col])
    base = pd.DataFrame({group_col: df[group_col], qid_col: df[qid_col],
                         "_log": log_level.values})
    lagged = base.rename(columns={"_log": "_log_lag", qid_col: "_qid_src"})
    lagged["_qid_tgt"] = lagged["_qid_src"] + lag
    merged = df[[group_col, qid_col]].merge(
        lagged[[group_col, "_qid_tgt", "_log_lag"]],
        left_on=[group_col, qid_col], right_on=[group_col, "_qid_tgt"], how="left",
    )
    g = log_level.values - merged["_log_lag"].values
    return pd.Series(g, index=df.index, name=out_col)


def reconstruct_level(saleq_lag4: pd.Series | np.ndarray,
                      growth_hat: pd.Series | np.ndarray) -> np.ndarray:
    """Reconstruct a revenue-level forecast from a seasonal-log-growth forecast.

    SALEQ_hat_{t+h} = SALEQ_{t+h-4} * exp(g_hat_{t+h})  .
    Lets scale-free models be reported in dollars and compared with I/B/E/S.
    """
    return np.asarray(saleq_lag4, dtype="float64") * np.exp(np.asarray(growth_hat, dtype="float64"))


def winsorize_by_group(df: pd.DataFrame, cols, group_col: str = "qid",
                       lower: float = 0.01, upper: float = 0.99,
                       quantiles: dict | None = None):
    """Winsorize ``cols`` within each ``group_col`` (e.g. each quarter).

    Parameters
    ----------
    quantiles : optional precomputed {col: (lo, hi)} thresholds. When provided,
        thresholds are *applied* rather than estimated. This is how the rolling
        pipeline avoids look-ahead: thresholds are estimated on the training
        window only, then applied to validation/test.

    Returns
    -------
    (winsorized_df, thresholds) where thresholds maps col -> (lo, hi).
    """
    out = df.copy()
    if quantiles is not None:
        for col, (lo, hi) in quantiles.items():
            if col in out:
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out, quantiles

    thresholds: dict = {}
    for col in cols:
        if col not in out:
            continue
        # Cross-sectional thresholds per group, then clipped within each group.
        lo = out.groupby(group_col)[col].transform(lambda s: s.quantile(lower))
        hi = out.groupby(group_col)[col].transform(lambda s: s.quantile(upper))
        out[col] = out[col].clip(lower=lo, upper=hi)
        thresholds[col] = (float(out[col].min()), float(out[col].max()))
    return out, thresholds
