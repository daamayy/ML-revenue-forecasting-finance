"""Set E — Analyst information from I/B/E/S (~10, IBES subsample only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger
from ..data import load_ibes_sal
from . import lags

log = get_logger(__name__)

COLUMNS = [
    "analyst_consensus", "analyst_forecast_growth", "analyst_consensus_scaled",
    "analyst_revision", "analyst_dispersion", "analyst_dispersion_scaled",
    "analyst_coverage", "analyst_coverage_change", "forecast_age",
]


def _latest_pre_announcement(ibes: pd.DataFrame) -> pd.DataFrame:
    """One consensus row per (cusip8, fpedats): the last snapshot before release.

    Keeps snapshots with statpers < anndats_act and picks the most recent one,
    i.e. the consensus a forecaster would have observed just before the actual.
    """
    df = ibes[ibes["statpers"] < ibes["anndats_act"]].copy()
    df = df.sort_values(["cusip8", "fpedats", "statpers"])
    latest = df.groupby(["cusip8", "fpedats"]).tail(1)

    # Snapshot one period earlier -> consensus revision and coverage change.
    prev = df.groupby(["cusip8", "fpedats"]).nth(-2)
    latest = latest.merge(
        prev[["cusip8", "fpedats", "meanest", "numest"]].rename(
            columns={"meanest": "_prev_mean", "numest": "_prev_numest"}),
        on=["cusip8", "fpedats"], how="left",
    )
    return latest


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge IBES consensus onto the panel by (CUSIP-8, fiscal quarter-end period).

    Matching key is the 8-digit CUSIP plus the calendar quarter of ``fpedats``
    (IBES) vs ``datadate`` (Compustat). CUSIP is far more reliable than the fuzzy
    name-based ticker->permno link; ``permno`` is still carried from the panel
    (via CCM) for later CRSP return merges (Q5).
    """
    cfg = load_config()
    ibes = load_ibes_sal()
    # IBES header CUSIP is 8-digit; Compustat CUSIP is 9-digit (last = check).
    ibes["cusip8"] = ibes["cusip"].astype(str).str.zfill(8).str[:8]

    consensus = _latest_pre_announcement(ibes)
    consensus["cqtr"] = consensus["fpedats"].dt.to_period("Q")
    before = len(consensus)
    consensus = (consensus.sort_values(["cusip8", "cqtr", "fpedats", "statpers"])
                          .drop_duplicates(["cusip8", "cqtr"], keep="last"))
    dropped = before - len(consensus)
    if dropped:
        log.info("Set E: dropped %d duplicate analyst consensus rows on cusip8-cqtr",
                 dropped)

    out = panel.copy()
    out["cusip8"] = out["cusip"].astype(str).str.zfill(9).str[:8]
    out["cqtr"] = out["datadate"].dt.to_period("Q")
    cols = ["cusip8", "cqtr", "meanest", "medest", "stdev", "numest", "actual",
            "numup", "numdown", "statpers", "anndats_act", "_prev_mean", "_prev_numest"]
    out = out.merge(consensus[cols], on=["cusip8", "cqtr"], how="left")

    # Derived analyst features.
    saleq_lag4 = lags.lag(out, "saleq", 4)
    out["analyst_consensus"] = out["medest"]                      # median = consensus
    out["ibes_actual"] = out["actual"]
    out["analyst_forecast_growth"] = out["analyst_consensus"] / saleq_lag4 - 1
    out["analyst_consensus_scaled"] = out["analyst_consensus"] / saleq_lag4
    out["analyst_revision"] = out["meanest"] - out["_prev_mean"]
    out["analyst_dispersion"] = out["stdev"]
    out["analyst_dispersion_scaled"] = out["stdev"] / out["meanest"].abs()
    out["analyst_coverage"] = out["numest"]
    out["analyst_coverage_change"] = out["numest"] - out["_prev_numest"]
    out["forecast_age"] = (out["anndats_act"] - out["statpers"]).dt.days

    n = out["analyst_consensus"].notna().sum()
    log.info("Set E: matched analyst consensus for %d firm-quarters", n)
    return out.drop(columns=["_prev_mean", "_prev_numest"])
