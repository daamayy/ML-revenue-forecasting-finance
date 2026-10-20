"""Merge Compustat + CRSP via CCM and apply the sample filters (plan 4.5)."""

from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..utils import get_logger, add_quarter_index

log = get_logger(__name__)


def _apply_sic_exclusions(df: pd.DataFrame, sic_col: str = "sic") -> pd.DataFrame:
    """Drop financials (6000-6999) and utilities (4900-4999)."""
    cfg = load_config()
    sic = pd.to_numeric(df[sic_col], errors="coerce")
    keep = pd.Series(True, index=df.index)
    for lo, hi in cfg.sample["exclude_sic_ranges"]:
        keep &= ~sic.between(lo, hi)
    return df[keep]


def link_gvkey_permno(compustat: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    """Attach CRSP ``permno`` to each Compustat row via the CCM date intervals."""
    merged = compustat.merge(ccm, on="gvkey", how="inner")
    in_window = (merged["datadate"] >= merged["linkdt"]) & (merged["datadate"] <= merged["linkenddt"])
    merged = merged[in_window].drop(columns=["linktype", "linkprim", "linkdt", "linkenddt"])
    # Guard against duplicate links for the same firm-quarter.
    merged = merged.drop_duplicates(["gvkey", "datadate"])
    return merged


def collapse_crsp_to_quarter(crsp: pd.DataFrame) -> pd.DataFrame:
    """Collapse monthly CRSP to quarter-end observations with derived market data.

    For each permno-quarter we keep the last month's price/shares (for market
    cap) plus rolling return / turnover / volatility features computed on the
    monthly series. Returns are kept monthly inside the function and aggregated.
    """
    cfg = load_config()
    df = crsp.copy()
    df = df[df["shrcd"].isin(cfg.sample["crsp_shrcd"])]
    df = df[df["exchcd"].isin(cfg.sample["crsp_exchcd"])]
    df["prc"] = df["prc"].abs()
    df = df[df["prc"] > cfg.sample["min_price"]]

    df = df.sort_values(["permno", "date"])
    df["mktcap"] = df["prc"] * df["shrout"]                # thousands of $
    df["turnover"] = df["vol"] / df["shrout"].replace(0, pd.NA)

    g = df.groupby("permno", group_keys=False)
    # Trailing cumulative returns and volatility on the monthly series.
    df["ret_1m"] = df["ret"]
    df["ret_3m"] = g["ret"].apply(lambda s: (1 + s).rolling(3).apply(lambda x: x.prod() - 1, raw=True))
    df["ret_12m"] = g["ret"].apply(lambda s: (1 + s).rolling(12).apply(lambda x: x.prod() - 1, raw=True))
    df["ret_vol_12m"] = g["ret"].rolling(12).std().reset_index(level=0, drop=True)
    df["turnover_12m"] = g["turnover"].rolling(12).mean().reset_index(level=0, drop=True)
    # Price momentum: month t-12 to t-2 (skip most recent month).
    df["mom_12_2"] = g["ret"].apply(
        lambda s: (1 + s).rolling(11).apply(lambda x: x.prod() - 1, raw=True).shift(1)
    )

    # Calendar quarter key for joining to Compustat fiscal quarter-end.
    df["cqtr"] = df["date"].dt.to_period("Q")
    last = df.groupby(["permno", "cqtr"]).tail(1)
    keep = ["permno", "cqtr", "date", "prc", "shrout", "mktcap", "siccd",
            "ret_1m", "ret_3m", "ret_12m", "ret_vol_12m", "turnover",
            "turnover_12m", "mom_12_2"]
    return last[keep].rename(columns={"date": "crsp_date"})


def build_merged_panel(compustat: pd.DataFrame, crsp: pd.DataFrame,
                       ccm: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Full merge + filter pipeline returning the base firm-quarter panel."""
    cfg = load_config()
    comp = _apply_sic_exclusions(compustat)
    comp = comp.dropna(subset=["saleq", "atq", "cshoq"])      # plan 4.5(1)
    before = len(comp)
    comp = comp.dropna(subset=["fyearq", "fqtr"])
    dropped = before - len(comp)
    if dropped:
        log.info("Dropped %d Compustat rows with missing fyearq/fqtr", dropped)
    comp = add_quarter_index(comp)
    before = len(comp)
    comp = (comp.sort_values(["gvkey", "qid", "datadate"])
                .drop_duplicates(["gvkey", "qid"], keep="last"))
    dropped = before - len(comp)
    if dropped:
        log.info("Dropped %d duplicate Compustat gvkey-qid rows", dropped)

    linked = link_gvkey_permno(comp, ccm)
    crsp_q = collapse_crsp_to_quarter(crsp)

    linked["cqtr"] = linked["datadate"].dt.to_period("Q")
    # Inner join enforces the CRSP sample filters (shrcd / exchcd / |prc|>$1),
    # which are applied inside collapse_crsp_to_quarter, as true sample
    # restrictions (plan 4.5): only firm-quarters with a valid CRSP common-share
    # observation are kept. Toggle ``sample.require_crsp_match`` to relax this.
    require_crsp = cfg.sample.get("require_crsp_match", True) if hasattr(cfg.sample, "get") \
        else cfg["sample"].get("require_crsp_match", True)
    how = "inner" if require_crsp else "left"
    before = len(linked)
    panel = linked.merge(crsp_q, on=["permno", "cqtr"], how=how)
    if require_crsp:
        log.info("CRSP sample filter: kept %d of %d firm-quarters with a valid "
                 "common-share CRSP match", len(panel), before)

    panel = panel.sort_values(["gvkey", "qid"]).reset_index(drop=True)
    log.info("Merged panel: %d firm-quarters, %d firms",
             len(panel), panel["gvkey"].nunique())

    if save:
        path = cfg.path("data_interim") / "panel_merged.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path, index=False)
        log.info("Saved %s", path.name)
    return panel
