"""Assemble the modeling panel: base panel -> target + Sets A-E -> processed parquet."""

from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..utils import get_logger
from .timing import add_availability_date
from .target import add_seasonal_log_growth, add_future_targets
from .clean import clean_panel
from . import clean as clean_module
from . import (set_a_revenue, set_b_financials, set_c_market,
               set_d_industry_macro, set_e_analyst)

log = get_logger(__name__)

# Populated by build_modeling_panel after the builders run (so the lists always
# reflect exactly what was produced).
FEATURE_BLOCKS: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": [], "E": []}


def build_modeling_panel(panel: pd.DataFrame, macro_raw: pd.DataFrame | None = None,
                         with_analyst: bool = True, save: bool = True) -> pd.DataFrame:
    """Run the full feature pipeline on a merged base panel.

    Parameters
    ----------
    panel : output of features.build_merged_panel.
    macro_raw : FRED quarterly frame (data.download_fred_macro). If None, Set D
        macro columns are skipped (industry columns still built).
    with_analyst : merge Set E (IBES). Set False to build the full (non-IBES)
        sample used for Q1-Q4.
    """
    cfg = load_config()
    horizons = cfg.target["horizons"]

    log.info("Building target + feature sets ...")
    df = add_availability_date(panel)
    df = add_seasonal_log_growth(df)

    df = set_a_revenue.build(df)
    FEATURE_BLOCKS["A"] = list(set_a_revenue.COLUMNS)

    df = set_b_financials.build(df)
    FEATURE_BLOCKS["B"] = list(set_b_financials.COLUMNS)

    df = set_c_market.build(df)
    FEATURE_BLOCKS["C"] = list(set_c_market.COLUMNS)

    df = set_d_industry_macro.build_industry(df)
    if macro_raw is not None:
        df = set_d_industry_macro.build_macro(df, macro_raw)
        FEATURE_BLOCKS["D"] = list(set_d_industry_macro.COLUMNS)
    else:
        FEATURE_BLOCKS["D"] = list(set_d_industry_macro.INDUSTRY_COLUMNS
                                   + set_d_industry_macro.QUARTER_DUMMIES)

    if with_analyst:
        df = set_e_analyst.build(df)
        FEATURE_BLOCKS["E"] = list(set_e_analyst.COLUMNS)

    # Future targets g_{s+h} and reconstruction bases for each horizon.
    df = add_future_targets(df, horizons=horizons)

    df = _apply_history_filter(df)

    # Source-level cleaning: quality filters, cross-sectional winsorization of the
    # target and ratio features, and macro missingness flags (plan 3.3 / 4.5).
    df = clean_panel(df)
    macro_flags = [f"{c}_isna" for c in clean_module.MACRO_FLAG if f"{c}_isna" in df]
    FEATURE_BLOCKS["D"] = list(dict.fromkeys(FEATURE_BLOCKS["D"] + macro_flags))

    log.info("Modeling panel ready: %d rows, %d features",
             len(df), sum(len(v) for v in FEATURE_BLOCKS.values()))

    if save:
        suffix = "ibes" if with_analyst else "full"
        path = cfg.path("data_processed") / f"modeling_panel_{suffix}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Saved %s", path.name)
    return df


def populate_feature_blocks_from_panel(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Reconstruct ``FEATURE_BLOCKS`` from a *saved* processed panel.

    The processed parquet stores the data but not the in-memory block map, so when
    a script loads it directly (e.g.) we rebuild the block →
    column mapping from each Set module's static ``COLUMNS`` list intersected with
    the panel's actual columns (plus the macro missingness flags). This lets the
    model scripts run from the processed panel alone, with no raw-data rebuild.
    """
    cols = set(panel.columns)
    present = lambda lst: [c for c in lst if c in cols]  # noqa: E731

    # Set B fills its ``COLUMNS`` dynamically inside build(); recover the exact list
    # by re-running it on the loaded panel (cheap arithmetic on existing columns;
    # output discarded so the panel is not mutated). A/C/D/E expose static COLUMNS.
    if not set_b_financials.COLUMNS:
        try:
            set_b_financials.build(panel)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not recover Set B columns from panel: %s", exc)

    FEATURE_BLOCKS["A"] = present(set_a_revenue.COLUMNS)
    FEATURE_BLOCKS["B"] = present(set_b_financials.COLUMNS)
    FEATURE_BLOCKS["C"] = present(set_c_market.COLUMNS)
    d_cols = present(getattr(set_d_industry_macro, "COLUMNS", []))
    if not d_cols:
        d_cols = present(list(set_d_industry_macro.INDUSTRY_COLUMNS)
                         + list(set_d_industry_macro.QUARTER_DUMMIES))
    macro_flags = [f"{c}_isna" for c in clean_module.MACRO_FLAG if f"{c}_isna" in cols]
    FEATURE_BLOCKS["D"] = list(dict.fromkeys(d_cols + macro_flags))
    FEATURE_BLOCKS["E"] = present(set_e_analyst.COLUMNS)
    return FEATURE_BLOCKS


def _apply_history_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the minimum-history / positive-revenue requirements (plan 4.5)."""
    cfg = load_config()
    min_hist = cfg.sample["min_history_quarters"]
    out = df.copy()
    # Need >= min_hist prior quarters: firm_age (quarters since first obs) >= min_hist.
    out = out[out["firm_age"] >= min_hist]
    if cfg.sample["require_positive_saleq"]:
        out = out[(out["saleq"] > 0) & (out["saleq_lag4"] > 0)]
    return out.reset_index(drop=True)


def all_feature_columns(blocks=("A", "B", "C", "D", "E")) -> list[str]:
    """Flatten the requested feature blocks into a single column list."""
    cols: list[str] = []
    for b in blocks:
        cols += FEATURE_BLOCKS.get(b, [])
    # De-duplicate while preserving order.
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]
