"""EDA & descriptive analysis on the built panels."""

import _bootstrap  # noqa: F401

import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin import eda
from as_fin.features.build import FEATURE_BLOCKS, build_modeling_panel
from as_fin.features import build_merged_panel
from as_fin.utils import get_logger

log = get_logger("eda")


def main():
    cfg = load_config()
    ensure_dirs()
    tdir, fdir = cfg.path("tables"), cfg.path("figures")
    raw = cfg.path("data_raw")

    raw_comp = pd.read_parquet(raw / "compustat_fundq.parquet")
    merged = pd.read_parquet(cfg.path("data_interim") / "panel_merged.parquet")
    full = pd.read_parquet(cfg.path("data_processed") / "modeling_panel_full.parquet")
    ibes = pd.read_parquet(cfg.path("data_processed") / "modeling_panel_ibes.parquet")

    # FEATURE_BLOCKS is populated by a (no-save) rebuild so missingness can be
    # reported per block without re-reading every feature name by hand.
    macro = pd.read_parquet(raw / "fred_macro.parquet")
    base = build_merged_panel(raw_comp, pd.read_parquet(raw / "crsp_msf.parquet"),
                              pd.read_parquet(raw / "ccm_link.parquet"), save=False)
    build_modeling_panel(base, macro_raw=macro, with_analyst=True, save=False)

    tables = {
        "eda_sample_construction": eda.sample_construction(raw_comp, merged, full),
        "eda_coverage_over_time": eda.coverage_over_time(ibes),
        "eda_target_description": eda.target_description(full),
        "eda_feature_missingness": eda.feature_missingness(ibes, FEATURE_BLOCKS),
        "eda_by_sector": eda.by_sector(full),
        "eda_compustat_ibes_check": eda.compustat_ibes_check(ibes),
    }
    for name, tbl in tables.items():
        tbl.to_csv(tdir / f"{name}.csv", index=False)
        log.info("wrote %s.csv (%d rows)", name, len(tbl))

    eda.plot_target_hist(full, fdir / "eda_target_hist.png")
    eda.plot_coverage(tables["eda_coverage_over_time"], fdir / "eda_coverage.png")

    print("\n=== Sample construction ===")
    print(tables["eda_sample_construction"].to_string(index=False))
    print("\n=== Target description ===")
    print(tables["eda_target_description"].round(3).to_string(index=False))
    print("\n=== Compustat vs IBES consistency (instructor check) ===")
    print(tables["eda_compustat_ibes_check"].T.to_string(header=False))


if __name__ == "__main__":
    main()
