"""Blocks 1-2 — Merge panel and build target + features."""

import _bootstrap  # noqa: F401

import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.features import build_merged_panel, build_modeling_panel
from as_fin.utils import get_logger

log = get_logger("features")


def main():
    cfg = load_config()
    ensure_dirs()
    raw = cfg.path("data_raw")

    compustat = pd.read_parquet(raw / "compustat_fundq.parquet")
    crsp = pd.read_parquet(raw / "crsp_msf.parquet")
    ccm = pd.read_parquet(raw / "ccm_link.parquet")
    macro = pd.read_parquet(raw / "fred_macro.parquet")

    panel = build_merged_panel(compustat, crsp, ccm)

    # Full sample for Q1-Q4 (no IBES) and IBES subsample for Q1/Q5.
    build_modeling_panel(panel, macro_raw=macro, with_analyst=False)
    build_modeling_panel(panel, macro_raw=macro, with_analyst=True)
    log.info("Feature build complete.")


if __name__ == "__main__":
    main()
