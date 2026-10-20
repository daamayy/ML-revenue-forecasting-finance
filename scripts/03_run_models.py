"""Blocks 2-4 — Run baselines + ML models over the rolling windows."""

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.features.build import (FEATURE_BLOCKS, all_feature_columns,
                                   build_modeling_panel,
                                   populate_feature_blocks_from_panel)
from as_fin.features import build_merged_panel
from as_fin.models.baselines import run_cross_validation as sf_run_cv
from as_fin.pipeline import run_tabular_model
from as_fin.runlog import record_run
from as_fin.utils import get_logger

log = get_logger("models")

# Tier 2 linear + Tier 3 tree models both run through the formation-date runner
# (raw sklearn / LightGBM / XGBoost) with the full feature set and correct
# formation-date timing. mlforecast (models/mlf.py) is retained only as an
# optional pure-time-series tree variant; it is not used for the main Q1 results
# because its target-aligned exogenous handling cannot use Sets B-E leakage-safely.
LINEAR = {"elastic_net": "elastic_net"}
TREES = {"lightgbm", "xgboost", "random_forest"}


def _load_panel(cfg, with_analyst: bool, rebuild: bool = False):
    """Load the modeling panel.

    Fast path (default): if the processed parquet exists, load it directly and
    reconstruct ``FEATURE_BLOCKS`` from the Set modules — no raw data needed (this
    is what the GPU host uses, shipping only the one parquet). Pass
    ``rebuild=True`` to rebuild from raw WRDS/FRED files instead.
    """
    suffix = "ibes" if with_analyst else "full"
    path = cfg.path("data_processed") / f"modeling_panel_{suffix}.parquet"
    if path.exists() and not rebuild:
        log.info("Loading processed panel %s (no raw rebuild)", path.name)
        panel = pd.read_parquet(path)
        populate_feature_blocks_from_panel(panel)
        return panel

    raw = cfg.path("data_raw")
    panel = build_merged_panel(
        pd.read_parquet(raw / "compustat_fundq.parquet"),
        pd.read_parquet(raw / "crsp_msf.parquet"),
        pd.read_parquet(raw / "ccm_link.parquet"),
        save=False,
    )
    macro = pd.read_parquet(raw / "fred_macro.parquet")
    # Rebuild to populate FEATURE_BLOCKS (cheap relative to model fitting).
    return build_modeling_panel(panel, macro_raw=macro, with_analyst=with_analyst, save=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["elastic_net", "lightgbm", "xgboost", "random_forest"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1])
    ap.add_argument("--ibes", action="store_true",
                    help="use the IBES subsample panel (adds Set E features)")
    ap.add_argument("--tune-models", nargs="+", default=["lightgbm"],
                    help="tree models to hyper-tune once on a validation window")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip tuning; use default hyper-parameters for all trees")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the panel from raw WRDS/FRED instead of loading the "
                         "processed parquet (default: load processed if present)")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    panel = _load_panel(cfg, with_analyst=args.ibes, rebuild=args.rebuild)
    blocks = ("A", "B", "C", "D", "E") if args.ibes else ("A", "B", "C", "D")
    feats = all_feature_columns(blocks)

    all_preds = []

    if "autoarima" in args.models or "autoets" in args.models:
        sf_models = [m for m, k in [("AutoARIMA", "autoarima"), ("AutoETS", "autoets")]
                     if k in args.models]
        # Rolling-origin OOS backtest (refit annually), same design as the tree
        # tier. This is the slow statistical baseline (hours for AutoARIMA).
        preds = sf_run_cv(panel, models=tuple(sf_models), horizons=tuple(args.horizons))
        all_preds.append(preds)

    # Tier 2: Elastic Net via the strict formation-date runner.
    for key in args.models:
        if key in LINEAR:
            grid = cfg.models.get(LINEAR[key], {})
            preds = run_tabular_model(panel, LINEAR[key], feats,
                                      horizons=tuple(args.horizons), grid=grid)
            all_preds.append(preds)

    # Tier 3: tree models as raw regressors through the formation-date runner, with
    # the FULL feature set (Sets A-D, +E in the IBES sample) and correct
    # formation-date timing. Hyper-parameters are tuned once per model (plan §3.5)
    # for the tuned models (default: LightGBM); the rest use robust defaults.
    tuned: dict = {}
    for key in [k for k in args.models if k in TREES]:
        do_tune = (not args.no_tune) and (key in args.tune_models)
        grid = cfg.models.get(key, {}) if do_tune else {}
        preds = run_tabular_model(panel, key, feats, horizons=tuple(args.horizons),
                                  grid=grid, tune="once")
        if do_tune:
            tuned[key] = "tuned_once"
        all_preds.append(preds)

    if all_preds:
        fresh = pd.concat(all_preds, ignore_index=True)
        suffix = "ibes" if args.ibes else "full"
        path = cfg.path("predictions") / f"predictions_{suffix}.parquet"
        # Incremental upsert: the §0b run plan executes several `03` invocations
        # (fast path, ARIMA/ETS, random forest, ...) that each compute a subset of
        # models. Merge into the existing file instead of overwriting it, replacing
        # only the rows for the models recomputed in *this* run, so the predictions
        # file accumulates across runs and step 3 (tables) sees every model.
        out = fresh
        new_models = set(fresh["model"].unique())
        if path.exists():
            prev = pd.read_parquet(path)
            prev = prev[~prev["model"].isin(new_models)]
            out = pd.concat([prev, fresh], ignore_index=True)
        out.to_parquet(path, index=False)
        log.info("Saved %s (%d rows, models=%s)", path.name, len(out),
                 sorted(out["model"].unique()))

        # Print + persist the run's hyper-parameters and accuracy for comparison.
        # Record only the models computed in this run (fresh) for a clean per-run
        # record; the on-disk file holds the accumulated set.
        record_run(
            tag=f"03_{suffix}",
            meta={"models": args.models, "horizons": args.horizons,
                  "sample": suffix, "tuned_hyperparams": tuned or "defaults",
                  "predictions_file": path.name},
            preds=fresh,
        )


if __name__ == "__main__":
    main()
