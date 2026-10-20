"""Tier 4 deep-learning models."""

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from as_fin.config import load_config, ensure_dirs
from as_fin.features.build import (populate_feature_blocks_from_panel,
                                   all_feature_columns)
from as_fin.models.deep import NeuralForecastRunner
from as_fin.models.mlf import panel_to_runs_long, _ds_to_qid, _UID_SEP
from as_fin.pipeline.rolling import generate_windows
from as_fin.utils import get_logger

log = get_logger("deep")


def _dl_long(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Gapless contiguous-run long frame with standardized A-D exogenous.

    The exogenous columns span very different scales (log levels ~10, ratios ~0.05,
    growth ~0.1). Feeding them raw lets the large-magnitude ones dominate and
    diverges the RNN, so we z-score each column on this window's data (inf -> nan
    first), then fill remaining gaps with 0 = the standardized mean.
    """
    long = panel_to_runs_long(panel, "seasonal_log_growth", exog=feat_cols)
    X = long[feat_cols].replace([np.inf, -np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0.0, 1.0)
    long[feat_cols] = ((X - mu) / sd).fillna(0.0)
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tft"], default="lstm")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--input-size", type=int, default=8,
                    help="LSTM/TFT lookback length. Contiguous series shorter than "
                         "input_size + horizon cannot form a training window and "
                         "are dropped before fitting (neuralforecast rejects them). "
                         "Lower it (e.g. 4) to cover more, shorter-history firms.")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    panel = pd.read_parquet(cfg.path("data_processed") / "modeling_panel_full.parquet")
    # Use the SAME curated feature design as the tree tier (scripts/03): Sets A-D.
    # The old "every numeric column" selection pulled in raw million-scale levels
    # (atq, mktcap, shrout), bare codes (siccd, fyearq) and inf-valued industry
    # columns, which made the LSTM/TFT diverge. Curated blocks keep the tiers
    # comparable.
    populate_feature_blocks_from_panel(panel)
    feat_cols = [c for c in all_feature_columns(("A", "B", "C", "D"))
                 if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    log.info("DL hist_exog: %d curated feature columns (Sets A-D)", len(feat_cols))

    ref = panel[["gvkey", "qid", "seasonal_log_growth", "saleq", "saleq_lag4",
                 "market_cap", "sic"]].copy()
    ref["gvkey"] = ref["gvkey"].astype(str)

    records = []
    for w in generate_windows(cfg):
        train_panel = panel[panel["qid"].between(w.train_qids.start, w.val_qids.stop - 1)]
        long = _dl_long(train_panel, feat_cols)
        # neuralforecast needs each contiguous series to have >= input_size + horizon
        # timestamps to form one training window; drop the shorter runs (they would
        # otherwise make nf.fit raise). The dropped firm-quarters are still modeled by
        # the cross-sectional tree tier (scripts/03), so DL runs on a smaller,
        # longer-history subsample.
        min_len = args.input_size + args.horizon
        vc = long["unique_id"].value_counts()
        long = long[long["unique_id"].isin(vc.index[vc >= min_len])]
        if long["unique_id"].nunique() < 50:
            continue
        runner = NeuralForecastRunner(model=args.model, horizon=args.horizon,
                                      hist_exog=feat_cols, max_steps=args.max_steps,
                                      input_size=args.input_size)
        fc = runner.fit_predict(long)             # unique_id, ds, <ModelName>
        # Exclude a spurious 'index' column (old neuralforecast reset_index artifact).
        pred_col = next(c for c in fc.columns if c not in ("unique_id", "ds", "index"))

        # Recover gvkey + the TARGET quarter from ds; keep the test block.
        fc["gvkey"] = fc["unique_id"].str.split(_UID_SEP).str[0]
        fc["target_qid"] = _ds_to_qid(fc["ds"])
        fc = fc[fc["target_qid"].between(w.test_qids.start, w.test_qids.stop - 1)]
        # Merge realized growth / level base at the TARGET quarter for y_true + recon.
        m = fc.merge(ref, left_on=["gvkey", "target_qid"], right_on=["gvkey", "qid"],
                     how="inner")
        if m.empty:
            continue
        horizon = (m["target_qid"] - w.test_qids.start + 1).clip(1, args.horizon)
        records.append(pd.DataFrame({
            "model": args.model.upper(), "gvkey": m["gvkey"].values,
            # Key by the FORMATION quarter (target - horizon) so DL rows align with
            # the formation-date ML / statsforecast rows for cross-model joins/DM.
            "qid": (m["target_qid"] - horizon).values,
            "horizon": horizon.values,
            "y_true": m["seasonal_log_growth"].values,
            "y_pred": m[pred_col].values, "y_pred_naive": 0.0,
            "saleq_future": m["saleq"].values,
            "saleq_pred": m["saleq_lag4"].values * np.exp(m[pred_col].values),
            "saleq_realbase": m["saleq_lag4"].values,
            "market_cap": m["market_cap"].values, "sic": m["sic"].values,
        }))
        log.info("DL window %d: %d preds (%d series)", w.test_year, len(m),
                 long["unique_id"].nunique())

    if records:
        out = pd.concat(records, ignore_index=True)
        path = cfg.path("predictions") / f"predictions_{args.model}.parquet"
        out.to_parquet(path, index=False)
        log.info("Saved %s (%d rows)", path.name, len(out))


if __name__ == "__main__":
    main()
