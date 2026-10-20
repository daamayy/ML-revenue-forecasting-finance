"""Tier 1 — Statistical benchmarks on the revenue series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import get_logger

log = get_logger(__name__)


class StatsForecastRunner:
    """Fit AutoARIMA / AutoETS per firm and produce h-step revenue forecasts."""

    def __init__(self, models: list[str] = ("AutoARIMA", "AutoETS"),
                 season_length: int = 4, freq: str = "QE"):
        self.model_names = list(models)
        self.season_length = season_length
        self.freq = freq

    def _make_models(self):
        from statsforecast.models import AutoARIMA, AutoETS
        # Fast AutoARIMA: stepwise + AIC approximation + tight order bounds keep
        # per-series order search ~3-5x cheaper than the exhaustive default, which
        # matters because we refit thousands of series across rolling windows.
        registry = {
            "AutoARIMA": AutoARIMA(season_length=self.season_length,
                                   stepwise=True, approximation=True,
                                   max_p=2, max_q=2, max_P=1, max_Q=1,
                                   max_d=1, max_D=1),
            "AutoETS": AutoETS(season_length=self.season_length),
        }
        return [registry[m] for m in self.model_names]

    def forecast(self, train_long: pd.DataFrame, h: int = 4) -> pd.DataFrame:
        """Forecast ``h`` quarters ahead for every series in ``train_long``.

        Parameters
        ----------
        train_long : columns ['unique_id', 'ds', 'y'] where y is revenue level.
        """
        from statsforecast import StatsForecast

        sf = StatsForecast(models=self._make_models(), freq=self.freq, n_jobs=-1)
        fc = sf.forecast(df=train_long, h=h)
        log.info("statsforecast produced %d series forecasts", fc["unique_id"].nunique())
        return fc.reset_index()


def run_cross_validation(panel: pd.DataFrame, models=("AutoARIMA", "AutoETS"),
                         horizons=(1, 2, 3, 4), season_length: int = 4) -> pd.DataFrame:
    """Rolling-origin OOS backtest of AutoARIMA / AutoETS via statsforecast.

    The series modeled is ``y = log(SALEQ)`` so that a seasonal (lag-4) difference
    of the forecast equals the project's seasonal-log-growth target.

    statsforecast's built-in ``cross_validation`` requires every series to be long
    enough for all windows, which fails on short-lived firms. We instead run a
    manual rolling loop: at each annual formation origin (the start of every
    rolling test block), we fit on each firm's history up to that quarter and
    forecast h = 1..H ahead — but only for firms **active at that origin** (whose
    last observation is exactly the origin), so the forecasts target the intended
    quarters. This mirrors the annual-origin Direct multi-step design of the tree
    tier and tolerates the highly unbalanced panel.

    Returns the project's tidy prediction schema in growth space.
    """
    from statsforecast import StatsForecast
    from .mlf import panel_to_runs_long, _ds_to_qid
    from ..pipeline.rolling import generate_windows

    cfg = load_config()
    H = max(horizons)
    min_len = max(2 * season_length, 8)               # need enough history to fit

    long = panel_to_runs_long(panel, "saleq", log_value=True)
    long["qid"] = _ds_to_qid(long["ds"])
    runner = StatsForecastRunner(models=list(models), season_length=season_length)

    # Annual formation origins = the first formation quarter of each rolling block.
    cutoffs = sorted({w.test_qids.start for w in generate_windows(cfg)})
    log.info("statsforecast rolling: %s, %d annual origins, h=%d (slow baseline)",
             list(models), len(cutoffs), H)

    frames = []
    for c in cutoffs:
        hist = long[long["qid"] <= c]
        last = hist.groupby("unique_id")["qid"].transform("max")
        size = hist.groupby("unique_id")["y"].transform("size")
        hist = hist[(last == c) & (size >= min_len)]      # firms active at origin c
        if hist.empty:
            continue
        sf = StatsForecast(models=runner._make_models(), freq="QE", n_jobs=-1)
        fc = sf.forecast(df=hist[["unique_id", "ds", "y"]], h=H).reset_index()
        fc["cutoff_qid"] = c
        frames.append(fc)

    if not frames:
        return pd.DataFrame()
    return _tidy_sf_cv(pd.concat(frames, ignore_index=True), panel, list(models), horizons)


def _tidy_sf_cv(cv: pd.DataFrame, panel: pd.DataFrame, model_names, horizons) -> pd.DataFrame:
    """Convert statsforecast CV output (log-revenue forecasts) into growth-space preds.

    For a target date t the seasonal-log-growth forecast is
    ``pred_log(t) - log(SALEQ_{t-4})`` and the level forecast is ``exp(pred_log)``.
    """
    from .mlf import _ds_to_qid, _UID_SEP

    df = cv.copy()
    # Recover gvkey (strip the run suffix) and the TARGET quarter qid from the
    # canonical QE grid; horizon is the distance from the formation origin.
    df["gvkey"] = df["unique_id"].str.split(_UID_SEP).str[0]
    df["target_qid"] = _ds_to_qid(df["ds"])
    df["horizon"] = df["target_qid"] - df["cutoff_qid"]
    df = df[df["horizon"].isin(list(horizons))]

    # Merge realized growth / reconstruction base at the *target* quarter.
    ref = panel[["gvkey", "qid", "saleq", "saleq_lag4",
                 "seasonal_log_growth", "market_cap", "sic"]].copy()
    ref["gvkey"] = ref["gvkey"].astype(str)
    df = df.merge(ref, left_on=["gvkey", "target_qid"], right_on=["gvkey", "qid"],
                  how="inner")
    df = df[df["saleq_lag4"] > 0]
    log_base = np.log(df["saleq_lag4"].values)

    frames = []
    for name in model_names:
        if name not in df:
            continue
        pred_log = df[name].values
        frames.append(pd.DataFrame({
            # Key by the FORMATION quarter (cutoff_qid), matching the formation-date
            # ML path, so cross-model joins/DM tests align the *same* forecast
            # (made at t for target t+h). Using the target qid here was a bug that
            # misaligned ARIMA/ETS vs ML by h quarters. See the project notes.
            "model": name, "gvkey": df["gvkey"].values, "qid": df["cutoff_qid"].values,
            "horizon": df["horizon"].values,
            "y_true": df["seasonal_log_growth"].values,
            "y_pred": pred_log - log_base,                 # growth space
            "y_pred_naive": 0.0,
            "saleq_future": df["saleq"].values,
            "saleq_pred": np.exp(pred_log),
            "saleq_realbase": df["saleq_lag4"].values,
            "market_cap": df["market_cap"].values, "sic": df["sic"].values,
        }))
    return pd.concat(frames, ignore_index=True).dropna(subset=["y_pred"]).reset_index(drop=True)


def panel_to_statsforecast_long(panel: pd.DataFrame, value_col: str = "saleq") -> pd.DataFrame:
    """Convert the firm-quarter panel into statsforecast long format.

    ``ds`` is the fiscal quarter-end timestamp; ``unique_id`` is the gvkey.
    """
    out = panel[["gvkey", "datadate", value_col]].dropna().copy()
    out = out.rename(columns={"gvkey": "unique_id", "datadate": "ds", value_col: "y"})
    return out.sort_values(["unique_id", "ds"])
