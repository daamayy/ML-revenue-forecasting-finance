"""Tier 4 — Deep learning models."""

from __future__ import annotations

import pandas as pd

from ..utils import get_logger

log = get_logger(__name__)


class NeuralForecastRunner:
    """Fit LSTM / TFT on the growth series and forecast horizons 1..H.

    Parameters
    ----------
    model : 'lstm' or 'tft'.
    horizon : maximum forecast horizon H (TFT outputs all 1..H jointly).
    hist_exog : optional list of exogenous feature columns to feed the model.
    """

    def __init__(self, model: str = "lstm", horizon: int = 4,
                 hist_exog: list[str] | None = None, max_steps: int = 500,
                 **hparams):
        self.model_name = model.lower()
        self.horizon = horizon
        self.hist_exog = hist_exog or []
        self.max_steps = max_steps
        self.hparams = hparams
        self.nf = None

    def _build(self):
        from neuralforecast import NeuralForecast
        from neuralforecast.models import LSTM, TFT

        common = dict(h=self.horizon, input_size=self.hparams.get("input_size", 8),
                      max_steps=self.max_steps, scaler_type="robust",
                      hist_exog_list=self.hist_exog or None)
        if self.model_name == "lstm":
            model = LSTM(
                encoder_hidden_size=self.hparams.get("hidden_size", 64),
                encoder_n_layers=self.hparams.get("num_layers", 2),
                **common,
            )
        elif self.model_name == "tft":
            model = TFT(
                hidden_size=self.hparams.get("hidden_size", 64),
                n_head=self.hparams.get("n_heads", 4),
                dropout=self.hparams.get("dropout", 0.1),
                **common,
            )
        else:
            raise ValueError(f"Unknown DL model {self.model_name}")
        return NeuralForecast(models=[model], freq="QE")

    def fit_predict(self, train_long: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``train_long`` (['unique_id','ds','y'] + exog) and forecast H steps."""
        self.nf = self._build()
        self.nf.fit(df=train_long)
        fc = self.nf.predict()
        # neuralforecast >= 1.7 already returns unique_id/ds as columns. Calling
        # reset_index() unconditionally (as older versions required) then injects a
        # spurious integer "index" column 0..N-1 that downstream code picks up as the
        # prediction (the model would appear to "predict row numbers"). Only reset
        # when unique_id is still on the index.
        if "unique_id" not in fc.columns:
            fc = fc.reset_index()
        log.info("%s forecast: %d series", self.model_name, fc["unique_id"].nunique())
        return fc
