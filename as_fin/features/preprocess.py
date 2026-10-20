"""Look-ahead-free preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """Stateful, fit-on-train / apply-on-test preprocessor."""

    feature_cols: list[str]
    target_col: str
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    standardize: bool = False

    # Learned parameters (populated by fit()).
    winsor_bounds_: dict = field(default_factory=dict)
    means_: dict = field(default_factory=dict)
    stds_: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        cols = self.feature_cols + [self.target_col]
        for c in cols:
            if c not in train:
                continue
            # Coerce to float (many features are nullable Int64) so quantiles are
            # plain floats; drop inf first so bounds are finite; all-NaN columns
            # yield NaN bounds, handled below.
            s = (pd.to_numeric(train[c], errors="coerce").astype("float64")
                 .replace([np.inf, -np.inf], np.nan))
            lo, hi = s.quantile(self.winsor_lower), s.quantile(self.winsor_upper)
            lo = 0.0 if pd.isna(lo) else float(lo)
            hi = 0.0 if pd.isna(hi) else float(hi)
            self.winsor_bounds_[c] = (lo, hi)
        if self.standardize:
            for c in self.feature_cols:
                if c not in train:
                    continue
                clipped = (pd.to_numeric(train[c], errors="coerce").astype("float64")
                           .replace([np.inf, -np.inf], np.nan).clip(*self.winsor_bounds_[c]))
                mean = clipped.mean()
                std = clipped.std()
                # An all-NaN column in this window -> neutral standardization.
                self.means_[c] = 0.0 if pd.isna(mean) else float(mean)
                std = 0.0 if pd.isna(std) else float(std)
                self.stds_[c] = std if std > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame, *, is_target: bool = True) -> pd.DataFrame:
        """Apply winsorization, imputation and (optional) standardization.

        ``is_target=False`` skips touching the target column (e.g. for the test
        rows whose realized target must stay raw for evaluation).
        """
        out = df.copy()
        # Coerce to float first: many engineered features are nullable Int64
        # (firm_age, *_isna flags, analyst_coverage, ...) which cannot be clipped
        # with float bounds or hold imputed/standardized values.
        for c in self.feature_cols:
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        for c, (lo, hi) in self.winsor_bounds_.items():
            if c == self.target_col and not is_target:
                continue
            if c in out:
                out[c] = out[c].astype("float64").clip(lo, hi)

        # Zero-impute features (missingness flags already encode the gaps).
        for c in self.feature_cols:
            if c in out:
                out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.standardize:
            for c in self.feature_cols:
                if c in out:
                    out[c] = (out[c] - self.means_[c]) / self.stds_[c]

        # Final safety net: guarantee finite feature inputs for sklearn estimators.
        for c in self.feature_cols:
            if c in out:
                out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)
