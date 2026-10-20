"""Common interface and registry for cross-sectional (tabular) models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class TabularModel(ABC):
    """Minimal fit/predict interface used by the rolling pipeline."""

    #: human-readable name used in result tables
    name: str = "tabular"
    #: whether this model wants standardized inputs (linear / neural -> True)
    needs_scaling: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.model: Any = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "TabularModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importance(self) -> pd.Series | None:
        """Native importance if the underlying model exposes one (else None)."""
        return None


# Registry maps config keys -> classes so the pipeline can build models by name.
MODEL_REGISTRY: dict[str, type[TabularModel]] = {}


def register(key: str):
    """Class decorator to add a model to the registry."""
    def deco(cls: type[TabularModel]) -> type[TabularModel]:
        MODEL_REGISTRY[key] = cls
        return cls
    return deco


def build_model(key: str, **params: Any) -> TabularModel:
    """Instantiate a registered model by its config key."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Registered: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**params)
