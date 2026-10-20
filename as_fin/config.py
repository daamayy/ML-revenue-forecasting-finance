"""Configuration loader."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Code/ directory (parent of the as_fin package).
CODE_ROOT = Path(__file__).resolve().parents[1]
# Repository root (AS_Fin/), one level above Code/.
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_CONFIG_PATH = CODE_ROOT / "config.yaml"


class Config(dict):
    """Dict that also supports attribute access and nested path resolution."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        # Recursively wrap nested dicts so ``cfg.sample.start_date`` works.
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def path(self, key: str) -> Path:
        """Resolve a key under the ``paths`` block to an absolute Path.

        Relative paths in the config are interpreted relative to ``Code/`` so
        that, e.g., ``"../Data/raw"`` points at ``AS_Fin/Data/raw``.
        """
        raw = self["paths"][key]
        p = Path(raw)
        if not p.is_absolute():
            p = (CODE_ROOT / p).resolve()
        return p


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load and cache the project configuration."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)


def ensure_dirs(cfg: Config | None = None) -> None:
    """Create all output / data directories declared in the config."""
    cfg = cfg or load_config()
    for key in cfg["paths"]:
        cfg.path(key).mkdir(parents=True, exist_ok=True)
