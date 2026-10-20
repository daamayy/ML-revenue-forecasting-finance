"""Minimal, consistent logging setup used across the package."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED = False
#: Absolute path of the log file created for the current process (or None).
LOG_PATH: Path | None = None

# Code/ directory = parents[2] of this file (as_fin/utils/logging.py).
_CODE_ROOT = Path(__file__).resolve().parents[2]


def _make_log_path() -> Path:
    """Timestamped log file path: Code/logs/as_fin_YYYYMMDD_HHMMSS.log."""
    log_dir = Path(os.environ.get("AS_FIN_LOG_DIR", _CODE_ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"as_fin_{stamp}.log"


def get_logger(name: str = "as_fin") -> logging.Logger:
    """Return a module logger with shared stdout + file handlers.

    Idempotent: repeated calls do not stack duplicate handlers. The first call
    creates the run's log file (unless ``AS_FIN_LOG_FILE=0``).
    """
    global _CONFIGURED, LOG_PATH
    if not _CONFIGURED:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        root = logging.getLogger("as_fin")
        root.setLevel(logging.INFO)

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        root.addHandler(stream)

        # Durable per-run file sink (full date in the record for cross-day runs).
        if os.environ.get("AS_FIN_LOG_FILE", "1") != "0":
            LOG_PATH = _make_log_path()
            file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"))
            root.addHandler(file_handler)
            root.info("Logging to %s", LOG_PATH)

        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
