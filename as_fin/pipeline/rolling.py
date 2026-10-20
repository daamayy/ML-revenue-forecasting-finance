"""Rolling-window split generator."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import load_config


@dataclass
class RollingWindow:
    """One rolling refit: qid ranges for train / validation / test blocks."""

    test_year: int
    train_qids: range
    val_qids: range
    test_qids: range

    def mask(self, qid: pd.Series, which: str) -> pd.Series:
        rng = {"train": self.train_qids, "val": self.val_qids, "test": self.test_qids}[which]
        return qid.between(rng.start, rng.stop - 1)


def _year_to_qid(year: int, quarter: int = 1) -> int:
    """Convert (calendar year, quarter) to the integer qid used in the panel."""
    return year * 4 + (quarter - 1)


def generate_windows(cfg=None) -> list[RollingWindow]:
    """Build the full list of rolling windows from the config (plan 3.5)."""
    cfg = cfg or load_config()
    train_q = cfg.rolling["train_quarters"]
    val_q = cfg.rolling["val_quarters"]
    refit = cfg.rolling["refit_every"]
    first_test = cfg.rolling["first_test_year"]
    last_year = pd.Timestamp(cfg.sample["end_date"]).year

    windows = []
    for year in range(first_test, last_year + 1, refit // 4 if refit >= 4 else 1):
        s = _year_to_qid(year, 1)                     # first formation qid of test block
        train_start = s - val_q - train_q
        train_stop = s - val_q                        # exclusive
        val_start = s - val_q
        val_stop = s                                  # exclusive
        test_start = s
        test_stop = s + refit                         # exclusive (one refit block)
        windows.append(RollingWindow(
            test_year=year,
            train_qids=range(train_start, train_stop),
            val_qids=range(val_start, val_stop),
            test_qids=range(test_start, test_stop),
        ))
    return windows
