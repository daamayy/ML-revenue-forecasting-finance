"""Set C — Market variables from CRSP (~10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import safe_log

COLUMNS = [
    "market_cap", "log_market_cap", "ret_1m", "ret_3m", "ret_12m",
    "ret_vol_12m", "turnover", "turnover_12m", "market_to_book", "mom_12_2",
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # market_cap (in $mn) from collapsed CRSP; mktcap there is price*shrout (000s).
    out["market_cap"] = out.get("mktcap", np.nan) / 1000.0
    out["log_market_cap"] = safe_log(out["market_cap"], log1p_nonneg=True)

    # Market-to-book = market equity / common book equity (Compustat ceqq, $mn).
    book = out["ceqq"].replace(0, np.nan)
    out["market_to_book"] = out["market_cap"] / book

    # The remaining return/turnover/momentum columns come straight from the
    # quarterly CRSP collapse; ensure they exist even if CRSP merge was partial.
    for col in ["ret_1m", "ret_3m", "ret_12m", "ret_vol_12m", "turnover",
                "turnover_12m", "mom_12_2"]:
        if col not in out:
            out[col] = np.nan
    return out
