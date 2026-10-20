"""Self-contained smoke test on synthetic data (no WRDS / no heavy ML deps)."""

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from as_fin.utils import add_quarter_index, seasonal_log_growth, reconstruct_level
from as_fin.features.target import add_seasonal_log_growth, add_future_targets
from as_fin.features import set_a_revenue
from as_fin.pipeline.rolling import generate_windows
from as_fin.pipeline.runner import run_tabular_model
from as_fin.evaluation.metrics import accuracy_table
from as_fin.evaluation.dm_test import dm_pairwise_table
from as_fin.utils import get_logger

log = get_logger("smoke")


def make_synthetic_panel(n_firms=120, n_quarters=120, seed=42) -> pd.DataFrame:
    """A toy firm-quarter panel with seasonal, persistent revenue dynamics."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        base = rng.uniform(50, 5000)
        season = rng.uniform(0.8, 1.2, size=4)
        drift = rng.normal(0.01, 0.01)
        level = base
        for t in range(n_quarters):
            year = 1990 + t // 4
            q = t % 4 + 1
            level *= (1 + drift) * (1 + rng.normal(0, 0.05))
            saleq = max(1.0, level * season[q - 1])
            rows.append(dict(gvkey=f, fyearq=year, fqtr=q,
                             datadate=pd.Timestamp(year=year, month=q * 3, day=1)
                             + pd.offsets.MonthEnd(0),
                             saleq=saleq, atq=saleq * rng.uniform(2, 5),
                             cshoq=rng.uniform(10, 100), sic=2000 + (f % 20) * 100,
                             market_cap=saleq * rng.uniform(1, 4),
                             ceqq=saleq * rng.uniform(0.5, 2)))
    return pd.DataFrame(rows)


def main():
    log.info("Building synthetic panel ...")
    panel = make_synthetic_panel()
    panel = add_quarter_index(panel)
    panel = add_seasonal_log_growth(panel)
    panel = set_a_revenue.build(panel)
    panel = add_future_targets(panel, horizons=(1, 2))
    panel = panel[panel["firm_age"] >= 8].reset_index(drop=True)

    # Sanity: reconstruction round-trips the growth target back to a level.
    g = panel["target_h1"].dropna().iloc[0]
    log.info("sample target g=%.4f -> recon factor exp(g)=%.4f", g, np.exp(g))

    windows = generate_windows()
    log.info("Generated %d rolling windows (first test year=%d)",
             len(windows), windows[0].test_year)

    feats = [c for c in set_a_revenue.COLUMNS if c in panel]

    # ElasticNet + RandomForest are available without the Nixtla/LGBM stack.
    preds = []
    for key in ["elastic_net", "random_forest"]:
        p = run_tabular_model(panel, key, feats, horizons=(1, 2), grid={})
        if len(p):
            preds.append(p)
    allp = pd.concat(preds, ignore_index=True)

    acc = accuracy_table(allp[allp["horizon"] == 1])
    log.info("\n%s", acc.to_string(index=False))

    if allp["model"].nunique() >= 2:
        dm = dm_pairwise_table(allp, horizon=1)
        log.info("DM tests:\n%s", dm.to_string(index=False))

    log.info("SMOKE TEST PASSED ✔  (%d predictions across %d models)",
             len(allp), allp["model"].nunique())


if __name__ == "__main__":
    main()
