"""Per-run record keeping: print + persist hyper-parameters and results."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .config import load_config
from .utils import get_logger
from .evaluation.metrics import accuracy_table

log = get_logger(__name__)


def record_run(tag: str, meta: dict, preds: pd.DataFrame,
               weight: str | None = "market_cap") -> dict:
    """Persist a run's settings + accuracy, print them, and index the run.

    Parameters
    ----------
    tag : short label for the run (e.g. "trees_h1", "baselines", "ablation").
    meta : settings to record — models, horizons, tuned hyper-parameters, notes.
    preds : tidy prediction frame (model x gvkey x qid x horizon + y_true/y_pred...).
    weight : column for value-weighted accuracy (None to skip).

    Returns the accuracy table as records (also written to disk).
    """
    cfg = load_config()
    runs_dir = cfg.path("outputs") / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    acc = accuracy_table(preds, weight=weight) if len(preds) else pd.DataFrame()

    # --- log to console + Code/logs/ ---
    log.info("=== RUN RECORD [%s] ===", tag)
    for k, v in meta.items():
        log.info("  %s: %s", k, v)
    if len(acc):
        log.info("Accuracy table:\n%s", acc.round(4).to_string(index=False))

    # --- human-readable per-run file ---
    path = runs_dir / f"{ts}_{tag}.md"
    lines = [f"# Run {ts} — {tag}", ""]
    lines += ["## Settings", "```json", json.dumps(meta, indent=2, default=str), "```", ""]
    if len(acc):
        lines += ["## Accuracy (out-of-sample)", "```",
                  acc.round(4).to_string(index=False), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved run record -> %s", path.name)

    # --- append to comparison index ---
    index = runs_dir / "INDEX.md"
    if not index.exists():
        index.write_text("# Run index (newest at bottom)\n\n"
                         "| timestamp | tag | models | horizons | best MAE (h1) | file |\n"
                         "|---|---|---|---|---|---|\n", encoding="utf-8")
    best_h1 = ""
    if len(acc) and "horizon" in acc:
        h1 = acc[acc["horizon"] == 1]
        if len(h1):
            best = h1.loc[h1["MAE"].idxmin()]
            best_h1 = f"{best['model']}={best['MAE']:.4f}"
    with index.open("a", encoding="utf-8") as fh:
        fh.write(f"| {ts} | {tag} | {meta.get('models','')} | {meta.get('horizons','')} "
                 f"| {best_h1} | {path.name} |\n")

    return acc.to_dict("records") if len(acc) else {}
