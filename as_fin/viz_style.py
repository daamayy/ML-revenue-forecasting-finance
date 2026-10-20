"""Single source of truth for all figure styling (colors, fonts, format)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# ── brand palette (coastal blues + orange) ───────────────────────────────────
# Five brand hues + a few brand-consistent derived steps ("近似色块") so the
# six-model charts still get colorblind-distinct lines. CVD separation of the
# model set validates at dE 14.4 (scripts/validate_palette.py); the hues are
# intentionally MUTED (low chroma), so the per-model markers below carry
# identity and the pale/light steps are reserved for fills, not thin lines.
PALETTE = {
    "deepblue":  "#3D5A80",   # brand
    "teal":      "#41766F",   # brand
    "lightblue": "#98C1D9",   # brand (light — faint as a line; weakest series only)
    "pale":      "#E0FBFC",   # brand (reserve for fills/spans, never a mark)
    "orange":    "#D1711F",   # brand
    "navy":      "#293241",   # derived dark blue
    "midblue":   "#5E7EA8",   # derived mid blue
    "brown":     "#9C4E1A",   # derived warm (deep orange)
    "amber":     "#E0A44A",   # derived light warm
}
# Order used for any plot that does not name its colors.
CYCLE = [PALETTE[k] for k in ("deepblue", "orange", "teal", "brown",
                              "midblue", "lightblue", "navy", "amber")]
GREY_REFERENCE = "#9AA7B0"   # naive / zero / axis reference lines (cool grey)
GREY_NEUTRAL   = "#5B6570"   # the human analyst benchmark (blue-grey neutral)
NEGATIVE       = PALETTE["brown"]   # loss / drawdown bars
FILL           = PALETTE["pale"]    # background spans / shaded regions

# ── fixed entity -> color (color follows the entity, never its rank) ─────────
# Co-occurring subsets validated all-pairs: the six ML/classical models (dE 12.9),
# the five trees+DL (floor band, markers required), analyst vs LGBM/EN.
MODEL_COLORS = {
    "LightGBM":     PALETTE["deepblue"],  # hero -> strongest brand color
    "XGBoost":      PALETTE["orange"],
    "RandomForest": PALETTE["teal"],
    "ElasticNet":   PALETTE["brown"],
    "AutoETS":      PALETTE["midblue"],   # key classical competitor
    "AutoARIMA":    PALETTE["lightblue"], # weakest series -> the faint light blue
    "LSTM":         PALETTE["navy"],
    "TFT":          PALETTE["amber"],
    "Analyst":      GREY_NEUTRAL,
    "AnalystConsensus": GREY_NEUTRAL,     # alias used in the prediction frames
    "SeasonalNaive": GREY_REFERENCE,
    # ensembles (rarely drawn as colored series; distinct brand-adjacent tones)
    "ML_Ensemble":  PALETTE["navy"],
    "Full_Ensemble": PALETTE["brown"],
    "DL_Ensemble":  PALETTE["teal"],
}

# Distinct marker per model = the secondary encoding that makes a floor-band CVD
# pair legal and keeps series separable in grayscale print.
MODEL_MARKERS = {
    "LightGBM": "o", "XGBoost": "s", "RandomForest": "^", "ElasticNet": "D",
    "AutoETS": "v", "AutoARIMA": "P", "LSTM": "X", "TFT": "*",
    "Analyst": "o", "AnalystConsensus": "o", "SeasonalNaive": None,
    "ML_Ensemble": "o", "Full_Ensemble": "s", "DL_Ensemble": "^",
}

# Role palettes for non-model series. To keep two-color splits consistent
# EVERYWHERE, any 2-way split is deepblue vs orange; a 3rd series adds teal.
EW_VW = {"EW": PALETTE["deepblue"], "VW": PALETTE["orange"]}
IMPORTANCE = {"gain": PALETTE["deepblue"], "permutation": PALETTE["orange"],
              "shap": PALETTE["teal"]}
Q4_BASELINE = {"naive": PALETTE["deepblue"], "arima": PALETTE["orange"],
               "ets": PALETTE["midblue"]}

_FALLBACK = CYCLE


def color_for(name: str) -> str:
    """Stable color for a model/entity; deterministic fallback for unknowns."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return _FALLBACK[hash(name) % len(_FALLBACK)]


def marker_for(name: str):
    return MODEL_MARKERS.get(name, "o")


def apply_style() -> None:
    """Set global rcParams so every figure shares one look (call once)."""
    from cycler import cycler
    mpl.rcParams.update({
        # Any plot that does not name colors draws from the brand cycle
        # (excludes the pale fill), instead of matplotlib's default cycle.
        "axes.prop_cycle": cycler(color=CYCLE),
        # Serif to match the Times New Roman paper body.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times",
                       "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        # Recessive grid + spines (thin marks, quiet chrome).
        "axes.grid": True,
        "grid.color": "#e6e6e3",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8a86",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save_fig(fig, path_stem, *, dpi: int = 200, close: bool = True) -> None:
    """Save a figure as BOTH .png (markdown/preview) and .pdf (vector, LaTeX).

    ``path_stem`` is a path without extension, e.g. ``fig_dir / "fig_q3"``.
    """
    stem = Path(path_stem)
    stem = stem.with_suffix("") if stem.suffix else stem
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
