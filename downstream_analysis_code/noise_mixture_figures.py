"""
scBONITA cell-level rule/noise mixture — figure generation
===========================================================

Python translation of the R/ggplot2 script
    "scBONITA cell-level rule/noise mixture: selected noise levels"

This module is standalone: it can be run directly from the command line to
reproduce all ten figures, OR imported by the trajectory-analysis pipeline,
which calls `generate_pipeline_panels()` to drop `tent.png` and `PPV.png`
into its output directory for `repatch_figure2_shane_version()`.

Inputs
------
Six ruleset CSV files produced by
    scbonita_cell_level_rule_null_mixture_FIXED.py
named
    positive_control_best_rules_randomness_{0,20,40,60,80,100}pct.csv

By default they are read from a `variable_noise_data` folder sitting next to
this file:

    <folder containing noise_mixture_figures.py>/
        noise_mixture_figures.py
        variable_noise_data/
            positive_control_best_rules_randomness_0pct.csv
            positive_control_best_rules_randomness_20pct.csv
            ...
            positive_control_best_rules_randomness_100pct.csv

Run `python noise_mixture_figures.py --check` to verify that folder before
committing to a full run.

Figures (identical definitions to the R script)
-----------------------------------------------
  1  target ON proportion            vs best-fitting rule error (%)   ["tent"]
  2  target ON proportion            vs PPV = TP / (TP + FP)          ["PPV"]
  3  target ON proportion            vs NPV = TN / (TN + FN)
  4  target ON proportion            vs sensitivity = TP / nON
  5  target ON proportion            vs specificity = TN / nOFF
  6  specificity                     vs sensitivity
  7  false positive rate = FP / nOFF vs sensitivity
  8  target ON proportion            vs fraction of tent-null error eliminated
  9  bar: % of targets where the selected rule was logically equivalent
 10  bar: the same, faceted by true parent count (optional)

Only raw points are drawn — no fitted lines, smoothers, binned medians, or
theoretical curves, matching the R script exactly.

Legend and frame styling
------------------------
`_scatter_by_noise()` accepts two presentation switches:

  box=True        draw all four spines instead of just left and bottom
  legend_kw=...   forwarded to `_noise_legend()`; use
                  dict(inside=True, loc=..., fontsize=..., bold=True)
                  to place the noise-level key inside the plotting area

The PPV panel (Figure 2) uses both: a full box and an inside 18 pt bold
legend in the lower-right corner, which PPV leaves empty because the point
cloud rises with p. Every other figure keeps the original two-spine frame
and the outside legend.

Palettes
--------
`palette=` selects the noise-level color scheme:

  "original"   the six-hue colorblind-friendly ramp from the R script
  "sequential" a single-hue blue ramp, light (0% noise) to dark (100%)
  "grey"       a greyscale ramp

Noise level is an *ordered* variable, so "sequential" is both semantically
truthful and far less colorful — useful when these panels sit next to the
pseudotime (coolwarm / plasma) and condition-bias (red-grey-green) t-SNE
panels in a combined manuscript figure, where a seventh categorical hue set
competes for the reader's attention. `generate_pipeline_panels()` therefore
defaults to "sequential"; pass palette="original" to restore the R colors.

Command line
------------
    # verify the input folder
    python noise_mixture_figures.py --check

    # full figure + table set
    python noise_mixture_figures.py --output-dir /path/to/figures

    # just refresh the two panels the trajectory pipeline repatches
    python noise_mixture_figures.py \
        --pipeline-outdir /path/to/cell_trajectories \
        --only-pipeline-panels

Requires: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ============================================================
# PATHS
# ============================================================

# `__file__` is undefined when this is pasted straight into a notebook cell,
# so fall back to the working directory rather than failing at import time.
try:
    MODULE_DIR = Path(__file__).resolve().parent
except NameError:
    MODULE_DIR = Path.cwd()

# Folder of ruleset CSVs, expected alongside this script.
INPUT_SUBDIR = "variable_noise_data"
DEFAULT_INPUT_DIR = str(MODULE_DIR / INPUT_SUBDIR)

# Full figure/table set goes here unless --output-dir says otherwise.
DEFAULT_OUTPUT_DIR = str(MODULE_DIR / "cell_level_mixture_figures")


# ============================================================
# USER SETTINGS  (mirrors the R script's settings block)
# ============================================================

NOISE_PERCENTAGES = [0, 20, 40, 60, 80, 100]

FILE_TEMPLATE = "positive_control_best_rules_randomness_{pct}pct.csv"

# Sanity checks from the R script. Set either to None to disable.
EXPECTED_N_CANDIDATE_RULES = 94
EXPECTED_N_UNIQUE_CANDIDATE_FUNCTIONS = 94

# ---- Palettes (index-aligned with NOISE_PERCENTAGES) ----
PALETTES = {
    # The exact six hues used by the R script.
    "original": [
        "#264653",  # 0%   deep teal
        "#2A9D8F",  # 20%  teal
        "#8AB17D",  # 40%  sage green
        "#E9C46A",  # 60%  warm gold
        "#F4A261",  # 80%  soft orange
        "#E76F51",  # 100% coral red
    ],
    # Single-hue ordered ramp: light = low noise, dark = high noise.
    "sequential": [
        "#9ECAE1",
        "#6BAED6",
        "#4292C6",
        "#2171B5",
        "#08519C",
        "#08306B",
    ],
    "grey": [
        "#BDBDBD",
        "#9E9E9E",
        "#757575",
        "#525252",
        "#333333",
        "#111111",
    ],
}
DEFAULT_PALETTE = "original"

# Point appearance. High-noise clouds get lower opacity because their
# distributions are broader and would otherwise obscure the others.
POINT_ALPHA_RANGE = (0.28, 0.09)   # first value -> lowest noise level

# ggplot2 `size` is a diameter in mm; matplotlib `s` is an area in points^2.
# 0.62 mm diameter -> ~1.76 pt -> ~2.4 pt^2. Rounded up slightly for parity.
POINT_SIZE_PT2 = 3.0

Y_AXIS_MIN_PERCENT = -2
Y_AXIS_MAX_PERCENT = 52

SCATTER_SIZE = (11.0, 7.5)
PREDICTIVE_VALUE_SIZE = (11.0, 7.5)
SENS_SPEC_SIZE = (11.0, 7.5)
JOINT_SENS_SPEC_SIZE = (8.5, 8.5)
FPR_SENS_SIZE = (8.5, 8.5)
NORMALIZED_SKILL_SIZE = (11.0, 7.5)
BAR_SIZE = (11.0, 6.5)
PARENT_BAR_SIZE = (12.0, 7.0)
FIGURE_DPI = 300

SHOW_BAR_LABELS = True
BAR_LABEL_DECIMALS = 2          # R: BAR_LABEL_ACCURACY = 0.01
SAVE_PARENT_COUNT_FIGURE = True

# ---- Inside-legend styling for the PPV panel ----
# PPV rises with p, so the cloud hugs the diagonal band and leaves the
# lower-right corner clear. Change PPV_LEGEND_LOC if your data fills it.
PPV_LEGEND_INSIDE = True
PPV_LEGEND_LOC = "lower right"
PPV_LEGEND_FONTSIZE = 18
PPV_LEGEND_BOLD = True
PPV_BOX_SPINES = True

# Sizes used for the two panels handed to the trajectory pipeline. The
# repatch grid rescales images anyway, but keeping them equal in aspect
# ratio stops one panel from being letterboxed next to the other.
PIPELINE_PANEL_SIZE = (8.0, 6.0)


# ---- Output filenames (unchanged from the R script) ----
OUT_NAMES = {
    "scatter":      "noise_0_20_40_60_80_100_on_fraction_vs_rule_error_points.png",
    "ppv":          "noise_0_20_40_60_80_100_on_fraction_vs_selected_rule_PPV_points.png",
    "npv":          "noise_0_20_40_60_80_100_on_fraction_vs_selected_rule_NPV_points.png",
    "sensitivity":  "noise_0_20_40_60_80_100_on_fraction_vs_selected_rule_sensitivity_points.png",
    "specificity":  "noise_0_20_40_60_80_100_on_fraction_vs_selected_rule_specificity_points.png",
    "joint":        "noise_0_20_40_60_80_100_specificity_vs_sensitivity_points.png",
    "fpr":          "noise_0_20_40_60_80_100_false_positive_rate_vs_sensitivity_points.png",
    "skill":        "noise_0_20_40_60_80_100_on_fraction_vs_fraction_null_error_eliminated.png",
    "recovery_bar": "noise_0_20_40_60_80_100_functional_rule_recovery_percent.png",
    "parent_bar":   "noise_0_20_40_60_80_100_recovery_by_parent_count.png",
    "combined_data":     "noise_0_20_40_60_80_100_combined_plot_data.csv.gz",
    "recovery_summary":  "noise_0_20_40_60_80_100_functional_recovery_summary.csv",
    "parent_summary":    "noise_0_20_40_60_80_100_recovery_by_parent_count.csv",
}

REQUIRED_COLUMNS = [
    "target_on_fraction",
    "target_on_count",
    "n_cells",
    "best_rule_error_percent",
    "selected_rule_tp",
    "selected_rule_fp",
    "selected_rule_tn",
    "selected_rule_fn",
    "selected_rule_ppv",
    "selected_rule_npv",
    "selected_functional_match",
    "n_candidate_rules",
    "n_unique_candidate_functions",
]


# ============================================================
# HELPERS
# ============================================================

def _header(text: str, verbose: bool = True) -> None:
    if verbose:
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)


def noise_labels(noise_percentages=None) -> list:
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    return [f"{p}% noise" for p in pcts]


def noise_colors(palette: str = DEFAULT_PALETTE, noise_percentages=None) -> dict:
    """Map noise label -> hex color for the requested palette."""
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    if palette not in PALETTES:
        raise ValueError(f"Unknown palette '{palette}'. "
                         f"Choose from {sorted(PALETTES)}.")
    base = PALETTES[palette]
    if len(pcts) > len(base):
        raise ValueError(f"Palette '{palette}' defines {len(base)} colors but "
                         f"{len(pcts)} noise levels were requested.")
    # Spread the ramp across however many levels were asked for.
    if len(pcts) == len(base):
        chosen = base
    else:
        idx = np.linspace(0, len(base) - 1, len(pcts)).round().astype(int)
        chosen = [base[i] for i in idx]
    return dict(zip(noise_labels(pcts), chosen))


def noise_alphas(noise_percentages=None) -> dict:
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    vals = np.linspace(POINT_ALPHA_RANGE[0], POINT_ALPHA_RANGE[1], len(pcts))
    return dict(zip(noise_labels(pcts), vals))


def _parse_logical(series: pd.Series) -> pd.Series:
    """R's parse_logical_column: tolerant true/false parsing -> object dtype
    holding True / False / np.nan."""
    if series.dtype == bool:
        return series.astype(object)

    def _one(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
        return np.nan

    return series.map(_one).astype(object)


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Numeric column if present, otherwise a full-length column of `default`."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _chr(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype("string")
    return pd.Series(pd.NA, index=df.index, dtype="string")


def _logical(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return _parse_logical(df[col])
    return pd.Series(np.nan, index=df.index, dtype=object)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division returning NaN wherever the denominator is <= 0."""
    out = pd.Series(np.nan, index=num.index, dtype=float)
    ok = den > 0
    out.loc[ok] = num.loc[ok] / den.loc[ok]
    return out


def _true_count(series: pd.Series) -> int:
    """Number of entries that are exactly True (R's `x %in% TRUE`)."""
    return int(sum(v is True or v == 1 for v in series if not pd.isna(v)))


def _pct_true(series: pd.Series) -> float:
    """Percent True among non-missing entries; NaN if nothing is evaluable."""
    evaluable = [v for v in series if not pd.isna(v)]
    if not evaluable:
        return np.nan
    return 100.0 * sum(bool(v) for v in evaluable) / len(evaluable)


def noise_file_paths(input_dir=None, noise_percentages=None) -> dict:
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    input_dir = Path(DEFAULT_INPUT_DIR if input_dir is None else input_dir)
    return {f"{p}% noise": input_dir / FILE_TEMPLATE.format(pct=p) for p in pcts}


# ============================================================
# INPUT INSPECTION
# ============================================================

def check_input_dir(input_dir=None, noise_percentages=None,
                    verbose: bool = True) -> dict:
    """
    Report on the ruleset folder without reading any data.

    Returns {"dir": Path, "exists": bool, "found": [...], "missing": [...],
             "extra": [...], "ok": bool}.

    The `extra` list is the useful part when something is wrong: if the CSV
    names are close but not exact — a stray suffix, a different percentage
    format — the loader would otherwise just raise on the first missing level
    and leave you guessing which file it wanted.
    """
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    data_dir = Path(DEFAULT_INPUT_DIR if input_dir is None else input_dir)

    if verbose:
        print(f"Looking in: {data_dir}")
        print(f"Exists: {data_dir.is_dir()}\n")

    result = {"dir": data_dir, "exists": data_dir.is_dir(),
              "found": [], "missing": [], "extra": [], "ok": False}

    if not data_dir.is_dir():
        if verbose:
            print("  Directory not found — check the folder name and location.")
            print(f"  Expected a '{INPUT_SUBDIR}' folder beside this script:")
            print(f"    {MODULE_DIR / INPUT_SUBDIR}")
        return result

    expected = {FILE_TEMPLATE.format(pct=p) for p in pcts}
    present = {f.name for f in data_dir.glob("*.csv")}

    for p in pcts:
        name = FILE_TEMPLATE.format(pct=p)
        f = data_dir / name
        if f.exists():
            result["found"].append(name)
            if verbose:
                print(f"  [ok]      {name}  ({f.stat().st_size / 1e6:.1f} MB)")
        else:
            result["missing"].append(name)
            if verbose:
                print(f"  [MISSING] {name}")

    result["extra"] = sorted(present - expected)
    if result["extra"] and verbose:
        print(f"\n  Other CSVs present (ignored): {result['extra']}")

    result["ok"] = not result["missing"]
    if verbose:
        print(f"\n  {len(result['found'])}/{len(pcts)} expected files present.")
    return result


# ============================================================
# THEME  (theme_classic(base_size = 14) equivalent)
# ============================================================

def _apply_theme(ax, base_size: int = 14, box: bool = False) -> None:
    """theme_classic() by default; `box=True` closes the frame on all four
    sides, which is what the PPV panel uses so its inside legend reads as
    sitting within a bounded field rather than floating."""
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False)

    for side in ("top", "right"):
        ax.spines[side].set_visible(box)

    sides = ("left", "bottom", "top", "right") if box else ("left", "bottom")
    for side in sides:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="both", which="major",
                   labelsize=base_size * 0.85, colors="black",
                   direction="out", length=4, width=0.9)
    ax.xaxis.label.set_size(base_size)
    ax.yaxis.label.set_size(base_size)


def _titles(ax, title: str, subtitle: str | None, base_size: int = 14) -> None:
    if not title and not subtitle:
        return
    pad = 24 if subtitle else 10
    if title:
        ax.set_title(title, fontsize=base_size * 1.05, fontweight="bold",
                     loc="left", pad=pad)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes,
                fontsize=10, ha="left", va="bottom", color="black")


def _decimal_formatter(decimals: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: f"{v:.{decimals}f}")


def _percent_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: f"{v * 100:.0f}%")


def _noise_legend(ax, colors: dict, legend_order: list,
                  title: str = "Noise level", *,
                  inside: bool = False,
                  loc: str = "upper left",
                  bbox_to_anchor=None,
                  fontsize: float = 11,
                  title_fontsize: float | None = None,
                  bold: bool = False,
                  frameon: bool = False,
                  markersize: float | None = None):
    """guide_legend(override.aes = list(alpha = 1, size = 3)).

    Default: outside the axes on the upper right, 11 pt, regular weight.
    Pass inside=True to drop the bbox anchor and let `loc` place the key
    within the plotting area instead.
    """
    if markersize is None:
        markersize = max(6.0, fontsize * 0.5)
    if title_fontsize is None:
        title_fontsize = fontsize * 1.05
    if bbox_to_anchor is None and not inside:
        bbox_to_anchor = (1.01, 1.0)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=markersize,
               markerfacecolor=colors[lbl], markeredgecolor=colors[lbl],
               alpha=1.0, label=lbl)
        for lbl in legend_order
    ]
    leg = ax.legend(handles=handles, title=title, loc=loc,
                    bbox_to_anchor=bbox_to_anchor, frameon=frameon,
                    fontsize=fontsize,
                    borderaxespad=0.8 if inside else 0.0,
                    labelspacing=0.35, handletextpad=0.5)

    if bold:
        for txt in leg.get_texts():
            txt.set_fontweight("bold")
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontsize(title_fontsize)
    return leg


def _save(fig, out_path, verbose: bool = True) -> str:
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    if verbose:
        print(f"Saved figure:\n {out_path}")
    return out_path


# ============================================================
# READING
# ============================================================

def read_noise_file(file_path, noise_label: str, expected_noise_percent: float,
                    verbose: bool = True) -> pd.DataFrame:
    """Read one ruleset CSV and return the tidy per-target frame."""
    if verbose:
        print(f"\nReading {noise_label}:\n{file_path}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Noise ruleset: {noise_label} not found:\n{file_path}")

    raw = pd.read_csv(file_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{noise_label} is missing required columns: "
                         f"{', '.join(missing)}")

    if EXPECTED_N_CANDIDATE_RULES is not None:
        vals = pd.to_numeric(raw["n_candidate_rules"], errors="coerce").dropna()
        if (vals != EXPECTED_N_CANDIDATE_RULES).any():
            raise ValueError(f"{noise_label} was not generated with "
                             f"{EXPECTED_N_CANDIDATE_RULES} candidate rules.")

    if EXPECTED_N_UNIQUE_CANDIDATE_FUNCTIONS is not None:
        vals = pd.to_numeric(raw["n_unique_candidate_functions"],
                             errors="coerce").dropna()
        if (vals != EXPECTED_N_UNIQUE_CANDIDATE_FUNCTIONS).any():
            raise ValueError(f"{noise_label} was not generated with "
                             f"{EXPECTED_N_UNIQUE_CANDIDATE_FUNCTIONS} "
                             f"distinct truth tables.")

    n_cells = _num(raw, "n_cells")
    n_on = _num(raw, "target_on_count")
    n_off = n_cells - n_on

    tp = _num(raw, "selected_rule_tp")
    fp = _num(raw, "selected_rule_fp")
    tn = _num(raw, "selected_rule_tn")
    fn = _num(raw, "selected_rule_fn")

    if "synthetic_id" in raw.columns:
        synthetic_id = pd.to_numeric(raw["synthetic_id"],
                                     errors="coerce").astype("Int64")
    else:
        synthetic_id = pd.Series(np.arange(1, len(raw) + 1), index=raw.index,
                                 dtype="Int64")

    out = pd.DataFrame({
        "synthetic_id": synthetic_id,
        "target_on_fraction": _num(raw, "target_on_fraction"),
        "n_cells": n_cells,
        "nON": n_on,
        "nOFF": n_off,
        "best_rule_error_percent": _num(raw, "best_rule_error_percent"),
        "selected_rule_tp": tp,
        "selected_rule_fp": fp,
        "selected_rule_tn": tn,
        "selected_rule_fn": fn,
        "selected_rule_ppv": _num(raw, "selected_rule_ppv"),
        "selected_rule_npv": _num(raw, "selected_rule_npv"),
        "selected_rule_sensitivity": _safe_div(tp, n_on),
        "selected_rule_specificity": _safe_div(tn, n_off),
        "selected_rule_false_positive_rate": _safe_div(fp, n_off),
        "noise_level_percent_from_file": _num(raw, "randomness_level_percent"),
        "realized_noise_percent": _num(raw, "realized_randomness_percent"),
        "true_rule_observed_error_percent": _num(raw, "true_rule_observed_error_percent"),
        "majority_null_error_percent": _num(raw, "majority_null_error_percent"),
        "true_parent_count": (
            pd.to_numeric(raw["true_parent_count"], errors="coerce").astype("Int64")
            if "true_parent_count" in raw.columns
            else pd.Series(pd.NA, index=raw.index, dtype="Int64")
        ),
        "selected_functional_match": _logical(raw, "selected_functional_match"),
        "selected_exact_rule_match": _logical(raw, "selected_exact_rule_match"),
        "true_function_among_tied_best": _logical(raw, "true_function_among_tied_best"),
        "n_tied_best_rules": (
            pd.to_numeric(raw["n_tied_best_rules"], errors="coerce").astype("Int64")
            if "n_tied_best_rules" in raw.columns
            else pd.Series(pd.NA, index=raw.index, dtype="Int64")
        ),
        "true_rule": _chr(raw, "true_rule"),
        "selected_rule": _chr(raw, "selected_rule"),
        "noise_label": noise_label,
        "noise_percent": float(expected_noise_percent),
    })

    keep = (
        np.isfinite(out["target_on_fraction"])
        & np.isfinite(out["best_rule_error_percent"])
        & out["target_on_fraction"].between(0, 1)
        & (out["best_rule_error_percent"] >= 0)
    )
    out = out.loc[keep].reset_index(drop=True)

    supplied = out["noise_level_percent_from_file"]
    supplied = supplied[np.isfinite(supplied)]
    if len(supplied) and (np.abs(supplied - expected_noise_percent) > 1e-8).any():
        warnings.warn(
            f"{noise_label} contains source randomness_level_percent values "
            f"that do not match the expected noise level "
            f"{expected_noise_percent}%"
        )

    if verbose:
        print(f"Observations retained: {len(out):,}")
    return out


def load_combined(input_dir=None, noise_percentages=None,
                  verbose: bool = True) -> pd.DataFrame:
    """Read every noise level and return the combined frame with the
    tent-null error and normalized-skill columns added."""
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    labels = noise_labels(pcts)
    input_dir = Path(DEFAULT_INPUT_DIR if input_dir is None else input_dir)
    paths = noise_file_paths(input_dir, pcts)

    _header("Checking input files", verbose)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Ruleset directory not found:\n{input_dir}\n"
            f"Expected a '{INPUT_SUBDIR}' folder beside "
            f"noise_mixture_figures.py, or pass input_dir=..."
        )
    for lbl in labels:
        p = paths[lbl]
        if not Path(p).exists():
            raise FileNotFoundError(f"Noise ruleset: {lbl} not found:\n{p}")
    if verbose:
        print(f"Ruleset directory:\n  {input_dir}")
        print("Noise ruleset files:")
        for lbl in labels:
            print(f"  {lbl}:\n    {paths[lbl].name}")

    _header("Reading noise-series rulesets", verbose)
    frames = [
        read_noise_file(paths[lbl], lbl, pct, verbose=verbose)
        for lbl, pct in zip(labels, pcts)
    ]

    combined = pd.concat(frames, ignore_index=True)
    combined["noise_label"] = pd.Categorical(combined["noise_label"],
                                             categories=labels, ordered=True)

    p = combined["target_on_fraction"]
    combined["tent_null_error_percent"] = 100.0 * np.minimum(p, 1.0 - p)
    combined["fraction_null_error_eliminated"] = _safe_div(
        combined["tent_null_error_percent"] - combined["best_rule_error_percent"],
        combined["tent_null_error_percent"],
    )
    combined["percent_null_error_eliminated"] = (
        100.0 * combined["fraction_null_error_eliminated"]
    )

    if verbose:
        print(f"\nTotal observations: {len(combined):,}")
        print(combined["noise_label"].value_counts().sort_index().to_string())

    return combined


# ============================================================
# SUMMARIES
# ============================================================

def summarize_functional_recovery(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, pct), g in combined.groupby(["noise_label", "noise_percent"],
                                            observed=True, sort=False):
        fm = g["selected_functional_match"]
        n_evaluable = int(fm.notna().sum())
        n_matches = _true_count(fm)
        rows.append({
            "noise_label": label,
            "noise_percent": pct,
            "n_synthetic_targets": len(g),
            "n_functionally_evaluable": n_evaluable,
            "functional_matches": n_matches,
            "functional_match_percent": (100.0 * n_matches / n_evaluable
                                         if n_evaluable > 0 else np.nan),
            "exact_matches": _true_count(g["selected_exact_rule_match"]),
            "exact_match_percent": _pct_true(g["selected_exact_rule_match"]),
            "true_function_among_tied_best_percent":
                _pct_true(g["true_function_among_tied_best"]),
            "mean_best_rule_error_percent": g["best_rule_error_percent"].mean(),
            "median_best_rule_error_percent": g["best_rule_error_percent"].median(),
            "mean_true_rule_observed_error_percent":
                g["true_rule_observed_error_percent"].mean(),
            "mean_realized_noise_percent": g["realized_noise_percent"].mean(),
        })
    return (pd.DataFrame(rows)
            .sort_values("noise_percent")
            .reset_index(drop=True))


def summarize_by_parent_count(combined: pd.DataFrame) -> pd.DataFrame:
    sub = combined[combined["true_parent_count"].notna()]
    rows = []
    for (label, pct, parents), g in sub.groupby(
            ["noise_label", "noise_percent", "true_parent_count"],
            observed=True, sort=False):
        rows.append({
            "noise_label": label,
            "noise_percent": pct,
            "true_parent_count": int(parents),
            "n_synthetic_targets": len(g),
            "functional_matches": _true_count(g["selected_functional_match"]),
            "functional_match_percent": _pct_true(g["selected_functional_match"]),
            "exact_match_percent": _pct_true(g["selected_exact_rule_match"]),
            "mean_best_rule_error_percent": g["best_rule_error_percent"].mean(),
        })
    if not rows:
        return pd.DataFrame(columns=["noise_label", "noise_percent",
                                     "true_parent_count", "n_synthetic_targets",
                                     "functional_matches",
                                     "functional_match_percent",
                                     "exact_match_percent",
                                     "mean_best_rule_error_percent"])
    return (pd.DataFrame(rows)
            .sort_values(["noise_percent", "true_parent_count"])
            .reset_index(drop=True))


# ============================================================
# CORE SCATTER BUILDER
# ============================================================

def _scatter_by_noise(combined: pd.DataFrame, x: str, y: str, *,
                      out_path, title: str, subtitle: str | None,
                      xlab: str, ylab: str,
                      xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                      xticks=None, yticks=None,
                      x_decimals: int = 1, y_decimals: int = 1,
                      y_percent: bool = False,
                      clamp_x_unit: bool = False, clamp_y_unit: bool = False,
                      figsize=SCATTER_SIZE, equal_aspect: bool = False,
                      hline_at=None,
                      palette: str = DEFAULT_PALETTE,
                      noise_percentages=None,
                      legend: bool = True,
                      box: bool = False,
                      legend_kw: dict | None = None,
                      verbose: bool = True):
    """
    One raw-point scatter with one color per noise level.

    High-noise clouds are drawn first and low-noise clouds last (R's
    PLOT_ORDER = rev(NOISE_LABELS)), so the sparser low-noise points stay
    visible on top of the broader high-noise distributions.

    `box=True` draws all four spines. `legend_kw` is forwarded to
    `_noise_legend()` — e.g. dict(inside=True, loc="lower right",
    fontsize=18, bold=True) for an inside key.
    """
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    legend_order = noise_labels(pcts)
    plot_order = list(reversed(legend_order))
    colors = noise_colors(palette, pcts)
    alphas = noise_alphas(pcts)

    fig, ax = plt.subplots(figsize=figsize)

    if hline_at is not None:
        ax.axhline(hline_at, linestyle="--", linewidth=0.6, color="#595959",
                   zorder=1)

    for label in plot_order:
        g = combined[combined["noise_label"] == label]
        keep = np.isfinite(g[x]) & np.isfinite(g[y])
        if clamp_x_unit:
            keep &= g[x].between(0, 1)
        if clamp_y_unit:
            keep &= g[y].between(0, 1)
        g = g.loc[keep]
        if g.empty:
            continue
        ax.scatter(g[x], g[y], s=POINT_SIZE_PT2, c=colors[label],
                   alpha=float(alphas[label]), linewidths=0, edgecolors="none",
                   rasterized=True, zorder=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.xaxis.set_major_formatter(_decimal_formatter(x_decimals))
    if y_percent:
        ax.yaxis.set_major_formatter(_percent_formatter())
    elif y_decimals is not None:
        ax.yaxis.set_major_formatter(_decimal_formatter(y_decimals))

    ax.set_xlabel(xlab, fontweight="normal")
    ax.set_ylabel(ylab, fontweight="normal")
    _apply_theme(ax, box=box)
    if legend:
        _noise_legend(ax, colors, legend_order, **(legend_kw or {}))

    return _save(fig, out_path, verbose)


# ============================================================
# FIGURES 1-8
# ============================================================

def plot_error_scatter(combined, out_path, *, palette=DEFAULT_PALETTE,
                       noise_percentages=None, figsize=SCATTER_SIZE,
                       legend=True, verbose=True):
    """Figure 1 — target ON proportion vs best-fitting rule error (%).

    This is the panel the trajectory pipeline repatches as `tent.png`: the
    point cloud is bounded above by the tent function 100*min(p, 1-p).
    """
    return _scatter_by_noise(
        combined, "target_on_fraction", "best_rule_error_percent",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Best-fitting rule error (%)",
        xlim=(0.0, 1.0), ylim=(Y_AXIS_MIN_PERCENT, Y_AXIS_MAX_PERCENT),
        xticks=np.arange(0.0, 1.01, 0.1),
        yticks=[Y_AXIS_MIN_PERCENT] + list(range(0, Y_AXIS_MAX_PERCENT + 1, 5)),
        x_decimals=1, y_decimals=0,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        legend=legend, verbose=verbose,
    )


def plot_ppv_scatter(combined, out_path, *, palette=DEFAULT_PALETTE,
                     noise_percentages=None, figsize=PREDICTIVE_VALUE_SIZE,
                     legend=True,
                     legend_loc=None, legend_fontsize=None,
                     box=None, verbose=True):
    """Figure 2 — target ON proportion vs PPV = TP / (TP + FP).

    Unlike the other panels, the noise-level key sits inside the axes at
    18 pt bold and the frame is closed on all four sides.

    The three style arguments default to None and fall back to the module
    constants at call time rather than at import time, so reassigning
    PPV_LEGEND_FONTSIZE / PPV_LEGEND_LOC / PPV_BOX_SPINES still takes effect.
    """
    legend_loc = PPV_LEGEND_LOC if legend_loc is None else legend_loc
    legend_fontsize = (PPV_LEGEND_FONTSIZE if legend_fontsize is None
                       else legend_fontsize)
    box = PPV_BOX_SPINES if box is None else box

    return _scatter_by_noise(
        combined, "target_on_fraction", "selected_rule_ppv",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Positive predictive value (PPV)",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_y_unit=True,
        box=box,
        legend_kw=dict(inside=PPV_LEGEND_INSIDE, loc=legend_loc,
                       fontsize=legend_fontsize,
                       title_fontsize=legend_fontsize,
                       bold=PPV_LEGEND_BOLD, markersize=10),
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        legend=legend, verbose=verbose,
    )


def plot_npv_scatter(combined, out_path, *, palette=DEFAULT_PALETTE,
                     noise_percentages=None, figsize=PREDICTIVE_VALUE_SIZE,
                     verbose=True):
    """Figure 3 — target ON proportion vs NPV = TN / (TN + FN)."""
    return _scatter_by_noise(
        combined, "target_on_fraction", "selected_rule_npv",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Negative predictive value (NPV)",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_y_unit=True,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


def plot_sensitivity_scatter(combined, out_path, *, palette=DEFAULT_PALETTE,
                             noise_percentages=None, figsize=SENS_SPEC_SIZE,
                             verbose=True):
    """Figure 4 — target ON proportion vs sensitivity = TP / nON."""
    return _scatter_by_noise(
        combined, "target_on_fraction", "selected_rule_sensitivity",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Sensitivity = TP / nON",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_y_unit=True,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


def plot_specificity_scatter(combined, out_path, *, palette=DEFAULT_PALETTE,
                             noise_percentages=None, figsize=SENS_SPEC_SIZE,
                             verbose=True):
    """Figure 5 — target ON proportion vs specificity = TN / nOFF."""
    return _scatter_by_noise(
        combined, "target_on_fraction", "selected_rule_specificity",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Specificity = TN / nOFF",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_y_unit=True,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


def plot_joint_sens_spec(combined, out_path, *, palette=DEFAULT_PALETTE,
                         noise_percentages=None, figsize=JOINT_SENS_SPEC_SIZE,
                         verbose=True):
    """Figure 6 — specificity vs sensitivity."""
    return _scatter_by_noise(
        combined, "selected_rule_specificity", "selected_rule_sensitivity",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Specificity = TN / nOFF",
        ylab="Sensitivity = TP / nON",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_x_unit=True, clamp_y_unit=True, equal_aspect=True,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


def plot_fpr_sensitivity(combined, out_path, *, palette=DEFAULT_PALETTE,
                         noise_percentages=None, figsize=FPR_SENS_SIZE,
                         verbose=True):
    """Figure 7 — false positive rate vs sensitivity."""
    return _scatter_by_noise(
        combined, "selected_rule_false_positive_rate", "selected_rule_sensitivity",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="False positive rate = FP / nOFF",
        ylab="Sensitivity = TP / nON",
        xticks=np.arange(0.0, 1.01, 0.1), yticks=np.arange(0.0, 1.01, 0.1),
        clamp_x_unit=True, clamp_y_unit=True, equal_aspect=True,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


def plot_null_error_eliminated(combined, out_path, *, palette=DEFAULT_PALETTE,
                               noise_percentages=None,
                               figsize=NORMALIZED_SKILL_SIZE, verbose=True):
    """Figure 8 — fraction of tent-function null error eliminated.

    1 means the rule eliminates all of the null error; 0 means it performs
    exactly at the tent-function null; negative means worse than the null.
    Cases where the null error is exactly 0 (p = 0 or p = 1) are undefined
    and were dropped when the column was built. Display is clipped to
    -100%..100%.
    """
    return _scatter_by_noise(
        combined, "target_on_fraction", "fraction_null_error_eliminated",
        out_path=out_path,
        title="",
        subtitle=None,
        xlab="Target ON proportion, p",
        ylab="Fraction of null error eliminated",
        xlim=(0.0, 1.0), ylim=(-1.0, 1.0),
        xticks=np.arange(0.0, 1.01, 0.1),
        yticks=np.arange(-1.0, 1.01, 0.25),
        y_percent=True, hline_at=0.0,
        figsize=figsize, palette=palette, noise_percentages=noise_percentages,
        verbose=verbose,
    )


# ============================================================
# FIGURES 9-10  (bars)
# ============================================================

def plot_recovery_bar(recovery_summary, out_path, *, palette=DEFAULT_PALETTE,
                      noise_percentages=None, figsize=BAR_SIZE,
                      show_labels=SHOW_BAR_LABELS, verbose=True):
    """Figure 9 — % of targets whose selected rule was logically equivalent
    to the true generating rule. Bar colors match the point colors."""
    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    legend_order = noise_labels(pcts)
    colors = noise_colors(palette, pcts)

    df = recovery_summary.set_index(
        recovery_summary["noise_label"].astype(str)).reindex(legend_order)
    values = df["functional_match_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(legend_order))
    ax.bar(xs, np.nan_to_num(values, nan=0.0), width=0.75,
           color=[colors[l] for l in legend_order],
           edgecolor="black", linewidth=0.9)

    if show_labels:
        for x, v in zip(xs, values):
            if np.isfinite(v):
                ax.text(x, v + 1.0, f"{v:.{BAR_LABEL_DECIMALS}f}%",
                        ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(legend_order, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.set_xlabel("Cell-level noise probability")
    ax.set_ylabel("Functionally equivalent rules recovered (%)")
    _apply_theme(ax)
    return _save(fig, out_path, verbose)


def plot_recovery_by_parent(parent_summary, out_path, *, palette=DEFAULT_PALETTE,
                            noise_percentages=None, figsize=PARENT_BAR_SIZE,
                            verbose=True):
    """Figure 10 — the same recovery percentage, faceted by true parent count."""
    if parent_summary is None or parent_summary.empty:
        if verbose:
            print("No true_parent_count data available — skipping "
                  "recovery-by-parent-count figure.")
        return None

    pcts = NOISE_PERCENTAGES if noise_percentages is None else noise_percentages
    legend_order = noise_labels(pcts)
    colors = noise_colors(palette, pcts)

    facets = sorted(parent_summary["true_parent_count"].unique())
    facet_titles = {1: "True rule uses 1 parent",
                    2: "True rule uses 2 parents",
                    3: "True rule uses 3 parents"}

    fig, axes = plt.subplots(len(facets), 1, figsize=figsize, sharex=True)
    if len(facets) == 1:
        axes = [axes]

    xs = np.arange(len(legend_order))
    for ax, parents in zip(axes, facets):
        sub = parent_summary[parent_summary["true_parent_count"] == parents]
        sub = sub.set_index(sub["noise_label"].astype(str)).reindex(legend_order)
        vals = sub["functional_match_percent"].to_numpy(dtype=float)
        ax.bar(xs, np.nan_to_num(vals, nan=0.0), width=0.75,
               color=[colors[l] for l in legend_order],
               edgecolor="black", linewidth=0.7)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
        ax.set_ylabel("")
        _apply_theme(ax)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(legend_order, rotation=45, ha="right")
    axes[-1].set_xlabel("Cell-level noise probability")
    fig.supylabel("Functionally equivalent rules recovered (%)", fontsize=14)
    fig.suptitle("Functional rule recovery by true rule complexity",
                 fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    return _save(fig, out_path, verbose)


# ============================================================
# DRIVERS
# ============================================================

def generate_all_figures(input_dir=None,
                         output_dir=None,
                         *, palette=DEFAULT_PALETTE,
                         noise_percentages=None,
                         save_parent_count_figure=SAVE_PARENT_COUNT_FIGURE,
                         save_tables=True,
                         combined=None,
                         verbose=True) -> dict:
    """Reproduce every figure and table from the R script.

    Returns a dict of {key: path} for everything written, plus the combined
    frame and both summary frames under the keys 'combined_df',
    'recovery_summary_df' and 'parent_count_summary_df'.
    """
    input_dir = DEFAULT_INPUT_DIR if input_dir is None else input_dir
    output_dir = Path(DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if combined is None:
        combined = load_combined(input_dir, noise_percentages, verbose=verbose)

    out = {}

    if save_tables:
        p = output_dir / OUT_NAMES["combined_data"]
        combined.to_csv(p, index=False)
        out["combined_data"] = str(p)
        if verbose:
            print(f"Saved combined data:\n {p}")

    _header("Calculating functional-rule recovery", verbose)
    recovery = summarize_functional_recovery(combined)
    parent = summarize_by_parent_count(combined)

    if save_tables:
        p = output_dir / OUT_NAMES["recovery_summary"]
        recovery.to_csv(p, index=False)
        out["recovery_summary"] = str(p)
        p = output_dir / OUT_NAMES["parent_summary"]
        parent.to_csv(p, index=False)
        out["parent_summary"] = str(p)
    if verbose:
        print(recovery.to_string(index=False))

    kw = dict(palette=palette, noise_percentages=noise_percentages,
              verbose=verbose)

    _header("Creating raw-point noise figure", verbose)
    out["scatter"] = plot_error_scatter(
        combined, output_dir / OUT_NAMES["scatter"], **kw)

    _header("Creating selected-rule PPV figure", verbose)
    out["ppv"] = plot_ppv_scatter(
        combined, output_dir / OUT_NAMES["ppv"], **kw)

    _header("Creating selected-rule NPV figure", verbose)
    out["npv"] = plot_npv_scatter(
        combined, output_dir / OUT_NAMES["npv"], **kw)

    _header("Creating selected-rule sensitivity figure", verbose)
    out["sensitivity"] = plot_sensitivity_scatter(
        combined, output_dir / OUT_NAMES["sensitivity"], **kw)

    _header("Creating selected-rule specificity figure", verbose)
    out["specificity"] = plot_specificity_scatter(
        combined, output_dir / OUT_NAMES["specificity"], **kw)

    _header("Creating joint specificity-vs-sensitivity figure", verbose)
    out["joint"] = plot_joint_sens_spec(
        combined, output_dir / OUT_NAMES["joint"], **kw)

    _header("Creating false-positive-rate-vs-sensitivity figure", verbose)
    out["fpr"] = plot_fpr_sensitivity(
        combined, output_dir / OUT_NAMES["fpr"], **kw)

    _header("Creating normalized null-error-elimination figure", verbose)
    out["skill"] = plot_null_error_eliminated(
        combined, output_dir / OUT_NAMES["skill"], **kw)

    _header("Creating functional-recovery bar graph", verbose)
    out["recovery_bar"] = plot_recovery_bar(
        recovery, output_dir / OUT_NAMES["recovery_bar"], **kw)

    if save_parent_count_figure:
        _header("Creating recovery-by-parent-count figure", verbose)
        path = plot_recovery_by_parent(
            parent, output_dir / OUT_NAMES["parent_bar"], **kw)
        if path:
            out["parent_bar"] = path

    _header("Done", verbose)

    out["combined_df"] = combined
    out["recovery_summary_df"] = recovery
    out["parent_count_summary_df"] = parent
    return out


def generate_pipeline_panels(outdir,
                             input_dir=None,
                             *, palette="sequential",
                             noise_percentages=None,
                             tent_name="tent.png",
                             ppv_name="PPV.png",
                             figsize=PIPELINE_PANEL_SIZE,
                             ppv_legend_fontsize=None,
                             combined=None,
                             strict=False,
                             verbose=True) -> dict:
    """
    Entry point for the trajectory-analysis pipeline.

    Writes exactly the two panels `repatch_figure2_shane_version()` expects —
    `tent.png` (Figure 1) and `PPV.png` (Figure 2) — directly into `outdir`,
    and returns {"tent": path, "ppv": path}.

    `input_dir` defaults to the `variable_noise_data` folder beside this file.

    Defaults to the single-hue "sequential" palette rather than the R script's
    six hues, because these panels are composited alongside t-SNE panels that
    already carry coolwarm, plasma, gold, and a red/grey/green diverging map.
    Pass palette="original" to reproduce the standalone R colors exactly.

    The PPV panel carries its noise-level key inside the axes at 18 pt bold.
    Six entries plus a title is roughly 2.5 in tall at that size, which is
    tight inside the 8x6 PIPELINE_PANEL_SIZE — drop `ppv_legend_fontsize`
    to ~14 if it crowds the point cloud in the composite.

    If the input CSVs are missing, this prints a warning and returns {}
    rather than raising, so a pipeline run that only needs the other figures
    still completes. Pass strict=True to raise instead.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if combined is None:
            combined = load_combined(input_dir, noise_percentages,
                                     verbose=verbose)
    except (FileNotFoundError, ValueError) as exc:
        if strict:
            raise
        print(f"[noise_mixture_figures] Skipping tent.png / PPV.png — {exc}")
        return {}

    kw = dict(palette=palette, noise_percentages=noise_percentages,
              figsize=figsize, legend=True,
              verbose=verbose)

    result = {
        "tent": plot_error_scatter(combined, outdir / tent_name, **kw),
        "ppv": plot_ppv_scatter(combined, outdir / ppv_name,
                                legend_fontsize=ppv_legend_fontsize, **kw),
    }
    if verbose:
        print(f"[noise_mixture_figures] Wrote {tent_name} and {ppv_name} "
              f"to {outdir}")
    return result


def generate_ppv_figure(out_path, input_dir=None, *,
                        palette=DEFAULT_PALETTE, noise_percentages=None,
                        legend_fontsize=None, legend_loc=None,
                        combined=None, verbose=True):
    """Convenience wrapper: build only the PPV panel at `out_path`."""
    if combined is None:
        combined = load_combined(input_dir, noise_percentages, verbose=verbose)
    return plot_ppv_scatter(combined, out_path, palette=palette,
                            noise_percentages=noise_percentages,
                            legend_fontsize=legend_fontsize,
                            legend_loc=legend_loc,
                            verbose=verbose)


def generate_tent_figure(out_path, input_dir=None, *,
                         palette=DEFAULT_PALETTE, noise_percentages=None,
                         combined=None, verbose=True):
    """Convenience wrapper: build only the error/tent panel at `out_path`."""
    if combined is None:
        combined = load_combined(input_dir, noise_percentages, verbose=verbose)
    return plot_error_scatter(combined, out_path, palette=palette,
                              noise_percentages=noise_percentages,
                              verbose=verbose)


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="scBONITA cell-level rule/noise mixture figures "
                    "(Python port of the R/ggplot2 script)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help="Directory holding the six ruleset CSV files")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory for the full figure/table set")
    p.add_argument("--pipeline-outdir", default=None,
                   help="If given, also write tent.png and PPV.png here for "
                        "the trajectory pipeline's repatch step")
    p.add_argument("--palette", default=DEFAULT_PALETTE,
                   choices=sorted(PALETTES),
                   help="Noise-level color scheme for the full figure set")
    p.add_argument("--pipeline-palette", default="sequential",
                   choices=sorted(PALETTES),
                   help="Color scheme for the two repatched panels")
    p.add_argument("--noise-percentages", nargs="+", type=int,
                   default=NOISE_PERCENTAGES,
                   help="Noise levels to read and plot")
    p.add_argument("--ppv-legend-fontsize", type=float,
                   default=None,
                   help="Font size of the inside noise-level key on the PPV panel")
    p.add_argument("--ppv-legend-loc", default=None,
                   choices=["upper left", "upper right",
                            "lower left", "lower right", "center"],
                   help="Where the inside key sits within the PPV axes")
    p.add_argument("--check", action="store_true",
                   help="Report on the input folder and exit without "
                        "reading any data")
    p.add_argument("--only-pipeline-panels", action="store_true",
                   help="Skip the full figure set; write only tent.png/PPV.png")
    p.add_argument("--no-parent-figure", action="store_true",
                   help="Skip the recovery-by-parent-count figure")
    p.add_argument("--no-tables", action="store_true",
                   help="Skip writing the combined data and summary CSVs")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    if args.check:
        status = check_input_dir(args.input_dir, args.noise_percentages,
                                 verbose=True)
        return 0 if status["ok"] else 1

    # CLI overrides for the PPV panel's inside key, applied before any figure
    # is built so both the full set and the pipeline panels pick them up.
    if args.ppv_legend_fontsize is not None:
        global PPV_LEGEND_FONTSIZE
        PPV_LEGEND_FONTSIZE = args.ppv_legend_fontsize
    if args.ppv_legend_loc is not None:
        global PPV_LEGEND_LOC
        PPV_LEGEND_LOC = args.ppv_legend_loc

    combined = load_combined(args.input_dir, args.noise_percentages,
                             verbose=verbose)

    if not args.only_pipeline_panels:
        generate_all_figures(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            palette=args.palette,
            noise_percentages=args.noise_percentages,
            save_parent_count_figure=not args.no_parent_figure,
            save_tables=not args.no_tables,
            combined=combined,
            verbose=verbose,
        )

    if args.pipeline_outdir:
        generate_pipeline_panels(
            args.pipeline_outdir,
            input_dir=args.input_dir,
            palette=args.pipeline_palette,
            noise_percentages=args.noise_percentages,
            ppv_legend_fontsize=args.ppv_legend_fontsize,
            combined=combined,
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
