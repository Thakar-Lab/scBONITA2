"""
Trajectory Analysis Pipeline
=============================
Directory layout assumed
------------------------
<project_root>/
└── scBONITA_output/
    ├── graphml_files/
    │   └── <dataset>/<pathway>_processed.graphml
    ├── importance_score_output/
    │   └── <dataset>/
    │       ├── text_files/<pathway>_importance_score.txt
    │       └── intermediate_files/<pathway>/
    │           ├── knockout_results_<GENE>.pkl
    │           └── knockin_results_<GENE>.pkl
    └── relative_abundance_output/
        └── <dataset>/AS+_vs_AS-/text_files/<pathway>_<dataset>_AS+_vs_AS-_relative_abundance.txt
└── trajectories/
    └── <dataset>_<pathway>/
        └── text_files/
            └── cell_trajectories/          ← SCRIPT_DIR (this file lives here)
                ├── Combined_Trajectory.txt
                ├── Combined_Unique_States.txt
                └── cell_1_trajectory.csv
└── input/
    └── tutorial_data/
        └── combined_metadata.txt


Final paper-figure repatching
------------------------------
After the individual Figure2 / Figure3 / Figure4 / Figure5 / Figure6 / Figure7
panels are generated, generate_paper_figures() combines them into the final
manuscript figure numbering:

    RepatchJuly_Figure2old_Combined.png (2x2): image_nullrandomness_pyramid,
                                                Figure2D, Figure2ANetwork, Figure2C
                                                — labeled a., b., c., d.
    RepatchJuly_Figure2new_Combined.png (2x2): Figure3A, Figure3B,
                                                Figure3EOrg, Figure3FOrg
                                                — labeled e., f., g., h.
    RepatchJuly_Figure3_Combined.png    (2x2): Figure4A, Figure4B,
                                                Figure5B, Figure5C
    RepatchJuly_Figure4_Combined.png    (1x2): Figure2D, Figure5A
    RepatchJuly_Figure5_Combined.png    (2x2): Figure7A, Figure7B,
                                                Figure6A, Figure6B
"""

import os
import re
import csv
import glob
import pickle
import argparse
import sys
import time
from collections import Counter
from pathlib import Path as FilePath

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, precision_score,
                              accuracy_score, confusion_matrix,
                              silhouette_score, adjusted_rand_score)
from sklearn.model_selection import KFold
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
# `wilcoxon` (paired signed-rank) is new here — it backs the across-gene
# Figure 5A tests. `ranksums` (unpaired) was already imported for Figure5B/5C
# and is now also used per-gene for Figure 5A.
from scipy.stats import (mannwhitneyu, chi2_contingency, ranksums, wilcoxon,
                         fisher_exact)

# ============================================================
# Path Setup
# ============================================================

NOTEBOOK_SCRIPT_DIR = FilePath(
    r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns"
    r"\T_cells_all_July_2025\scBONITA_output\importance_score_output"
    r"\tutorial_dataset\..\..\..\..\trajectories"
    r"\tutorial_dataset_hsa05417\text_files\cell_trajectories"
)

try:
    SCRIPT_DIR = FilePath(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = NOTEBOOK_SCRIPT_DIR.resolve()

SCBONITA_OUT = (SCRIPT_DIR / ".." / ".." / ".." / "..").resolve()
PROJECT_ROOT = (SCBONITA_OUT / "..").resolve()

_PATHWAY_DIR = SCRIPT_DIR.parent.parent
_PATHWAY_TAG = _PATHWAY_DIR.name
_parts       = _PATHWAY_TAG.rsplit("_hsa", 1)
_DATASET_TAG = _parts[0]
_PATHWAY_ID  = "hsa" + _parts[1] if len(_parts) == 2 else _PATHWAY_TAG

os.chdir(str(SCRIPT_DIR))
print(f"[config] Working directory : {SCRIPT_DIR}")
print(f"[config] Dataset tag       : {_DATASET_TAG}")
print(f"[config] Pathway ID        : {_PATHWAY_ID}")

_INPUT_DIR = PROJECT_ROOT / "input" / "tutorial_data"
_KO_DIR    = (SCBONITA_OUT / "importance_score_output"
              / _DATASET_TAG / "intermediate_files" / _PATHWAY_ID)

_DEFAULT_PATHS = dict(
    trajectories         = str(SCRIPT_DIR / "Combined_Trajectory.txt"),
    unique_states        = str(SCRIPT_DIR / "Combined_Unique_States.txt"),
    dissimilarity_matrix = str(SCRIPT_DIR / "cell_dissimilarity_matrix.pkl"),
    metadata             = str(_INPUT_DIR  / "combined_metadata.txt"),
    ko_pattern           = str(_KO_DIR     / "knockout_results_{gene}.pkl"),
    ki_pattern           = str(_KO_DIR     / "knockin_results_{gene}.pkl"),
    outdir               = str(SCRIPT_DIR),
    selected_gene        = "XBP1",
    genes                = ["RXRB","NRAS","HSPA5","MAP3K5","EIF2AK3","ERN1",
                             "EIF2S1","DDIT3","ITPR1","TRAF2","XBP1","ATF4",
                             "PLCG1","NFATC1","GSK3B","AGER"],
    # ---- Figure 2 inputs ----
    cell_trajectory   = str(SCRIPT_DIR / "cell_1_trajectory.csv"),
    network           = str(SCBONITA_OUT / "graphml_files" / _DATASET_TAG
                            / f"hsa05417_processed.graphml"),
    importance_scores = str(SCBONITA_OUT / "importance_score_output" / _DATASET_TAG
                            / "text_files" / f"{_PATHWAY_ID}_importance_score.txt"),
    foldchange        = str(SCBONITA_OUT / "relative_abundance_output" / _DATASET_TAG
                            / "AS+_vs_AS-" / "text_files"
                            / f"hsa05417_tutorial_dataset_AS+_vs_AS-_relative_abundance.txt"),
    # ---- Noise-mixture panels (tent.png / PPV.png) ----
    noise_input_dir   = str(SCRIPT_DIR / "variable_noise_data"),
    noise_palette     = "sequential",   # or "original" / "grey"
    skip_noise_panels = False,
    # ---- Clustering ----
    n_clusters          = 2,   # integer, or "auto" to sweep
    cluster_k_min       = 2,
    cluster_k_max       = 10,
    # ---- Statistics reporting ----
    stats_report        = "statistics_summary",
    annotate_stats      = False,   # draw p-value boxes on the panels themselves
    figure5a_stats      = True,    # draw the Wilcoxon results on Figure 5A/5B/5C
    show_stat_boxes     = False,   # False = stars only, no text boxes
)


# ============================================================
# CLI / Configuration
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trajectory analysis pipeline (barcode-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectories",
                   default=_DEFAULT_PATHS["trajectories"],
                   help="Path to Combined_Trajectory.txt")
    p.add_argument("--unique-states",
                   default=_DEFAULT_PATHS["unique_states"],
                   help="Path to Combined_Unique_States.txt")
    p.add_argument("--metadata",
                   default=_DEFAULT_PATHS["metadata"],
                   help="combined_metadata.txt  "
                        "(cols: cell_1based_idx  participant_id  condition)")
    p.add_argument("--ko-pattern",
                   default=_DEFAULT_PATHS["ko_pattern"],
                   help="Path pattern for knockout pkl files. "
                        "Use {gene} as placeholder.")
    p.add_argument("--ki-pattern",
                   default=_DEFAULT_PATHS["ki_pattern"],
                   help="Path pattern for knock-in pkl files. "
                        "Use {gene} as placeholder.")
    p.add_argument("--dissimilarity-matrix",
                   default=_DEFAULT_PATHS["dissimilarity_matrix"],
                   help="Path to pre-computed dissimilarity matrix pkl "
                        "(generated by generate_figure2 if it doesn't exist yet)")
    p.add_argument("--outdir",
                   default=_DEFAULT_PATHS["outdir"],
                   help="Output directory")
    p.add_argument("--selected-gene",
                   default=_DEFAULT_PATHS["selected_gene"],
                   help="Gene used for bifurcation / Sankey figures")
    p.add_argument("--genes", nargs="+",
                   default=_DEFAULT_PATHS["genes"],
                   help="Genes to process in Figure 5A")
    p.add_argument("--cell-trajectory",
                   default=_DEFAULT_PATHS["cell_trajectory"],
                   help="Path to a single cell_*_trajectory.csv "
                        "(used only to read the ordered gene name list)")
    p.add_argument("--network",
                   default=_DEFAULT_PATHS["network"],
                   help="Path to the processed pathway .graphml file")
    p.add_argument("--importance-scores",
                   default=_DEFAULT_PATHS["importance_scores"],
                   help="Path to the scBONITA gene importance score text file")
    p.add_argument("--foldchange",
                   default=_DEFAULT_PATHS["foldchange"],
                   help="Path to the AS+ vs AS- relative abundance text file")
    p.add_argument("--force-recompute-dissimilarity", action="store_true",
                   default=False,
                   help="Force recomputation of the O(n^2) pairwise cell "
                        "dissimilarity matrix even if cell_dissimilarity_matrix.pkl "
                        "already exists in outdir")
    p.add_argument("--noise-input-dir",
                   default=_DEFAULT_PATHS["noise_input_dir"],
                   help="Folder with positive_control_best_rules_randomness_*pct.csv")
    p.add_argument("--noise-palette",
                   default=_DEFAULT_PATHS["noise_palette"],
                   choices=["sequential", "original", "grey"],
                   help="Color scheme for tent.png / PPV.png")
    p.add_argument("--skip-noise-panels", action="store_true", default=False,
                   help="Reuse existing tent.png / PPV.png instead of regenerating")
    p.add_argument("--n-clusters",
                   default=_DEFAULT_PATHS["n_clusters"],
                   help="Number of clusters for the spectral/KMeans "
                        "clustering in Figure 3. An integer forces that k; "
                        "'auto' sweeps --cluster-k-min..--cluster-k-max and "
                        "picks the best silhouette in dissimilarity space.")
    p.add_argument("--cluster-k-min", type=int,
                   default=_DEFAULT_PATHS["cluster_k_min"],
                   help="Smallest k tried when --n-clusters auto")
    p.add_argument("--cluster-k-max", type=int,
                   default=_DEFAULT_PATHS["cluster_k_max"],
                   help="Largest k tried when --n-clusters auto")
    p.add_argument("--show-stat-boxes", action="store_true",
                   default=_DEFAULT_PATHS["show_stat_boxes"],
                   help="Also draw the text boxes on Figure5A/5B/5C (the "
                        "across-gene footer and the per-panel p-value box). "
                        "Off by default: significance STARS only.")
    p.add_argument("--stats-report",
                   default=_DEFAULT_PATHS["stats_report"],
                   help="Basename (no extension) for the statistics report "
                        "written to outdir as <name>.txt and <name>.csv")
    p.add_argument("--annotate-stats", action="store_true",
                   default=_DEFAULT_PATHS["annotate_stats"],
                   help="Draw the p-value / silhouette boxes ON the figure "
                        "panels (Figure3A/3B/3EOrg/3FOrg, Figure6A/6B, "
                        "Figure7A/7B). Off by default; the values always go "
                        "to the statistics report regardless. Does NOT affect "
                        "Figure 5A, which has its own switch.")
    # Figure 5A gets its own switch because, unlike the panels above, its
    # Wilcoxon annotations are ON by default.
    p.add_argument("--no-figure5a-stats", dest="figure5a_stats",
                   action="store_false",
                   default=_DEFAULT_PATHS["figure5a_stats"],
                   help="Suppress the Wilcoxon p-value box and the per-gene "
                        "significance stars on Figure5A.png. The tests still "
                        "run and still go to the statistics report.")
    return p

def _is_real_cli() -> bool:
    if len(sys.argv) <= 1:
        return False
    suspicious = {"-f", "--ip", "--stdin", "--control", "--hb",
                  "--Session.signature_scheme", "--shell", "--transport",
                  "--iopub", "--heartbeat"}
    for arg in sys.argv[1:]:
        key = arg.split("=")[0]
        if key in suspicious or arg.endswith(".json"):
            return False
    return True


def parse_args(argv=None):
    if argv is None and not _is_real_cli():
        print("[config] Notebook/interactive mode – using auto-derived _DEFAULT_PATHS.")
        ns = argparse.Namespace(**_DEFAULT_PATHS)
        return ns
    return build_arg_parser().parse_args(argv)


# ============================================================
# Statistics registry
# ============================================================
# Single collection point for every statistic the pipeline computes.
#
# Values are appended to _STATS_RECORDS as they are computed and flushed to
# disk by write_stats_report(), which is called at the end of each figure
# stage (so a crash part-way through the pipeline still leaves a valid
# report) and again at the end of main().
#
# Records are keyed by (figure, panels) so the report reads in the same
# order the manuscript panels do.

STATS_REPORT_BASENAME = "statistics_summary"
_STATS_RECORDS = []


# ------------------------------------------------------------------
# >>> MASTER SWITCH for on-figure statistic boxes <<<
#
# False (default) : no p-value / silhouette box is drawn on any panel.
#                   Values go only to statistics_summary.txt / .csv.
# True            : the boxes are drawn on Figure3A, Figure3B, Figure3EOrg,
#                   Figure3FOrg, Figure6A, Figure6B, Figure7A and Figure7B,
#                   exactly as they used to be. The text file is still
#                   written either way.
#
#
# NOTE: this switch does NOT cover Figure 5A — see the next switch.
# ------------------------------------------------------------------
ANNOTATE_STATS_ON_FIGURES = False


# ------------------------------------------------------------------
# >>> SWITCH for the Figure 5A Wilcoxon annotations <<<
#
# True (default) : Figure5A.png carries (a) a box with the three across-gene
#                  signed-rank p-values and (b) a significance star at the
#                  tip of each knockin / knockout bar from that gene's
#                  rank-sum test against baseline.
# False          : bare panel, exactly as before this revision.
#
# Either way the Wilcoxon tests run and their p-values are written to
# statistics_summary.txt / .csv. Separate from ANNOTATE_STATS_ON_FIGURES so
# the Figure 5A statistics can be shown without un-suppressing the older
# Figure 3 / 6 / 7 boxes. CLI equivalent: --no-figure5a-stats.
# ------------------------------------------------------------------
ANNOTATE_FIGURE5A_WILCOXON = True


# ------------------------------------------------------------------
# >>> STARS ONLY, OR STARS + TEXT BOXES? <<<
#
# ANNOTATE_FIGURE5A_WILCOXON above is the master switch for all Figure
# 5A/5B/5C statistical annotation. This second switch decides how much of
# it gets drawn:
#
#   False (default) : STARS ONLY. The significance markers appear at the
#                     Figure 5A bar tips and on the Figure5B/5C titles, and
#                     nothing else — no footer summary box on 5A, no
#                     "p = ... / Wilcoxon rank-sum vs. normal" box in the
#                     corner of 5B/5C.
#   True            : stars PLUS those text boxes.
#
# Turning the boxes off also drops the blank strip Figure 5A used to
# reserve underneath the bars for the footer, so the bars fill the panel
# normally again.
#
# The p-values themselves always go to statistics_summary.txt / .csv
# regardless, so nothing is lost by hiding the boxes — but note the star
# THRESHOLDS then only exist in the figure legend you write yourself, since
# PSTAR_THRESHOLDS is no longer printed anywhere on the panel. State them
# in the caption.
#
# CLI equivalent: --show-stat-boxes
# ------------------------------------------------------------------
SHOW_PERTURBATION_STAT_BOXES = False
ANNOTATE_FIGURE5BC_STARS = False


def set_stats_annotation(enabled: bool):
    """Set the on-figure annotation switch. Called by main() from --annotate-stats."""
    global ANNOTATE_STATS_ON_FIGURES
    ANNOTATE_STATS_ON_FIGURES = bool(enabled)
    print(f"[config] On-figure statistic boxes: "
          f"{'ON' if ANNOTATE_STATS_ON_FIGURES else 'OFF'} "
          f"(statistics_summary.txt is written either way)")
    return ANNOTATE_STATS_ON_FIGURES


def set_figure5a_annotation(enabled: bool):
    """Set the Figure 5A Wilcoxon annotation switch (main() / --no-figure5a-stats)."""
    global ANNOTATE_FIGURE5A_WILCOXON
    ANNOTATE_FIGURE5A_WILCOXON = bool(enabled)
    print(f"[config] Figure 5A Wilcoxon annotations: "
          f"{'ON' if ANNOTATE_FIGURE5A_WILCOXON else 'OFF'} "
          f"(statistics_summary.txt is written either way)")
    return ANNOTATE_FIGURE5A_WILCOXON


def set_perturbation_stat_boxes(enabled: bool):
    """Show/hide the Figure 5A footer box and the Figure5B/5C corner box.

    Stars are unaffected — they follow ANNOTATE_FIGURE5A_WILCOXON.
    """
    global SHOW_PERTURBATION_STAT_BOXES
    SHOW_PERTURBATION_STAT_BOXES = bool(enabled)
    print(f"[config] Figure 5A/5B/5C text boxes: "
          f"{'ON' if SHOW_PERTURBATION_STAT_BOXES else 'OFF (stars only)'}")
    return SHOW_PERTURBATION_STAT_BOXES


def _stat_box(ax, x, y, text, use_axes_coords=True,
              fontsize=14, ha="left", va="top", facecolor="white"):
    """
    Draw one statistic annotation box — but ONLY if ANNOTATE_STATS_ON_FIGURES
    is True. When it is False this is a no-op, which is what keeps every
    panel clean by default.

    Every on-figure statistic in this pipeline except the Figure 5A Wilcoxon
    annotations goes through here, so this single function is the only place
    that needs changing to alter the look of all of them at once. Figure 5A
    uses `_figure5a_stat_box()` instead, so it can be toggled independently.
    """
    if not ANNOTATE_STATS_ON_FIGURES:
        return None
    kwargs = dict(fontsize=fontsize, fontweight="bold", ha=ha, va=va,
                  bbox=dict(boxstyle="round", facecolor=facecolor,
                            edgecolor="black", alpha=0.85))
    if use_axes_coords:
        kwargs["transform"] = ax.transAxes
    return ax.text(x, y, text, **kwargs)


def _figure5a_stat_box(ax, x, y, text, fontsize=12,
                       ha="right", va="bottom", facecolor="white"):
    """
    Figure 5A's Wilcoxon summary box. Gated on ANNOTATE_FIGURE5A_WILCOXON
    rather than ANNOTATE_STATS_ON_FIGURES, and ON by default.
    """
    if not (ANNOTATE_FIGURE5A_WILCOXON and SHOW_PERTURBATION_STAT_BOXES):
        return None
    return ax.text(x, y, text, transform=ax.transAxes,
                   fontsize=fontsize, fontweight="bold", ha=ha, va=va,
                   bbox=dict(boxstyle="round", facecolor=facecolor,
                             edgecolor="black", alpha=0.85))


def _fmt_p(value):
    """Format a p-value for an on-figure box (matches the historical '%.3g')."""
    return "p = n/a" if not np.isfinite(value) else f"p = {value:.3g}"


# ------------------------------------------------------------------
# Significance-star scheme, shared by Figure5A, Figure5B and Figure5C
# ------------------------------------------------------------------
# Thresholds are checked in order, so list them from strictest to loosest.
# The default is the two-level scheme:
#       **   p < 0.001
#       *    p < 0.05
# To go back to the conventional three-level convention, use:
#       PSTAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
# PSTAR_NONSIG is what gets drawn when nothing is met — set it to "" if you
# would rather leave those bars unmarked entirely.
PSTAR_THRESHOLDS = ((0.001, "**"), (0.05, "*"))
PSTAR_NONSIG     = "ns"


# ------------------------------------------------------------------
# >>> WHICH TEST DRIVES THE FIGURE 5A BAR STARS? <<<
#
# "proportion" (default)
#     Fisher exact test on the SAME 2x2 AS+/AS- counts the plotted delta is
#     computed from: baseline (states where the gene is ON) vs. perturbed
#     (cells). This is the test that matches the bar — a long bar means a
#     large shift in the AS+ share, and this test asks whether that shift in
#     share is bigger than chance. Stars and bar lengths therefore agree.
#
# "ranksum"
#     Wilcoxon rank-sum on the underlying continuous condition-bias values.
#     This asks a DIFFERENT question: has the whole distribution shifted in
#     rank, regardless of how many values crossed the zero threshold. Its
#     power depends heavily on how many baseline states have the gene ON, so
#     a gene that is ON in only a handful of states can show a huge bar and
#     still come out "ns".
#
# Both are computed and both go into statistics_summary.txt / .csv on every
# run; this only selects which one is drawn on the panel. Whichever you use,
# name it in the figure caption.
# ------------------------------------------------------------------
FIGURE5A_STAR_TEST = "proportion"      # "proportion" | "ranksum"

# Registry family names for the two per-gene test families. The one selected
# by FIGURE5A_STAR_TEST is what gets drawn AND what statistics_summary_pvalues
# .csv contains, so the stars and that CSV always describe the same test.
FIG5A_FAMILY_PROPORTION = "Figure 5A (per-gene proportion test)"
FIG5A_FAMILY_RANKSUM    = "Figure 5A (per-gene Wilcoxon)"


def _figure5a_star_family():
    """Registry `figure` name of the test currently drawn as the 5A stars."""
    return (FIG5A_FAMILY_RANKSUM if FIGURE5A_STAR_TEST == "ranksum"
            else FIG5A_FAMILY_PROPORTION)


def _drawn_flag(family):
    """'yes (drawn as the Figure5A stars)' / 'no (reported for comparison)'."""
    return ("yes (drawn as the Figure5A stars)"
            if family == _figure5a_star_family()
            else "no (reported for comparison only)")


def write_contingency_tables_txt(tables, outdir,
                                 filename="figure5A_contingency_tables.txt",
                                 star_test_label=""):
    """
    Render every gene/arm 2x2 as an actual table in a plain-text file.

    `tables` is the list of dicts collected in generate_figure5 (the same
    rows that go to figure5A_contingency_tables.csv). The CSV is for
    machines; this is the version to read, paste into a supplementary
    document, or hand to a collaborator who wants to see the counts laid
    out rather than parse columns.

    Genes appear in the order they were processed, KNOCKIN before KNOCKOUT
    within each gene. Tables built on fewer than 10 baseline states are
    flagged inline and listed again at the end, since a large delta resting
    on a handful of states is the main thing worth eyeballing here.
    """
    if not tables:
        print("  No contingency tables to write.")
        return None

    W = 74
    lines = []
    lines.append("=" * W)
    lines.append("FIGURE 5A — CONTINGENCY TABLES")
    lines.append("=" * W)
    lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Output dir: {os.path.abspath(outdir)}")
    if star_test_label:
        lines.append(f"Panel stars: {star_test_label}")
        thin = []
    current_gene = None
    for row in tables:
        gene = row["Gene"]
        if gene != current_gene:
            current_gene = gene
            lines.append("")
            lines.append("-" * W)
            lines.append(f"{gene}")
            lines.append("-" * W)

        b_ap, b_am = row["baseline_AS_plus"],  row["baseline_AS_minus"]
        p_ap, p_am = row["perturbed_AS_plus"], row["perturbed_AS_minus"]
        b_n,  p_n  = row["baseline_n_states"], row["perturbed_n_cells"]
        b_pct = row["baseline_AS_plus_pct"]
        p_pct = row["perturbed_AS_plus_pct"]
        delta = row["delta_AS_plus_pct"]

        lines.append("")
        lines.append(f"  {gene} — {row['Arm']}")
        lines.append("")
        lines.append("                            AS+        AS-      total     AS+ %")
        lines.append("      " + "-" * 58)
        lines.append(f"      baseline (states) {b_ap:>8} {b_am:>10} {b_n:>10}"
                     f" {b_pct if b_pct == '' else f'{b_pct:>9.2f}'}")
        lines.append(f"      perturbed (cells) {p_ap:>8} {p_am:>10} {p_n:>10}"
                     f" {p_pct if p_pct == '' else f'{p_pct:>9.2f}'}")
        lines.append("      " + "-" * 58)
        if delta != "":
            lines.append(f"      plotted delta (normal - perturbed) = "
                         f"{delta:+.2f} percentage points")
        lines.append("")
        _fp = row["fisher_p"]
        lines.append(f"      Fisher exact, two-sided : p = {_fp or 'not computable'}"
                     f"   {row['fisher_stars']}")
        #if row.get("fisher_odds_ratio"):
        #    lines.append(f"      odds ratio              : {row['fisher_odds_ratio']}")
        #_wp = row["wilcoxon_ranksum_p"]
        #lines.append(f"      Wilcoxon rank-sum       : p = {_wp or 'not computable'}"
        #             f"   {row['wilcoxon_stars']}   (comparison only, not plotted)")

        if isinstance(b_n, (int, float)) and b_n < 10:
            lines.append(f"      NOTE: only {b_n} baseline states — this bar rests "
                         f"on very little data.")
            thin.append(f"{gene} ({row['Arm']}, n={b_n})")

    lines.append("")
    lines.append("=" * W)
    lines.append(f"{len(tables)} table(s): "
                 f"{len({r['Gene'] for r in tables})} gene(s) x KNOCKIN/KNOCKOUT.")
    if thin:
        lines.append("")
        lines.append("Tables built on fewer than 10 baseline states:")
        for t in thin:
            lines.append(f"  - {t}")
    lines.append("=" * W)
    lines.append("")

    path = _outpath(outdir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Wrote {path}")
    return path

def write_figure4_legend_pvalues(outdir, panel_a_genes, all_gene_results,
                                 across_gene, selected_gene,
                                 ki_pvalue, ko_pvalue, filename="Fig4.txt"):
    """
    P-values behind paper Figure 4 (RepatchJuly_Figure4_Combined.png =
    Figure5A / Figure5B / Figure5C as panels a. / b. / c.), laid out for
    pasting into the figure legend.

    Panels b. and c. no longer carry significance stars
    (ANNOTATE_FIGURE5BC_STARS), so this file and statistics_summary.txt are
    the only places those two p-values appear.
    """
    if FIGURE5A_STAR_TEST == "ranksum":
        star_label = ("Wilcoxon rank-sum, two-sided, vs. the baseline states "
                      "where that gene is ON")
        ki_key, ko_key = "ki_p_gene_on", "ko_p_gene_on"
    else:
        star_label = ("Fisher exact, two-sided, on the AS+ share — the same "
                      "2x2 counts the plotted bar is computed from")
        ki_key, ko_key = "ki_p_prop", "ko_p_prop"

    def _p(v):
        return _fmt_or_blank(v, "{:.3g}") or "n/a"

    W = 74
    L = ["=" * W, "FIGURE 4 - P-VALUES FOR THE LEGEND", "=" * W]
    L.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Output dir: {os.path.abspath(outdir)}")
    L.append("Panels    : a. Figure5A.png   b. Figure5B.png   c. Figure5C.png")
    L.append("")

    # ---- panel a ----
    L.append("-" * W)
    L.append("a.  Per-gene perturbation effect (stars at the bar tips)")
    L.append("-" * W)
    L.append(f"Test: {star_label}.")
    L.append("Genes are listed bottom-to-top, in the order the bars appear.")
    L.append("")
    L.append(f"  {'Gene':<10} {'KNOCKIN p':>12} {'':>4} {'KNOCKOUT p':>12} {'':>4}")
    L.append("  " + "-" * 46)
    for gene in (panel_a_genes or []):
        wc = (all_gene_results.get(gene, {}) or {}).get("wilcoxon", {})
        p_ki, p_ko = wc.get(ki_key, np.nan), wc.get(ko_key, np.nan)
        L.append(f"  {gene:<10} {_p(p_ki):>12} {_p_stars(p_ki):>4} "
                 f"{_p(p_ko):>12} {_p_stars(p_ko):>4}")
    L.append("")
    L.append(f"  Star key: {_pstar_legend_text()}   {PSTAR_NONSIG} = not significant")
    L.append("  Uncorrected. Correct across genes before quoting.")

    if across_gene:
        L.append("")
        L.append("  Across genes (Wilcoxon signed-rank, two-sided, paired by gene,")
        L.append(f"  n = {across_gene.get('n_genes', 'n/a')} genes):")
        L.append(f"    knockin delta vs 0    p = {_p(across_gene.get('ki_vs_zero'))}")
        L.append(f"    knockout delta vs 0   p = {_p(across_gene.get('ko_vs_zero'))}")
        L.append(f"    knockin vs knockout   p = {_p(across_gene.get('ki_vs_ko'))}")

    # ---- panels b / c ----
    L.append("")
    L.append("-" * W)
    L.append(f"b.  {selected_gene} Knockin      c.  {selected_gene} Knockout")
    L.append("-" * W)
    L.append("Test: Wilcoxon rank-sum, two-sided; predicted condition bias of the")
    L.append("perturbed cells vs. the baseline condition bias of the unperturbed")
    L.append("states. Baseline is per-STATE and perturbed is per-CELL, so the two")
    L.append("samples are independent and of unequal length.")
    L.append("")
    L.append(f"  b.  knockin   p = {_p(ki_pvalue)}   {_p_stars(ki_pvalue) or 'n/a'}")
    L.append(f"  c.  knockout  p = {_p(ko_pvalue)}   {_p_stars(ko_pvalue) or 'n/a'}")
    L.append("")
    L.append("  No significance stars are drawn on these two panels - quote the")
    L.append("  values above in the legend text.")
    L.append("")
    L.append("=" * W)
    L.append("")

    path = _outpath(outdir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"  Wrote {path}")
    return path

def _fmt_or_blank(value, fmt="{:.10g}"):
    """Full-precision number for a data file, or "" when not computable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return fmt.format(v) if np.isfinite(v) else ""


def _p_stars(value):
    """
    Compact significance marker for the Figure 5A bar tips and the
    Figure5B/5C titles, per PSTAR_THRESHOLDS above.

    Returns an empty string when the test could not be computed, so a nan
    is silently left unmarked rather than being drawn as "ns".
    """
    if value is None or not np.isfinite(value):
        return ""
    for cutoff, mark in PSTAR_THRESHOLDS:
        if value < cutoff:
            return mark
    return PSTAR_NONSIG


def _pstar_legend_text():
    """One-line key describing the current scheme, for the Figure 5A footer."""
    parts = [f"{mark} p<{cutoff:g}" for cutoff, mark in PSTAR_THRESHOLDS]
    return "   ".join(parts)


def reset_stats():
    """Clear the registry. Called once at the start of main()."""
    _STATS_RECORDS.clear()


def record_stat(figure, panels, test, value,
                value_name="p-value", notes="", **details):
    """
    Record one statistic.

    Parameters
    ----------
    figure     : str   e.g. "Figure 3"
    panels     : str   which PNG(s) the value used to be annotated on,
                       e.g. "Figure3A.png, Figure3EOrg.png"
    test       : str   human-readable description of the test performed
    value      : float the statistic itself (may be np.nan)
    value_name : str   what `value` is ("p-value", "silhouette score", ...)
    """
    rec = {
        "figure":     figure,
        "panels":     panels,
        "test":       test,
        "value_name": value_name,
        "value":      float(value) if value is not None else float("nan"),
        "notes":      notes,
        "details":    dict(details),
    }
    _STATS_RECORDS.append(rec)

    val = rec["value"]
    val_str = "nan" if not np.isfinite(val) else f"{val:.6g}"
    print(f"  [Stats] {figure} ({panels}) — {test}: {value_name} = {val_str}")
    return rec


def _format_stat_value(value_name, value):
    if not np.isfinite(value):
        return "not computable (insufficient data in one or both groups)"
    if "p-value" in value_name.lower():
        if value < 1e-4:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return f"{value:.6f}"


def write_stats_report(outdir, basename=None):
    """
    Flush every recorded statistic to <outdir>/<basename>.txt (human-readable)
    and <outdir>/<basename>.csv (machine-readable). Overwrites on each call,
    so the files always reflect everything computed so far.
    """
    if basename is None:
        basename = STATS_REPORT_BASENAME

    txt_path = _outpath(outdir, f"{basename}.txt")
    csv_path = _outpath(outdir, f"{basename}.csv")

    if not _STATS_RECORDS:
        print("  [Stats] Nothing recorded yet — report not written.")
        return None, None

    # ---- human-readable ----
    lines = []
    lines.append("=" * 74)
    lines.append("STATISTICS SUMMARY")
    lines.append("=" * 74)
    lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Output dir: {os.path.abspath(outdir)}")
    lines.append("")
    lines.append("Except for the Figure 5A Wilcoxon results, these values are")
    lines.append("NOT drawn on any figure. Each entry names the panel(s) it")
    lines.append("describes so it can be quoted in the corresponding figure")
    lines.append("legend or methods section.")
    lines.append("")

    current_figure = None
    for rec in _STATS_RECORDS:
        if rec["figure"] != current_figure:
            current_figure = rec["figure"]
            lines.append("")
            lines.append("-" * 74)
            lines.append(current_figure)
            lines.append("-" * 74)

        lines.append("")
        lines.append(f"  Panel(s) : {rec['panels']}")
        lines.append(f"  Test     : {rec['test']}")
        lines.append(f"  {rec['value_name']:<9}: "
                     f"{_format_stat_value(rec['value_name'], rec['value'])}")
        if rec["details"]:
            lines.append("  Details  :")
            for k, v in rec["details"].items():
                lines.append(f"      {k} = {v}")
        if rec["notes"]:
            lines.append(f"  Note     : {rec['notes']}")

    lines.append("")
    lines.append("=" * 74)
    lines.append(f"{len(_STATS_RECORDS)} statistic(s) recorded.")
    lines.append("=" * 74)
    lines.append("")

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Figure", "Panels", "Test", "Value_Name", "Value",
                    "Details", "Notes"])
        for rec in _STATS_RECORDS:
            det = "; ".join(f"{k}={v}" for k, v in rec["details"].items())
            w.writerow([rec["figure"], rec["panels"], rec["test"],
                        rec["value_name"], f"{rec['value']:.10g}",
                        det, rec["notes"]])

    print(f"  [Stats] Wrote {txt_path}")
    print(f"  [Stats] Wrote {csv_path}")

    # Dedicated p-values-only table, regenerated on every flush.
    write_pvalue_table(outdir, basename)

    return txt_path, csv_path


# ------------------------------------------------------------------
# P-VALUE-ONLY TABLE
# ------------------------------------------------------------------
# statistics_summary.csv holds everything the pipeline computes, which mixes
# hypothesis tests in with accuracies, R2 values and the silhouette score.
# This writes a second, deliberately minimal file: four columns, one row per
# test, nothing else.
#
#   Test            full description of the comparison
#   Value_Name      always "p-value"
#   p_value         full precision, e.g. 0.036523221 or 6.98e-13
#   Label           "<GENE> KNOCKIN" / "<GENE> KNOCKOUT"
#
# Scope is controlled by PVALUE_TABLE_FIGURE5A_ONLY below. The default is
# True: only the per-gene Figure 5A rank-sum tests are written, in the order
# they were computed (each gene's KNOCKIN row immediately followed by its
# KNOCKOUT row). Set it to False to get every p-value in the pipeline
# (Figure 3, 5, 6, 7) in the same four columns.

PVALUE_TABLE_SUFFIX = "_pvalues"

# Only the per-gene Figure 5A rank-sum tests (True), or every p-value the
# pipeline records (False).
PVALUE_TABLE_FIGURE5A_ONLY = True

# Write the "Test,Value_Name,p_value,Label" header line. Set False for a
# bare, header-less table.
PVALUE_TABLE_HEADER = True

# Add a Benjamini-Hochberg q_value column (corrected within figure family).
# Off by default to keep the table to the four columns above; the machinery
# is still there in _benjamini_hochberg() if you want it back.
PVALUE_TABLE_INCLUDE_BH = False


def _benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR adjustment. Returns q-values in the input order.

    NaNs are passed through untouched and excluded from the ranking, so a
    test that could not be computed does not inflate the correction for the
    tests that could. Implemented directly rather than via statsmodels to
    avoid adding a dependency.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q

    p_ok = p[finite]
    n    = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downwards, then clip at 1.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)

    q_ok = np.empty(n)
    q_ok[order] = ranked
    q[finite] = q_ok
    return q


def write_pvalue_table(outdir, basename=None, alpha=0.05):
    """
    Write <outdir>/<basename>_pvalues.csv — a minimal four-column table of
    p-values: Test, Value_Name, p_value, Label.
    """
    if basename is None:
        basename = STATS_REPORT_BASENAME

    rows = [r for r in _STATS_RECORDS if "p-value" in r["value_name"].lower()]
    if PVALUE_TABLE_FIGURE5A_ONLY:
        # Exactly the family drawn as the Figure 5A stars — so this file and
        # the panel can never disagree. The other per-gene family stays in
        # statistics_summary.txt / .csv, flagged "no (reported for
        # comparison only)" in its on_figure detail.
        star_family = _figure5a_star_family()
        rows = [r for r in rows
                if r["details"].get("gene") and r["figure"] == star_family]

    if not rows:
        scope = (f"tests for {_figure5a_star_family()}"
                 if PVALUE_TABLE_FIGURE5A_ONLY else "p-values")
        print(f"  [Stats] No {scope} recorded yet — p-value table not written.")
        return None

    # ---- optional BH q-values, within figure family ----
    q_lookup = {}
    if PVALUE_TABLE_INCLUDE_BH:
        families = {}
        for r in rows:
            families.setdefault(r["figure"], []).append(r)
        for _fig, recs in families.items():
            for r, q in zip(recs, _benjamini_hochberg([x["value"] for x in recs])):
                q_lookup[id(r)] = q

    def _label(rec):
        """'<GENE> KNOCKIN' / '<GENE> KNOCKOUT', derived from the record."""
        gene = rec["details"].get("gene", "")
        test = rec["test"].upper()
        if "KNOCKIN" in test:
            arm = "KNOCKIN"
        elif "KNOCKOUT" in test:
            arm = "KNOCKOUT"
        else:
            arm = ""
        return " ".join(x for x in (gene, arm) if x)

    out_path = _outpath(outdir, f"{basename}{PVALUE_TABLE_SUFFIX}.csv")

    header = ["Test", "Value_Name", "p_value", "Label"]
    if PVALUE_TABLE_INCLUDE_BH:
        header.append("q_value_BH")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if PVALUE_TABLE_HEADER:
            w.writerow(header)
        for r in rows:
            p = r["value"]
            row = [r["test"],
                   r["value_name"],
                   f"{p:.10g}" if np.isfinite(p) else "",
                   _label(r)]
            if PVALUE_TABLE_INCLUDE_BH:
                q = q_lookup.get(id(r), np.nan)
                row.append(f"{q:.10g}" if np.isfinite(q) else "")
            w.writerow(row)

    scope = (f"the test drawn as the Figure5A stars: {_figure5a_star_family()}"
             if PVALUE_TABLE_FIGURE5A_ONLY else "all figures")
    print(f"  [Stats] Wrote {out_path} ({len(rows)} row(s), {scope})")
    return out_path


# ============================================================
# Metadata helpers – barcode-based
# ============================================================

def parse_metadata(filepath):
    """
    Parse combined_metadata.txt and return three lookup dicts:
      num_to_label          { "0": "AS+" }
      barcode_to_label      { "AAACCTGA.Participant_1": "AS+" }
      barcode_to_participant { "AAACCTGA.Participant_1": "Participant_1" }
    """
    num_to_label           = {}
    barcode_to_label       = {}
    barcode_to_participant = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip('"').strip("'") for p in re.split(r'[\t ]+', line)]
            if len(parts) < 3:
                continue
            num_id      = parts[0]
            barcode     = parts[1]
            label       = parts[2]
            participant = barcode.rsplit(".", 1)[-1] if "." in barcode else barcode
            num_to_label[num_id]            = label
            barcode_to_label[barcode]       = label
            barcode_to_participant[barcode] = participant

    return num_to_label, barcode_to_label, barcode_to_participant


def trajectory_key_to_barcode(cell_key: str) -> str:
    """
    Extract the barcode from a trajectory key.

    Examples:
        "cell_AAACCTGAGACAAAGG.Participant_1_trajectory"
            -> "AAACCTGAGACAAAGG.Participant_1"
        "cell_AAACCTGAGACAAAGG.Participant_1"
            -> "AAACCTGAGACAAAGG.Participant_1"
        "AAACCTGAGACAAAGG.Participant_1"
            -> "AAACCTGAGACAAAGG.Participant_1"
    """
    key = cell_key
    if key.lower().startswith("cell_"):
        key = key[5:]
    if key.lower().endswith("_trajectory"):
        key = key[:-11]
    return key


# ============================================================
# Utility – trajectory processing
# ============================================================

def load_pickle(file_path):
    with open(file_path, 'rb') as fh:
        return pickle.load(fh)


def read_trajectories(file_path: str) -> dict:
    with open(file_path, "r") as fh:
        return {
            cell.strip(): list(map(int, traj.strip().strip("[]").split(",")))
            for cell, traj in (line.strip().split(":") for line in fh)
        }


def clean_trajectory(trajectory: list) -> list:
    cleaned, seen = [], set()
    for state in trajectory:
        if state in seen:
            break
        cleaned.append(state)
        seen.add(state)
    return cleaned


def find_attractors(trajectory: list):
    visited = {}
    for i, state in enumerate(trajectory):
        if state in visited:
            cycle_start  = visited[state]
            cycle_states = trajectory[cycle_start:i]
            attractor_state = cycle_states[-1]
            return attractor_state, cycle_states, attractor_state, trajectory[0]
        visited[state] = i
    return trajectory[-1], [], trajectory[-1], trajectory[0]


def compute_weighted_pseudotime(trajectories: dict, attractor_value_dict: dict) -> dict:
    """Lightweight pseudotime-only version (used by Figures 3, 5)."""
    pseudotime_dict     = {}
    observed_steps_dict = {}

    for cell, trajectory in trajectories.items():
        cleaned  = clean_trajectory(trajectory)
        single_att, cyclic_att, att_state, _ = find_attractors(cleaned)
        attractor = single_att if single_att is not None else cyclic_att[0]
        observed_steps_dict[cell] = cleaned.index(attractor) + 1

    max_s = max(observed_steps_dict.values())
    min_s = min(observed_steps_dict.values())

    for cell, steps in observed_steps_dict.items():
        if max_s != min_s:
            pseudotime = 1 - (abs(steps - min_s) / (max_s - min_s))
        else:
            pseudotime = 0
        pseudotime_dict[cell] = pseudotime

    return pseudotime_dict


def generalized_jaccard_similarity(arr1, arr2):
    arr1, arr2 = np.array(arr1), np.array(arr2)
    matching = np.sum(arr1 == arr2)
    return matching / len(arr1)


def calculate_similarity_matching(*arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = []
    for arr in arrays:
        arr = list(arr)
        if len(arr) < max_length:
            arr = [arr[0]] * (max_length - len(arr)) + arr if arr else [0] * max_length
        elif len(arr) > max_length:
            arr = arr[-max_length:]
        padded_arrays.append(np.array(arr))
    stacked = np.vstack(padded_arrays)
    matching = np.all(stacked == stacked[0], axis=0).sum()
    return matching / max_length


def find_most_traveled_paths(basins, trajectories, output_file="most_traveled_paths.txt"):
    most_traveled_paths = {}
    for attractor, cells in basins.items():
        path_counts = {}
        for cell in cells:
            traj = trajectories[cell]
            traj_tuple = tuple(traj)
            path_counts[traj_tuple] = path_counts.get(traj_tuple, 0) + 1
        if path_counts:
            most_common_path = max(path_counts, key=path_counts.get)
            most_common_count = path_counts[most_common_path]
            most_traveled_paths[attractor] = (list(most_common_path), most_common_count)
        else:
            most_traveled_paths[attractor] = None
    with open(output_file, "w", encoding="utf-8") as f:
        for attractor, result in most_traveled_paths.items():
            if result is not None:
                path, count = result
                f.write(f"Attractor {attractor}: most traveled path is {path} with {count} occurrences\n")
            else:
                f.write(f"Attractor {attractor}: no paths recorded\n")
    return most_traveled_paths


def compute_weighted_pseudotime_full(cleaned_trajectories: dict, attractor_value_dict: dict,
                                      outdir: str = "."):
    """
    Full pseudotime + pairwise cell dissimilarity matrix, ported from the
    standalone Figure-2 generation script. Two fixes relative to that
    original script:

      1. `attractor_type_dict` is now actually computed and returned
         ("single" vs "cyclic" attractor per cell), since the call site
         unpacks three values but the original function only returned two.
      2. Basin membership and the most-traveled-path lookup are hoisted
         OUTSIDE the O(n^2) cell-pair loop – they don't depend on the loop
         indices, so recomputing them on every (i, j) pair (as the original
         script did) was pure O(n) wasted work per pair for no benefit.
    """
    print("Computing weighted pseudotime + pairwise cell dissimilarity matrix "
          "(this can take a while for large cell counts)...")

    pseudotime_dict     = {}
    observed_steps_dict = {}
    unique_attractors   = set()
    attractor_type_dict = {}

    trajectory_keys = list(cleaned_trajectories.keys())
    n = len(trajectory_keys)
    dissimilarity_matrix = np.zeros((n, n))

    for cell, trajectory in cleaned_trajectories.items():
        cleaned = clean_trajectory(trajectory)
        single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned)
        attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
        observed_steps_dict[cell] = cleaned.index(attractor) + 1
        unique_attractors.add(attractor)
        attractor_type_dict[cell] = "cyclic" if cyclic_attractor else "single"

    max_observed_steps = max(observed_steps_dict.values())
    min_observed_steps = min(observed_steps_dict.values())
    for cell, observed_steps in observed_steps_dict.items():
        if max_observed_steps != min_observed_steps:
            pseudotime = 1 - (abs(observed_steps - min_observed_steps)
                               / (max_observed_steps - min_observed_steps))
        else:
            pseudotime = 0
        pseudotime_dict[cell] = pseudotime

    sorted_attractors = sorted(unique_attractors)

    # Computed ONCE (fix #2 above).
    basins = {att: [c for c, traj in cleaned_trajectories.items() if att in traj]
              for att in unique_attractors}
    most_traveled_paths = find_most_traveled_paths(
        basins, cleaned_trajectories,
        output_file=_outpath(outdir, "most_traveled_paths.txt")
    )

    for i, cell in enumerate(trajectory_keys):
        cleaned_i = clean_trajectory(cleaned_trajectories[cell])
        att_i = find_attractors(cleaned_i)[0]
        if att_i is None:
            att_i = find_attractors(cleaned_i)[1][0]

        mtp_entry = most_traveled_paths.get(att_i)
        most_traveled_path = clean_trajectory(mtp_entry[0]) if mtp_entry else []

        for j, cell2 in enumerate(trajectory_keys):
            if i == j:
                dissimilarity_matrix[i, j] = 0
                continue
            cleaned_j = clean_trajectory(cleaned_trajectories[cell2])

            max_len   = max(len(cleaned_i), len(cleaned_j), len(most_traveled_path))
            padded_i  = [cleaned_i[0]] * (max_len - len(cleaned_i)) + cleaned_i
            padded_j  = [cleaned_j[0]] * (max_len - len(cleaned_j)) + cleaned_j
            padded_mp = ([most_traveled_path[0]] * (max_len - len(most_traveled_path))
                         + most_traveled_path) if most_traveled_path else []

            similarity_weight1_ij = calculate_similarity_matching(padded_i, padded_j)
            if most_traveled_path:
                similarity_weight2_ijl = (
                    calculate_similarity_matching(padded_i, padded_mp)
                    if cleaned_i in most_traveled_path
                    else calculate_similarity_matching(padded_j, padded_mp)
                )
            else:
                similarity_weight2_ijl = similarity_weight1_ij

            attractor1 = attractor_value_dict.get(cell, [])
            attractor2 = attractor_value_dict.get(cell2, [])
            similarity_weight3_ij = calculate_similarity_matching(attractor1, attractor2)

            attractor3 = attractor_value_dict.get(sorted_attractors[0], [])
            similarity_weight4_ijk = (
                1 if attractor1 == attractor3 or attractor2 == attractor3
                else calculate_similarity_matching(attractor1, attractor2, attractor3)
            )

            similarity_score = (similarity_weight1_ij + similarity_weight2_ijl
                                 + similarity_weight3_ij + similarity_weight4_ijk) / 4
            tolerance = 1e-2
            pseudotime_diff = abs(pseudotime_dict[cell] - pseudotime_dict[cell2])
            if similarity_score > 0.999 and pseudotime_diff < tolerance:
                distance_score = 0
            else:
                distance_score = 1 - (similarity_score * (1 - pseudotime_diff))
            if pseudotime_dict[cell] == 1 or pseudotime_dict[cell2] == 1:
                distance_score = 1 - similarity_score
            if similarity_weight1_ij == 1:
                distance_score = 0
            dissimilarity_matrix[i, j] = distance_score

    return pseudotime_dict, dissimilarity_matrix, attractor_type_dict


def jitter_positions(pos, jitter=0.01, min_dist=0.05, max_iter=100):
    nodes  = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    for _ in range(max_iter):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < min_dist:
                    direction = coords[i] - coords[j]
                    if np.all(direction == 0):
                        direction = np.random.randn(2)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    coords[i] += direction * jitter
                    coords[j] -= direction * jitter
                    moved = True
        if not moved:
            break
    return {n: tuple(coords[i]) for i, n in enumerate(nodes)}


def plot_stg(trajectories):
    stg = nx.DiGraph()
    for cell, traj in trajectories.items():
        if len(set(traj)) == 1:
            continue
        for i in range(len(traj) - 1):
            stg.add_edge(traj[i], traj[i + 1])
            label = stg[traj[i]][traj[i + 1]].get('label', '')
            stg[traj[i]][traj[i + 1]]['label'] = f"{label}, {cell.split('_')[1]}".strip(', ')
    return stg


def plot_tsne(data, title, filename, marker, show=False):
    tsne_results = TSNE(n_components=2, random_state=24, perplexity=150).fit_transform(data)
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=1, marker=marker)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return tsne_results


def attractor_ASplus_minus_analysis(attractor_value_to_check, cell_attractor_file, metadata_file):
    as_plus_count = as_minus_count = total_cells = 0
    metadata = pd.read_csv(metadata_file, sep=" ", header=None).reset_index(drop=True)
    with open(cell_attractor_file, 'r') as f:
        lines = f.readlines()
    as_plus_indices, as_minus_indices = [], []
    for line in lines:
        match = re.search(r"Attractor:\s*(\d+),", line)
        if match:
            attractor_value = int(match.group(1))
            if attractor_value == attractor_value_to_check:
                total_cells += 1
                cell_num_trajectory = line.split(':')[0].split('_')[1]
                cell_state = metadata.iloc[int(cell_num_trajectory), 2]
                if cell_state == "AS+":
                    as_plus_count += 1
                    as_plus_indices.append(cell_num_trajectory)
                elif cell_state == "AS-":
                    as_minus_count += 1
                    as_minus_indices.append(cell_num_trajectory)
    if total_cells > 0:
        as_plus_percentage  = (as_plus_count  / total_cells) * 100
        as_minus_percentage = (as_minus_count / total_cells) * 100
    else:
        as_plus_percentage = as_minus_percentage = 0
    return (as_plus_count, as_minus_count, total_cells,
            as_plus_percentage, as_minus_percentage,
            as_plus_indices, as_minus_indices)


# ============================================================
# Transition matrices
# ============================================================

def compute_state_and_cell_transition_matrices(cleaned_trajectories, state_dict, trajectories):
    unique_states = [
        state_dict[key]
        for key in sorted(state_dict.keys(), key=lambda x: int(x.split()[1]))
    ]
    num_states = len(unique_states)
    num_cells  = len(cleaned_trajectories)

    state_transition_matrix    = np.zeros((num_states, num_states))
    state_occupancy            = np.zeros((num_cells,  num_states))
    weighted_transition_matrix = np.zeros((num_states, num_states))

    for cell_id, traj in enumerate(cleaned_trajectories.values()):
        if not traj:
            continue
        for state in traj:
            if 1 <= state <= num_states:
                state_occupancy[cell_id, state - 1] = 1
        for i in range(len(traj) - 1):
            si, sj = traj[i] - 1, traj[i + 1] - 1
            state_transition_matrix[si, sj]    += 1
            weighted_transition_matrix[si, sj] += 1

    row_sums = state_transition_matrix.sum(axis=1, keepdims=True)
    nz = row_sums.flatten() != 0
    state_transition_matrix[nz] /= row_sums[nz]

    cell_transition_matrix = state_occupancy @ state_transition_matrix @ state_occupancy.T
    np.fill_diagonal(cell_transition_matrix, 0)
    rs  = cell_transition_matrix.sum(axis=1, keepdims=True)
    nz2 = rs.flatten() != 0
    if nz2.any():
        cell_transition_matrix[nz2] /= rs[nz2]

    return (state_transition_matrix, state_occupancy,
            cell_transition_matrix, weighted_transition_matrix)


# ============================================================
# Plotting helpers
# ============================================================

def _outpath(outdir: str, filename: str) -> str:
    return os.path.join(outdir, filename)


def plot_binary_connectivity_only(cell_transition_matrix, output_file,
                                   threshold=0.05, max_cells=500):
    print(f"\nCreating binary connectivity visualisation → {output_file}")
    num_cells     = cell_transition_matrix.shape[0]
    binary_matrix = (cell_transition_matrix > threshold).astype(int)

    out_degree   = binary_matrix.sum(axis=1)
    in_degree    = binary_matrix.sum(axis=0)
    total_degree = out_degree + in_degree

    if num_cells > max_cells:
        top_indices = np.argsort(total_degree)[-max_cells:]
    else:
        top_indices = np.arange(num_cells)
        max_cells   = num_cells

    plot_matrix = binary_matrix[np.ix_(top_indices, top_indices)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(plot_matrix, cmap="binary", aspect="auto", interpolation="nearest")
    ax.set_xlabel("To Cell",   fontsize=12, fontweight="bold")
    ax.set_ylabel("From Cell", fontsize=12, fontweight="bold")

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="black",            label="Connection Present"),
        Rectangle((0, 0), 1, 1, fc="white", ec="black", label="No Connection"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="black")

    # Descriptive counts, not a test result — kept on the panel.
    stats_text = (f"Total cells: {num_cells:,}\n"
                  f"Shown: {max_cells}\n"
                  f"Connections: {plot_matrix.sum()}\n"
                  f"Density: {plot_matrix.sum()/(max_cells**2)*100:.2f}%")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


def plot_binary_connectivity_with_regions(cell_transition_matrix, cell_labels,
                                          barcode_to_label,
                                          output_file,
                                          threshold=0.05, max_cells=500):
    """
    Binary connectivity sorted by AS- / Mixed / AS+.
    Uses barcode_to_label (from combined_metadata) to look up each cell's condition.
    """
    print(f"\nCreating region-annotated connectivity → {output_file}")
    num_cells = cell_transition_matrix.shape[0]

    cell_conditions = []
    for cell in cell_labels:
        bc   = trajectory_key_to_barcode(cell)
        cond = barcode_to_label.get(bc, "Mixed")
        cell_conditions.append(cond)

    as_minus_cells = [i for i, c in enumerate(cell_conditions) if c == "AS-"]
    as_plus_cells  = [i for i, c in enumerate(cell_conditions) if c == "AS+"]
    mixed_cells    = [i for i, c in enumerate(cell_conditions)
                      if c not in ("AS-", "AS+")]

    sorted_indices = as_minus_cells + mixed_cells + as_plus_cells
    if len(sorted_indices) > max_cells:
        total = len(sorted_indices)
        n_m = max(1, int(max_cells * len(as_minus_cells) / total))
        n_x = max(1, int(max_cells * len(mixed_cells)    / total))
        n_p = max(1, max_cells - n_m - n_x)
        sorted_indices = (as_minus_cells[:n_m] + mixed_cells[:n_x]
                          + as_plus_cells[:n_p])
        n_am, n_mx, n_ap = n_m, n_x, n_p
    else:
        n_am = len(as_minus_cells)
        n_mx = len(mixed_cells)
        n_ap = len(as_plus_cells)

    binary_matrix = (cell_transition_matrix > threshold).astype(int)
    plot_matrix   = binary_matrix[np.ix_(sorted_indices, sorted_indices)]
    num_shown     = len(sorted_indices)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(plot_matrix, cmap="binary", aspect="auto", interpolation="nearest")
    ax.set_xlabel("To Cell",   fontsize=12, fontweight="bold")
    ax.set_ylabel("From Cell", fontsize=12, fontweight="bold")

    b1, b2 = n_am, n_am + n_mx
    for boundary in (b1, b2):
        if 0 < boundary < num_shown:
            ax.axvline(x=boundary - 0.5, color="gray", linestyle="--", linewidth=1.5)
            ax.axhline(y=boundary - 0.5, color="gray", linestyle="--", linewidth=1.5)

    for label, start, end in [("AS-", 0, b1), ("Mixed", b1, b2), ("AS+", b2, num_shown)]:
        if end > start:
            mid = (start + end) / 2
            ax.text(-0.02, mid / num_shown, label, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", ha="right", va="center")

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="black",            label="Connection Present"),
        Rectangle((0, 0), 1, 1, fc="white", ec="black", label="No Connection"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="black")

    # Descriptive counts, not a test result — kept on the panel.
    stats_text = (f"Total cells: {num_cells:,}\n"
                  f"Shown: {num_shown}\n"
                  f"Connections: {plot_matrix.sum()}\n"
                  f"Density: {plot_matrix.sum()/(num_shown**2)*100:.2f}%\n\n"
                  f"AS-: {n_am}\nMixed: {n_mx}\nAS+: {n_ap}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


# ============================================================
# Figure 2  (network diagram, basin composition, gene heatmap,
#            3D t-SNE pseudotime landscape, basin sub-STG)
# ============================================================

def generate_figure2(args, outdir: str):
    """
    Generates:
      Figure2ANetwork.png  – pathway network coloured by log2 fold-change,
                              node size by scBONITA importance score
      Figure2B.png          – state-transition subgraph for the 5th-largest
                              attractor basin
      Figure2C.png          – bar chart of top-5 attractor basin sizes
      Figure2D.png          – ON/OFF heatmap of genes across top-5 attractors
      Figure2E.png          – 3D t-SNE of unique states coloured by pseudotime
      FigureSTG.png         – full combined-trajectory state transition graph
      Figure2_Combined.png  – a.–e. combined panel (2x2 + full-width row)

    Also computes and saves (needed downstream by generate_all_figures3):
      cell_dissimilarity_matrix.pkl
      attractor_type_dict.pkl

    No statistical tests are performed for Figure 2, so nothing is added to
    the statistics registry here.
    """
    print("\n" + "=" * 60)
    print("Generating Figure 2 : Network / Basins / Gene Heatmap / 3D t-SNE")
    print("=" * 60)

    # ---------------- Importance scores ----------------
    gene_names = []
    importance_scores = []
    with open(args.importance_scores, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                gene, score = parts
                try:
                    importance_scores.append(float(score))
                    gene_names.append(gene)
                except ValueError:
                    print(f"  Warning: could not convert score '{score}' for gene '{gene}'. Skipping.")

    print(f"  Loaded importance scores for {len(importance_scores)} genes.")
    if importance_scores:
        top_30_indices = np.argsort(importance_scores)[-30:][::-1]
        print("  Top 30 important genes:")
        for idx in top_30_indices:
            gene  = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
            score = importance_scores[idx]
            print(f"    {gene}: {score:.4f}")

    # ---------------- Fold change ----------------
    foldchange_dict = {}
    with open(args.foldchange) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 4:
                key = parts[0]
                try:
                    value = float(parts[4].strip())
                except ValueError:
                    value = 0.0
                foldchange_dict[key] = np.log2(value + 1e-6)

    # ---------------- Figure 2A: network diagram ----------------
    def _resolve_node_overlaps(pos, node_list, node_sizes,
                                fig_w_in=10, fig_h_in=7.5,
                                padding=1.25, iterations=500):
        """
        Push nodes apart (in the same coordinate system spring_layout uses)
        until circles sized per `node_sizes` (matplotlib scatter's points^2
        area convention) stop overlapping, given the eventual figure size.

        Returns a pos dict covering ALL nodes originally in `pos` — nodes in
        `node_list` get their de-overlapped coordinates, every other node
        (e.g. genes with no importance score) keeps its original spring_layout
        position untouched. This matters because draw_networkx_edges needs a
        position for every node touched by an edge, not just the ones we're
        resizing.
        """
        new_pos = dict(pos)  # start from a full copy — nothing gets dropped

        if len(node_list) < 2:
            # Nothing to de-overlap.
            radii_data = (np.sqrt(np.asarray(node_sizes) / np.pi)
                        if len(node_list) else np.array([]))
            return new_pos, radii_data

        coords = np.array([pos[n] for n in node_list], dtype=float)
        radii_pts = np.sqrt(np.asarray(node_sizes) / np.pi)   # matplotlib radius in points

        x_span = coords[:, 0].max() - coords[:, 0].min() or 1.0
        y_span = coords[:, 1].max() - coords[:, 1].min() or 1.0
        pts_per_inch = 72.0
        dux = x_span / (fig_w_in * pts_per_inch)   # data-units per point, x
        duy = y_span / (fig_h_in * pts_per_inch)   # data-units per point, y
        radii_data = radii_pts * ((dux + duy) / 2)

        for _ in range(iterations):
            moved = False
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    delta = coords[i] - coords[j]
                    dist  = np.linalg.norm(delta)
                    min_dist = (radii_data[i] + radii_data[j]) * padding
                    if dist < min_dist:
                        if dist < 1e-9:
                            delta = np.random.randn(2) * 1e-3
                            dist  = np.linalg.norm(delta)
                        push      = (min_dist - dist) / 2
                        direction = delta / dist
                        coords[i] += direction * push
                        coords[j] -= direction * push
                        moved = True
            if not moved:
                break

        for k, n in enumerate(node_list):
            new_pos[n] = tuple(coords[k])

        return new_pos, radii_data

    # Manual per-gene nudges (data-space fractions), applied AFTER the automatic
    # overlap resolution. Add entries here for any node that's still crowded —
    # this is cheaper and more reliable than trying to tune the algorithm further.
    MANUAL_NODE_OFFSETS = {
        # "XBP1":  (0.06, -0.04),
        # "ATF4":  (-0.05, 0.05),
    }

    G = nx.read_graphml(args.network)
    node_list = [n for n in G.nodes() if n in gene_names]

    if node_list and importance_scores:
        node_sizes = [
            300 + 7000 * (importance_scores[gene_names.index(n)] / max(importance_scores))
            if n in gene_names else 300
            for n in node_list
        ]
    else:
        node_sizes = [300 for _ in node_list]
    node_sizes = np.array(node_sizes)

    log_fold_changes = np.array([foldchange_dict.get(n, 0) for n in node_list])
    log_fold_changes = np.clip(log_fold_changes, -1.5, 1.5)

    cmap = plt.cm.coolwarm
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    node_colors = [cmap(norm(val)) for val in log_fold_changes]

    FIG_W, FIG_H = 12, 9   # bumped up from 10x7.5 to give the layout more room
    plt.figure(figsize=(FIG_W, FIG_H))

    # More spacing in the base layout: larger k, more iterations, fixed seed.
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(len(G), 1)), iterations=300)

    # Size-aware de-overlap pass, then manual per-node nudges.
    pos, radii_data = _resolve_node_overlaps(pos, node_list, node_sizes,
                                            fig_w_in=FIG_W, fig_h_in=FIG_H)
    for gene, (dx, dy) in MANUAL_NODE_OFFSETS.items():
        if gene in pos:
            x, y = pos[gene]
            pos[gene] = (x + dx, y + dy)
    # Nodes in G but not in node_list (no importance score) still need a position
    # for edge drawing — spring_layout already gave them one; leave as-is.

    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=node_sizes,
                            node_color=node_colors, edgecolors='black', linewidths=0.6)
    # Slight curvature on edges reduces edge-over-label collisions and makes
    # parallel/hub edges visually distinguishable.
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=1.0, arrows=True,
                            arrowsize=30, arrowstyle='-|>',
                            connectionstyle='arc3,rad=0.08',
                            min_source_margin=12, min_target_margin=12)

    # Draw labels just above each node (offset by its own radius) instead of
    # dead-center, with a white halo so they stay legible over edges/nodes.
    import matplotlib.patheffects as pe
    radius_lookup = dict(zip(node_list, radii_data))
    for n in node_list:
        x, y = pos[n]
        r = radius_lookup.get(n, 0.02)
        txt = ax.text(x, y + r * 1.15, n, fontsize=11, fontweight='bold',
                    color='black', ha='center', va='bottom', zorder=20)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.axis('off')
    plt.tight_layout()
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label("Log2 Fold Change", fontsize=18, fontweight='bold')
    cbar.outline.set_linewidth(2)
    plt.savefig(_outpath(outdir, "Figure2ANetwork.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved Figure2ANetwork.png")

    # ---------------- Trajectories ----------------
    trajectories = read_trajectories(args.trajectories)

    cleaned_trajectories = {}
    with open(_outpath(outdir, "cleaned_trajectory.txt"), 'w') as f:
        for key, value in trajectories.items():
            cleaned = clean_trajectory(value)
            cleaned_trajectories[key] = cleaned
            f.write(f"{key}: {cleaned}\n")

    # ---------------- FigureSTG: full state transition graph ----------------
    stg_combined = plot_stg(trajectories)
    plt.figure(figsize=(10, 8))
    pos_stg = nx.spring_layout(stg_combined)
    nx.draw(stg_combined, pos_stg, with_labels=True, node_size=500, node_color='lightgray',
            edge_color='blue', font_size=8, font_color='black', connectionstyle="arc3,rad=0.11")
    plt.savefig(_outpath(outdir, "FigureSTG.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved FigureSTG.png")

    # ---------------- Unique states ----------------
    with open(args.unique_states, 'r') as f:
        columns_dict = {
            key.strip(): list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(',')))
            for key, value in (line.strip().split(':') for line in f)
        }
    unique_states = list(columns_dict.values())
    state_labels  = list(columns_dict.keys())
    state_data    = np.array(unique_states)

    tsne_results = plot_tsne(state_data, "t-SNE Plot of Unique States",
                              None, marker='s')

    # ---------------- Attractors ----------------
    attractors           = {}
    attractor_value_dict = {}
    with open(_outpath(outdir, "cell_and_attractor.txt"), 'w') as f:
        for cell, trajectory in trajectories.items():
            cleaned = clean_trajectory(trajectory)
            single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned)
            attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
            attractors.setdefault(attractor, []).append(cell)
            formatted_attractor_state = f"State {attractor_state}"
            if formatted_attractor_state in columns_dict:
                attractor_value = columns_dict[formatted_attractor_state]
                f.write(f"{cell}: {trajectory} -> Attractor: {attractor_state}, Value: {attractor_value}\n")
                attractor_value_dict[cell] = attractor_value
            else:
                f.write(f"{cell}: {trajectory} -> Attractor: {attractor_state}\n")
                attractor_value_dict[cell] = attractor_state

    with open(_outpath(outdir, "attractor_value_dict.txt"), 'w') as f:
        print(attractor_value_dict, file=f)

    unique_attractors = list(attractors.keys())

    # ---------------- Basin analysis ----------------
    basins = {attractor: [cell for cell, trajectory in trajectories.items() if attractor in trajectory]
              for attractor in unique_attractors}
    sorted_attractors = sorted(basins, key=lambda x: len(basins[x]), reverse=True)
    sorted_sizes      = [len(basins[attractor]) for attractor in sorted_attractors]

    # ---------------- Figure 2C: basin bar chart ----------------
    cmap_blues  = plt.get_cmap('Blues')
    colors_bars = [cmap_blues(i / 4) for i in reversed(range(5))]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111)
    bars = ax.bar(range(5), sorted_sizes[:5], tick_label=sorted_attractors[:5],
                  color=colors_bars, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Attr {i+1}" for i in range(5)], rotation=45, fontsize=18)
    ax.set_xlabel("Top Attractors", fontsize=18, fontweight='bold')
    ax.set_ylabel("Number of Cells", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(_outpath(outdir, "Figure2C.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved Figure2C.png")

    # ---------------- Figure 2D: ON/OFF gene heatmap ----------------
    state_genes = {int(k.split()[1]): np.array(v) for k, v in columns_dict.items()}

    cell_traj_path = FilePath(args.cell_trajectory)
    if not cell_traj_path.exists():
        traj_dir  = FilePath(args.trajectories).parent
        fallbacks = sorted(glob.glob(str(traj_dir / "cell_*trajectory*.csv")))
        if fallbacks:
            cell_traj_path = FilePath(fallbacks[0])
        else:
            raise FileNotFoundError(
                f"cell_trajectory file not found: {args.cell_trajectory} "
                f"(no cell_*trajectory*.csv fallback found in {traj_dir} either)"
            )

    with open(cell_traj_path, 'r') as f:
        gene_names_list = [line.strip().split(',')[0] for line in f.readlines()]
    print(f"  Total genes in gene_names_list: {len(gene_names_list)}")

    num_genes_total       = len(gene_names_list)
    attractor_gene_matrix = np.zeros((num_genes_total, 5), dtype=int)
    for col_idx, attractor in enumerate(sorted_attractors[:5]):
        if attractor in state_genes:
            state_vector = state_genes[attractor]
            copy_length  = min(len(state_vector), num_genes_total)
            attractor_gene_matrix[:copy_length, col_idx] = state_vector[:copy_length]

    genes_on_mask        = np.any(attractor_gene_matrix == 1, axis=1)
    filtered_matrix      = attractor_gene_matrix[genes_on_mask]
    filtered_gene_names  = [gene_names_list[i] for i in range(len(gene_names_list)) if genes_on_mask[i]]

    gene_list_path = os.path.join(outdir, "filtered_gene_names.txt")

    with open(gene_list_path, "w", encoding="utf-8") as f:
        for gene in filtered_gene_names:
            f.write(f"{gene}\n")
        print(f"  Genes ON in at least one of top-5 attractors: {len(filtered_gene_names)}")

    if len(filtered_gene_names) > 100:
        filtered_matrix     = filtered_matrix[:100]
        filtered_gene_names = filtered_gene_names[:100]

    set1_colors    = sns.color_palette("Set1", 10)
    cmap_heatmap   = ListedColormap([set1_colors[8], set1_colors[1]])
    bounds         = [-0.5, 0.5, 1.5]
    norm_heatmap   = plt.cm.colors.BoundaryNorm(bounds, cmap_heatmap.N)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111)
    ax.imshow(filtered_matrix, cmap=cmap_heatmap, norm=norm_heatmap, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Attr {i+1}" for i in range(5)], rotation=45, fontsize=18)
    ax.set_xlabel("Top Attractors", fontsize=18, fontweight='bold')

    gene_labels_without_scores = []
    gene_labels_with_scores = []
    for gene in filtered_gene_names:
        if gene in gene_names:
            score = importance_scores[gene_names.index(gene)]
            gene_labels_with_scores.append(f"{gene} ({score:.2f})")
            gene_labels_without_scores.append(gene)
        else:
            gene_labels_with_scores.append(gene)
    ax.set_yticks(range(len(filtered_gene_names)))
    ax.set_yticklabels(gene_labels_without_scores, fontsize=18)
    ax.set_ylabel("Selected Genes", fontsize=18, fontweight='bold')

    for (i, j), val in np.ndenumerate(filtered_matrix):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', lw=2))

    legend_elements = [
        Patch(facecolor=set1_colors[8], edgecolor='black', label='Off'),
        Patch(facecolor=set1_colors[1], edgecolor='black', label='On'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(_outpath(outdir, "Figure2D.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved Figure2D.png")
    print(f"  Genes shown in Figure 2D: {filtered_gene_names}")

    # ---------------- Figure 2B: 5th-largest attractor sub-STG ----------------
    G_attractor    = nx.DiGraph()
    terminal_states = set()

    if len(sorted_attractors) >= 5:
        attractor        = sorted_attractors[4]
        attractor_color  = 'blue'
        attractor_cells  = basins[attractor]
        attractor_trajs  = [cleaned_trajectories[cell] for cell in attractor_cells
                            if cell in cleaned_trajectories]

        edge_to_traj = {}
        for traj_idx, traj in enumerate(attractor_trajs):
            if len(traj) == 0:
                continue
            terminal_states.add(traj[-1])
            for i in range(len(traj) - 1):
                edge = (traj[i], traj[i + 1])
                G_attractor.add_edge(traj[i], traj[i + 1])
                edge_to_traj[edge] = traj_idx
            final_state = traj[-1]
            if final_state == attractor or traj.count(final_state) > 1:
                G_attractor.add_edge(final_state, final_state)

        print(f"  Figure 2B: Attractor {attractor} basin — "
              f"{len(attractor_trajs)} trajectories, terminal states {terminal_states}")

        mismatch_count = sum(1 for traj in attractor_trajs if len(traj) > 0 and traj[-1] != attractor)
        if mismatch_count > 0:
            print(f"  WARNING: {mismatch_count} trajectories don't end at attractor {attractor} — "
                  f"basin assignment may be inconsistent.")

    if len(G_attractor) > 0:
        plt.figure(figsize=(20, 20))
        pos_att = nx.kamada_kawai_layout(G_attractor)
        pos_att = jitter_positions(pos_att, jitter=0.15, min_dist=0.08, max_iter=200)

        attractor_states = terminal_states.copy()
        for node in G_attractor.nodes():
            if G_attractor.has_edge(node, node):
                attractor_states.add(node)

        if attractor_states:
            center = np.array([0.0, 0.0])
            if len(attractor_states) == 1:
                pos_att[list(attractor_states)[0]] = center
            else:
                radius = 0.1
                for i, state in enumerate(attractor_states):
                    angle = 2 * np.pi * i / len(attractor_states)
                    pos_att[state] = center + radius * np.array([np.cos(angle), np.sin(angle)])

        labels = {node: f"S{node}" for node in G_attractor.nodes()}

        positions   = np.array([pos_att[node] for node in G_attractor.nodes()])
        node_alphas = []
        for i, node in enumerate(G_attractor.nodes()):
            distances = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
            nearby    = np.sum(distances < 0.2)
            alpha = 1.0 if node in attractor_states else max(0.3, 1.0 - (nearby - 1) * 0.05)
            node_alphas.append(alpha)

        regular_nodes    = [(n, a) for n, a in zip(G_attractor.nodes(), node_alphas) if n not in attractor_states]
        attractor_nodes  = [(n, a) for n, a in zip(G_attractor.nodes(), node_alphas) if n in attractor_states]

        for node, alpha in regular_nodes:
            nx.draw_networkx_nodes(G_attractor, pos_att, nodelist=[node], node_size=3000,
                                    node_color=attractor_color, alpha=alpha, node_shape='s')

        num_trajs = len(attractor_trajs) if len(sorted_attractors) >= 5 else 0
        for traj_idx in range(num_trajs):
            traj_edges = [edge for edge, idx in edge_to_traj.items() if idx == traj_idx]
            if traj_edges:
                nx.draw_networkx_edges(
                    G_attractor, pos_att, edgelist=traj_edges,
                    edge_color='black', width=6, arrows=True,
                    arrowsize=40, arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
                    alpha=0.6, min_source_margin=15, min_target_margin=15,
                )

        for node, alpha in regular_nodes:
            nx.draw_networkx_labels(G_attractor, pos_att, labels={node: labels[node]},
                                    font_size=12, font_color='white', alpha=min(1.0, alpha + 0.2))

        for node, alpha in attractor_nodes:
            nx.draw_networkx_nodes(G_attractor, pos_att, nodelist=[node], node_size=4000,
                                    node_color='gold', alpha=1.0, node_shape='s',
                                    edgecolors='darkgoldenrod', linewidths=3)

        for node, alpha in attractor_nodes:
            nx.draw_networkx_labels(G_attractor, pos_att, labels={node: labels[node]},
                                    font_size=14, font_color='black', font_weight='bold', alpha=1.0)

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(_outpath(outdir, "Figure2B.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved Figure2B.png")
    else:
        print("  Skipping Figure2B – fewer than 5 attractor basins or empty basin graph.")

    # ---------------- Pseudotime + dissimilarity matrix ----------------
    normalized_steps_to_attractor = {
        state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1)
               / len(clean_trajectory(trajectory))
        for trajectory in cleaned_trajectories.values()
        for state in clean_trajectory(trajectory)
        if len(clean_trajectory(trajectory)) > 1
    }
    steps_to_attractor = {
        state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1)
        for trajectory in cleaned_trajectories.values()
        for state in clean_trajectory(trajectory)
        if len(clean_trajectory(trajectory)) > 1
    }

    with open(_outpath(outdir, "report_steps_to_attractor.txt"), "w") as f:
        for state, steps in steps_to_attractor.items():
            f.write(f"State {state}: Steps to Attractor: {steps}\n")
    with open(_outpath(outdir, "report_normalized_steps.txt"), "w") as f:
        for state, steps in normalized_steps_to_attractor.items():
            f.write(f"State {state}: Normalized Steps to Attractor: {steps}\n")

    dissimilarity_pkl_path   = _outpath(outdir, "cell_dissimilarity_matrix.pkl")
    dissimilarity_alt_path   = _outpath(outdir, "dissimilarity_matrix.pkl")
    attractor_type_pkl_path  = _outpath(outdir, "attractor_type_dict.pkl")
    pseudotime_pkl_path      = _outpath(outdir, "weighted_pseudotime_dict.pkl")
    force_recompute = getattr(args, "force_recompute_dissimilarity", False)

    dissimilarity_source = None
    if os.path.exists(dissimilarity_pkl_path):
        dissimilarity_source = dissimilarity_pkl_path
    elif os.path.exists(dissimilarity_alt_path):
        dissimilarity_source = dissimilarity_alt_path

    have_cached_dissim = (dissimilarity_source is not None
                          and os.path.exists(attractor_type_pkl_path))

    if not force_recompute and have_cached_dissim:
        print(f"  Found existing dissimilarity matrix → reusing "
              f"{dissimilarity_source} instead of recomputing.")
        print("  (delete this file, or pass --force-recompute-dissimilarity, "
              "to force a fresh O(n^2) recomputation)")
        with open(dissimilarity_source, "rb") as pkl_f:
            cell_dissimilarity_matrix = pickle.load(pkl_f)
        with open(attractor_type_pkl_path, "rb") as pkl_f:
            attractor_type_dict = pickle.load(pkl_f)

        n_cells = len(cleaned_trajectories)
        if cell_dissimilarity_matrix.shape != (n_cells, n_cells):
            print(f"  WARNING: cached matrix shape {cell_dissimilarity_matrix.shape} "
                  f"does not match current cell count ({n_cells}, {n_cells}). "
                  f"The cached file is likely from a different trajectories.txt — "
                  f"delete it and rerun to recompute for this dataset.")

        if dissimilarity_source != dissimilarity_pkl_path:
            with open(dissimilarity_pkl_path, "wb") as pkl_f:
                pickle.dump(cell_dissimilarity_matrix, pkl_f)
            print(f"  Also wrote canonical copy → {dissimilarity_pkl_path}")

        if os.path.exists(pseudotime_pkl_path):
            print(f"  Found existing weighted_pseudotime_dict.pkl → reusing "
                  f"{pseudotime_pkl_path}.")
            with open(pseudotime_pkl_path, "rb") as pkl_f:
                pseudotime_dict = pickle.load(pkl_f)
        else:
            pseudotime_dict = compute_weighted_pseudotime(cleaned_trajectories, attractor_value_dict)
            with open(pseudotime_pkl_path, "wb") as pkl_f:
                pickle.dump(pseudotime_dict, pkl_f)
            print(f"  Saved weighted pseudotime dict → {pseudotime_pkl_path}")
    else:
        pseudotime_dict, cell_dissimilarity_matrix, attractor_type_dict = \
            compute_weighted_pseudotime_full(cleaned_trajectories, attractor_value_dict, outdir)

        with open(dissimilarity_pkl_path, "wb") as pkl_f:
            pickle.dump(cell_dissimilarity_matrix, pkl_f)
        print(f"  Saved cell dissimilarity matrix ({cell_dissimilarity_matrix.shape}) → {dissimilarity_pkl_path}")

        with open(attractor_type_pkl_path, "wb") as pkl_f:
            pickle.dump(attractor_type_dict, pkl_f)
        print(f"  Saved attractor type dict ({len(attractor_type_dict)} cells) → {attractor_type_pkl_path}")

        with open(pseudotime_pkl_path, "wb") as pkl_f:
            pickle.dump(pseudotime_dict, pkl_f)
        print(f"  Saved weighted pseudotime dict → {pseudotime_pkl_path}")

    state_vector_with_pseudotime = np.array([
        np.array(unique_states[i]) * (1 - normalized_steps_to_attractor.get(i, 0))
        for i in range(len(unique_states))
    ])

    tsne_results = plot_tsne(
        state_vector_with_pseudotime,
        "t-SNE Plot of Unique States with Pseudotime",
        _outpath(outdir, "t-SNE_Plot_Colored_with_Pseudotime.png"),
        marker='s',
    )

    pseudotime_values = [
        normalized_steps_to_attractor.get(int(state.split()[1]), 0)
        for state in state_labels
    ]

    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    for i, state in enumerate(state_labels):
        color = plt.cm.coolwarm(1 - pseudotime_values[i])
        ax_2d.scatter(tsne_results[i, 0], tsne_results[i, 1],
                      color=color, alpha=0.6, marker='s', edgecolors='none', s=100)
    ax_2d.set_xlabel("t-SNE 1")
    ax_2d.set_ylabel("t-SNE 2")
    ax_2d.set_title("2D t-SNE Plot of Unique States with Pseudotime")
    sm = plt.cm.ScalarMappable(cmap='coolwarm_r',
                                norm=plt.Normalize(vmin=min(pseudotime_values), vmax=max(pseudotime_values)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_2d, label='Pseudotime')
    plt.tight_layout()
    plt.savefig(_outpath(outdir, "t-SNE_2D_Plot_Colored_with_Pseudotime.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Figure 2E: 3D t-SNE pseudotime landscape ----------------
    fig = plt.figure(figsize=(12, 10))
    tsne_3d = TSNE(n_components=3, random_state=42,
                   perplexity=min(30, max(2, len(unique_states) - 1)),
                   max_iter=1000, learning_rate=200, early_exaggeration=12.0)
    tsne_results_3d = tsne_3d.fit_transform(state_vector_with_pseudotime)

    spread_factor = 5.0
    tsne_results_3d = tsne_results_3d * spread_factor

    ax = fig.add_subplot(111, projection='3d')
    for i, state in enumerate(state_labels):
        state_index = int(state.split()[1])
        color = plt.cm.coolwarm(1 - normalized_steps_to_attractor.get(state_index, 0))
        ax.scatter(tsne_results_3d[i, 0], tsne_results_3d[i, 1], tsne_results_3d[i, 2],
                   color=color, alpha=0.6, marker='s', edgecolors='none', s=100)

    x_min, x_max = np.min(tsne_results_3d[:, 0]), np.max(tsne_results_3d[:, 0])
    y_min, y_max = np.min(tsne_results_3d[:, 1]), np.max(tsne_results_3d[:, 1])
    z_min, z_max = np.min(tsne_results_3d[:, 2]), np.max(tsne_results_3d[:, 2])
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    ax.view_init(elev=15, azim=-110)

    indices_pseudotime_0 = [
        i for i, state in enumerate(state_labels)
        if normalized_steps_to_attractor.get(int(state.split()[1]), 0) == 0
    ]
    attractor_coordinates_3d = {}
    for i in indices_pseudotime_0:
        state_index = int(state_labels[i].split()[1])
        attractor_coordinates_3d[state_index] = (
            tsne_results_3d[i, 0], tsne_results_3d[i, 1], tsne_results_3d[i, 2]
        )
    print(f"  Identified {len(attractor_coordinates_3d)} attractor coordinates in 3D")

    label_positions = []
    for idx, attractor in enumerate(sorted_attractors[:5]):
        rank = idx + 1
        if attractor not in attractor_coordinates_3d:
            print(f"  Warning: Attractor {rank} (state {attractor}) missing 3D coordinates → skipped.")
            continue
        text_x, text_y, text_z = attractor_coordinates_3d[attractor]
        ax.scatter([text_x], [text_y], [text_z], marker='s', s=2000,
                   facecolors='gold', edgecolors='black', linewidths=3,
                   alpha=1.0, depthshade=False)
        label_positions.append((text_x, text_y, text_z, rank))
        print(f"  Attractor {rank}: (State idx {attractor}) "
              f"Position=({text_x:.2f}, {text_y:.2f}, {text_z:.2f}), "
              f"Basin size={len(basins[attractor])}")

    print("  Distances between attractors:")
    for i in range(len(label_positions)):
        for j in range(i + 1, len(label_positions)):
            x1, y1, z1, rank1 = label_positions[i]
            x2, y2, z2, rank2 = label_positions[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            print(f"    Attr {rank1} <-> Attr {rank2}: distance = {dist:.2f}")

    manual_offsets = {
        1: (0, 0, 0),
        2: (0, 0, 0),
        3: (-10, -5, 10),
        4: (10, 5, -10),
        5: (0, 0, 0),
    }
    for x, y, z, rank in label_positions:
        offset_x, offset_y, offset_z = manual_offsets.get(rank, (0, 0, 0))
        ax.text(x + offset_x, y + offset_y, z + offset_z, f"Attr {rank}",
                fontsize=18, fontweight='bold', color='black', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', linewidth=2, pad=3),
                zorder=10000)

    for axis in ([ax.w_xaxis, ax.w_yaxis, ax.w_zaxis] if hasattr(ax, 'w_xaxis')
                 else [ax.xaxis, ax.yaxis, ax.zaxis]):
        try:
            axis.line.set_linewidth(2)
        except AttributeError:
            try:
                axis.pane.set_linewidth(2)
            except Exception:
                pass
    for line in ax.get_lines():
        line.set_linewidth(2)

    ax.set_xlabel("t-SNE Component 1", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel("t-SNE Component 2", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_zlabel("t-SNE Component 3", fontsize=18, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='z', which='major', labelsize=18)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Normalized Pseudotime", shrink=0.5, pad=0)
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(_outpath(outdir, "Figure2E.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved Figure2E.png")

    # ---------------- Figure 2 combined (a.–e.) ----------------
    fnames = ["Figure2ANetwork.png", "Figure2B.png", "Figure2C.png",
              "Figure2D.png", "Figure2E.png"]
    paths  = [_outpath(outdir, f) for f in fnames]
    if all(os.path.exists(p) for p in paths):
        img_data = [mpimg.imread(p) for p in paths]
        labels   = ["a.", "b.", "c.", "d.", "e."]

        fig = plt.figure(figsize=(12, 16))
        gs  = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)

        for i in range(4):
            row, col = i // 2, i % 2
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_data[i]); ax.axis('off')
            label_y = 0.98 if i == 1 else 1.1
            ax.text(0.03, label_y, labels[i], transform=ax.transAxes,
                    fontsize=20, fontweight='bold', color='black',
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        ax5 = fig.add_subplot(gs[2, :])
        ax5.imshow(img_data[4]); ax5.axis('off')
        ax5.text(0.03, 0.88, labels[4], transform=ax5.transAxes,
                 fontsize=20, fontweight='bold', color='black',
                 ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0)
        plt.savefig(_outpath(outdir, "Figure2_Combined.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("  Saved Figure2_Combined.png")
    else:
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"  Skipping Figure2_Combined.png – missing: {missing}")

    print("Figure 2 complete.")
    return cleaned_trajectories, attractor_value_dict, filtered_gene_names


# ============================================================
# Cluster-count selection
# ============================================================
# k used to be hard-coded as `n_clusters = 5` inside generate_all_figures3,
# which quietly matched the "top 5 attractors" used elsewhere even though the
# two are unrelated quantities. It is now resolved from the config:
#
#   --n-clusters 7      force k = 7
#   --n-clusters auto   sweep k over [--cluster-k-min, --cluster-k-max] and
#                       take the k with the best silhouette, measured in the
#                       precomputed dissimilarity space the clustering is
#                       actually performed in (NOT the t-SNE plot space —
#                       see the diagnostics note in generate_all_figures3)
#
# The sweep writes cluster_k_selection.csv / .png so the choice is auditable,
# and every k's silhouette is pushed into the statistics registry.

def _cluster_palette(n):
    """n visually distinct colours, cycling tab20 past 20 clusters."""
    base = plt.get_cmap("tab10").colors + plt.get_cmap("tab20").colors
    return [base[i % len(base)] for i in range(n)]


def sweep_n_clusters(affinity_matrix, D_sym, k_min=2, k_max=10,
                     random_state=42, outdir=".", record=True):
    """
    Fit spectral clustering for each k in [k_min, k_max] and score it with
    the silhouette on the symmetrized precomputed distance.

    Returns (best_k, results_dataframe). Failures at a given k are recorded
    as nan rather than aborting the sweep, so one bad k doesn't cost you the
    whole run.
    """
    # Defensive: silhouette_score(metric="precomputed") rejects negative
    # entries outright, and a distance matrix should not have any. Clip
    # rather than crash mid-sweep, and say so if anything was clipped.
    D_sym = np.asarray(D_sym, dtype=float)
    _n_neg = int(np.sum(D_sym < 0))
    if _n_neg:
        print(f"  WARNING: {_n_neg} negative entries in the distance matrix "
              f"(min {D_sym.min():.4g}) — clipped to 0.")
        D_sym = np.clip(D_sym, 0.0, None)
        np.fill_diagonal(D_sym, 0.0)

    n_samples = D_sym.shape[0]
    k_max = int(min(k_max, max(2, n_samples - 1)))
    k_min = int(max(2, min(k_min, k_max)))

    print(f"\n  Sweeping k = {k_min}..{k_max} "
          f"(silhouette on the precomputed dissimilarity)...")

    rows = []
    for k in range(k_min, k_max + 1):
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                    random_state=random_state, n_init=10)
            lbl = sc.fit_predict(affinity_matrix)
            sil = silhouette_score(D_sym, lbl, metric="precomputed")
            sizes = np.bincount(lbl, minlength=k)
            smallest = int(sizes.min())
        except Exception as exc:
            print(f"    k={k:2d}: failed ({exc})")
            rows.append({"k": k, "silhouette": np.nan,
                         "smallest_cluster": np.nan, "cluster_sizes": ""})
            continue

        rows.append({"k": k, "silhouette": sil,
                     "smallest_cluster": smallest,
                     "cluster_sizes": " ".join(str(int(s)) for s in sizes)})
        print(f"    k={k:2d}: silhouette={sil:+.4f}   "
              f"smallest cluster={smallest:5d}   sizes=[{rows[-1]['cluster_sizes']}]")

        if record:
            record_stat(
                figure="Figure 3 (cluster-count sweep)",
                panels="cluster_k_selection.png",
                test=f"Silhouette of spectral clustering at k={k}, on the "
                     f"symmetrized precomputed dissimilarity",
                value=sil, value_name="score",
                k=k, smallest_cluster=smallest,
                cluster_sizes=rows[-1]["cluster_sizes"],
            )

    df = pd.DataFrame(rows)
    df.to_csv(_outpath(outdir, "cluster_k_selection.csv"), index=False)

    valid = df.dropna(subset=["silhouette"])
    if valid.empty:
        print("    All k failed — falling back to k=5.")
        return 5, df
    best_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])

    # Sweep plot
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid["k"], valid["silhouette"], marker="o", lw=2, color="#2c7fb8")
        ax.axvline(best_k, color="crimson", ls="--", lw=2,
                   label=f"selected k = {best_k}")
        ax.set_xlabel("Number of clusters (k)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Silhouette (dissimilarity space)",
                      fontsize=14, fontweight="bold")
        ax.set_xticks(list(valid["k"]))
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=12)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        plt.tight_layout()
        plt.savefig(_outpath(outdir, "cluster_k_selection.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"    (sweep plot failed: {exc})")

    print(f"  Selected k = {best_k} "
          f"(silhouette {valid['silhouette'].max():+.4f})")
    print(f"  Wrote cluster_k_selection.csv / .png")
    return best_k, df


def resolve_n_clusters(args, affinity_matrix, D_sym, outdir="."):
    """
    Turn the --n-clusters setting into an integer k.

    Accepts an int, a numeric string, or "auto" (sweep). Anything
    unrecognised falls back to the historical value of 5 with a warning
    rather than crashing mid-pipeline.
    """
    raw = getattr(args, "n_clusters", "auto")
    k_min = int(getattr(args, "cluster_k_min", 2))
    k_max = int(getattr(args, "cluster_k_max", 10))

    if isinstance(raw, str) and raw.strip().lower() == "auto":
        best_k, _ = sweep_n_clusters(affinity_matrix, D_sym,
                                     k_min=k_min, k_max=k_max, outdir=outdir)
        return best_k

    try:
        k = int(raw)
    except (TypeError, ValueError):
        print(f"  WARNING: could not interpret --n-clusters={raw!r}; using 5.")
        return 5

    if k < 2:
        print(f"  WARNING: --n-clusters={k} is below 2; using 2.")
        return 2

    print(f"  Using fixed k = {k} (no sweep; pass --n-clusters auto to search).")
    return k


# ============================================================
# Figure 3
# ============================================================
def generate_all_figures3(args, outdir: str):
    """
    Figure 3 panels.

    Statistics: the Mann-Whitney U test on pseudotime (AS- vs AS+ cells) and
    the silhouette score of the spectral clustering are both still computed on
    every run, but they are NO LONGER drawn onto Figure3A/Figure3EOrg or
    Figure3B/Figure3FOrg. They are handed to record_stat() and written to
    statistics_summary.txt / .csv instead.
    """
    print("\n" + "=" * 60)
    print("Generating Figure 3 : State / Cell Transition Matrices")
    print("=" * 60)

    # ------ load data ------
    trajectories         = read_trajectories(args.trajectories)
    cleaned_trajectories = {k: clean_trajectory(v) for k, v in trajectories.items()}

    with open(args.unique_states, "r") as fh:
        columns_dict = {
            key.strip(): list(map(int,
                value.strip().strip("[]").replace("(","").replace(")","").split(",")))
            for key, value in (line.strip().split(":") for line in fh)
        }

    unique_states = list(columns_dict.values())
    state_labels  = list(columns_dict.keys())

    num_to_label, barcode_to_label, barcode_to_participant = parse_metadata(args.metadata)

    _traj_keys   = list(trajectories.keys())[:5]
    _cond_counts = {}
    for _v in barcode_to_label.values():
        _cond_counts[_v] = _cond_counts.get(_v, 0) + 1
    print("  [DIAG] First 5 trajectory keys :", _traj_keys)
    print("  [DIAG] Extracted barcodes      :",
          [trajectory_key_to_barcode(k) for k in _traj_keys])
    print("  [DIAG] Condition counts in metadata:", _cond_counts)
    _hits = sum(1 for c in trajectories
                if barcode_to_label.get(trajectory_key_to_barcode(c)) is not None)
    print(f"  [DIAG] Cells with successful condition lookup: {_hits} / {len(trajectories)}")

    # ------ attractors / basins ------
    attractors           = {}
    basins               = {}
    attractor_value_dict = {}

    for cell, trajectory in trajectories.items():
        cleaned    = clean_trajectory(trajectory)
        single_att, cyclic_att, att_state, _ = find_attractors(cleaned)
        attractor  = single_att if single_att is not None else cyclic_att[0]

        attractors.setdefault(attractor, []).append(cell)
        basins.setdefault(attractor, []).append(cell)

        fmt_key = f"State {att_state}"
        bc      = trajectory_key_to_barcode(cell)
        attractor_value_dict[bc] = (
            columns_dict[fmt_key] if fmt_key in columns_dict else att_state
        )

    normalized_steps_to_attractor = {
        state: (
            (len(clean_trajectory(traj)) - clean_trajectory(traj).index(state) - 1)
            / len(clean_trajectory(traj))
        )
        for traj in cleaned_trajectories.values()
        for state in clean_trajectory(traj)
        if len(clean_trajectory(traj)) > 1
    }

    state_vector_with_pseudotime = np.array([
        np.array(unique_states[i]) * (1 - normalized_steps_to_attractor.get(i, 0))
        for i in range(len(unique_states))
    ])

    tsne_results = TSNE(n_components=2, random_state=24,
                        perplexity=150).fit_transform(state_vector_with_pseudotime)

    # ---- Stats formerly annotated on Figure3A/Figure3B/Figure3EOrg/Figure3FOrg ----
    # p-value: Mann-Whitney U test comparing pseudotime between AS- and AS+ cells.
    # Computed early (attractor_value_dict / barcode_to_label are already
    # available). The value goes to the statistics registry, NOT onto the panels.
    _pseudotime_dict_early        = compute_weighted_pseudotime(trajectories, attractor_value_dict)
    _cell_labels_early            = list(trajectories.keys())
    _cell_pseudotime_values_early = [_pseudotime_dict_early.get(c, 0) for c in _cell_labels_early]
    _cell_conditions_early = [barcode_to_label.get(trajectory_key_to_barcode(c), None)
                               for c in _cell_labels_early]
    _pt_as_minus_early = [_cell_pseudotime_values_early[i]
                          for i, c in enumerate(_cell_conditions_early) if c == "AS-"]
    _pt_as_plus_early  = [_cell_pseudotime_values_early[i]
                          for i, c in enumerate(_cell_conditions_early) if c == "AS+"]
    if len(_pt_as_minus_early) > 0 and len(_pt_as_plus_early) > 0:
        _u_stat, pseudotime_pvalue = mannwhitneyu(_pt_as_minus_early, _pt_as_plus_early,
                                                  alternative="two-sided")
    else:
        _u_stat, pseudotime_pvalue = np.nan, np.nan

    record_stat(
        figure="Figure 3",
        panels="Figure3A.png, Figure3EOrg.png",
        test="Mann-Whitney U, two-sided: cell pseudotime, AS- vs AS+",
        value=pseudotime_pvalue,
        value_name="p-value",
        n_AS_minus_cells=len(_pt_as_minus_early),
        n_AS_plus_cells=len(_pt_as_plus_early),
        median_pseudotime_AS_minus=(f"{np.median(_pt_as_minus_early):.6f}"
                                    if _pt_as_minus_early else "n/a"),
        median_pseudotime_AS_plus=(f"{np.median(_pt_as_plus_early):.6f}"
                                   if _pt_as_plus_early else "n/a"),
        U_statistic=(f"{_u_stat:.6g}" if np.isfinite(_u_stat) else "n/a"),
    )

    # ---- 3A ----
    indices_pseudotime1 = [
        i for i, s in enumerate(state_labels)
        if normalized_steps_to_attractor.get(int(s.split()[1]), 0) == 0
    ]
    attractor_coordinates = {}
    norm_sm = plt.Normalize(vmin=0, vmax=1)
    sm      = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm_sm)
    sm.set_array([])

    if indices_pseudotime1:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for i, s in enumerate(state_labels):
            sidx  = int(s.split()[1])
            color = plt.cm.coolwarm(1 - normalized_steps_to_attractor.get(sidx, 0))
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                        color=color, alpha=0.3, marker="s", s=120)
        for i in indices_pseudotime1:
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                        color="gold", edgecolor="black", s=120,
                        marker="s", linewidths=2)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight="bold")
        cbar.ax.tick_params(labelsize=18)
        cbar.outline.set_linewidth(2)
        for sp in ax.spines.values(): sp.set_linewidth(2)
        plt.xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
        plt.ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
        plt.xticks(fontsize=18); plt.yticks(fontsize=18)
        # Drawn only when ANNOTATE_STATS_ON_FIGURES is True.
        _stat_box(ax, 0.02, 0.98, _fmt_p(pseudotime_pvalue))
        plt.savefig(_outpath(outdir, "Figure3A.png"), dpi=300, bbox_inches="tight")
        plt.close()
        for i in indices_pseudotime1:
            sidx = int(state_labels[i].split()[1])
            attractor_coordinates[sidx] = (tsne_results[i, 0], tsne_results[i, 1])

    # ---- 3B ----
    cell_labels      = list(trajectories.keys())
    cell_tsne_coords = np.zeros((len(trajectories), 2))
    # Per-cell attractor identity, kept for the clustering diagnostics below.
    # `cell_attractor_ids` is the basin each cell actually falls into;
    # `_coord_missing` flags cells whose attractor has NO entry in
    # attractor_coordinates and therefore silently lands on (0, 0).
    cell_attractor_ids = []
    _coord_missing     = np.zeros(len(trajectories), dtype=bool)
    for idx, cell in enumerate(cell_labels):
        att_state = find_attractors(clean_trajectory(trajectories[cell]))[0]
        if att_state is None:
            att_state = find_attractors(clean_trajectory(trajectories[cell]))[1][0]
        cell_attractor_ids.append(att_state)
        _coord_missing[idx] = att_state not in attractor_coordinates
        cell_tsne_coords[idx] = attractor_coordinates.get(att_state, (0, 0))
    cell_attractor_ids = np.array(cell_attractor_ids)

    rng     = np.random.default_rng(seed=42)
    jitter  = rng.normal(0, 1.0, cell_tsne_coords.shape)
    cell_tsne_coords_jittered = cell_tsne_coords + jitter

    pseudotime_dict        = compute_weighted_pseudotime(trajectories, attractor_value_dict)
    cell_pseudotime_values = [pseudotime_dict.get(c, 0) for c in cell_labels]

    # ---- 3C spectral ----
    sorted_attractors = sorted(basins, key=lambda x: len(basins[x]), reverse=True)
    # NOTE: this is the top-5 ATTRACTOR BASINS (used for the convex hulls and
    # the gold anchor markers), which is a separate thing from the number of
    # CLUSTERS. Changing k below does not change how many basins are outlined.
    top_attractors    = sorted_attractors[:5]

    dissimilarity_path = getattr(args, "dissimilarity_matrix",
                                  getattr(args, "dissimilarity-matrix",
                                          "cell_dissimilarity_matrix.pkl"))
    print(f"Loading dissimilarity matrix from {dissimilarity_path} …")
    with open(dissimilarity_path, "rb") as fh:
        dissimilarity_matrix = pickle.load(fh)

    sigma           = np.median(dissimilarity_matrix[dissimilarity_matrix > 0])
    affinity_matrix = np.exp(-dissimilarity_matrix**2 / (2 * sigma**2))
    similarity_matrix = 1 - dissimilarity_matrix / np.max(dissimilarity_matrix)

    # Symmetrized distance matrix, built ONCE and reused for every silhouette
    # evaluation and for the k sweep. compute_weighted_pseudotime_full builds
    # the matrix asymmetrically (the most-traveled-path term depends on cell
    # i's basin, so d[i,j] != d[j,i]), and silhouette_score with
    # metric="precomputed" assumes symmetry.
    _D      = np.asarray(dissimilarity_matrix, dtype=float)
    _D_sym  = 0.5 * (_D + _D.T)
    _D_sym  = np.clip(_D_sym, 0.0, None)   # silhouette rejects negatives
    np.fill_diagonal(_D_sym, 0.0)
    _asym   = float(np.abs(_D - _D.T).max())

    X_coords = cell_tsne_coords_jittered

    # ---- how many clusters? ----
    # k is no longer hard-coded. `--n-clusters auto` (the default) sweeps
    # k over [--cluster-k-min, --cluster-k-max] and picks the k with the
    # highest silhouette measured in the dissimilarity space the clustering
    # is performed in. Pass an integer instead to force a specific k.
    n_clusters = resolve_n_clusters(args, affinity_matrix, _D_sym, outdir)

    spectral = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                   random_state=42, n_init=10)
    cluster_labels_spectral = spectral.fit_predict(affinity_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_kmeans = kmeans.fit_predict(similarity_matrix)

    # Palettes sized to k rather than the old five hard-coded colour names.
    cluster_palette = _cluster_palette(n_clusters)
    cluster_cmap    = ListedColormap(cluster_palette)

    spectral_centroids      = np.zeros((n_clusters, 2))
    spectral_cluster_counts = []
    for i in range(n_clusters):
        mask = cluster_labels_spectral == i
        pts  = X_coords[mask]
        spectral_cluster_counts.append(mask.sum())
        if len(pts): spectral_centroids[i] = pts.mean(axis=0)

    kmeans_centroids      = np.zeros((n_clusters, 2))
    kmeans_cluster_counts = []
    for i in range(n_clusters):
        mask = cluster_labels_kmeans == i
        pts  = X_coords[mask]
        kmeans_cluster_counts.append(mask.sum())
        if len(pts): kmeans_centroids[i] = pts.mean(axis=0)

    # ================================================================
    # Clustering quality — several numbers, because the original one was
    # answering the wrong question
    # ================================================================
    # (1) silhouette in Figure3B's JITTERED t-SNE EMBEDDING (the historical
    #     number). Figure3FOrg is a different embedding and gets its own
    #     score further down, next to where that panel is drawn.
    # (2) silhouette in the DISSIMILARITY SPACE the clustering was actually
    #     performed in — this is the number that describes the clustering
    # (3) adjusted Rand index of the spectral labels against basin identity
    #
    # Why (1) is near zero even when the clustering is fine:
    # `X_coords` is NOT an embedding of the cells. Every cell is placed at
    # its ATTRACTOR's t-SNE coordinate plus N(0, 1) jitter, so all cells
    # sharing a basin collapse onto one point, and there are only as many
    # distinct anchor positions as there are attractors with coordinates.
    # Cells whose attractor has no entry in `attractor_coordinates` all land
    # on (0, 0) in one undifferentiated pile. The spectral labels, meanwhile,
    # come from the dissimilarity affinity — a different space entirely.
    # Wherever two clusters share an anchor, their points are interleaved at
    # jitter scale, a(i) ≈ b(i), and the silhouette goes to ~0 by
    # construction. It is measuring anchor overlap, not cluster structure.
    _n_origin   = int(np.sum(_coord_missing))
    _n_anchors  = len({tuple(r) for r in cell_tsne_coords})
    _jitter_sd  = 1.0
    _anchor_span = float(np.ptp(cell_tsne_coords)) if _n_anchors > 1 else 0.0

    print("\n=== CLUSTERING DIAGNOSTICS ===")
    print(f"  Cells: {len(cell_labels)}   distinct anchor positions: {_n_anchors}")
    print(f"  Cells on the (0,0) fallback (attractor had no t-SNE coord): "
          f"{_n_origin} ({100*_n_origin/max(len(cell_labels),1):.1f}%)")
    print(f"  Anchor coordinate span: {_anchor_span:.2f}   jitter sd: {_jitter_sd}")

    # (1) historical: embedding-space silhouette
    try:
        silhouette_val = silhouette_score(X_coords, cluster_labels_spectral)
    except Exception:
        silhouette_val = np.nan

    # (2) the meaningful one: silhouette on the precomputed dissimilarity.
    # The matrix is symmetrized first — compute_weighted_pseudotime_full
    # builds it asymmetrically (the most-traveled-path term depends on cell
    # i's basin, so d[i,j] != d[j,i]), and silhouette_score with
    # metric="precomputed" assumes a symmetric distance. The diagonal is
    # forced to exactly zero for the same reason.
    try:
        silhouette_precomputed = silhouette_score(
            _D_sym, cluster_labels_spectral, metric="precomputed")
    except Exception as _exc:
        print(f"  (precomputed silhouette failed: {_exc})")
        silhouette_precomputed = np.nan

    # (3) do the clusters recover the basins at all?
    try:
        ari_vs_basin = adjusted_rand_score(cell_attractor_ids,
                                           cluster_labels_spectral)
    except Exception:
        ari_vs_basin = np.nan

    print(f"  Silhouette (t-SNE embedding, historical): {silhouette_val:.4f}")
    print(f"  Silhouette (dissimilarity space, correct): {silhouette_precomputed:.4f}")
    print(f"  Adjusted Rand index vs basin identity:     {ari_vs_basin:.4f}")
    print(f"  Max |d(i,j) - d(j,i)| before symmetrizing: {_asym:.4f}")

    record_stat(
        figure="Figure 3",
        panels="Figure3B.png, Figure3FOrg.png",
        test=("Silhouette score of the k-cluster spectral clustering, evaluated "
              "on the PRECOMPUTED cell dissimilarity matrix (symmetrized) — "
              "i.e. in the space the clustering was performed in"),
        value=silhouette_precomputed,
        value_name="score",
        n_clusters=n_clusters,
        n_cells=len(cell_labels),
        cluster_sizes=", ".join(str(int(c)) for c in spectral_cluster_counts),
        max_asymmetry_before_symmetrizing=(f"{_asym:.6f}"
                                           if np.isfinite(_asym) else "n/a"),
    )

    record_stat(
        figure="Figure 3",
        panels="Figure3B.png",
        test=("Silhouette score of the same clustering, evaluated in the "
              "jittered pseudotime t-SNE space plotted in Figure3B "
              "(historical value; Figure3FOrg has its own, separate score)"),
        value=silhouette_val,
        value_name="score",
        n_clusters=n_clusters,
        n_cells=len(cell_labels),
        distinct_anchor_positions=_n_anchors,
        cells_on_origin_fallback=_n_origin,
        pct_cells_on_origin_fallback=f"{100*_n_origin/max(len(cell_labels),1):.2f}",
        anchor_coordinate_span=f"{_anchor_span:.4f}",
        jitter_sd=_jitter_sd,
    )

    record_stat(
        figure="Figure 3",
        panels="Figure3B.png, SimilarityClustering.png",
        test=("Adjusted Rand index: 5-cluster spectral labels vs. attractor "
              "basin identity"),
        value=ari_vs_basin,
        value_name="score",
        n_clusters=n_clusters,
        n_distinct_basins=int(len(np.unique(cell_attractor_ids))),
        n_cells=len(cell_labels),
    )

    # ---- Figure 3B ----
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter = plt.scatter(
        cell_tsne_coords_jittered[:, 0], cell_tsne_coords_jittered[:, 1],
        c=cell_pseudotime_values, cmap="plasma", alpha=0.7,
        edgecolor="white", s=120, linewidths=2
    )
    for sp in ax.spines.values(): sp.set_linewidth(2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("1-Normalized Steps to Attractors", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=18); cbar.outline.set_linewidth(2)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    # Drawn only when ANNOTATE_STATS_ON_FIGURES is True.
    _stat_box(ax, 0.02, 0.98, f"Silhouette = {silhouette_val:.3f}")
    plt.savefig(_outpath(outdir, "Figure3B.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter = plt.scatter(
        X_coords[:, 0], X_coords[:, 1],
        c=cluster_labels_spectral,
        cmap=cluster_cmap,                       # sized to k, not five names
        vmin=-0.5, vmax=n_clusters - 0.5,
        alpha=0.7, edgecolor="white", s=120, linewidths=2
    )
    for sp in ax.spines.values(): sp.set_linewidth(2)

    # One colour per outlined ATTRACTOR BASIN (still the top 5), independent
    # of the cluster palette above.
    hull_colors  = ["r","g","b","m","c"]
    legend_elems = []
    for i, att in enumerate(top_attractors):
        indices = [idx for idx, cell in enumerate(cell_labels)
                   if find_attractors(clean_trajectory(trajectories[cell]))[0] == att]
        if len(indices) >= 3:
            pts = X_coords[indices]
            try:
                hull = ConvexHull(pts)
                hp   = pts[hull.vertices]
                plt.plot(np.append(hp[:,0], hp[0,0]),
                         np.append(hp[:,1], hp[0,1]),
                         color=hull_colors[i % len(hull_colors)],
                         linestyle="--", lw=3)
                legend_elems.append(
                    Line2D([0],[0], color=hull_colors[i % len(hull_colors)],
                           lw=3, linestyle="--", label=f"Attractor {i+1}"))
            except Exception:
                pass

    for att in top_attractors:
        coord = attractor_coordinates.get(att)
        if coord:
            plt.scatter(*coord, color="gold", edgecolor="black", s=180, marker="s")

    for i in range(n_clusters):
        if spectral_cluster_counts[i] > 0:
            plt.scatter(spectral_centroids[i,0], spectral_centroids[i,1],
                        color=cluster_palette[i], edgecolor="black",
                        s=400, marker="X", linewidths=3, zorder=10)
    legend_elems.append(
        Line2D([0],[0], marker="X", color="w",
               markerfacecolor="black", markeredgecolor="black",
               markersize=15, label="Spectral Centroid"))

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label("Cluster Label (Spectral)", fontsize=16, fontweight="bold")
    cbar.ax.set_yticklabels([f"C{i}" for i in range(n_clusters)])
    cbar.ax.tick_params(labelsize=14 if n_clusters <= 8 else 10)
    cbar.outline.set_linewidth(2)
    if legend_elems: plt.legend(handles=legend_elems, loc="upper right", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.savefig(_outpath(outdir, "SimmilarityClustering.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- save cluster / centroid files ----
    with open(_outpath(outdir, "Cluster_Assignments_Spectral.txt"), "w") as fh:
        fh.write("Cell_ID\tCluster_Label\n")
        for idx, cell in enumerate(cell_labels):
            fh.write(f"{cell}\tCluster_{cluster_labels_spectral[idx]}\n")

    with open(_outpath(outdir, "Cluster_Assignments_KMeans.txt"), "w") as fh:
        fh.write("Cell_ID\tCluster_Label\n")
        for idx, cell in enumerate(cell_labels):
            fh.write(f"{cell}\tCluster_{cluster_labels_kmeans[idx]}\n")

    # ---- Cell Transition Matrix ----
    print("\nComputing cell transition matrix …")
    state_tm, state_occ, cell_tm, weighted_tm = \
        compute_state_and_cell_transition_matrices(
            cleaned_trajectories, columns_dict, trajectories)

    np.save(_outpath(outdir, "state_transition_matrix.npy"),   state_tm)
    np.save(_outpath(outdir, "state_occupancy_matrix.npy"),    state_occ)
    np.save(_outpath(outdir, "weighted_transition_matrix.npy"), weighted_tm)
    np.save(_outpath(outdir, "cell_transition_matrix.npy"),    cell_tm)

    plot_binary_connectivity_only(
        cell_tm,
        output_file=_outpath(outdir, "Cell_Transition_Matrix.png"),
        threshold=0.01, max_cells=500)

    plot_binary_connectivity_with_regions(
        cell_tm, cell_labels,
        barcode_to_label,
        output_file=_outpath(outdir, "Cell_Transition_Matrix_Region.png"),
        threshold=0.01, max_cells=500)

    # ---- top-500 transitions ----
    flat       = cell_tm.flatten()
    top_idx    = np.argsort(flat)[::-1][:500]
    top_500    = []
    att_counts = {}

    for fi in top_idx:
        row, col = np.unravel_index(fi, cell_tm.shape)
        if cell_tm[row, col] <= 0:
            continue
        cell_from = cell_labels[row]
        cell_to   = cell_labels[col]

        traj_f = clean_trajectory(trajectories[cell_from])
        traj_t = clean_trajectory(trajectories[cell_to])
        att_f  = find_attractors(traj_f)[0] or find_attractors(traj_f)[1][0]
        att_t  = find_attractors(traj_t)[0] or find_attractors(traj_t)[1][0]

        att_pair = (att_f, att_t)
        att_counts[att_pair] = att_counts.get(att_pair, 0) + 1
        top_500.append({"from": cell_from, "to": cell_to,
                        "att_from": att_f, "att_to": att_t,
                        "prob": cell_tm[row, col]})

    with open(_outpath(outdir, "top_500_transitions_by_attractor.txt"), "w") as fh:
        fh.write("Source_Cell\tTarget_Cell\tSource_Attractor\t"
                 "Target_Attractor\tTransition_Probability\n")
        for t in top_500:
            fh.write(f"{t['from']}\t{t['to']}\t"
                     f"{t['att_from']}\t{t['att_to']}\t{t['prob']:.6f}\n")

    # ============================================================
    # condition-bias / t-SNE (Figs 3E, 3F)
    #
    # NOTE: "Mixed" states have been removed from the condition-bias
    # classification. Every state is now binary-classified purely on the
    # SIGN of its condition_bias value:
    #     condition_bias  > 0  ->  AS-
    #     condition_bias <= 0  ->  AS+
    # (no threshold band, no third "Mixed" class).
    # ============================================================

    state_pseudotime_dict = {
        state: 1 - normalized_steps_to_attractor.get(state, 0)
        for state in normalized_steps_to_attractor
    }

    state_pt_items = [
        (int(s.split()[1]),
         state_pseudotime_dict.get(int(s.split()[1]), 0),
         unique_states[i])
        for i, s in enumerate(state_labels)
    ]
    state_pt_items.sort(key=lambda x: x[1])
    num_states       = len(state_pt_items)
    sorted_idx       = [x[0] for x in state_pt_items]
    sorted_pt        = [x[1] for x in state_pt_items]
    sorted_state_vec = [x[2] for x in state_pt_items]

    state_to_sorted = {sorted_idx[k]: k for k in range(num_states)}

    condition_bias_array = np.zeros(num_states)
    for i in range(num_states):
        sidx    = sorted_idx[i]
        cells_v = [c for c, traj in trajectories.items() if sidx in traj]
        healthy = diseased = total = 0
        for cell in cells_v:
            bc   = trajectory_key_to_barcode(cell)
            cond = barcode_to_label.get(bc, "")
            if cond == "AS-":   healthy  += 1
            elif cond == "AS+": diseased += 1
            total += 1
        if total > 0:
            condition_bias_array[i] = (healthy - diseased) / total

    print("=== CB DIAGNOSTICS ===")
    print(f"  min={condition_bias_array.min():.4f}  max={condition_bias_array.max():.4f}"
          f"  std={condition_bias_array.std():.4f}")
    print(f"  unique values: {len(np.unique(np.round(condition_bias_array, 3)))}")
    print(f"  states with 0 cells: "
          f"{sum(1 for v in condition_bias_array if v == 0)}/{num_states}")
    print(f"  AS-={int(np.sum(condition_bias_array > 0))}  "
          f"AS+={int(np.sum(condition_bias_array <= 0))}")

    np.save(_outpath(outdir, "baseline_condition_bias.npy"), condition_bias_array)
    print(f"  → baseline_condition_bias.npy saved.")

    pseudotime_norm = np.array(sorted_pt)
    cb_norm_arr = (condition_bias_array - condition_bias_array.min()) \
                  / (np.ptp(condition_bias_array) + 1e-8)
    pt_norm_arr = (pseudotime_norm - pseudotime_norm.min()) \
                  / (np.ptp(pseudotime_norm) + 1e-8)
    w_cb = 0.99

    dist_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            dist_matrix[i, j] = (
                w_cb * (1 - np.exp(-1 * abs(cb_norm_arr[i] - cb_norm_arr[j])))
                + (1 - w_cb) * abs(pt_norm_arr[i] - pt_norm_arr[j])
            )

    tsne_dist = TSNE(n_components=2, metric="precomputed", random_state=24,
                     perplexity=450, init="random",
                     early_exaggeration=48.0).fit_transform(dist_matrix)

    as_plus_idx  = [i for i, v in enumerate(condition_bias_array) if v <= 0]
    as_minus_idx = [i for i, v in enumerate(condition_bias_array) if v >  0]

    print(f"  AS-={len(as_minus_idx)}  AS+={len(as_plus_idx)}")

    # ---- Figure 3E ----
    indices_pt1 = [i for i, s in enumerate(state_labels)
                   if normalized_steps_to_attractor.get(int(s.split()[1]), 0) == 0]

    _, ax_dist = plt.subplots(figsize=(8, 6))
    plt.setp(ax_dist.spines.values(), linewidth=2)

    for i, s in enumerate(state_labels):
        orig_idx = int(s.split()[1])
        k = state_to_sorted.get(orig_idx)
        if k is None:
            continue
        ax_dist.scatter(tsne_dist[k, 0], tsne_dist[k, 1],
                        color=plt.cm.coolwarm(pt_norm_arr[k]),
                        alpha=1, marker="s", linewidths=2)

    for i in indices_pt1:
        orig_idx = int(state_labels[i].split()[1])
        k = state_to_sorted.get(orig_idx)
        if k is None:
            continue
        ax_dist.scatter(tsne_dist[k, 0], tsne_dist[k, 1],
                        color="gold", edgecolor="black",
                        s=120, marker="s", linewidths=2)

    ax_dist.set_xlabel("t-SNE Component 1", fontsize=18)
    ax_dist.set_ylabel("t-SNE Component 2", fontsize=18)
    ax_dist.tick_params(axis="both", labelsize=18)

    def _hull_region(ax, idx_list, color, label):
        if len(idx_list) < 3:
            return
        pts = np.array([tsne_dist[i] for i in idx_list])
        try:
            hull = ConvexHull(pts)
            hv   = np.append(hull.vertices, hull.vertices[0])
            ax.plot(pts[hv, 0], pts[hv, 1], color=color, linestyle="--", lw=2)
            ax.fill(pts[hv, 0], pts[hv, 1], color=color, alpha=0.15)
            cx = pts[hull.vertices].mean(axis=0)
            ax.annotate(label, xy=cx, color=color, fontsize=18,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=color, lw=2, alpha=0.7))
        except Exception:
            pass

    _hull_region(ax_dist, as_plus_idx,  "red",   "AS+")
    _hull_region(ax_dist, as_minus_idx, "green",  "AS-")

    sm2 = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cbar = plt.colorbar(sm2, ax=ax_dist)
    cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=18); cbar.outline.set_linewidth(2)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    # Drawn only when ANNOTATE_STATS_ON_FIGURES is True.
    _stat_box(ax_dist, 0.02, 0.98, _fmt_p(pseudotime_pvalue))
    plt.savefig(_outpath(outdir, "Figure3EOrg.png"), dpi=300, bbox_inches="tight")
    plt.close()

    np.save(_outpath(outdir, "condition_bias_array.npy"), condition_bias_array)
    np.save(_outpath(outdir, "sorted_pseudotimes.npy"),   np.array(sorted_pt))

    # ---- Attractor coordinates for 3F ----
    attractor_coordinates_dist = {}
    if indices_pt1:
        with open(_outpath(outdir,
                           "highlighted_states_coordinates_of_attractors.txt"), "w") as fh:
            for i in indices_pt1:
                sidx     = int(state_labels[i].split()[1])
                k        = state_to_sorted.get(sidx)
                if k is None:
                    continue
                x, y     = tsne_dist[k, 0], tsne_dist[k, 1]
                attractor_coordinates_dist[sidx] = (x, y)
                fh.write(f"{x}\t{y}\n")

    cell_tsne_dist = np.zeros((len(trajectories), 2))
    for idx, cell in enumerate(cell_labels):
        att_s = find_attractors(clean_trajectory(trajectories[cell]))[0]
        if att_s in attractor_coordinates_dist:
            cell_tsne_dist[idx] = attractor_coordinates_dist[att_s]

    rng2    = np.random.default_rng(seed=42)
    jitter2 = rng2.normal(0, 2, cell_tsne_dist.shape)
    cell_tsne_j2 = cell_tsne_dist + jitter2

    participant_col = []
    for cell in cell_labels:
        bc  = trajectory_key_to_barcode(cell)
        pid = barcode_to_participant.get(bc, "NA")
        participant_col.append(pid)

    with open(_outpath(outdir, "cell_tsne_embeddings.txt"), "w") as fh:
        for pid, (x, y) in zip(participant_col, cell_tsne_j2):
            fh.write(f"{pid}\t{x}\t{y}\n")

    pseudotime_dict2 = compute_weighted_pseudotime(trajectories, attractor_value_dict)
    cell_pt_values2  = [pseudotime_dict2.get(c, 0) for c in cell_labels]

    with open(_outpath(outdir, "cell_pseudotime_values.txt"), "w") as fh:
        for cell, pt in zip(cell_labels, cell_pt_values2):
            bc   = trajectory_key_to_barcode(cell)
            pid  = barcode_to_participant.get(bc, "NA")
            cond = barcode_to_label.get(bc, "NA")
            fh.write(f"{cell}\t{pid}\t{cond}\t{pt}\n")

    # ---- Figure 3F ----
    # Figure3FOrg lives in a DIFFERENT embedding from Figure3B: cells are
    # anchored on `tsne_dist` (the condition-bias distance t-SNE over states)
    # with sd-2 jitter, not on `tsne_results` with sd-1 jitter. Reusing 3B's
    # silhouette here was therefore reporting a number computed on other
    # coordinates. It now gets its own.
    _cell_tsne_dist_missing = np.array(
        [find_attractors(clean_trajectory(trajectories[c]))[0]
         not in attractor_coordinates_dist for c in cell_labels])
    _n_origin_3f  = int(np.sum(_cell_tsne_dist_missing))
    _n_anchors_3f = len({tuple(r) for r in cell_tsne_dist})

    try:
        silhouette_val_3f = silhouette_score(cell_tsne_j2,
                                             cluster_labels_spectral)
    except Exception:
        silhouette_val_3f = np.nan

    print(f"  Silhouette (Figure3FOrg embedding):        {silhouette_val_3f:.4f}"
          f"   anchors={_n_anchors_3f}  origin-fallback cells={_n_origin_3f}")

    record_stat(
        figure="Figure 3",
        panels="Figure3FOrg.png",
        test=("Silhouette score of the spectral clustering, evaluated in the "
              "jittered condition-bias t-SNE space plotted in Figure3FOrg"),
        value=silhouette_val_3f,
        value_name="score",
        n_clusters=n_clusters,
        n_cells=len(cell_labels),
        distinct_anchor_positions=_n_anchors_3f,
        cells_on_origin_fallback=_n_origin_3f,
        pct_cells_on_origin_fallback=f"{100*_n_origin_3f/max(len(cell_labels),1):.2f}",
        jitter_sd=2.0,
    )

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter = plt.scatter(
        cell_tsne_j2[:, 0], cell_tsne_j2[:, 1],
        c=cell_pt_values2, cmap="plasma",
        alpha=0.7, edgecolor="white", s=120, linewidths=2
    )
    for sp in ax.spines.values(): sp.set_linewidth(2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("1-Normalized Steps to Attractors", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=18); cbar.outline.set_linewidth(2)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    # Drawn only when ANNOTATE_STATS_ON_FIGURES is True. Uses 3F's OWN score.
    _stat_box(ax, 0.02, 0.98, f"Silhouette = {silhouette_val_3f:.3f}")
    plt.savefig(_outpath(outdir, "Figure3FOrg.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Figure 3A coloured by Condition Bias
    # ============================================================
    cb_by_state = {sorted_idx[i]: condition_bias_array[i]
                   for i in range(num_states)}

    cb_values = np.array([
        cb_by_state.get(int(s.split()[1]), 0.0)
        for s in state_labels
    ])

    cb_cmap = LinearSegmentedColormap.from_list(
        "cb_rgg",
        [
            (0.0, "#d62728"),
            (0.5, "#aaaaaa"),
            (1.0, "#2ca02c"),
        ],
    )

    cb_normalize = Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in ax.spines.values():
        sp.set_linewidth(2)

    sc = ax.scatter(
        tsne_results[:, 0], tsne_results[:, 1],
        c=cb_values,
        cmap=cb_cmap,
        norm=cb_normalize,
        alpha=0.7,
        marker="s",
        s=120,
        edgecolors="none",
    )

    for i in indices_pseudotime1:
        ax.scatter(
            tsne_results[i, 0], tsne_results[i, 1],
            color="gold", edgecolor="black",
            s=140, marker="s", linewidths=2, zorder=5,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Condition Bias  (AS− → AS+)", fontsize=18, fontweight="bold")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=16)
    cbar.outline.set_linewidth(2)

    ax.set_xlabel("t-SNE Component 1", fontsize=18, fontweight="bold")
    ax.set_ylabel("t-SNE Component 2", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)

    plt.savefig(_outpath(outdir, "Figure3A_ConditionBias.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  → Figure3A_ConditionBias.png saved.")

    write_stats_report(outdir, getattr(args, "stats_report", None))

    print("All Figure-3 panels generated.")
    return cell_labels, barcode_to_label


# ============================================================
# Figure 4
# ============================================================

def generate_figure4(outdir: str):
    """
    Figure 4: condition-bias distribution.

    No hypothesis test is performed here. The "AS-: xx.x% / AS+: xx.x%" box on
    Figure4B is a descriptive composition label, not a test result, so it is
    deliberately kept on the panel.
    """
    print("\n" + "=" * 60)
    print("Generating Figure 4: Condition Bias Analysis")
    print("=" * 60)

    cb_path = _outpath(outdir, "condition_bias_array.npy")
    pt_path = _outpath(outdir, "sorted_pseudotimes.npy")

    if not os.path.exists(cb_path) or not os.path.exists(pt_path):
        print("ERROR: Run generate_all_figures3 first."); return

    condition_bias_array= np.load(cb_path)
    sorted_pseudotimes  = np.load(pt_path)
    num_states = len(condition_bias_array)

    # 4A
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.coolwarm(sorted_pseudotimes)
    for i in range(num_states):
        ax.scatter(sorted_pseudotimes[i], condition_bias_array[i],
                   c=[colors[i]], marker="s", s=100, alpha=0.7,
                   edgecolors="white", linewidths=0.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    for label, bias, color, va in [("AS-\nStates", 0.85, "green", "bottom"),
                                    ("AS+\nStates", 0.15, "red",   "top")]:
        ax.annotate(label, xy=(1.05, bias), xytext=(1.05, bias),
                    textcoords="axes fraction", fontsize=14,
                    fontweight="bold", color=color, ha="left", va=va)
    ax.set_xlabel("Normalized Pseudotime",        fontsize=16, fontweight="bold")
    ax.set_ylabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight="bold")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis="both", labelsize=14)
    for sp in ax.spines.values(): sp.set_linewidth(2)
    plt.savefig(_outpath(outdir, "Figure4A.png"), dpi=300); plt.close()

    # 4B
    fig, ax = plt.subplots(figsize=(8, 6))
    n_am  = np.sum(condition_bias_array >  0)
    n_ap  = np.sum(condition_bias_array <= 0)
    bins  = np.linspace(-1.1, 1.1, 25)
    ax.hist(condition_bias_array[condition_bias_array <= 0],
            bins=bins, color="red",  alpha=0.8, label="AS+",   edgecolor="black")
    ax.hist(condition_bias_array[condition_bias_array >  0],
            bins=bins, color="teal", alpha=0.8, label="AS-",   edgecolor="black")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    mean_b   = np.mean(condition_bias_array)
    median_b = np.median(condition_bias_array)
    ax.axvline(mean_b,   color="navy",        linestyle="-",  linewidth=2)
    ax.axvline(median_b, color="darkmagenta", linestyle=":",  linewidth=2)
    # Descriptive composition, not a test result — kept on the panel.
    stats_text = (f"AS-: {100*n_am/num_states:.1f}%\n"
                  f"AS+: {100*n_ap/num_states:.1f}%")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))
    legend_elems = [
        Line2D([0],[0], color="navy",        linestyle="-", linewidth=2,
               label=f"Mean: {mean_b:.3f}"),
        Line2D([0],[0], color="darkmagenta", linestyle=":", linewidth=2,
               label=f"Median: {median_b:.3f}"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=12)
    ax.set_xlabel("Condition Bias (AS- vs. AS+)",        fontsize=16, fontweight="bold")
    ax.set_ylabel("Frequency (Number of States)",        fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    for sp in ax.spines.values(): sp.set_linewidth(2)
    plt.savefig(_outpath(outdir, "Figure4B.png"), dpi=300); plt.close()

    # Combined
    imgs = [mpimg.imread(_outpath(outdir, f)) for f in ("Figure4A.png","Figure4B.png")]
    fig  = plt.figure(figsize=(8, 6))
    gs   = GridSpec(2, 1, figure=fig, hspace=0.05)
    for i, (label, img) in enumerate(zip(["a.","b."], imgs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img); ax.axis("off")
        ax.text(0.02, 1.02, label, transform=ax.transAxes,
                fontsize=20, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    plt.savefig(_outpath(outdir, "Figure4_Combined.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("Figure 4 complete.")


# ============================================================
# MLP Held-Out Experiments (class-specific AS+/AS- holdout)
# ============================================================

def run_class_specific_holdout_experiments(X, Y_multi, condition_bias_classes,
                                            experiments=None, random_state=42, outdir="."):
    """
    Four held-out experiments for the condition-bias / pseudotime MLP, where
    the AS+ holdout percentage and the AS- holdout percentage are each set
    directly and independently. Classification is now strictly binary
    (condition_bias > 0 -> AS-, condition_bias <= 0 -> AS+); there is no
    longer a "Mixed" class to keep in the training pool.

    Unlike the 5-fold CV elsewhere in generate_figure5 (where every state
    eventually gets predicted, but the model that predicted it still saw 80%
    of the data), each experiment here trains ONCE on a fixed split and
    evaluates ONLY on states walled off from training entirely — a stricter
    test of whether the model generalizes to genuinely unseen states.

    Default experiments:
      1. 5%  AS+ held out, 5%  AS- held out   (balanced, very light)
      2. 10% AS+ held out, 10% AS- held out   (balanced, light)

    Reports overall accuracy/precision plus PER-CLASS recall for AS+ and
    AS- separately, since an unbalanced holdout can hide one class's
    degradation behind the other's.

    Results are written to mlp_class_specific_holdout_experiments.csv/.png as
    before, and each experiment's accuracy is also registered with
    record_stat() so it appears in statistics_summary.txt.
    """
    if experiments is None:
        experiments = [
            {"name": "5pct_each",     "as_plus_frac": 0.05, "as_minus_frac": 0.05},
            {"name": "10pct_each",    "as_plus_frac": 0.10, "as_minus_frac": 0.10},
        ]

    def _classify(bias):
        return "AS-" if bias > 0 else "AS+"

    rng = np.random.default_rng(random_state)
    condition_bias_classes = np.array(condition_bias_classes)
    idx_as_plus  = np.where(condition_bias_classes == "AS+")[0]
    idx_as_minus = np.where(condition_bias_classes == "AS-")[0]

    print(f"  Total states available — AS+: {len(idx_as_plus)}  AS-: {len(idx_as_minus)}")

    results = []
    for exp in experiments:
        n_plus  = int(round(len(idx_as_plus)  * exp["as_plus_frac"]))
        n_minus = int(round(len(idx_as_minus) * exp["as_minus_frac"]))
        n_plus  = max(1, min(n_plus,  len(idx_as_plus)  - 1)) if len(idx_as_plus)  > 1 else 0
        n_minus = max(1, min(n_minus, len(idx_as_minus) - 1)) if len(idx_as_minus) > 1 else 0

        if n_plus == 0 or n_minus == 0:
            print(f"  [{exp['name']}] SKIPPED — not enough AS+ or AS- states "
                  f"to hold out (AS+={len(idx_as_plus)}, AS-={len(idx_as_minus)}).")
            continue

        holdout_plus  = rng.choice(idx_as_plus,  size=n_plus,  replace=False)
        holdout_minus = rng.choice(idx_as_minus, size=n_minus, replace=False)
        test_idx  = np.concatenate([holdout_plus, holdout_minus])
        train_idx = np.setdiff1d(np.arange(len(condition_bias_classes)), test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y_multi[train_idx], Y_multi[test_idx]
        cls_test = condition_bias_classes[test_idx]

        model = MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu',
                         max_iter=10000, random_state=10000)
        )
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        pred_classes = np.array([_classify(np.tanh(v)) for v in Y_pred[:, 0]])

        acc        = accuracy_score(cls_test, pred_classes)
        prec_macro = precision_score(cls_test, pred_classes,
                                     labels=["AS-", "AS+"],
                                     average='macro', zero_division=0)
        r2_pt = r2_score(Y_test[:, 1], Y_pred[:, 1])

        recall_plus  = (np.mean(pred_classes[cls_test == "AS+"] == "AS+")
                        if np.any(cls_test == "AS+") else np.nan)
        recall_minus = (np.mean(pred_classes[cls_test == "AS-"] == "AS-")
                        if np.any(cls_test == "AS-") else np.nan)

        results.append({
            "experiment":             exp["name"],
            "as_plus_frac_target":    exp["as_plus_frac"],
            "as_minus_frac_target":   exp["as_minus_frac"],
            "n_as_plus_held_out":     n_plus,
            "n_as_minus_held_out":    n_minus,
            "n_train":                len(train_idx),
            "accuracy":               acc,
            "precision_macro":        prec_macro,
            "recall_AS+":             recall_plus,
            "recall_AS-":             recall_minus,
            "pseudotime_r2":          r2_pt,
        })
        print(f"  [{exp['name']}] AS+ held out={n_plus} ({exp['as_plus_frac']*100:.0f}%)  "
              f"AS- held out={n_minus} ({exp['as_minus_frac']*100:.0f}%)  "
              f"Acc={acc:.4f}  Recall(AS+)={recall_plus:.4f}  Recall(AS-)={recall_minus:.4f}  "
              f"R²={r2_pt:.4f}")

        record_stat(
            figure="Figure 5 (MLP held-out experiments)",
            panels="mlp_class_specific_holdout_experiments.png",
            test=f"Held-out experiment '{exp['name']}' — classification accuracy "
                 f"on states walled off from training",
            value=acc,
            value_name="accuracy",
            n_AS_plus_held_out=n_plus,
            n_AS_minus_held_out=n_minus,
            n_train=len(train_idx),
            precision_macro=f"{prec_macro:.6f}",
            recall_AS_plus=f"{recall_plus:.6f}",
            recall_AS_minus=f"{recall_minus:.6f}",
            pseudotime_R2=f"{r2_pt:.6f}",
        )

    df = pd.DataFrame(results)
    df.to_csv(_outpath(outdir, "mlp_class_specific_holdout_experiments.csv"), index=False)

    if len(df):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(df))
        width = 0.2
        ax.bar(x - 1.5*width, df["accuracy"],        width, label="Accuracy")
        ax.bar(x - 0.5*width, df["recall_AS+"],       width, label="Recall (AS+)")
        ax.bar(x + 0.5*width, df["recall_AS-"],       width, label="Recall (AS-)")
        ax.bar(x + 1.5*width, df["pseudotime_r2"],    width, label="Pseudotime R²")
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment"], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("MLP performance — class-specific held-out experiments")
        ax.legend()
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        plt.tight_layout()
        plt.savefig(_outpath(outdir, "mlp_class_specific_holdout_experiments.png"), dpi=300)
        plt.close()

    print("  Saved mlp_class_specific_holdout_experiments.csv / .png")
    return df


# ============================================================
# Figure 5
# ============================================================
def generate_figure5(args, outdir):
    """
    Generate Figure 5: Perturbation Analysis
    - Figure 5A: Delta AS+ percentage bar chart for multiple genes
                 (now carrying Wilcoxon statistics — see below)
    - Figure 5B: KI bifurcation diagram with histogram
    - Figure 5C: KO bifurcation diagram with histogram

    Consistency rule:
    - Baseline : state-based (states where gene is ON → condition_bias_array lookup)
    - KI / KO  : cell-based (perturbed attractor → ML model → condition bias prediction)
    - Figure 5A delta uses the same KI/KO counts as the % shown in 5B/5C

    Condition-bias classification is strictly binary: condition_bias > 0 is
    AS-, condition_bias <= 0 is AS+. There is no "Mixed" class anywhere in
    this pipeline.

    Also runs two class-specific held-out experiments for the MLP (5% and
    10% AS+/AS- held out — see run_class_specific_holdout_experiments)
    right after the model's training features/classes are built, before
    the 5-fold CV.

    Statistics: cross-validation and training metrics are still printed to
    stdout exactly as before, and are additionally registered with
    record_stat() so they land in statistics_summary.txt.

    NEW — Figure 5A Wilcoxon tests. Two levels:
      * per gene, a rank-sum (unpaired Wilcoxon / Mann-Whitney) test of that
        gene's predicted perturbed condition-bias values against the baseline
        condition-bias values, once for the knockin arm and once for the
        knockout arm. Rendered as significance stars at each bar tip.
      * across genes, three signed-rank (paired Wilcoxon) tests on the plotted
        deltas themselves: delta_KI vs 0, delta_KO vs 0, and delta_KI vs
        delta_KO. Rendered in a summary box on the panel.
    Both are gated on ANNOTATE_FIGURE5A_WILCOXON (default True) for the
    on-figure part, and always written to statistics_summary.txt.

    The "AS-: xx.x% / AS+: xx.x%" box in _bifurcation_fig is a descriptive
    composition label, not a test result, and is kept.

    BUGFIX (earlier revision): the baseline AS-/AS+ counting loop below used
    to index the gene ON/OFF vector with `unique_states[i]` (file order)
    while pairing it with `condition_bias_array[i]` (pseudotime-sorted-rank
    order). Those two orderings only coincide by accident, so the two
    values being compared usually described two *different* states,
    silently corrupting `baseline_am`/`baseline_ap` (and hence the Figure 5A
    deltas) for every gene. Fixed by looking the state vector up by its
    actual state number, `sorted_indices[i]`, exactly as the bifurcation-
    dict construction earlier in this function already does.
    """

    print("\n" + "="*60)
    print("Generating Figure 5: Perturbation Analysis")
    print("="*60)

    # ========== Load Required Data ==========
    print("\n[1/6] Loading trajectories...")
    trajectories         = read_trajectories(args.trajectories)
    cleaned_trajectories = {cell: clean_trajectory(traj) for cell, traj in trajectories.items()}

    print("[2/6] Loading unique states...")
    with open(args.unique_states, 'r') as f:
        columns_dict = {
            key.strip(): list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(',')))
            for key, value in (line.strip().split(':') for line in f)
        }
    unique_states = list(columns_dict.values())
    state_labels  = list(columns_dict.keys())

    # ========== Pseudotime ==========
    print("[3/6] Calculating pseudotime...")
    normalized_steps_to_attractor = {
        state: ((len(clean_trajectory(traj)) - clean_trajectory(traj).index(state)) - 1)
               / len(clean_trajectory(traj))
        for traj in cleaned_trajectories.values()
        for state in clean_trajectory(traj)
        if len(clean_trajectory(traj)) > 1
    }
    state_pseudotime_dict = {s: 1 - normalized_steps_to_attractor[s]
                             for s in normalized_steps_to_attractor}

    state_pseudotime_items = [
        (state_index, state_pseudotime_dict.get(state_index, 0), unique_states[i])
        for i, state_index in enumerate([int(s.split()[1]) for s in state_labels])
    ]
    state_pseudotime_items.sort(key=lambda x: x[1])
    num_states      = len(state_pseudotime_items)
    sorted_indices  = [x[0] for x in state_pseudotime_items]
    state_to_sorted = {state_num: i for i, state_num in enumerate(sorted_indices)}

    # ========== Load Condition Bias from Figure 3 ==========
    print("[4/6] Loading condition bias from Figure 3...")
    condition_bias_array = np.load(_outpath(outdir, "baseline_condition_bias.npy"))

    print(f"  shape        : {condition_bias_array.shape}")
    print(f"  min={condition_bias_array.min():.4f}  max={condition_bias_array.max():.4f}  "
          f"mean={condition_bias_array.mean():.4f}  std={condition_bias_array.std():.4f}")
    print(f"  zeros        : {int(np.sum(condition_bias_array == 0))} / {len(condition_bias_array)}")
    print(f"  unique values: {len(np.unique(condition_bias_array))}")

    if len(condition_bias_array) != num_states:
        raise RuntimeError(
            f"condition_bias_array has {len(condition_bias_array)} entries "
            f"but num_states={num_states}. Ensure Figure 3 used the same unique states file."
        )

    baseline_as_minus_states = int(np.sum(condition_bias_array >  0))
    baseline_as_plus_states  = int(np.sum(condition_bias_array <= 0))
    print(f"  AS-={baseline_as_minus_states}  AS+={baseline_as_plus_states}")

    # ---- Normal (unperturbed) signaling reference points ----
    # condition_bias_array[i] belongs to state sorted_indices[i], so the
    # pseudotime must be looked up with that same state number (same pairing
    # rule as the bifurcation dicts and the Figure 5A baseline counts).
    baseline_x = np.array([state_pseudotime_dict.get(s, 0) for s in sorted_indices],
                          dtype=float)
    baseline_y = np.asarray(condition_bias_array, dtype=float)
    # Build bifurcation dicts for model training
    # NOTE: correctly keyed by state number (sorted_indices[i]) already.
    bifurcation_y_values = {}
    bifurcation_x_values = {}
    for i, state in enumerate(sorted_indices):
        binary_value = tuple(columns_dict[f"State {state}"])
        bifurcation_y_values[binary_value] = condition_bias_array[i]
        bifurcation_x_values[binary_value] = state_pseudotime_dict.get(state, 0)

    # ========== Train Model ==========
    print("[5/6] Training prediction model with cross-validation...")
    bifurcation_features = [
        (list(bv), bifurcation_x_values[bv], bifurcation_y_values[bv])
        for bv in bifurcation_x_values
    ]
    X_raw              = np.array([f[0] for f in bifurcation_features])
    Y_raw              = np.array([[f[2], f[1]] for f in bifurcation_features])
    x_scaler           = StandardScaler()
    X                  = x_scaler.fit_transform(X_raw)
    condition_bias_raw = Y_raw[:, 0]
    pseudotime_raw     = Y_raw[:, 1]
    Y_multi            = np.column_stack([condition_bias_raw, pseudotime_raw])

    def get_condition_class(bias):
        return "AS-" if bias > 0 else "AS+"

    condition_bias_classes = np.array([get_condition_class(cb) for cb in condition_bias_raw])

    # ---- class-specific held-out experiments ----
    # Runs BEFORE the 5-fold CV below. Each experiment trains once on a
    # fixed split and evaluates only on the AS+/AS- states held out for
    # that experiment — a stricter test than the rotating 5-fold CV, since
    # no held-out state ever contributes to that experiment's training.
    print("\n  Running 2 class-specific held-out experiments "
          "(5%/5%, 10%/10%)...")
    holdout_results_df = run_class_specific_holdout_experiments(
        X, Y_multi, condition_bias_classes,
        outdir=outdir,
    )

    # 5-fold CV
    n_folds              = 5
    kfold                = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores_pseudotime = []
    cv_precision_macro   = []
    cv_precision_weighted= []
    cv_accuracy          = []
    all_true_classes     = []
    all_pred_classes     = []
    all_raw_scores       = []

    print(f"\n  Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Performing {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_model = MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu',
                         max_iter=10000, random_state=10000)
        )
        fold_model.fit(X[train_idx], Y_multi[train_idx])
        Y_pred      = fold_model.predict(X[val_idx])
        y_class_val = condition_bias_classes[val_idx]

        r2_pt          = r2_score(Y_multi[val_idx, 1], Y_pred[:, 1])
        pred_classes   = np.array([get_condition_class(np.tanh(v))
                                   for v in Y_pred[:, 0]])
        all_true_classes.extend(y_class_val)
        all_pred_classes.extend(pred_classes)
        all_raw_scores.extend(np.tanh(Y_pred[:, 0]))

        pm  = precision_score(y_class_val, pred_classes, labels=["AS-","AS+"],
                              average='macro',    zero_division=0)
        pw  = precision_score(y_class_val, pred_classes, labels=["AS-","AS+"],
                              average='weighted', zero_division=0)
        acc = accuracy_score(y_class_val, pred_classes)
        cv_scores_pseudotime.append(r2_pt)
        cv_precision_macro.append(pm)
        cv_precision_weighted.append(pw)
        cv_accuracy.append(acc)
        print(f"    Fold {fold+1}: Precision (macro)={pm:.4f}  (weighted)={pw:.4f}  "
              f"Accuracy={acc:.4f}  R²={r2_pt:.4f}")

    print(f"\n  Cross-Validation Results ({n_folds}-fold):")
    print(f"    Precision (macro):    {np.mean(cv_precision_macro):.4f} ± {np.std(cv_precision_macro):.4f}")
    print(f"    Precision (weighted): {np.mean(cv_precision_weighted):.4f} ± {np.std(cv_precision_weighted):.4f}")
    print(f"    Accuracy:             {np.mean(cv_accuracy):.4f} ± {np.std(cv_accuracy):.4f}")
    print(f"    Pseudotime R²:        {np.mean(cv_scores_pseudotime):.4f} ± {np.std(cv_scores_pseudotime):.4f}")

    all_true_classes = np.array(all_true_classes)
    all_pred_classes = np.array(all_pred_classes)
    all_raw_scores   = np.array(all_raw_scores)
    class_labels     = ["AS-", "AS+"]
    cm = confusion_matrix(all_true_classes, all_pred_classes, labels=class_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=["AS-","AS+"])
    disp.plot(cmap="Blues")
    plt.savefig(_outpath(outdir, "Figure5_ConfusionMatrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Contingency Table:")
    print(f"                    {'AS-':>8} {'AS+':>8}")
    print(f"         " + "-" * 26)
    for i, label in enumerate(class_labels):
        print(f"  Actual {label:>5} |{cm[i,0]:>8} {cm[i,1]:>8}")
    print(f"\n  Per-class Precision:")
    for i, label in enumerate(class_labels):
        col_sum = cm[:, i].sum()
        if col_sum > 0:
            print(f"    {label}: {cm[i,i]/col_sum:.4f} ({cm[i,i]}/{col_sum})")
        else:
            print(f"    {label}: N/A (no predictions)")

    # ---- AUROC on pooled out-of-fold scores ----
    y_true_binary = np.array([1 if c == "AS-" else 0 for c in all_true_classes])
    auroc = roc_auc_score(y_true_binary, all_raw_scores)
    print(f"\n  Pooled out-of-fold AUROC: {auroc:.4f}")

    fpr, tpr, _ = roc_curve(y_true_binary, all_raw_scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Condition Bias Classification (pooled CV)")
    plt.legend()
    plt.savefig(_outpath(outdir, "Figure5_ROC.png"), dpi=300, bbox_inches="tight")
    plt.close()

    record_stat(
        figure="Figure 5 (MLP cross-validation)",
        panels="Figure5B.png, Figure5C.png (model used to predict perturbed bias)",
        test=f"{n_folds}-fold cross-validated classification accuracy "
             f"(condition-bias class from state gene vector)",
        value=float(np.mean(cv_accuracy)),
        value_name="accuracy",
        accuracy_sd=f"{np.std(cv_accuracy):.6f}",
        precision_macro=f"{np.mean(cv_precision_macro):.6f} "
                        f"+/- {np.std(cv_precision_macro):.6f}",
        precision_weighted=f"{np.mean(cv_precision_weighted):.6f} "
                           f"+/- {np.std(cv_precision_weighted):.6f}",
        pseudotime_R2=f"{np.mean(cv_scores_pseudotime):.6f} "
                      f"+/- {np.std(cv_scores_pseudotime):.6f}",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        confusion_AS_minus_row=f"[{cm[0,0]}, {cm[0,1]}]",
        confusion_AS_plus_row=f"[{cm[1,0]}, {cm[1,1]}]",
        AUROC=f"{auroc:.6f}"
    )

    print("\n  Training final model on all data...")
    model_multi = MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu',
                     max_iter=10000, random_state=10000)
    )

    model_multi.fit(X, Y_multi)
    Y_pred_train         = model_multi.predict(X)
    pred_cond_bias_train = np.tanh(Y_pred_train[:, 0])
    pred_classes_train   = np.array([get_condition_class(cb)
                                     for cb in pred_cond_bias_train])
    train_acc  = accuracy_score(condition_bias_classes, pred_classes_train)
    train_prec = precision_score(condition_bias_classes, pred_classes_train,
                                 labels=class_labels, average='macro', zero_division=0)
    train_r2   = r2_score(Y_multi[:,1], Y_pred_train[:,1])
    print(f"  Training Accuracy:  {train_acc:.4f}")
    print(f"  Training Precision: {train_prec:.4f}")
    print(f"  Training R²:        {train_r2:.4f}")

    record_stat(
        figure="Figure 5 (MLP cross-validation)",
        panels="Figure5B.png, Figure5C.png (final model, all data)",
        test="Final-model in-sample (training) classification accuracy",
        value=float(train_acc),
        value_name="accuracy",
        precision_macro=f"{train_prec:.6f}",
        pseudotime_R2=f"{train_r2:.6f}",
    )

    # ========== Process Perturbations ==========
    print("\n[6/6] Processing perturbations...")

    gene_list_path = os.path.join(outdir, "filtered_gene_names.txt")

    with open(gene_list_path, "r") as f:
        GENE_ORDER = [line.strip() for line in f if line.strip()]

    all_gene_results = {}
    # Every gene's 2x2 contingency table, exported for review as
    # figure5A_contingency_tables.csv — these are the exact counts Fisher's
    # exact test consumes and the exact counts the Figure 5A bar is derived
    # from, side by side with both p-values.
    figure5a_tables  = []
    ki_cell_ids_selected = []
    ko_cell_ids_selected = []
    selected_gene = args.selected_gene
    ki_data_selected = None
    ko_data_selected = None

    from pathlib import Path as PlPath

    traj_dir = PlPath(args.trajectories).parent
    traj_csvs = sorted(glob.glob(str(traj_dir / "cell_*.csv")))

    if not traj_csvs:
        raise RuntimeError(f"No cell_*.csv files found in {traj_dir}")

    gene_order = (
        pd.read_csv(
            traj_csvs[0],
            header=None,
            index_col=0,
            sep=","
        )
        .index
        .str.strip()
        .tolist()
    )

    gene_to_col_index = {g: i for i, g in enumerate(gene_order)}

    print(f"  Loaded {len(gene_order)} genes from {PlPath(traj_csvs[0]).name}")
    print(f"  Gene interest found:   {[g for g in GENE_ORDER if g in gene_to_col_index]}")
    print(f"  Gene interest missing: {[g for g in GENE_ORDER if g not in gene_to_col_index]}")

    num_to_label, barcode_to_label, barcode_to_participant = parse_metadata(args.metadata)

    metadata = pd.read_csv(
        args.metadata,
        sep=" ",
        header=None,
        skiprows=0
    )

    cell_sample_indices = [
        int(str(row[0]).strip('"')) - 1
        for _, row in metadata.iterrows()
    ]

    # Row-index -> barcode / condition, so the Figure5B/5C source-data files
    # can identify each point rather than emitting anonymous coordinates.
    # Same 0-based indexing as cell_sample_indices above.
    _idx_to_barcode, _idx_to_condition = {}, {}
    for _, _row in metadata.iterrows():
        try:
            _i = int(str(_row[0]).strip('"')) - 1
        except (TypeError, ValueError):
            continue
        _idx_to_barcode[_i]   = str(_row[1]).strip('"') if len(_row) > 1 else ""
        _idx_to_condition[_i] = str(_row[2]).strip('"') if len(_row) > 2 else ""

    for gene in GENE_ORDER:
        print(f"  Processing gene: {gene}")

        knockout_results_path = args.ko_pattern.format(gene=gene)
        knockin_results_path = args.ki_pattern.format(gene=gene)

        if (
            not os.path.exists(knockout_results_path)
            or not os.path.exists(knockin_results_path)
        ):
            print(f"    Warning: Perturbation files not found for {gene}, skipping...")
            continue

        with open(knockin_results_path, "rb") as f:
            knockin_results = pickle.load(f)

        with open(knockout_results_path, "rb") as f:
            knockout_results = pickle.load(f)

        # -------------------------------------------------------------
        # BASELINE AS-/AS+ counts for this gene (FIXED)
        # -------------------------------------------------------------
        # condition_bias_array[i] describes the state whose state-number
        # is sorted_indices[i] (pseudotime-sorted rank ordering — see the
        # bifurcation-dict construction above, which uses this exact same
        # pairing). The gene's ON/OFF value for that SAME state must be
        # looked up via columns_dict[f"State {sorted_indices[i]}"], NOT
        # via unique_states[i], which is in file order and only lines up
        # with condition_bias_array's ordering by coincidence.
        baseline_am = baseline_ap = 0
        if gene not in gene_to_col_index:
            print(f"    Warning: {gene} not found in gene order, skipping...")
            continue
        gene_col = gene_to_col_index[gene]
        # Baseline condition-bias values restricted to the states where this
        # gene is ON. Collected so the per-gene Wilcoxon test below compares
        # like with like: the perturbed arrays are what happens when the gene
        # is forced ON (KI) or OFF (KO), so the natural reference is the
        # gene-ON subset of the baseline rather than all states. The full
        # baseline is also tested, and both p-values are reported.
        baseline_gene_on_bias = []
        for i, state_num in enumerate(sorted_indices):
            state_vec = columns_dict[f"State {state_num}"]
            cb        = condition_bias_array[i]
            if state_vec[gene_col] == 1:
                baseline_gene_on_bias.append(cb)
                if cb > 0: baseline_am += 1
                else:      baseline_ap += 1
        baseline_gene_on_bias = np.asarray(baseline_gene_on_bias, dtype=float)
        print(f"    baseline: AS-={baseline_am}  AS+={baseline_ap}")

        y_values_ki  = []
        perturbed_x_ki  = []
        perturbed_pt_ki = []
        cell_ids_ki     = []      # which cell each KI prediction came from
        for cell_idx in cell_sample_indices:
            if cell_idx in knockin_results:
                knockin_traj     = np.array(knockin_results[cell_idx])
                perturbed_binary = [int(bit) for bit in knockin_traj[:, -1]]
                X_test           = x_scaler.transform([perturbed_binary])
                raw_prediction   = model_multi.predict(X_test)[0]
                cond_pred        = np.tanh(raw_prediction[0])
                pseudotime_pred  = np.clip(raw_prediction[1], 0, 1)
                y_values_ki.append(cond_pred)
                perturbed_x_ki.append(pseudotime_pred)
                perturbed_pt_ki.append(pseudotime_pred)
                cell_ids_ki.append(cell_idx)

        y_values_ko  = []
        perturbed_x_ko  = []
        perturbed_pt_ko = []
        cell_ids_ko     = []      # which cell each KO prediction came from
        for cell_idx in cell_sample_indices:
            if cell_idx in knockout_results:
                knockout_traj    = np.array(knockout_results[cell_idx])
                perturbed_binary = [int(bit) for bit in knockout_traj[:, -1]]
                X_test           = x_scaler.transform([perturbed_binary])
                raw_prediction   = model_multi.predict(X_test)[0]
                cond_pred        = np.tanh(raw_prediction[0])
                pseudotime_pred  = np.clip(raw_prediction[1], 0, 1)
                y_values_ko.append(cond_pred)
                perturbed_x_ko.append(pseudotime_pred)
                perturbed_pt_ko.append(pseudotime_pred)
                cell_ids_ko.append(cell_idx)

        ki_arr      = np.array(y_values_ki)
        ko_arr      = np.array(y_values_ko)
        ki_as_minus = int(np.sum(ki_arr >  0))
        ki_as_plus  = int(np.sum(ki_arr <= 0))
        ko_as_minus = int(np.sum(ko_arr >  0))
        ko_as_plus  = int(np.sum(ko_arr <= 0))
        print(f"    ki: AS-={ki_as_minus}  AS+={ki_as_plus}  |  ko: AS-={ko_as_minus}  AS+={ko_as_plus}")

        # -------------------------------------------------------------
        # PER-GENE WILCOXON RANK-SUM TESTS  (feed Figure 5A's bar stars)
        # -------------------------------------------------------------
        # Unpaired (rank-sum) rather than paired (signed-rank): the baseline
        # sample is one value per STATE and the perturbed sample is one value
        # per CELL, so the two samples are independent and generally of
        # different length — there is nothing to pair. This is the same test
        # and the same caveat already applied to Figure5B/5C for the selected
        # gene; it is simply now run for every gene in the panel.
        #
        # Two references are computed per arm:
        #   *_vs_gene_on : baseline restricted to states where the gene is ON
        #                  (the like-for-like comparison; this is the one the
        #                  Figure 5A stars use)
        #   *_vs_all     : baseline over all states (matches Figure5B/5C)
        def _ranksum_p(sample, reference):
            sample    = np.asarray(sample,    dtype=float)
            reference = np.asarray(reference, dtype=float)
            if sample.size == 0 or reference.size == 0:
                return np.nan, np.nan
            try:
                stat, p = ranksums(sample, reference)
            except Exception:
                return np.nan, np.nan
            return stat, p

        # -------------------------------------------------------------
        # PER-GENE PROPORTION TEST (Fisher exact) — the test that MATCHES
        # the bar being drawn.
        # -------------------------------------------------------------
        # The Figure 5A bar is baseline_AS+% minus perturbed_AS+%, so the
        # quantity on the panel is a difference in the AS+ SHARE. Fisher's
        # exact test on the same 2x2 counts asks exactly that question,
        # which is why a long bar now gets a small p and a short bar does
        # not. The rank-sum below asks a different question (has the
        # continuous distribution shifted in rank) and can disagree.
        #
        # Unit mismatch caveat, unchanged: the baseline row is per-STATE and
        # the perturbed row is per-CELL, so this is not two samples of the
        # same kind of thing. It is the test that corresponds to the plotted
        # number, not a clean design.
        def _fisher_p(base_plus, base_minus, pert_plus, pert_minus):
            table = np.array([[int(base_plus), int(base_minus)],
                              [int(pert_plus), int(pert_minus)]])
            if table.sum(axis=1).min() == 0 or table.sum(axis=0).min() == 0:
                return np.nan, np.nan
            try:
                odds, p = fisher_exact(table)
            except Exception:
                return np.nan, np.nan
            return odds, p

        _or_ki, p_ki_prop = _fisher_p(baseline_ap, baseline_am,
                                      ki_as_plus,  ki_as_minus)
        _or_ko, p_ko_prop = _fisher_p(baseline_ap, baseline_am,
                                      ko_as_plus,  ko_as_minus)

        _s_ki_on,  p_ki_gene_on  = _ranksum_p(ki_arr, baseline_gene_on_bias)
        _s_ko_on,  p_ko_gene_on  = _ranksum_p(ko_arr, baseline_gene_on_bias)
        _s_ki_all, p_ki_gene_all = _ranksum_p(ki_arr, baseline_y)
        _s_ko_all, p_ko_gene_all = _ranksum_p(ko_arr, baseline_y)

        print(f"    fisher  (AS+ share, matches the bar):    "
              f"KI p={p_ki_prop:.3g}  KO p={p_ko_prop:.3g}")
        print(f"    wilcoxon (rank-sum vs gene-ON baseline): "
              f"KI p={p_ki_gene_on:.3g}  KO p={p_ko_gene_on:.3g}")

        for _arm, _p, _or, _pp, _pm in (
                ("KNOCKIN",  p_ki_prop, _or_ki, ki_as_plus, ki_as_minus),
                ("KNOCKOUT", p_ko_prop, _or_ko, ko_as_plus, ko_as_minus)):
            record_stat(
                figure=FIG5A_FAMILY_PROPORTION,
                panels="Figure5A.png",
                test=f"Fisher exact, two-sided: {gene} {_arm} AS+ share vs. "
                     f"baseline AS+ share among states where {gene} is ON",
                value=_p, value_name="p-value",
                gene=gene,
                on_figure=_drawn_flag(FIG5A_FAMILY_PROPORTION),
                baseline_AS_plus=baseline_ap, baseline_AS_minus=baseline_am,
                perturbed_AS_plus=int(_pp), perturbed_AS_minus=int(_pm),
                odds_ratio=(f"{_or:.6g}" if np.isfinite(_or) else "n/a"),
            )

        record_stat(
            figure=FIG5A_FAMILY_RANKSUM,
            panels="Figure5A.png",
            test=f"Wilcoxon rank-sum, two-sided: {gene} KNOCKIN predicted "
                 f"condition bias vs. baseline condition bias of the states "
                 f"where {gene} is ON",
            value=p_ki_gene_on,
            value_name="p-value",
            gene=gene,
            on_figure=_drawn_flag(FIG5A_FAMILY_RANKSUM),
            n_perturbed_cells=int(ki_arr.size),
            n_baseline_states_gene_on=int(baseline_gene_on_bias.size),
            median_perturbed=(f"{np.median(ki_arr):.6f}" if ki_arr.size else "n/a"),
            median_baseline_gene_on=(f"{np.median(baseline_gene_on_bias):.6f}"
                                     if baseline_gene_on_bias.size else "n/a"),
            statistic=(f"{_s_ki_on:.6g}" if np.isfinite(_s_ki_on) else "n/a"),
            p_value_vs_all_states=(f"{p_ki_gene_all:.6g}"
                                   if np.isfinite(p_ki_gene_all) else "n/a"),
        )
        record_stat(
            figure=FIG5A_FAMILY_RANKSUM,
            panels="Figure5A.png",
            test=f"Wilcoxon rank-sum, two-sided: {gene} KNOCKOUT predicted "
                 f"condition bias vs. baseline condition bias of the states "
                 f"where {gene} is ON",
            value=p_ko_gene_on,
            value_name="p-value",
            gene=gene,
            on_figure=_drawn_flag(FIG5A_FAMILY_RANKSUM),
            n_perturbed_cells=int(ko_arr.size),
            n_baseline_states_gene_on=int(baseline_gene_on_bias.size),
            median_perturbed=(f"{np.median(ko_arr):.6f}" if ko_arr.size else "n/a"),
            median_baseline_gene_on=(f"{np.median(baseline_gene_on_bias):.6f}"
                                     if baseline_gene_on_bias.size else "n/a"),
            statistic=(f"{_s_ko_on:.6g}" if np.isfinite(_s_ko_on) else "n/a"),
            p_value_vs_all_states=(f"{p_ko_gene_all:.6g}"
                                   if np.isfinite(p_ko_gene_all) else "n/a"),
        )

        # ---- contingency tables for review ----
        for _arm, _pp, _pm, _fp, _wp, _or in (
                ("KNOCKIN",  ki_as_plus, ki_as_minus, p_ki_prop, p_ki_gene_on, _or_ki),
                ("KNOCKOUT", ko_as_plus, ko_as_minus, p_ko_prop, p_ko_gene_on, _or_ko)):
            _bn = baseline_ap + baseline_am
            _pn = _pp + _pm
            _b_pct = (baseline_ap / _bn * 100) if _bn else np.nan
            _p_pct = (_pp / _pn * 100) if _pn else np.nan
            figure5a_tables.append({
                "Gene": gene,
                "Arm": _arm,
                # --- the 2x2 Fisher consumes (counts) ---
                "baseline_AS_plus":   baseline_ap,
                "baseline_AS_minus":  baseline_am,
                "baseline_n_states":  _bn,
                "perturbed_AS_plus":  int(_pp),
                "perturbed_AS_minus": int(_pm),
                "perturbed_n_cells":  int(_pn),
                # --- what the bar shows ---
                "baseline_AS_plus_pct":  round(_b_pct, 4) if _bn else "",
                "perturbed_AS_plus_pct": round(_p_pct, 4) if _pn else "",
                "delta_AS_plus_pct": (round(_b_pct - _p_pct, 4)
                                      if (_bn and _pn) else ""),
                # --- both tests, so they can be compared directly ---
                "fisher_p":   f"{_fp:.10g}" if np.isfinite(_fp) else "",
                "fisher_stars": _p_stars(_fp),
                #"fisher_odds_ratio": f"{_or:.6g}" if np.isfinite(_or) else "",
                #"wilcoxon_ranksum_p": f"{_wp:.10g}" if np.isfinite(_wp) else "",
                #"wilcoxon_stars": _p_stars(_wp),
            })

        all_gene_results[gene] = {
            'baseline': {'as_minus': baseline_am,  'as_plus': baseline_ap},
            'ki':       {'as_minus': ki_as_minus,  'as_plus': ki_as_plus},
            'ko':       {'as_minus': ko_as_minus,  'as_plus': ko_as_plus},
            'wilcoxon': {'ki_p_gene_on':  p_ki_gene_on,
                         'ko_p_gene_on':  p_ko_gene_on,
                         'ki_p_all':      p_ki_gene_all,
                         'ko_p_all':      p_ko_gene_all,
                         # Fisher exact on the AS+ share — matches the bar
                         'ki_p_prop':     p_ki_prop,
                         'ko_p_prop':     p_ko_prop},
        }

        if gene == selected_gene:
            ki_data_selected = (perturbed_x_ki, y_values_ki, perturbed_pt_ki, gene)
            ko_data_selected = (perturbed_x_ko, y_values_ko, perturbed_pt_ko, gene)
            ki_cell_ids_selected = list(cell_ids_ki)
            ko_cell_ids_selected = list(cell_ids_ko)

    # ========== Save for Figure 7 ==========
    np.save(_outpath(outdir, "baseline_condition_bias.npy"), condition_bias_array)

    if ki_data_selected is not None and ko_data_selected is not None:
        _, y_ki, _, gene_sel = ki_data_selected
        _, y_ko, _, _        = ko_data_selected
        np.save(_outpath(outdir, f"{gene_sel}_ki_condition_bias.npy"), np.array(y_ki))
        np.save(_outpath(outdir, f"{gene_sel}_ko_condition_bias.npy"), np.array(y_ko))
        print(f"  Saved perturbation condition bias arrays for {gene_sel}")
    else:
        print("  Warning: Selected gene perturbation data not found.")

    if len(all_gene_results) == 0:
        print("Warning: No perturbation results found. Skipping Figure 5 generation.")
        write_stats_report(outdir, getattr(args, "stats_report", None))
        return

    # ========== Stacked bar figures ==========
    def plot_stacked_figure(all_gene_results, condition="baseline", title="Normal Signaling"):
        entry_map = {}
        for gene, data in all_gene_results.items():
            minus = data[condition]["as_minus"]
            plus  = data[condition]["as_plus"]
            total = minus + plus
            if total == 0:
                continue
            entry_map[gene] = (minus / total * 100, plus / total * 100)

        ordered = [(g, *entry_map[g]) for g in reversed(GENE_ORDER) if g in entry_map]
        if not ordered:
            print(f"  No data to plot for condition='{condition}'.")
            return

        names, as_minus_pct, as_plus_pct = zip(*ordered)
        fig, ax    = plt.subplots(figsize=(12, max(6, len(names) * 0.55)))
        y          = np.arange(len(names))
        bar_height = 0.75

        ax.barh(y, as_minus_pct, bar_height, color="#2ca02c", edgecolor="white", linewidth=0.5)
        ax.barh(y, as_plus_pct,  bar_height, left=as_minus_pct,
                color="#d62728", edgecolor="white", linewidth=0.5)

        for i, (m, p) in enumerate(zip(as_minus_pct, as_plus_pct)):
            if m > 8:
                ax.text(m / 2,     i, f"{m:.1f}%", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
            if p > 8:
                ax.text(m + p / 2, i, f"{p:.1f}%", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

        ax.axvline(50, color="grey", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=13)
        ax.set_xlabel("ON Percentage", fontsize=15, fontweight="bold")
        ax.set_ylabel("Gene",          fontsize=15, fontweight="bold")
        ax.set_title(title,            fontsize=16, fontweight="bold", pad=12)
        ax.tick_params(labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(1.5)
        ax.legend(
            handles=[
                Patch(facecolor="#2ca02c", edgecolor="black", label="AS-"),
                Patch(facecolor="#d62728", edgecolor="black", label="AS+"),
            ],
            fontsize=12, loc="upper right", frameon=True, edgecolor="black",
        )
        plt.tight_layout()
        out_name = f"{condition}_stacked.png"
        plt.savefig(_outpath(outdir, out_name), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_name}")

    plot_stacked_figure(all_gene_results, condition="baseline", title="Normal Signaling")
    plot_stacked_figure(all_gene_results, condition="ki",       title="Knockin Signaling")
    plot_stacked_figure(all_gene_results, condition="ko",       title="Knockout Signaling")

    # ========== Figure 5A: Delta AS+ Bar Chart ==========
    #
    # WHAT THIS PANEL SHOWS
    # ---------------------
    # One horizontal bar pair per gene. For a given gene:
    #
    #   baseline_pct = % of the states where that gene is ON that are
    #                  classified AS+ (condition bias <= 0)
    #   ki_pct       = % of cells whose XBP1-style KNOCKIN-perturbed attractor
    #                  the MLP predicts as AS+
    #   ko_pct       = same for the KNOCKOUT-perturbed attractor
    #
    #   purple bar (Knockin)  = baseline_pct - ki_pct
    #   yellow bar (Knockout) = baseline_pct - ko_pct
    #
    # So the x-axis is a DIFFERENCE OF PERCENTAGES in percentage points:
    # "how much of the AS+ (disease-biased) fraction does perturbing this gene
    # remove?". Positive (right) = perturbation moves the population AWAY from
    # AS+ (fewer disease-biased states — a putatively protective perturbation).
    # Negative (left) = perturbation pushes the population TOWARD AS+.
    # Zero = the perturbation leaves the AS+/AS- split unchanged.
    # The dashed guides at 0, +/-10 and +/-20 are eyeball thresholds for
    # "negligible / modest / large" shifts, not test results.
    #
    # STATISTICS ON THIS PANEL (new)
    # ------------------------------
    #  * stars at each bar tip  : that gene's Wilcoxon RANK-SUM p-value,
    #    perturbed cells vs. the baseline states where the gene is ON
    #    (computed in the perturbation loop above)
    #  * summary box            : three Wilcoxon SIGNED-RANK tests taken
    #    across the genes shown, on the plotted deltas themselves
    # ---- export the contingency tables for review ----
    if figure5a_tables:
        _ct_path = _outpath(outdir, "figure5A_contingency_tables.csv")
        pd.DataFrame(figure5a_tables).to_csv(_ct_path, index=False)
        print(f"\n  Wrote {_ct_path}")
        print(f"    {len(figure5a_tables)} rows "
              f"({len(figure5a_tables)//2} genes x KNOCKIN/KNOCKOUT). "
              f"Each row is one 2x2 table: the baseline_* counts are STATES "
              f"where the gene is ON, the perturbed_* counts are CELLS.")
        write_contingency_tables_txt(
            figure5a_tables, outdir,
            star_test_label=("Wilcoxon rank-sum" if FIGURE5A_STAR_TEST == "ranksum"
                             else "Fisher exact (AS+ delta)"))
        _thin = [r for r in figure5a_tables if r["baseline_n_states"] < 10]
        if _thin:
            print(f"    NOTE: {len(_thin)} row(s) have fewer than 10 baseline "
                  f"states — their bars rest on very little data: "
                  f"{sorted({r['Gene'] for r in _thin})}")

    print("\n  Generating Figure 5A: Delta AS+ Percentage Bar Chart...")

    delta_map = {}
    for gene, data in all_gene_results.items():
        baseline_total = data['baseline']['as_minus'] + data['baseline']['as_plus']
        ki_total       = data['ki']['as_minus']       + data['ki']['as_plus']
        ko_total       = data['ko']['as_minus']       + data['ko']['as_plus']
        if baseline_total == 0 or ki_total == 0 or ko_total == 0:
            continue
        baseline_pct    = data['baseline']['as_plus'] / baseline_total * 100
        delta_map[gene] = (baseline_pct - data['ki']['as_plus'] / ki_total * 100,
                        baseline_pct - data['ko']['as_plus'] / ko_total * 100)

    gene_names_sorted       = [g for g in reversed(GENE_ORDER) if g in delta_map]
    delta_ki_as_plus_sorted = [delta_map[g][0] for g in gene_names_sorted]
    delta_ko_as_plus_sorted = [delta_map[g][1] for g in gene_names_sorted]

    _fig4_across = None
    ki_pvalue = ko_pvalue = np.nan
    if not gene_names_sorted:
        print("  No data for Figure 5A, skipping.")
    else:
        # -------------------------------------------------------------
        # ACROSS-GENE WILCOXON SIGNED-RANK TESTS
        # -------------------------------------------------------------
        # Paired here, unlike the per-gene tests: each gene contributes one
        # delta to each arm, so delta_KI[g] and delta_KO[g] are two
        # measurements of the same unit (the gene) and the pairing is real.
        #
        #   1. delta_KI vs 0  — is knock-in shifting AS+ composition at all,
        #                       taking the panel as a whole?
        #   2. delta_KO vs 0  — same question for knock-out.
        #   3. delta_KI vs delta_KO — do the two perturbation directions
        #                       produce different shifts?
        #
        # scipy raises when every difference is exactly zero (nothing left to
        # rank after dropping ties), and warns for very small n, so each call
        # is guarded and falls back to nan -> "p = n/a" on the panel.
        d_ki = np.asarray(delta_ki_as_plus_sorted, dtype=float)
        d_ko = np.asarray(delta_ko_as_plus_sorted, dtype=float)
        n_genes_tested = len(d_ki)

        def _signed_rank_p(a, b=None):
            """Two-sided Wilcoxon signed-rank, nan-safe. b=None tests a vs 0."""
            try:
                if b is None:
                    if np.allclose(a, 0):
                        return np.nan, np.nan
                    stat, p = wilcoxon(a, alternative="two-sided")
                else:
                    if np.allclose(a - b, 0):
                        return np.nan, np.nan
                    stat, p = wilcoxon(a, b, alternative="two-sided")
                    _fig4_across = {"n_genes": n_genes_tested,
                        "ki_vs_zero": p_ki_vs_zero,
                        "ko_vs_zero": p_ko_vs_zero,
                        "ki_vs_ko":   p_ki_vs_ko}
            except Exception:
                return np.nan, np.nan
            return stat, p

        _w_ki,    p_ki_vs_zero  = _signed_rank_p(d_ki)
        _w_ko,    p_ko_vs_zero  = _signed_rank_p(d_ko)
        _w_kiko,  p_ki_vs_ko    = _signed_rank_p(d_ki, d_ko)

        print(f"    Wilcoxon signed-rank across {n_genes_tested} genes:")
        print(f"      Knockin  delta vs 0 : p = {p_ki_vs_zero:.4g}")
        print(f"      Knockout delta vs 0 : p = {p_ko_vs_zero:.4g}")
        print(f"      Knockin vs Knockout : p = {p_ki_vs_ko:.4g}")


        record_stat(
            figure="Figure 5A (across-gene Wilcoxon)",
            panels="Figure5A.png",
            test="Wilcoxon signed-rank, two-sided: knockin delta AS+ "
                 "percentage vs. 0, paired across genes",
            value=p_ki_vs_zero, value_name="p-value",
            n_genes=n_genes_tested,
            median_delta_KI=f"{np.median(d_ki):.6f}" if d_ki.size else "n/a",
            mean_delta_KI=f"{np.mean(d_ki):.6f}" if d_ki.size else "n/a",
            n_genes_positive_delta=int(np.sum(d_ki > 0)),
            n_genes_negative_delta=int(np.sum(d_ki < 0)),
            statistic=(f"{_w_ki:.6g}" if np.isfinite(_w_ki) else "n/a"),
        )
        record_stat(
            figure="Figure 5A (across-gene Wilcoxon)",
            panels="Figure5A.png",
            test="Wilcoxon signed-rank, two-sided: knockout delta AS+ "
                 "percentage vs. 0, paired across genes",
            value=p_ko_vs_zero, value_name="p-value",
            n_genes=n_genes_tested,
            median_delta_KO=f"{np.median(d_ko):.6f}" if d_ko.size else "n/a",
            mean_delta_KO=f"{np.mean(d_ko):.6f}" if d_ko.size else "n/a",
            n_genes_positive_delta=int(np.sum(d_ko > 0)),
            n_genes_negative_delta=int(np.sum(d_ko < 0)),
            statistic=(f"{_w_ko:.6g}" if np.isfinite(_w_ko) else "n/a"),
            
        )
        record_stat(
            figure="Figure 5A (across-gene Wilcoxon)",
            panels="Figure5A.png",
            test="Wilcoxon signed-rank, two-sided: knockin delta vs. knockout "
                 "delta, paired by gene",
            value=p_ki_vs_ko, value_name="p-value",
            n_genes=n_genes_tested,
            median_difference_KI_minus_KO=(f"{np.median(d_ki - d_ko):.6f}"
                                           if d_ki.size else "n/a"),
            n_genes_KI_greater=int(np.sum(d_ki > d_ko)),
            n_genes_KO_greater=int(np.sum(d_ko > d_ki)),
            statistic=(f"{_w_kiko:.6g}" if np.isfinite(_w_kiko) else "n/a"),
            
        )

        # ------------------------------------------------------------------
        # SOURCE DATA — Figure5A_source_data.csv
        # ------------------------------------------------------------------
        # Exactly what the panel draws: one row per gene, in the order the
        # bars appear from bottom to top, with each bar's length, the
        # p-value behind its star and the star itself. The raw 2x2 counts
        # these are derived from live in figure5A_contingency_tables.csv.
        _star_key_ki = ("ki_p_gene_on" if FIGURE5A_STAR_TEST == "ranksum"
                        else "ki_p_prop")
        _star_key_ko = ("ko_p_gene_on" if FIGURE5A_STAR_TEST == "ranksum"
                        else "ko_p_prop")
        _a_rows = []
        for _i, _g in enumerate(gene_names_sorted):
            _wc = all_gene_results.get(_g, {}).get("wilcoxon", {})
            _bd = all_gene_results.get(_g, {})
            _a_rows.append({
                "plot_order_bottom_to_top": _i + 1,
                "Gene": _g,
                "delta_AS_plus_pct_KNOCKIN":  round(delta_ki_as_plus_sorted[_i], 6),
                "delta_AS_plus_pct_KNOCKOUT": round(delta_ko_as_plus_sorted[_i], 6),
                "baseline_AS_plus_states":  _bd.get("baseline", {}).get("as_plus", ""),
                "baseline_AS_minus_states": _bd.get("baseline", {}).get("as_minus", ""),
                "KI_AS_plus_cells":  _bd.get("ki", {}).get("as_plus", ""),
                "KI_AS_minus_cells": _bd.get("ki", {}).get("as_minus", ""),
                "KO_AS_plus_cells":  _bd.get("ko", {}).get("as_plus", ""),
                "KO_AS_minus_cells": _bd.get("ko", {}).get("as_minus", ""),
                "p_KNOCKIN":  _fmt_or_blank(_wc.get(_star_key_ki, np.nan)),
                "p_KNOCKOUT": _fmt_or_blank(_wc.get(_star_key_ko, np.nan)),
                "stars_KNOCKIN":  _p_stars(_wc.get(_star_key_ki, np.nan)),
                "stars_KNOCKOUT": _p_stars(_wc.get(_star_key_ko, np.nan)),
                "test_shown": ("Wilcoxon rank-sum" if FIGURE5A_STAR_TEST == "ranksum"
                               else "Fisher exact"),
            })
        _a_path = _outpath(outdir, "Figure5A_source_data.csv")
        pd.DataFrame(_a_rows).to_csv(_a_path, index=False)
        print(f"  Wrote {_a_path} ({len(_a_rows)} genes)")

        fig, ax = plt.subplots(figsize=(8, 6))
        x       = np.arange(len(gene_names_sorted))
        width   = 0.35

        ax.barh(x - width/2, delta_ki_as_plus_sorted, width,
                label='Knockin',  color='purple', alpha=0.8, edgecolor='black', linewidth=2)
        ax.barh(x + width/2, delta_ko_as_plus_sorted, width,
                label='Knockout', color='yellow', alpha=0.8, edgecolor='black', linewidth=2)

        ax.axvline(0,   color='black', linewidth=2, linestyle='--')
        ax.axvline( 20, color='black', linewidth=2, linestyle='--', alpha=0.5)
        ax.axvline(-20, color='black', linewidth=2, linestyle='--', alpha=0.5)
        ax.axvline( 10, color='blue',  linewidth=2, linestyle=':',  alpha=0.5)
        ax.axvline(-10, color='blue',  linewidth=2, linestyle=':',  alpha=0.5)

        max_abs = max(max(abs(d) for d in delta_ki_as_plus_sorted),
                      max(abs(d) for d in delta_ko_as_plus_sorted))
        # A little extra headroom when the stars are being drawn, so the
        # markers at the bar tips don't collide with the axes frame.
        span_mult = 1.45 if ANNOTATE_FIGURE5A_WILCOXON else 1.3
        ax.set_xlim(-max(max_abs * span_mult, 35), max(max_abs * span_mult, 35))
        ax.set_yticks(x)
        ax.set_yticklabels(gene_names_sorted, fontsize=14)
        if ANNOTATE_FIGURE5A_WILCOXON and SHOW_PERTURBATION_STAT_BOXES:
            # Reserve a blank strip below the bottom gene for the summary box,
            # so it can never sit on top of a bar however the deltas fall.
            # Not needed in stars-only mode — the bars fill the panel.
            ax.set_ylim(-1.9, len(gene_names_sorted) - 0.5)
        ax.set_xlabel('Delta AS+ Percentage (Normal - Perturbed)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=16, fontweight='bold')
        ax.legend(fontsize=14, loc='upper left', frameon=True, edgecolor='black')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # ---- per-gene significance stars at the bar tips ----
        if ANNOTATE_FIGURE5A_WILCOXON:
            x_lo, x_hi = ax.get_xlim()
            pad = 0.012 * (x_hi - x_lo)
            # Which p-value the stars report — see FIGURE5A_STAR_TEST.
            if FIGURE5A_STAR_TEST == "ranksum":
                _ki_key, _ko_key = "ki_p_gene_on", "ko_p_gene_on"
            else:
                _ki_key, _ko_key = "ki_p_prop", "ko_p_prop"
            print(f"    Figure5A stars are showing the "
                  f"{'Wilcoxon rank-sum' if FIGURE5A_STAR_TEST == 'ranksum' else 'Fisher exact (AS+ share)'}"
                  f" p-values.")
            for i, gene in enumerate(gene_names_sorted):
                wc = all_gene_results.get(gene, {}).get("wilcoxon", {})
                for y_pos, delta, p_key in (
                        (x[i] - width/2, delta_ki_as_plus_sorted[i], _ki_key),
                        (x[i] + width/2, delta_ko_as_plus_sorted[i], _ko_key)):
                    stars = _p_stars(wc.get(p_key, np.nan))
                    if not stars:
                        continue
                    # Sit the marker just beyond the end of the bar, on the
                    # side the bar points to.
                    if delta >= 0:
                        tx, ha = delta + pad, "left"
                    else:
                        tx, ha = delta - pad, "right"
                    ax.text(tx, y_pos, stars, fontsize=11, fontweight="bold",
                            color="black", ha=ha, va="center")

        # ---- across-gene signed-rank summary box ----
        _figure5a_stat_box(
            ax, 0.99, 0.015,
            "Wilcoxon signed-rank (across genes)    "
            f"KI vs 0: {_fmt_p(p_ki_vs_zero)}    "
            f"KO vs 0: {_fmt_p(p_ko_vs_zero)}    "
            f"KI vs KO: {_fmt_p(p_ki_vs_ko)}\n"
            f"n = {n_genes_tested} genes    "
            "bar tips = per-gene rank-sum vs baseline:  "
            + _pstar_legend_text(),
            fontsize=9, ha="right", va="bottom",
        )

        plt.savefig(_outpath(outdir, "Figure5A.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved Figure5A.png")

    # ========== Figures 5B & 5C ==========
    def _bifurcation_fig(x_vals, y_vals, title, output_file, marker,
                         x_normal=None, y_normal=None, point_color=None,
                         pvalue=None):
        """
        `pvalue`, when given, is that panel's Wilcoxon rank-sum p-value
        (perturbed vs. baseline condition bias). It is rendered as a
        significance star appended to the title, plus the numeric value in
        the top-right corner — both gated on ANNOTATE_FIGURE5A_WILCOXON, the
        same switch that controls the Figure 5A annotations. Pass None (the
        default) for panels with no associated test, e.g. Figure5Normal.
        """
        fig = plt.figure(figsize=(8, 6))
        gs  = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        axm = fig.add_subplot(gs[0])
        axh = fig.add_subplot(gs[1], sharey=axm)

        # Normal baseline = grey squares, drawn underneath the perturbed points
        if x_normal is not None and y_normal is not None and len(x_normal):
            axm.scatter(x_normal, y_normal, color="gray", edgecolor="dimgray",
                        s=150, marker="s", alpha=0.45, linewidths=1,
                        zorder=1, label="Normal signaling")

        if point_color is None:
            axm.scatter(x_vals, y_vals, c=x_vals, cmap="coolwarm",
                        edgecolor="black", s=150, marker=marker,
                        alpha=0.6, zorder=3)
        else:
            axm.scatter(x_vals, y_vals, color=point_color,
                        edgecolor="black", s=150, marker=marker,
                        alpha=0.6, zorder=3)

        mean_v   = np.mean(y_vals)
        median_v = np.median(y_vals)
        axm.axhline(mean_v,   color="blue",     linestyle=":",  linewidth=2,
                    label=f"Mean: {mean_v:.3f}")
        axm.axhline(median_v, color="darkblue", linestyle="--", linewidth=2,
                    label=f"Median: {median_v:.3f}")
        axm.axhline(0,        color="black",    linestyle="--", linewidth=2)
        axm.legend(loc="lower left", fontsize=12)

        # Histogram: perturbed distribution, with the normal (unperturbed)
        # distribution underneath in grey when y_normal is supplied.
        hist_bins = np.linspace(-1, 1, 21)   # same 0.1-wide bins as before

        if y_normal is not None and len(y_normal):
            # Baseline has one entry per STATE while the perturbed arrays have
            # one per CELL, so rescale the grey counts to the perturbed sample
            # size — otherwise the two histograms aren't visually comparable.
            norm_w = np.full(len(y_normal), len(y_vals) / len(y_normal))
            axh.hist(y_normal, bins=hist_bins, weights=norm_w,
                     orientation="horizontal", color="gray",
                     edgecolor="dimgray", alpha=0.45, linewidth=1,
                     zorder=1, label="Normal signaling")
            #axh.legend(loc="upper left", fontsize=9, framealpha=0.6)

        counts, bin_edges, patches = axh.hist(
            y_vals, bins=hist_bins,
            orientation="horizontal", edgecolor="black", alpha=0.85, zorder=2)
        for patch, edge in zip(patches, bin_edges[:-1]):
            center = edge + 0.05
            patch.set_facecolor("red" if center < 0 else "teal")

        axh.axhline(0,        color="black",    linestyle="--", linewidth=2)
        axh.axhline(mean_v,   color="blue",     linestyle=":",  linewidth=2)
        axh.axhline(median_v, color="darkblue", linestyle="--", linewidth=2)
        axh.yaxis.set_visible(False)

        y_arr = np.array(y_vals)
        am_p  = np.sum(y_arr >  0) / len(y_vals) * 100
        ap_p  = np.sum(y_arr <= 0) / len(y_vals) * 100
        # Descriptive composition, not a test result — kept on the panel.
        axm.text(0.02, 0.98,
                 f"AS-: {am_p:.1f}%\nAS+: {ap_p:.1f}%",
                 transform=axm.transAxes, fontsize=14, va="top", fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axm.set_xlim(-0.1, 1.1); axm.set_ylim(-1.1, 1.1)
        axm.set_xlabel("Normalized Pseudotime",        fontsize=16, fontweight="bold")
        axm.set_ylabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight="bold")

        # Significance star on the title, numeric p in the free top-right
        # corner (the wheat composition box sits top-left, the mean/median
        # legend bottom-left, so this corner is clear).
        panel_title = title
        if pvalue is not None and ANNOTATE_FIGURE5A_WILCOXON:
            if ANNOTATE_FIGURE5BC_STARS:
                stars = _p_stars(pvalue)
                if stars:
                    panel_title = f"{title}  {stars}"
            if SHOW_PERTURBATION_STAT_BOXES:
                axm.text(0.98, 0.98,
                         f"{_fmt_p(pvalue)}\nWilcoxon rank-sum vs. normal",
                         transform=axm.transAxes, fontsize=10,
                         fontweight="bold", ha="right", va="top",
                         bbox=dict(boxstyle="round", facecolor="white",
                                   edgecolor="black", alpha=0.85))
        axm.set_title(panel_title, fontsize=18, fontweight="bold")
        axm.tick_params(labelsize=14)
        axh.set_xlabel("Frequency", fontsize=14, fontweight="bold")
        axh.tick_params(labelsize=14)
        for axes in (axm, axh):
            for sp in axes.spines.values(): sp.set_linewidth(2)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    # ---- Figure 5Normal: unperturbed baseline on its own ----
    # (no y_normal passed — otherwise the baseline would be drawn twice)
    _bifurcation_fig(baseline_x, baseline_y, "Normal Signaling",
                     _outpath(outdir, "Figure5Normal.png"), "s",
                     point_color="gray")
    print("  Saved Figure5Normal.png")

    if ki_data_selected and ko_data_selected:
        x_ki, y_ki, _, gene_sel = ki_data_selected
        x_ko, y_ko, _, _        = ko_data_selected

        # ---- Wilcoxon rank-sum test: perturbed vs. normal condition bias ----
        # baseline_y is one value per STATE; y_ki/y_ko are one value per CELL —
        # independent, unequal-length samples, so this is the rank-sum
        # (unpaired) Wilcoxon variant, not the signed-rank (paired) one.
        if len(y_ki) > 0 and len(baseline_y) > 0:
            _stat_ki, ki_pvalue = ranksums(y_ki, baseline_y)
        else:
            _stat_ki, ki_pvalue = np.nan, np.nan
        if len(y_ko) > 0 and len(baseline_y) > 0:
            _stat_ko, ko_pvalue = ranksums(y_ko, baseline_y)
        else:
            _stat_ko, ko_pvalue = np.nan, np.nan

        record_stat(
            figure="Figure 5", panels="Figure5B.png",
            test=f"Wilcoxon rank-sum test, two-sided: {gene_sel} knockin "
                 f"condition bias vs. normal (baseline) condition bias",
            value=ki_pvalue, value_name="p-value",
            n_perturbed_cells=len(y_ki), n_baseline_states=len(baseline_y),
            median_perturbed=f"{np.median(y_ki):.6f}" if len(y_ki) else "n/a",
            median_baseline=f"{np.median(baseline_y):.6f}" if len(baseline_y) else "n/a",
            statistic=(f"{_stat_ki:.6g}" if np.isfinite(_stat_ki) else "n/a"),
        )
        record_stat(
            figure="Figure 5", panels="Figure5C.png",
            test=f"Wilcoxon rank-sum test, two-sided: {gene_sel} knockout "
                 f"condition bias vs. normal (baseline) condition bias",
            value=ko_pvalue, value_name="p-value",
            n_perturbed_cells=len(y_ko), n_baseline_states=len(baseline_y),
            median_perturbed=f"{np.median(y_ko):.6f}" if len(y_ko) else "n/a",
            median_baseline=f"{np.median(baseline_y):.6f}" if len(baseline_y) else "n/a",
            statistic=(f"{_stat_ko:.6g}" if np.isfinite(_stat_ko) else "n/a"),
        )

        # ------------------------------------------------------------------
        # SOURCE DATA — Figure5B_source_data.csv / Figure5C_source_data.csv
        # ------------------------------------------------------------------
        # One row per plotted point. `series` separates the two things drawn
        # on each panel:
        #   perturbed_cell  – the coloured triangles, one per CELL, from the
        #                     MLP's prediction on that cell's perturbed
        #                     attractor
        #   baseline_state  – the grey squares, one per STATE, the
        #                     unperturbed reference
        # The side histogram is just a binning of the condition_bias column,
        # so no separate file is needed for it. Panel-level numbers (mean,
        # median, AS-/AS+ split, p-value) go to Figure5BC_panel_summary.csv.
        def _write_bc_source(fname, arm, x_vals, y_vals, cell_ids):
            rows = []
            for _j, (_x, _y) in enumerate(zip(x_vals, y_vals)):
                _cid = cell_ids[_j] if _j < len(cell_ids) else ""
                rows.append({
                    "series": "perturbed_cell",
                    "gene": gene_sel,
                    "arm": arm,
                    "unit": "cell",
                    "cell_row_index_0based": _cid,
                    "barcode": _idx_to_barcode.get(_cid, ""),
                    "condition": _idx_to_condition.get(_cid, ""),
                    #"state_number": "",
                    "normalized_pseudotime": round(float(_x), 6),
                    "condition_bias": round(float(_y), 6),
                    #"class_at_zero_threshold": "AS-" if _y > 0 else "AS+",
                })
            for _i, _state in enumerate(sorted_indices):
                rows.append({
                    "series": "baseline_state",
                    "gene": gene_sel,
                    "arm": arm,
                    "unit": "state",
                    "cell_row_index_0based": "",
                    "barcode": "",
                    "condition": "",
                    "state_number": _state,
                    "normalized_pseudotime": round(float(baseline_x[_i]), 6),
                    "condition_bias": round(float(baseline_y[_i]), 6),
                    #"class_at_zero_threshold": ("AS-" if baseline_y[_i] > 0
                                                #else "AS+"),
                })
            _path = _outpath(outdir, fname)
            pd.DataFrame(rows).to_csv(_path, index=False)
            _n_pert = sum(1 for r in rows if r["series"] == "perturbed_cell")
            print(f"  Wrote {_path} "
                  f"({_n_pert} perturbed cells + {len(rows)-_n_pert} baseline states)")

        _write_bc_source("Figure5B_source_data.csv", "KNOCKIN",
                         x_ki, y_ki, ki_cell_ids_selected)
        _write_bc_source("Figure5C_source_data.csv", "KNOCKOUT",
                         x_ko, y_ko, ko_cell_ids_selected)

        # Panel-level summary: the numbers printed in the boxes / legend.
        _bc_summary = []
        for _panel, _arm, _yv, _pv in (("Figure5B.png", "KNOCKIN",  y_ki, ki_pvalue),
                                       ("Figure5C.png", "KNOCKOUT", y_ko, ko_pvalue)):
            _ya = np.asarray(_yv, dtype=float)
            _bc_summary.append({
                "panel": _panel, "gene": gene_sel, "arm": _arm,
                "n_perturbed_cells": _ya.size,
                "mean_condition_bias":   round(float(np.mean(_ya)), 6) if _ya.size else "",
                "median_condition_bias": round(float(np.median(_ya)), 6) if _ya.size else "",
                "pct_AS_minus": round(float(np.sum(_ya > 0) / _ya.size * 100), 4) if _ya.size else "",
                "pct_AS_plus":  round(float(np.sum(_ya <= 0) / _ya.size * 100), 4) if _ya.size else "",
                "n_baseline_states": int(np.size(baseline_y)),
                "mean_baseline_condition_bias":   round(float(np.mean(baseline_y)), 6),
                "median_baseline_condition_bias": round(float(np.median(baseline_y)), 6),
                "test": "Wilcoxon rank-sum, two-sided (perturbed cells vs baseline states)",
                "p_value": _fmt_or_blank(_pv),
                "stars": _p_stars(_pv),
            })
        _sum_path = _outpath(outdir, "Figure5BC_panel_summary.csv")
        pd.DataFrame(_bc_summary).to_csv(_sum_path, index=False)
        print(f"  Wrote {_sum_path}")

        _bifurcation_fig(x_ki, y_ki, f"{gene_sel} Knockin",
                         _outpath(outdir, "Figure5B.png"), "^",
                         y_normal=baseline_y, pvalue=ki_pvalue)
        _bifurcation_fig(x_ko, y_ko, f"{gene_sel} Knockout",
                         _outpath(outdir, "Figure5C.png"), "v",
                         y_normal=baseline_y, pvalue=ko_pvalue)
        print("  Saved Figure5B.png and Figure5C.png")

    # ========== Combined Figure 5 ==========
    # (this block appeared twice in the previous revision — the duplicate was
    #  a no-op that redrew and re-saved the identical composite, so it has
    #  been removed.)
    fnames = ["Figure5A.png", "Figure5B.png", "Figure5C.png"]
    if all(os.path.exists(_outpath(outdir, f)) for f in fnames):
        imgs = [mpimg.imread(_outpath(outdir, f)) for f in fnames]
        fig  = plt.figure(figsize=(8, 6))
        gs   = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1],
                        hspace=0.15, wspace=0.1)
        for gsidx, label, img in [
                ((0, slice(None)), "a.", imgs[0]),
                ((1, 0),          "b.", imgs[1]),
                ((1, 1),          "c.", imgs[2])]:
            ax = fig.add_subplot(gs[gsidx])
            ax.imshow(img); ax.axis("off")
            ax.text(0.02, 1.02, label, transform=ax.transAxes,
                    fontsize=24, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
        plt.savefig(_outpath(outdir, "Figure5_Combined.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print("  Saved Figure5_Combined.png")

    write_figure4_legend_pvalues(
        outdir, gene_names_sorted, all_gene_results, _fig4_across,
        args.selected_gene, ki_pvalue, ko_pvalue)
    write_stats_report(outdir, getattr(args, "stats_report", None))

    print("Figure 5 complete.")


# ============================================================
# Shared condition palette
# ============================================================
# Single source of truth for AS- / AS+ across Figure 6B and Figure 7A/7B,
# which land side by side in RepatchJuly_Figure5_Combined.png.
# Explicit hex — named colours were the source of the mismatch.
CONDITION_COLORS = {
    "AS-": "#008080",   # teal
    "AS+": "#FF0000",   # red   (swap both to #2C7FB8 / #D62728 for softer print tones)
}
CONDITION_MARKERS = {"AS-": "o", "AS+": "s"}

RIBBON_ALPHA = 0.4   # historical Sankey ribbon transparency, now baked in


def _blend(color, alpha, background="white"):
    """Flatten `color` at `alpha` over `background` into an opaque RGB tuple.

    Drawing the Sankey ribbons with this instead of alpha=0.4 keeps the fill
    identical to what the old code *looked* like, while making the value
    explicit and preventing overlapping ribbons from compounding.
    """
    fr, fg, fb = mcolors.to_rgb(color)
    br, bg, bb = mcolors.to_rgb(background)
    return (fr * alpha + br * (1 - alpha),
            fg * alpha + bg * (1 - alpha),
            fb * alpha + bb * (1 - alpha))


CONDITION_RIBBONS = {k: _blend(v, RIBBON_ALPHA) for k, v in CONDITION_COLORS.items()}


# ============================================================
# Figure 6  (attractor distribution heatmap + scatter)
# ============================================================

def _participant_attractor_distributions(cell_labels, trajectories,
                                          sorted_attractors, columns_dict,
                                          barcode_to_participant,
                                          barcode_to_label):
    """
    Returns
    -------
    participant_distributions : dict { pid -> np.array of len 5 percentages }
    pid_to_condition          : dict { pid -> 'AS-' | 'AS+' }
    pids_ordered              : list of pids in sorted order
    """
    top_attrs = sorted_attractors[:5]
    att_tuples = {}
    for att in top_attrs:
        key = f"State {att}"
        if key in columns_dict:
            att_tuples[att] = tuple(columns_dict[key])

    pid_cell_map = {}
    pid_condition= {}
    for cell in cell_labels:
        bc   = trajectory_key_to_barcode(cell)
        pid  = barcode_to_participant.get(bc, "NA")
        cond = barcode_to_label.get(bc, "NA")
        pid_cell_map.setdefault(pid, []).append(cell)
        pid_condition[pid] = cond

    participant_distributions = {}
    for pid, cells in pid_cell_map.items():
        counts = {att: 0 for att in top_attrs}
        for cell in cells:
            traj    = trajectories[cell]
            if not traj: continue
            final_s = traj[-1]
            key     = f"State {final_s}"
            if key in columns_dict:
                fs_tuple = tuple(columns_dict[key])
                for att, att_t in att_tuples.items():
                    if fs_tuple == att_t:
                        counts[att] += 1
                        break
        total = len(cells)
        participant_distributions[pid] = np.array(
            [counts[att] / total * 100 if total else 0 for att in top_attrs])

    pids_ordered = sorted(participant_distributions,
                           key=lambda x: int(x) if x.isdigit() else x)
    return participant_distributions, pid_condition, pids_ordered


def _short_pid(pid) -> str:
    """'Participant 12' / 'Participant12' / 'P12' -> 'P12'."""
    s = str(pid).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return f"P{m.group(1)}"
    return re.sub(r"(?i)^participants?[\s_\-]*", "P", s)


def _robust_limits(vals, lo_pct=5, hi_pct=95, pad_frac=0.15,
                   min_span=1.0, include_all=False):
    """Axis limits framed on where the bulk of the points sit."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi - lo < min_span:
        center = 0.5 * (lo + hi)
        lo, hi = center - min_span / 2, center + min_span / 2
    pad = pad_frac * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if include_all:
        lo, hi = min(lo, vals.min() - pad), max(hi, vals.max() + pad)
    return lo, hi


def generate_figure6(args, outdir: str):
    """
    Figure 6: participant-level attractor distribution.

    The two Mann-Whitney U tests still run on every invocation. Their
    p-values are no longer drawn on Figure6A/Figure6B — they go to
    statistics_summary.txt via record_stat().

    (The old `add_stats = getattr(args, "statistics", False)` gate has been
    removed. No `statistics` option was ever defined in _DEFAULT_PATHS or the
    argument parser, so that gate was permanently False and the annotations
    never actually rendered.)
    """
    print("\n" + "=" * 60)
    print("Generating Figure 6: Attractor Distribution")
    print("=" * 60)

    trajectories    = read_trajectories(args.trajectories)
    cleaned_traj    = {k: clean_trajectory(v) for k, v in trajectories.items()}

    with open(args.unique_states, "r") as fh:
        columns_dict = {
            key.strip(): list(map(int,
                value.strip().strip("[]").replace("(","").replace(")","").split(",")))
            for key, value in (line.strip().split(":") for line in fh)
        }

    cell_labels = list(trajectories.keys())
    num_to_label, barcode_to_label, barcode_to_participant = parse_metadata(args.metadata)

    attractors = {}
    for cell, traj in trajectories.items():
        cleaned = clean_trajectory(traj)
        if not cleaned: continue
        sa, ca, _, _ = find_attractors(cleaned)
        att = sa if sa is not None else (ca[0] if ca else None)
        if att is None: continue
        attractors.setdefault(att, []).append(cell)

    basins = {
        att: [c for c, t in trajectories.items() if att in t]
        for att in attractors
    }
    sorted_attractors = sorted(basins, key=lambda x: len(basins[x]), reverse=True)

    dist_dict, pid_cond, pids = _participant_attractor_distributions(
        cell_labels, trajectories, sorted_attractors, columns_dict,
        barcode_to_participant, barcode_to_label)

    distributions = np.array([dist_dict[p] for p in pids])

    # ---- Mann-Whitney U comparing AS- vs AS+ participants ----
    as_minus_mask = np.array([pid_cond[p] == "AS-" for p in pids])
    as_plus_mask  = np.array([pid_cond[p] == "AS+" for p in pids])

    # Figure6A (heatmap): pool all 5 attractor percentages per participant
    # and compare the AS- group's values against the AS+ group's values.
    if as_minus_mask.sum() > 0 and as_plus_mask.sum() > 0:
        _u_heat, heatmap_pvalue = mannwhitneyu(
            distributions[as_minus_mask].ravel(),
            distributions[as_plus_mask].ravel(),
            alternative="two-sided")
    else:
        _u_heat, heatmap_pvalue = np.nan, np.nan

    record_stat(
        figure="Figure 6",
        panels="Figure6A.png",
        test=("Mann-Whitney U, two-sided: top-5 attractor occupancy "
              "percentages pooled across attractors, AS- vs AS+ participants"),
        value=heatmap_pvalue,
        value_name="p-value",
        n_AS_minus_participants=int(as_minus_mask.sum()),
        n_AS_plus_participants=int(as_plus_mask.sum()),
        n_values_AS_minus=int(distributions[as_minus_mask].size),
        n_values_AS_plus=int(distributions[as_plus_mask].size),
        U_statistic=(f"{_u_heat:.6g}" if np.isfinite(_u_heat) else "n/a"),
    )

    # 6A heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(distributions, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(pids)))
    # Short participant labels: P1 (AS-), P2 (AS+), ...
    ax.set_yticklabels([f"{_short_pid(p)} ({pid_cond[p]})" for p in pids], fontsize=18)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Attr{i+1}" for i in range(5)], fontsize=18)
    ax.set_xlabel("Top 5 Attractors", fontsize=18, fontweight="bold")
    ax.set_ylabel("Participants",     fontsize=18, fontweight="bold")
    for i in range(len(pids)):
        for j in range(5):
            ax.text(j, i, f"{distributions[i,j]:.1f}",
                    ha="center", va="center", color="black", fontsize=18)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)", fontsize=18, weight="bold")
    cbar.ax.tick_params(labelsize=18); cbar.outline.set_linewidth(2)
    for sp in ax.spines.values(): sp.set_linewidth(2)
    # Centered above the heatmap. Drawn only when ANNOTATE_STATS_ON_FIGURES.
    _stat_box(ax, 0.5, 1.04, _fmt_p(heatmap_pvalue), ha="center", va="bottom")
    plt.savefig(_outpath(outdir, "Figure6A.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 6B scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    x_vals = distributions[:, 0] + distributions[:, 1]
    y_vals = distributions[:, 2] + distributions[:, 3]
    seen   = set()
    for i, pid in enumerate(pids):
        cond   = pid_cond[pid]
        # Same palette as the Figure 7 Sankey nodes — solid, no alpha, so the
        # marker fill is byte-identical to the 7A/7B node rectangles.
        color  = CONDITION_COLORS[cond]
        marker = CONDITION_MARKERS[cond]
        lbl    = cond if cond not in seen else None
        seen.add(cond)
        ax.scatter(x_vals[i], y_vals[i], c=color, marker=marker, s=800,
                   edgecolors="black", linewidths=2, label=lbl)
        # White label text: black-on-#FF0000 is unreadable once the panel is
        # downscaled into the 2x2 composite.
        ax.text(x_vals[i], y_vals[i], f"{_short_pid(pid)}", fontsize=18,
                weight="bold", ha="center", va="center", color="white")
    ax.set_xlabel("Combined % (Attr1 + Attr2)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Combined % (Attr3 + Attr4)", fontsize=18, fontweight="bold")

    # Axes auto-scaled to where the majority of the points fall
    ax.set_xlim(*_robust_limits(x_vals))
    ax.set_ylim(*_robust_limits(y_vals))

    ax.xaxis.set_tick_params(labelsize=18); ax.yaxis.set_tick_params(labelsize=18)
    # Legend parked outside the axes on the right so it can't sit on top of
    # the participant markers/labels wherever _robust_limits puts them.
    #ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    #          borderaxespad=0, frameon=True, edgecolor="black")
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values(): sp.set_linewidth(2)

    # Figure6B (scatter): compare the combined Attr1+2 % between AS- and AS+
    # participants — the primary axis being plotted.
    if as_minus_mask.sum() > 0 and as_plus_mask.sum() > 0:
        _u_scat, scatter_pvalue = mannwhitneyu(
            x_vals[as_minus_mask], x_vals[as_plus_mask], alternative="two-sided")
    else:
        _u_scat, scatter_pvalue = np.nan, np.nan

    record_stat(
        figure="Figure 6",
        panels="Figure6B.png",
        test=("Mann-Whitney U, two-sided: combined Attr1+Attr2 occupancy %, "
              "AS- vs AS+ participants"),
        value=scatter_pvalue,
        value_name="p-value",
        n_AS_minus_participants=int(as_minus_mask.sum()),
        n_AS_plus_participants=int(as_plus_mask.sum()),
        median_AS_minus=(f"{np.median(x_vals[as_minus_mask]):.4f}"
                         if as_minus_mask.sum() else "n/a"),
        median_AS_plus=(f"{np.median(x_vals[as_plus_mask]):.4f}"
                        if as_plus_mask.sum() else "n/a"),
        U_statistic=(f"{_u_scat:.6g}" if np.isfinite(_u_scat) else "n/a"),
    )
    # Centered above the scatter. Drawn only when ANNOTATE_STATS_ON_FIGURES.
    # NOTE: the pre-rewrite version passed facecolor="black" here, which made
    # the black annotation text invisible. Left as white to match Figure6A.
    _stat_box(ax, 0.5, 1.04, _fmt_p(scatter_pvalue), ha="center", va="bottom")
    plt.savefig(_outpath(outdir, "Figure6B.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Combined
    imgs = [mpimg.imread(_outpath(outdir, f)) for f in ("Figure6A.png","Figure6B.png")]
    fig  = plt.figure(figsize=(8, 6))
    for i, (label, img) in enumerate(zip(["a.","b."], imgs)):
        ax = plt.subplot(2, 1, i+1)
        ax.imshow(img); ax.axis("off")
        ax.text(0.02, 1.02, label, transform=ax.transAxes, fontsize=20,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    plt.savefig(_outpath(outdir, "Figure6_Combined.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    with open(_outpath(outdir, "Participant_Attractor_Distribution_Normal.txt"), "w") as fh:
        fh.write("Participant\tCondition\t"
                 + "\t".join(f"Attractor_{i+1}_%" for i in range(5)) + "\n")
        for pid in pids:
            cond = pid_cond[pid]
            vals = "\t".join(f"{v:.2f}" for v in dist_dict[pid])
            fh.write(f"{pid}\t{cond}\t{vals}\n")

    write_stats_report(outdir, getattr(args, "stats_report", None))

    print("Figure 6 complete.")


# ============================================================
# Figure 7  (Sankey)
# ============================================================

def generate_figure7(args, outdir: str):
    """
    Figure 7: Sankey plots of baseline -> perturbed condition-bias class.

    The chi-square test still runs for each Sankey. Its p-value is no longer
    drawn between the node columns — it goes to statistics_summary.txt via
    record_stat(). (The permanently-False `add_stats` gate has been removed;
    see generate_figure6's docstring.)
    """
    print("\n" + "=" * 60)
    print("Generating Figure 7: Sankey Plots")
    print("=" * 60)

    selected_gene = args.selected_gene
    required = [
        _outpath(outdir, "baseline_condition_bias.npy"),
        _outpath(outdir, f"{selected_gene}_ki_condition_bias.npy"),
        _outpath(outdir, f"{selected_gene}_ko_condition_bias.npy"),
    ]
    for f in required:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}"); return

    baseline_bias   = np.load(_outpath(outdir, "baseline_condition_bias.npy"))
    ki_bias         = np.load(_outpath(outdir, f"{selected_gene}_ki_condition_bias.npy"))
    ko_bias         = np.load(_outpath(outdir, f"{selected_gene}_ko_condition_bias.npy"))

    def classify(val):
        return "AS-" if val > 0 else "AS+"

    baseline_cls = [classify(v) for v in baseline_bias]
    ki_cls       = [classify(v) for v in ki_bias]
    ko_cls       = [classify(v) for v in ko_bias]

    print("Baseline:", Counter(baseline_cls))
    print("KI:      ", Counter(ki_cls))
    print("KO:      ", Counter(ko_cls))

    # Heads-up on a pre-existing pairing issue: baseline_bias has one entry
    # per STATE (pseudotime-sorted rank order) while ki_bias/ko_bias have one
    # entry per CELL (cell_sample_indices order). zip() therefore pairs state
    # i with cell i, and truncates to the shorter array. This affects the
    # ribbons AND the contingency table, not just the p-value. Flagged rather
    # than silently changed, since fixing it alters the published panels.
    if len(baseline_cls) != len(ki_cls) or len(baseline_cls) != len(ko_cls):
        print(f"  WARNING: length mismatch — baseline={len(baseline_cls)} (states), "
              f"KI={len(ki_cls)} (cells), KO={len(ko_cls)} (cells). "
              f"Sankey pairing and the chi-square table are built by zip(), "
              f"which pairs unrelated units and truncates to the shorter array.")

    # Shared with Figure 6B — do not redefine locally.
    color_map = CONDITION_COLORS

    def _sankey(baseline_classes, perturbed_classes, title, output_file,
                panel_label=""):
        transitions = Counter(zip(baseline_classes, perturbed_classes))
        total       = len(baseline_classes)

        before_counts = Counter(baseline_classes)
        after_counts  = Counter(perturbed_classes)
        scale = 0.7

        # ---- chi-square test of independence on the 2x2
        # (baseline class x perturbed class) contingency table ----
        contingency_table = np.array([
            [transitions.get(("AS-", "AS-"), 0), transitions.get(("AS-", "AS+"), 0)],
            [transitions.get(("AS+", "AS-"), 0), transitions.get(("AS+", "AS+"), 0)],
        ])
        try:
            _chi2, sankey_pvalue, _dof, _exp = chi2_contingency(contingency_table)
        except Exception:
            _chi2, sankey_pvalue, _dof = np.nan, np.nan, np.nan

        record_stat(
            figure="Figure 7",
            panels=os.path.basename(output_file),
            test=f"Chi-square test of independence, 2x2 baseline class x "
                 f"perturbed class ({title})",
            value=sankey_pvalue,
            value_name="p-value",
            chi2_statistic=(f"{_chi2:.6g}" if np.isfinite(_chi2) else "n/a"),
            dof=_dof,
            # ASCII "->" rather than a Unicode arrow: this string is written
            # into statistics_summary.txt/.csv, and keeping the report pure
            # ASCII means it survives being opened under any codec.
            table_AS_minus_row=f"[AS- -> AS-: {contingency_table[0,0]}, "
                               f"AS- -> AS+: {contingency_table[0,1]}]",
            table_AS_plus_row=f"[AS+ -> AS-: {contingency_table[1,0]}, "
                              f"AS+ -> AS+: {contingency_table[1,1]}]",
            n_pairs=total,
        )

        def _heights(counts):
            tot = sum(counts.values())
            return {k: (v/tot)*scale for k, v in counts.items()}

        b_h = _heights(before_counts)
        a_h = _heights(after_counts)
        gap = 0.03
        x_before, x_after = 0.15, 0.85
        node_w = 0.08

        def _y_positions(h):
            pos = {}; cur = 0.85
            for s in ["AS-","AS+"]:
                pos[s] = (cur, cur - h.get(s, 0))
                cur   -= h.get(s, 0) + gap
            return pos

        b_y = _y_positions(b_h)
        a_y = _y_positions(a_h)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")

        for state in ["AS-","AS+"]:
            for (x, y_dict, cnt_dict) in [
                    (x_before, b_y, before_counts),
                    (x_after,  a_y, after_counts)]:
                h = y_dict[state][0] - y_dict[state][1]
                if h > 0:
                    rect = plt.Rectangle((x - node_w/2, y_dict[state][1]),
                                          node_w, h,
                                          fc=color_map[state], ec="black", lw=2)
                    ax.add_patch(rect)
                    yc = (y_dict[state][0] + y_dict[state][1]) / 2
                    side = x - node_w/2 - 0.02 if x == x_before else x + node_w/2 + 0.02
                    ha   = "right" if x == x_before else "left"
                    ax.text(side, yc, f"{state}\n({cnt_dict[state]})",
                            ha=ha, va="center", fontsize=14, fontweight="bold")

        b_off = {s: b_y[s][0] for s in ["AS-","AS+"]}
        a_off = {s: a_y[s][0] for s in ["AS-","AS+"]}

        for (bf, af), cnt in sorted(transitions.items(), key=lambda x: -x[1]):
            if cnt == 0: continue
            fh_val = (cnt / total) * scale
            y1t = b_off[bf]; y1b = y1t - fh_val; b_off[bf] = y1b
            y2t = a_off[af]; y2b = y2t - fh_val; a_off[af] = y2b

            x1, x2 = x_before + node_w/2, x_after - node_w/2
            mx = (x1 + x2) / 2
            verts = [(x1,y1t),(mx,y1t),(x2,y2t),(x2,y2b),(mx,y1b),(x1,y1b),(x1,y1t)]
            codes = [Path.MOVETO,Path.CURVE3,Path.CURVE3,
                     Path.LINETO,Path.CURVE3,Path.CURVE3,Path.CLOSEPOLY]
            # Opaque pre-blended fill (was fc=color_map[bf], alpha=0.4). Same
            # apparent tint, but overlapping ribbons no longer compound, and
            # the edge stays the full-strength condition colour so it reads as
            # the same hue family as the nodes and the 6B markers.
            patch = PathPatch(Path(verts, codes),
                              fc=CONDITION_RIBBONS[bf], alpha=1.0,
                              ec=color_map[bf], linewidth=0.5)
            ax.add_patch(patch)

        ax.set_xlim(0,1); ax.set_ylim(-0.05,1.0)
        ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
        ax.text(x_before, 0.92, "Before\nPerturbation",
                ha="center", fontsize=16, fontweight="bold")
        ax.text(x_after,  0.92, "After\nPerturbation",
                ha="center", fontsize=16, fontweight="bold")
        # Centered between the two node columns, level with the Before/After
        # headers. Data coords, not axes coords. Drawn only when
        # ANNOTATE_STATS_ON_FIGURES is True.
        _stat_box(ax, (x_before + x_after) / 2, 0.92, _fmt_p(sankey_pvalue),
                  use_axes_coords=False, fontsize=16, ha="center", va="center")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_file}")
        return transitions

    _sankey(baseline_cls, ki_cls,
            f"{selected_gene} Knockin: State Transitions",
            _outpath(outdir, "Figure7A.png"))
    _sankey(baseline_cls, ko_cls,
            f"{selected_gene} Knockout: State Transitions",
            _outpath(outdir, "Figure7B.png"))

    det_file = _outpath(outdir, f"{selected_gene}_state_transitions_detailed.csv")
    n_detail = min(len(baseline_cls), len(ki_cls), len(ko_cls))
    if n_detail < len(baseline_cls):
        print(f"  NOTE: detailed CSV truncated to {n_detail} rows "
              f"(shortest of baseline/KI/KO arrays).")
    with open(det_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["State_Index","Baseline_Bias","Baseline_Class",
                    "KI_Bias","KI_Class","KO_Bias","KO_Class"])
        for i in range(n_detail):
            w.writerow([i, f"{baseline_bias[i]:.6f}", baseline_cls[i],
                        f"{ki_bias[i]:.6f}",  ki_cls[i],
                        f"{ko_bias[i]:.6f}",  ko_cls[i]])

    for suffix, classes, biases, label in [
            ("KI", ki_cls,  ki_bias,  "KI"),
            ("KO", ko_cls,  ko_bias, "KO")]:
        trans_file = _outpath(outdir, f"{selected_gene}_{suffix}_transitions_summary.csv")
        with open(trans_file, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Baseline_Class",f"{label}_Class","Count","Percentage"])
            cnts  = Counter(zip(baseline_cls, classes))
            total = len(baseline_cls)
            for (before_cls, after_cls), cnt in sorted(cnts.items()):
                w.writerow([before_cls, after_cls, cnt, f"{cnt/total*100:.2f}%"])

    imgs = [mpimg.imread(_outpath(outdir, f))
            for f in ("Figure7A.png","Figure7B.png")]
    fig  = plt.figure(figsize=(8, 6))
    gs   = GridSpec(1, 2, figure=fig, wspace=0.1)
    for i, (label, img) in enumerate(zip(["a.","b."], imgs)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img); ax.axis("off")
        ax.text(0.02, 1.02, label, transform=ax.transAxes,
                fontsize=24, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    plt.savefig(_outpath(outdir, "Figure7_Combined.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    write_stats_report(outdir, getattr(args, "stats_report", None))

    print("Figure 7 complete.")


def generate_noise_mixture_panels(args, outdir: str):
    """
    Generate tent.png and PPV.png, the two panels
    repatch_figure2_shane_version() composites as a. and b.

    Produced by noise_mixture_figures.py from the six ruleset CSVs in
    args.noise_input_dir (default: variable_noise_data/ beside this script).
    Imported lazily; missing inputs are non-fatal.
    """
    print("\n" + "=" * 60)
    print("Generating noise-mixture panels (tent.png / PPV.png)")
    print("=" * 60)

    if getattr(args, "skip_noise_panels", False):
        print("  skip_noise_panels set - reusing existing panels.")
        return {}

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from noise_mixture_figures_no_titles import generate_pipeline_panels
    except ImportError as exc:
        print(f"  Could not import noise_mixture_figures ({exc}).")
        print(f"  Place noise_mixture_figures.py in {SCRIPT_DIR}")
        return {}

    noise_input_dir = getattr(args, "noise_input_dir", None)
    palette = getattr(args, "noise_palette", "sequential")

    if not noise_input_dir or not os.path.isdir(noise_input_dir):
        print(f"  Ruleset folder not found: {noise_input_dir}")
        return {}

    print(f"  Ruleset folder : {noise_input_dir}")
    print(f"  Palette        : {palette}")

    written = generate_pipeline_panels(
        outdir, input_dir=noise_input_dir, palette=palette,
        tent_name="tent.png", ppv_name="PPV.png",
        strict=False, verbose=True,
    )

    if written:
        print("  Noise-mixture panels complete.")
    else:
        print("  Skipped - Figure2new composite will skip too unless "
              "tent.png / PPV.png already exist in outdir.")
    return written


# ============================================================
# Combine Figure 3 (intermediate 6-panel combine, unchanged)
# ============================================================

def combine_figures(outdir: str):
    fnames = ["Figure3A.png","Figure3B.png","SimmilarityClustering.png",
              "Cell_Transition_Matrix_Region.png",
              "Figure3EOrg.png","Figure3FOrg.png"]
    paths  = [_outpath(outdir, f) for f in fnames]
    if not all(os.path.exists(p) for p in paths):
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"Skipping Figure 3 combine – missing: {missing}"); return

    imgs = [mpimg.imread(p) for p in paths]
    fig  = plt.figure(figsize=(8, 6))
    gs   = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)
    for i in range(6):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(imgs[i]); ax.axis("off")
        ax.text(0.03, 1.05, chr(ord("a") + i) + ".", transform=ax.transAxes,
                fontsize=20, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.savefig(_outpath(outdir, "Figure3_Combined.png"),
                dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Figure 3 combined panel saved.")


# ============================================================
# RepatchJuly – final paper-figure repatching
# ============================================================

def _repatch_panel_grid(outdir: str, image_names, out_name: str,
                         nrows: int, ncols: int, figsize=(12, 12),
                         label_x: float = 0.03, label_y: float = 1.05,
                         label_start: str = "a"):
    """
    Combine `image_names` into an nrows x ncols grid saved as `out_name`,
    with panels lettered starting at `label_start` (default "a") instead
    of always restarting at "a" — lets a later composite continue the
    lettering sequence of an earlier one (e.g. "e." following a prior
    "a.-d." composite).
    """
    paths   = [_outpath(outdir, f) for f in image_names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[RepatchJuly] Skipping {out_name} – missing source panel(s): {missing}")
        return None

    imgs = [mpimg.imread(p) for p in paths]
    fig  = plt.figure(figsize=figsize)
    gs   = GridSpec(nrows, ncols, figure=fig, hspace=0, wspace=0)

    start_ord = ord(label_start)
    for i, img in enumerate(imgs):
        row, col = i // ncols, i % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, aspect='auto')
        ax.axis('off')

        label = chr(start_ord + i) + "."
        ax.text(
            label_x, label_y, label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight='bold',
            color='black',
            ha='left',
            va='top',
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='none',
                pad=2
            )
        )
    out_path = _outpath(outdir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[RepatchJuly] Saved {out_path}")
    return out_path


def repatch_figure2_shane_version(outdir: str):
    """
    Shane's version of the "new" Figure 2 composite
    """
    return _repatch_panel_grid(
            outdir,
            ["PPV.png", "Figure2C.png", "Figure2D.png"],
            "RepatchJuly_Figure2ShaneVersion_Combined.png",
            nrows=1, ncols=3, figsize=(30, 8),
        )


def repatch_figure3_shane_version(outdir: str):
    """
    Figure3A/Figure3B/Figure3EOrg/Figure3FOrg
    """
    return _repatch_panel_grid(
        outdir,
        ["Figure3A.png", "Figure3B.png",
         "Figure3EOrg.png", "Figure3FOrg.png"],
        "RepatchJuly_Figure3ShaneVersion_Combined.png",
        nrows=2, ncols=2, figsize=(20, 16),
    )


# def repatch_paper_figure3(outdir: str):
#     """
#     Figure3A/Figure3B/Figure3EOrg/Figure3FOrg
#     """
#     return _repatch_panel_grid(
#         outdir,
#         ["Figure3A.png", "Figure3B.png",
#          "Figure3EOrg.png", "Figure3FOrg.png",
#         "Figure4A.png", "Figure4B.png"],
#         "RepatchJuly_Figure3_Combined.png",
#         nrows=3, ncols=2, figsize=(20, 10),
#     )

def repatch_paper_figure3(outdir: str):
    return _repatch_panel_grid(
        outdir,
        ["Figure5B.png", "Figure5C.png"],
        "RepatchJuly_Figure3_Combined.png",
        nrows=1, ncols=2, figsize=(20, 8),
    )


def repatch_paper_figure4(outdir: str):
    return _repatch_panel_grid(
        outdir,
        ["Figure5A.png", "Figure5B.png", "Figure5C.png"],
        "RepatchJuly_Figure4_Combined.png",
        nrows=1, ncols=3, figsize=(30, 8),
    )


def repatch_paper_figure5(outdir: str):
    return _repatch_panel_grid(
        outdir,
        ["Figure7A.png", "Figure7B.png",
         "Figure6A.png", "Figure6B.png"],
        "RepatchJuly_Figure5_Combined.png",
        nrows=2, ncols=2, figsize=(20,16),
    )


def generate_paper_figures(outdir: str):
    print("\n" + "=" * 60)
    print("RepatchJuly: repatching final paper figures")
    print("=" * 60)
    repatch_figure2_shane_version(outdir)
    repatch_figure3_shane_version(outdir)
    repatch_paper_figure3(outdir)
    repatch_paper_figure4(outdir)
    repatch_paper_figure5(outdir)
    print("RepatchJuly: paper figure repatching complete.")


# ============================================================
# Main
# ============================================================

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    reset_stats()
    set_stats_annotation(getattr(args, "annotate_stats", False))
    set_figure5a_annotation(getattr(args, "figure5a_stats", True))
    set_perturbation_stat_boxes(getattr(args, "show_stat_boxes", False))

    stats_basename = getattr(args, "stats_report", STATS_REPORT_BASENAME)

    print("=" * 60)
    print("Trajectory Analysis Pipeline")
    print(f"  trajectories      : {args.trajectories}")
    print(f"  unique_states     : {args.unique_states}")
    print(f"  metadata          : {args.metadata}")
    print(f"  ko_pattern        : {args.ko_pattern}")
    print(f"  ki_pattern        : {args.ki_pattern}")
    print(f"  outdir            : {args.outdir}")
    print(f"  selected_gene     : {args.selected_gene}")
    print(f"  network           : {args.network}")
    print(f"  importance_scores : {args.importance_scores}")
    print(f"  foldchange        : {args.foldchange}")
    print(f"  cell_trajectory   : {args.cell_trajectory}")
    print(f"  stats report      : {_outpath(args.outdir, stats_basename)}.txt / .csv")
    print(f"  p-value table     : {_outpath(args.outdir, stats_basename)}{PVALUE_TABLE_SUFFIX}.csv")
    print(f"  n_clusters        : {getattr(args, 'n_clusters', '2')} "
          f"(sweep range {getattr(args, 'cluster_k_min', 2)}"
          f"-{getattr(args, 'cluster_k_max', 10)} when 'auto')")
    print(f"  boxes on figures  : {ANNOTATE_STATS_ON_FIGURES}")
    print(f"  Fig 5A Wilcoxon   : {ANNOTATE_FIGURE5A_WILCOXON} "
          f"({'stars + boxes' if SHOW_PERTURBATION_STAT_BOXES else 'stars only'})")
    print("=" * 60)

    generate_figure2(args, args.outdir)

    generate_all_figures3(args, args.outdir)
    combine_figures(args.outdir)
    generate_figure4(args.outdir)
    generate_figure5(args, args.outdir)
    generate_figure6(args, args.outdir)
    generate_figure7(args, args.outdir)

    generate_noise_mixture_panels(args, args.outdir)

    generate_paper_figures(args.outdir)

    print("\n" + "=" * 60)
    print("Statistics report")
    print("=" * 60)
    write_stats_report(args.outdir, stats_basename)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()