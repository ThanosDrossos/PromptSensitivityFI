"""Generate the paper's figures as vector PDFs.

Every number is either read from a committed artifact or taken from the declared
source-of-truth files, cited inline:

  data/stats_hygiene.md        -- the declared primary family (Fig. 1a)
  data/width_dial_cells.parquet-- the generator-width arms (Fig. 1b)
  figures/v3_metric_corr.npy   -- 14-metric within-stratum Spearman matrix (Fig. 2)
  data/probe_eval_hardened.md  -- hardened probe evaluation (Fig. 3)

Usage:  uv run python -m prompt_sensitivity.scripts.make_paper_figures [--out DIR]

Writes fig1_dissociation.pdf, fig2_metric_structure.pdf, fig3_probe.pdf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FIGS = REPO / "figures"

MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
NICE = {"qwen_2_5_7b": "Qwen2.5-7B", "llama_3_1_8b": "Llama-3.1-8B", "mistral_7b_v03": "Mistral-7B"}
CMAP = {"qwen_2_5_7b": "#1b6ca8", "llama_3_1_8b": "#c1121f", "mistral_7b_v03": "#2a9d8f"}

# --- data/stats_hygiene.md, "Declared primary family" table -------------------
# effect (L1-L0) and question-clustered 95% CI, n = 150 per model.
SPEC = {
    "accuracy": {
        "qwen_2_5_7b": (+0.0639, -0.0002, +0.1279),
        "llama_3_1_8b": (+0.1251, +0.0667, +0.1859),
        "mistral_7b_v03": (+0.1189, +0.0559, +0.1818),
    },
    "hsem": {
        "qwen_2_5_7b": (-0.1239, -0.2049, -0.0454),
        "llama_3_1_8b": (-0.4333, -0.5831, -0.2842),
        "mistral_7b_v03": (-0.1393, -0.2774, +0.0011),
    },
    "rhof": {
        "qwen_2_5_7b": (-0.0019, -0.0172, +0.0137),
        "llama_3_1_8b": (+0.0127, -0.0031, +0.0286),
        "mistral_7b_v03": (+0.0134, -0.0111, +0.0382),
    },
}

# --- data/width_dial_analysis.md, "P2b rho_F (MoM, covered cells)" ------------
# These are the numbers in the paper's width table. They are computed on the
# cells covered in ALL three arms (paired), which is NOT the same as averaging
# each arm over its own covered set -- the unpaired version is non-monotone for
# Llama. Read them from the source of truth so figure and table cannot drift.
WIDTH_RHOF = {
    "qwen_2_5_7b": [0.3563, 0.4631, 0.5193],
    "llama_3_1_8b": [0.1130, 0.1348, 0.1382],
    "mistral_7b_v03": [0.2110, 0.2303, 0.2761],
}

# --- data/probe_eval_hardened.md ----------------------------------------------
PROBE_IN = {"qwen_2_5_7b": 0.874, "llama_3_1_8b": 0.873, "mistral_7b_v03": 0.873}
PROBE_IN_BASE = 0.756                      # baseline_length, identical across models
PROBE_OOD = {"qwen_2_5_7b": 0.678, "llama_3_1_8b": 0.670, "mistral_7b_v03": 0.678}
PROBE_OOD_BASE = 0.587                     # best frozen text baseline (TF-IDF char)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def fig1(out: Path) -> None:
    """The double dissociation: each dial moves its own axis and only its own."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.4, 2.45))

    # (a) specificity dial -----------------------------------------------------
    axes_order = [("accuracy", "Competence\n(accuracy)"), ("hsem", "Dispersion\n($H_{sem}$)"),
                  ("rhof", "Formulation\nsens. ($\\rho_F$)")]
    ypos, ylab = [], []
    for gi, (key, lab) in enumerate(axes_order):
        for mi, m in enumerate(MODELS):
            eff, lo, hi = SPEC[key][m]
            y = gi * 4 + (2 - mi)
            ypos.append(y)
            ax0.errorbar(eff, y, xerr=[[eff - lo], [hi - eff]], fmt="o", ms=3.4,
                         color=CMAP[m], ecolor=CMAP[m], elinewidth=1.1, capsize=2,
                         label=NICE[m] if gi == 0 else None)
        ylab.append((gi * 4 + 1, lab))
    ax0.axvline(0, color="0.35", lw=0.8, ls="--", zorder=0)
    ax0.set_yticks([y for y, _ in ylab])
    ax0.set_yticklabels([l for _, l in ylab])
    ax0.set_xlabel("change from ambiguous to disambiguated (L1 $-$ L0)")
    ax0.set_title("(a) Specificity dial", loc="left", fontweight="bold")
    ax0.set_ylim(-1.2, 11.2)
    ax0.legend(frameon=False, loc="lower left", handletextpad=0.4, borderpad=0.1)

    # (b) width dial -----------------------------------------------------------
    xlab = ["narrow", "production", "wide"]
    for m in MODELS:
        ax1.plot(range(3), WIDTH_RHOF[m], "o-", ms=3.4, lw=1.4,
                 color=CMAP[m], label=NICE[m])
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(xlab)
    ax1.set_xlabel("paraphrase-generator width")
    ax1.set_ylabel("$\\rho_F$ (method of moments)")
    ax1.set_title("(b) Generator-width dial", loc="left", fontweight="bold")
    ax1.set_xlim(-0.25, 2.25)
    ax1.legend(frameon=False, loc="upper left", handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out / "fig1_dissociation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_dissociation.pdf")


def fig2(out: Path) -> None:
    """The 14 candidate metrics, ordered by axis: three blocks, not fourteen."""
    corr = np.load(FIGS / "v3_metric_corr.npy")
    meta = json.loads((FIGS / "v3_metric_corr_labels.json").read_text())
    labels = meta["labels"]

    # order: dispersion family, competence family, sensitivity family
    groups = [
        ("Dispersion", ["H_sem", "S_tau (Errica)", "TVD-sens  [M4]", "|A_q| observed",
                        "variation ratio", "Var[FI_out]  [M4]", "FI_out_fixed"]),
        ("Competence", ["accuracy", "AUFI (graded)", "FI premium  [M2]"]),
        ("Formulation\nsensitivity", ["rho_F  [M1]", "rho_u (Cox)", "spread (Cao)", "ESS_in"]),
    ]
    order, bounds, gnames = [], [], []
    for gname, members in groups:
        gnames.append(gname)
        for mlab in members:
            order.append(labels.index(mlab))
        bounds.append(len(order))
    C = corr[np.ix_(order, order)]
    disp = [labels[i].replace("  [M1]", "").replace("  [M2]", "").replace("  [M4]", "")
            for i in order]

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    im = ax.imshow(np.abs(C), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(disp, rotation=55, ha="right")
    ax.set_yticklabels(disp)
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="k", lw=1.3)
        ax.axvline(b - 0.5, color="k", lw=1.3)
    start = 0
    for gname, b in zip(gnames, bounds):
        ax.text(len(order) - 0.25, (start + b - 1) / 2, gname, va="center", ha="left",
                fontsize=7.5, fontweight="bold", linespacing=0.95)
        start = b
    cb = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.30)
    cb.set_label("|Spearman| (mean within stratum)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Candidate metrics group into three blocks", loc="left",
                 fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out / "fig2_metric_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_metric_structure.pdf")


def fig3(out: Path) -> None:
    """The underspecification head: in-distribution and zero-shot, against matched baselines."""
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    x = np.arange(len(MODELS))
    w = 0.2
    ax.bar(x - 1.5 * w, [PROBE_IN[m] for m in MODELS], w, label="head, in-distribution",
           color="#1b6ca8")
    ax.bar(x - 0.5 * w, [PROBE_IN_BASE] * 3, w, label="best text baseline, in-dist.",
           color="#a8c8e0")
    ax.bar(x + 0.5 * w, [PROBE_OOD[m] for m in MODELS], w, label="head, zero-shot holdout",
           color="#c1121f")
    ax.bar(x + 1.5 * w, [PROBE_OOD_BASE] * 3, w, label="frozen text baseline, holdout",
           color="#e8a0a6")
    ax.axhline(0.5, color="0.35", lw=0.9, ls="--", zorder=0)
    ax.text(2.52, 0.505, "chance", fontsize=6.8, color="0.35", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in MODELS])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Underspecification from one forward pass", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.2,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig3_probe.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_probe.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output directory for the PDFs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    fig1(args.out)
    fig2(args.out)
    fig3(args.out)


if __name__ == "__main__":
    main()
