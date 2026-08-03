"""Presentation figures for the 2026-08-03 supervisor deck.

Graph-heavy, low-text: every panel is built from the committed final-run data
(v3 three-model grid, evidence dial, POSIX arms, holdout, Part-B analyses).

    uv run python -m prompt_sensitivity.scripts.make_supervisor_figures
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from ..config import load_config
from ..logging_setup import configure_logging

MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
NICE = {"qwen_2_5_7b": "Qwen-2.5-7B", "llama_3_1_8b": "Llama-3.1-8B",
        "mistral_7b_v03": "Mistral-7B"}
SHORT = ["Qwen", "Llama", "Mistral"]
# Colour-blind-safe, consistent across the deck.
C_L0, C_L1 = "#D95F02", "#1B7837"        # vague (orange) vs specific (green)
C_ABIL, C_SENS, C_DISP, C_DIAL = "#B2182B", "#2166AC", "#01665E", "#555555"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "legend.frameon": False,
})


def _v3(root):
    return {m: pd.read_parquet(root / f"data/specificity_v3_{m}.parquet") for m in MODELS}


# --- 1. headline: what one turn of the dial buys ---------------------------
def fig_headline(root, out):
    d = _v3(root)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        ("f_graded_mean", r"$\bar{F}$  (graded accuracy)",
         r"mean over rephrasings of $P(\mathrm{correct}\,|\,x)$ · higher = better", False),
        ("aufi_in_graded", r"$AUFI_{in}$  (bits)",
         r"$\int_0^1 FI_{in}(q,k)\,dk$ · lower = better", True),
        ("h_sem_mean", r"$H_{sem}$  (bits)",
         "semantic entropy of the answers · lower = better", True),
    ]
    x = np.arange(3)
    for ax, (col, title, sub, lower_better) in zip(axes, panels):
        l0 = [d[m][d[m].spec_level == 0][col].mean() for m in MODELS]
        l1 = [d[m][d[m].spec_level == 1][col].mean() for m in MODELS]
        ax.bar(x - 0.2, l0, 0.38, label="vague question", color=C_L0)
        ax.bar(x + 0.2, l1, 0.38, label="specific question", color=C_L1)
        for i, (a, b) in enumerate(zip(l0, l1)):
            ax.annotate("", xy=(i + 0.2, b), xytext=(i - 0.2, a),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))
            ax.text(i, max(a, b) * 1.14, f"{'−' if b < a else '+'}{abs(b-a):.2f}",
                    ha="center", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, fontsize=13)
        ax.set_title(title, pad=16)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center",
                fontsize=11, color="#666666")
        ax.set_ylim(0, max(max(l0), max(l1)) * 1.30)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               ncol=2, fontsize=14)
    fig.suptitle(r"$FI_{spec}$: 0 $\rightarrow$ 1.58 bits   ·   149 questions, 3 models,"
                 " N=10 rephrasings, k=10 samples", fontsize=16, y=1.03)
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 2. the evidence interaction ------------------------------------------
def fig_dial(root, out):
    f00 = pd.read_parquet(root / "data/evidence_dial_f00_qwen_2_5_7b.parquet")
    f05 = pd.read_parquet(root / "data/evidence_dial_f05_qwen_2_5_7b.parquet")
    v3 = pd.read_parquet(root / "data/specificity_v3_qwen_2_5_7b.parquet")
    qids = set(f00.question_id.astype(str))
    f10 = v3[v3.question_id.astype(str).isin(qids)]
    fr = [0.0, 0.5, 1.0]
    acc = {lvl: [df[df.spec_level == lvl].f_graded_mean.mean()
                 for df in (f00, f05, f10)] for lvl in (0, 1)}
    fi = {lvl: [df[df.spec_level == lvl].aufi_in_graded.mean()
                for df in (f00, f05, f10)] for lvl in (0, 1)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, data, ttl, ylab in [
        (axes[0], acc, r"$\bar{F}$  (graded accuracy)", r"$\bar{F}$"),
        (axes[1], fi, r"$AUFI_{in}$  (bits)", "bits (lower = better)"),
    ]:
        ax.plot(fr, data[0], "o-", lw=3, ms=11, color=C_L0, label="vague question")
        ax.plot(fr, data[1], "o-", lw=3, ms=11, color=C_L1, label="specific question")
        ax.fill_between(fr, data[0], data[1], alpha=0.13,
                        color=C_L1 if "bar{F}" in ttl else C_L0)
        ax.set_xticks(fr)
        ax.set_xticklabels(["none", "half", "full"])
        ax.set_xlabel("evidence given to the model")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, pad=10)
    gap = acc[1][2] - acc[0][2]
    axes[0].annotate(f"gap = {gap:+.2f}", xy=(1.0, (acc[0][2] + acc[1][2]) / 2),
                     xytext=(0.66, (acc[0][2] + acc[1][2]) / 2), fontsize=13,
                     fontweight="bold", va="center")
    axes[0].legend(loc="upper left", fontsize=12)
    fig.suptitle("Specificity x evidence interaction  (Qwen-2.5-7B, 50 questions, k=10, N=10)",
                 fontsize=15, y=1.04)
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 3. the three axes are independent ------------------------------------
def fig_independence(root, out):
    C = np.load(root / "figures/v3_metric_corr.npy")
    meta = json.loads((root / "figures/v3_metric_corr_labels.json").read_text("utf-8"))
    names = meta["labels"]
    order = ["accuracy", "AUFI (graded)", "FI premium  [M2]",
             "rho_F  [M1]", "rho_u (Cox)",
             "H_sem", "S_tau (Errica)", "TVD-sens  [M4]", "|A_q| observed",
             "variation ratio", "Var[FI_out]  [M4]", "FI_out_fixed"]
    idx = [names.index(n) for n in order]
    M = np.nan_to_num(C[np.ix_(idx, idx)], nan=0.0)
    short = {"accuracy": "accuracy", "AUFI (graded)": "AUFI", "FI premium  [M2]": "ΔFI premium",
             "rho_F  [M1]": "ρ_F", "rho_u (Cox)": "ρ_u (Cox)", "H_sem": "H_sem",
             "S_tau (Errica)": "S_τ (Errica)", "TVD-sens  [M4]": "TVD", "|A_q| observed": "|A_q|",
             "variation ratio": "variation ratio", "Var[FI_out]  [M4]": "Var[FI_out]",
             "FI_out_fixed": "FI_out"}
    lab = [short[n] for n in order]

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(np.abs(M), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(lab, fontsize=11)
    for (a, b) in [(0, 3), (3, 5), (5, 12)]:
        ax.add_patch(plt.Rectangle((a - .5, a - .5), b - a, b - a, fill=False,
                                   ec=[C_ABIL, C_SENS, C_DISP][[0, 3, 5].index(a)], lw=3.5))
    for pos, txt, col in [(1.0, "ABILITY", C_ABIL), (3.6, "SENSITIVITY", C_SENS),
                          (8.0, "DISPERSION", C_DISP)]:
        ax.text(pos, -0.85, txt, ha="center", fontsize=12.5, fontweight="bold", color=col)
    fig.colorbar(im, ax=ax, shrink=0.78, label="|correlation|")
    ax.set_title("Three blocks, almost nothing between them", pad=52, fontsize=16)
    fig.text(0.5, -0.03,
             "ρ_F vs accuracy = .08     ρ_F vs H_sem = .03     →  the axes cannot substitute for each other",
             ha="center", fontsize=12.5, color="#333333")
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 4. POSIX: the literature measures dispersion --------------------------
def fig_posix(root, out):
    rows = []
    for m in MODELS:
        d = pd.read_parquet(root / f"data/posix_arm_{m}.parquet")
        d = d.assign(tvd_sens=1 - d.consistency_mean)
        def r(col):
            s = d[["posix_psi", col]].dropna()
            return spearmanr(s.posix_psi, s[col])[0]
        rows.append([r("tvd_sens"), r("h_sem_mean"), r("s_tau_mean"), r("rho_f")])
    A = np.array(rows)
    labs = ["TVD", "H_sem", "S_τ", "ρ_F"]
    cols = [C_DISP, C_DISP, C_DISP, C_SENS]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(labs))
    w = 0.26
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * w, A[i], w, label=NICE[m],
               color=cols, alpha=[0.55, 0.78, 1.0][i], edgecolor="white")
    ax.axhspan(0, 0.35, color="#EEEEEE", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["TVD\n(dispersion)", "H_sem\n(dispersion)", "S_τ\n(dispersion)",
                        "ρ_F\n(our sensitivity axis)"], fontsize=12.5)
    ax.set_ylabel("correlation with POSIX")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=12, ncol=3, loc="upper right")
    ax.set_title("POSIX — a published 'prompt sensitivity index' — measures answer scatter,\n"
                 "not phrasing sensitivity   (3 models, 100 cells each)", fontsize=15, pad=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 5. the deliverable: a prompt checker ---------------------------------
def fig_feedback(root, out):
    ver = pd.read_parquet(root / "data/feedback_verification.parquet")
    hold = pd.read_parquet(root / "data/vagueness_holdout_results.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    ax = axes[0]
    heads = ["vagueness", "dispersion"]
    x = np.arange(len(heads))
    # NB: ver["head"] — `.head` is a DataFrame METHOD; attribute access would
    # silently return the function instead of the column.
    per_model = {h: [ver[(ver.model_key == m) & (ver["head"] == h)].auroc.iloc[0]
                     for m in MODELS] for h in heads}
    means = [np.mean(per_model[h]) for h in heads]
    ax.bar(x, means, 0.5, color=[C_SENS, C_DISP], zorder=2)
    for i, h in enumerate(heads):                       # each model as a dot
        ax.scatter([i] * 3, per_model[h], s=55, color="white", edgecolor="#222222",
                   zorder=4, linewidth=1.4)
    ax.axhline(0.5, ls=":", color="#999999", zorder=1)
    ax.text(1.34, 0.515, "chance", fontsize=11, color="#777777")
    ax.axhline(ver[ver["head"] == "vagueness"].auroc_length_baseline.iloc[0],
               ls="--", color="#B22222", lw=2, zorder=3)
    ax.text(-0.45, 0.772, "length-only baseline", fontsize=11.5, color="#B22222")
    ax.set_xticks(x)
    ax.set_xticklabels(['"is it too vague?"', '"will answers scatter?"'], fontsize=13)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("One forward pass predicts prompt quality", fontsize=15, pad=10)
    ax.text(0.5, 0.955, "bar = 3-model mean,  dots = individual models",
            transform=ax.transAxes, ha="center", fontsize=10.5, color="#666666")

    ax = axes[1]
    ind = [ver[(ver.model_key == m) & (ver["head"] == "vagueness")].auroc.iloc[0] for m in MODELS]
    ood = [hold[hold.model_key == m].auroc.iloc[0] for m in MODELS]
    x2 = np.arange(3)
    ax.bar(x2 - 0.19, ind, 0.36, color=C_SENS, label="same setting")
    ax.bar(x2 + 0.19, ood, 0.36, color="#9ECAE1", label="unseen questions,\nhuman labels")
    ax.axhline(hold.auroc_length_baseline.iloc[0], ls="--", color="#B22222", lw=2)
    ax.text(-0.45, 0.40, "length-only baseline", fontsize=11, color="#B22222")
    ax.axhline(0.5, ls=":", color="#999999")
    ax.set_xticks(x2)
    ax.set_xticklabels(SHORT, fontsize=13)
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title('"Is this prompt too vague?" — held-out test', fontsize=15, pad=10)
    ax.legend(fontsize=11, loc="upper right")
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 6. the design, as a picture ------------------------------------------
def fig_design(root, out):
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=12.5, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=2.2,
                                   zorder=2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                zorder=3, fontweight="bold" if bold else "normal", wrap=True)

    box(0.01, 0.63, 0.28, 0.25, "VAGUE question\n“Who won the mayor race\nin St. Petersburg?”",
        "#FDEBD9", C_L0, fs=11.5)
    box(0.01, 0.09, 0.28, 0.25, "SPECIFIC question\n“Who won the 2017 mayor race\nin St. Petersburg?”",
        "#E3F1E6", C_L1, fs=11.5)
    # the dial, in the gap between the two question boxes
    ax.annotate("", xy=(0.10, 0.36), xytext=(0.10, 0.61),
                arrowprops=dict(arrowstyle="-|>", lw=3.2, color=C_DIAL))
    ax.text(0.135, 0.485, "+1.6 bits\nof specificity\n(human annotation)",
            fontsize=11, va="center", color="#333333")

    box(0.42, 0.33, 0.17, 0.34, "10 rephrasings\n×\n10 samples\n\n3 open models",
        "#F2F2F2", "#888888", fs=11.5)
    for y_from in (0.74, 0.20):
        ax.annotate("", xy=(0.415, 0.50), xytext=(0.295, y_from),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#777777"))

    labels = [("ABILITY\ncan it answer?", C_ABIL, 0.70),
              ("SENSITIVITY\ndoes phrasing decide?", C_SENS, 0.41),
              ("DISPERSION\nhow scattered?", C_DISP, 0.12)]
    for txt, col, y in labels:
        box(0.69, y, 0.30, 0.18, txt, "white", col, fs=12, bold=True)
        ax.annotate("", xy=(0.685, y + 0.09), xytext=(0.595, 0.50),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.04, 0.94)
    ax.set_title("The experiment: turn one dial, watch three independent measures",
                 fontsize=16, pad=6)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    configure_logging("make_supervisor_figures")
    root = load_config().repo_root()
    outdir = root / "figures" / "supervisor_2026-08-03"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("design", fig_design), ("headline", fig_headline), ("dial", fig_dial),
        ("independence", fig_independence), ("posix", fig_posix),
        ("feedback", fig_feedback),
    ]
    for name, fn in jobs:
        p = outdir / f"{name}.png"
        fn(root, p)
        logger.info("wrote {}", p.name)
    print(f"DONE -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
