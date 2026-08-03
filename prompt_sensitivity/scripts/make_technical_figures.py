"""Two explanatory figures for the KIT deck: metric definitions + probe design.

Answers the two questions a supervisor will ask out loud:
  "what exactly is the phrasing-luck cost, and how do you quantify it?"  -> definitions
  "how does the probe work?"                                             -> probe

    uv run python -m prompt_sensitivity.scripts.make_technical_figures
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging

C_ABIL, C_SENS, C_DISP, C_DIAL = "#B2182B", "#2166AC", "#01665E", "#4D4D4D"
KIT_GREEN = "#009682"
OUTDIR = "figures/supervisor_2026-08-03"

plt.rcParams.update({
    "font.size": 13, "figure.dpi": 170, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
})


def fig_definitions(out):
    """Four rows: axis · formula · what it means in words."""
    rows = [
        (C_ABIL, "ABILITY",
         r"$FI_{in}(q,k)\;=\;-\log_2\!\left(\dfrac{N_k(q)}{|U_q|}\right)$",
         "Of all rephrasings $U_q$ of question $q$, what fraction still reaches quality $k$?\n"
         "0 bits = every phrasing works · 3.3 bits = only 1 in 10 works.\n"
         r"Area under the curve over $k$ = $AUFI$ = the “phrasing-luck cost”."),
        (C_SENS, "SENSITIVITY",
         r"$\rho_F\;=\;\dfrac{MS_B-MS_W}{MS_B+(k-1)\,MS_W}$",
         "One-way ICC over the rephrasings: the share of success variance caused by\n"
         "WHICH phrasing was used, rather than by random sampling noise.\n"
         r"$\rho_F=0$: phrasing irrelevant · $\rho_F=1$: phrasing decides everything."),
        (C_DISP, "DISPERSION",
         r"$H_{sem}=-\sum_c p(c)\log_2 p(c)$" "\n" r"$FI_{out}=\log_2|\mathcal{A}_q|-H_{sem}$",
         "Sample the model $k$ times, cluster answers by meaning (NLI), take the entropy.\n"
         "High = the answers scatter over many meanings; 0 = it commits to one.\n"
         "(Farquhar et al., Nature 2024 — semantic entropy.)"),
        (C_DIAL, "THE DIAL",
         r"$FI_{spec}=\log_2\!\left(\dfrac{m_0}{m_{valid}}\right)$",
         "How much ambiguity the QUESTION TEXT itself removes. $m_0$ = number of valid\n"
         "readings a human annotator found; $m_{valid}$ = readings the wording still allows.\n"
         "Model-free — this is the variable we manipulate, not an outcome."),
    ]
    # Two columns: [name + formula] | [plain-words reading]. Keeping the formula
    # under its own heading (not beside it) avoids any horizontal collision.
    fig, ax = plt.subplots(figsize=(13.0, 5.9))
    ax.axis("off")
    tops = [0.985, 0.735, 0.475, 0.185]        # dispersion row carries 2 formulas
    for (col, name, formula, words), y in zip(rows, tops):
        ax.add_patch(plt.Rectangle((0.0, y - 0.20), 0.0055, 0.20,
                                   color=col, transform=ax.transAxes, clip_on=False))
        ax.text(0.018, y, name, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color=col, va="top")
        ax.text(0.028, y - 0.062, formula, transform=ax.transAxes, fontsize=15.5,
                va="top", color="#111111", linespacing=1.7)
        ax.text(0.44, y - 0.004, words, transform=ax.transAxes, fontsize=12,
                va="top", color="#333333", linespacing=1.6)
    fig.text(0.012, -0.005, "every quantity is in BITS:   "
             r"$\mathrm{bits}=-\log_2(\mathrm{fraction\ that\ still\ works})$"
             "   — Szostak 2003 / Hazen et al. 2007",
             fontsize=12.5, color=KIT_GREEN, fontweight="bold")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_probe(out):
    """The prompt-checker: architecture on top, training protocol below."""
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, txt, fc, ec, fs=11.5, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=2.0, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                zorder=3, fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(x0, x1, y=0.70):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#777777"))

    box(0.005, 0.58, 0.155, 0.25, "your prompt\n" r"$x$", "#F7F7F7", "#999999")
    arrow(0.168, 0.203)
    box(0.21, 0.55, 0.20, 0.31,
        "FROZEN LLM\none forward pass\n(no answer generated)", "#FDEBD9", "#D95F02", bold=True)
    arrow(0.418, 0.453)
    box(0.46, 0.55, 0.215, 0.31,
        "hidden state of the\nLAST prompt token\n"
        r"$h^{(l)}\in\mathbb{R}^{4096}$" "\nat 50/75/100 % depth", "#EAF1F8", C_SENS)
    arrow(0.683, 0.718)
    box(0.725, 0.55, 0.18, 0.31,
        "LINEAR head\nlogistic regression\n+ isotonic calibration\n(~12k features)",
        "#E3F1E6", "#1B7837", bold=True)

    outs = [("“too vague?”", "AUROC .85–.87"), ("“answer correct?”", "ρ .41–.52"),
            ("“answers scatter?”", "AUROC .69–.79"), ("“phrasing-fragile?”", "≈ chance")]
    x = 0.005
    for label, score in outs:
        box(x, 0.225, 0.213, 0.155, f"{label}\n{score}", "white", "#BBBBBB", fs=11)
        x += 0.259
    ax.annotate("", xy=(0.815, 0.395), xytext=(0.815, 0.545),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#777777"))
    ax.text(0.84, 0.47, "4 calibrated gauges", fontsize=11.5, color="#555555", va="center")

    ax.text(0.005, 0.135,
            "Trained per model on 2,980 prompts from 149 questions  ·  "
            "labels come from the measured metrics (no new annotation)",
            fontsize=12, color="#333333")
    ax.text(0.005, 0.055,
            "Split: GroupKFold BY QUESTION — all rephrasings of a question stay on one side, "
            "so every number is on unseen questions.",
            fontsize=12, color="#333333")
    ax.text(0.005, -0.025,
            "Controls: label permutation + a length-only baseline  ·  "
            "calibration checked out-of-fold (ECE .04–.08)",
            fontsize=12, color="#333333")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> int:
    configure_logging("make_technical_figures")
    root = load_config().repo_root()
    outdir = root / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn in [("definitions", fig_definitions), ("probe", fig_probe)]:
        fn(outdir / f"{name}.png")
        logger.info("wrote {}.png", name)
    print(f"DONE -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
