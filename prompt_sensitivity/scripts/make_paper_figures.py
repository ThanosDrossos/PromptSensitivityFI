"""Generate the paper's figures as vector PDFs.

Every number is READ from a committed artifact at run time — there are no
hand-copied literals, so a regenerated artifact regenerates the figures:

  data/stats_hygiene.json        -- the declared primary family (Fig. 1a)
  data/width_dial_cells.parquet  -- the generator-width arms (Fig. 1b)
  figures/v3_metric_corr.npy     -- 14-metric within-stratum Spearman (Fig. 2)
  data/probe_eval_hardened_*.parquet + data/probe_eval_ood.json (Fig. 3)

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

_ARMS = ["narrow", "medium", "wide"]
_SPEC_KEYS = {
    "accuracy (union gold)  [PRIMARY]": "accuracy",
    "H_sem": "hsem",
    "rho_F (hierarchical, union gold)": "rhof",
}


def load_spec() -> dict[str, dict[str, tuple[float, float, float]]]:
    """Fig. 1a inputs from data/stats_hygiene.json (effect, ci_lo, ci_hi)."""
    payload = json.loads((DATA / "stats_hygiene.json").read_text(encoding="utf-8"))
    spec: dict[str, dict[str, tuple[float, float, float]]] = {v: {} for v in _SPEC_KEYS.values()}
    for r in payload["endpoints"]:
        key = _SPEC_KEYS.get(r["endpoint"])
        if key:
            spec[key][r["model"]] = (r["effect"], r["ci_lo"], r["ci_hi"])
    for key, per_model in spec.items():
        missing = [m for m in MODELS if m not in per_model]
        if missing:
            raise ValueError(f"stats_hygiene.json lacks {key} for {missing}")
    return spec


def load_width(*, n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Fig. 1b inputs: paired-covered MoM rho_F per arm, bootstrap CI, and n.

    The paired-covered set (MoM defined in ALL three arms) is outcome-selected
    and small — the n goes into the panel so the figure cannot imply n = 100.
    """
    cells = pd.read_parquet(DATA / "width_dial_cells.parquet")
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for m in MODELS:
        d = cells[cells.model == m]
        piv = d.pivot_table(index=["question_id", "spec_level"], columns="arm", values="rho_f_mom")[
            _ARMS
        ].dropna()
        boots = np.array(
            [piv.iloc[rng.integers(0, len(piv), len(piv))].mean().to_numpy() for _ in range(n_boot)]
        )
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out[m] = {
            "means": piv.mean().to_numpy(),
            "lo": lo,
            "hi": hi,
            "n": len(piv),
        }
    return out


def load_probe() -> tuple[dict, dict, dict]:
    """Fig. 3 inputs: nested-CV head + best text baseline (in-dist), OOD json."""
    in_head, in_base = {}, {}
    for m in MODELS:
        df = pd.read_parquet(DATA / f"probe_eval_hardened_{m}.parquet")
        v = df[df["target"] == "vagueness"]
        in_head[m] = float(v.loc[v["head"] == "nested_cv", "auroc"].iloc[0])
        in_base[m] = float(v.loc[v["head"].str.startswith("baseline"), "auroc"].max())
    ood = {
        r["model"]: r
        for r in json.loads((DATA / "probe_eval_ood.json").read_text(encoding="utf-8"))
    }
    return in_head, in_base, ood


def _style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def fig1(out: Path) -> None:
    """The two dials: each moves its own axis; ns and CIs shown, not implied."""
    spec = load_spec()
    width = load_width()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.4, 2.45))

    # (a) specificity dial — units differ per row and are named on the labels
    axes_order = [
        ("accuracy", "Competence\n($\\Delta$ accuracy)"),
        ("hsem", "Dispersion\n($\\Delta H_{sem}$, bits)"),
        ("rhof", "Formulation sens.\n($\\Delta\\rho_F$, share)"),
    ]
    ylab = []
    for gi, (key, lab) in enumerate(axes_order):
        for mi, m in enumerate(MODELS):
            eff, lo, hi = spec[key][m]
            y = gi * 4 + (2 - mi)
            ax0.errorbar(
                eff,
                y,
                xerr=[[eff - lo], [hi - eff]],
                fmt="o",
                ms=3.4,
                color=CMAP[m],
                ecolor=CMAP[m],
                elinewidth=1.1,
                capsize=2,
                label=NICE[m] if gi == 0 else None,
            )
        ylab.append((gi * 4 + 1, lab))
    ax0.axvline(0, color="0.35", lw=0.8, ls="--", zorder=0)
    ax0.set_yticks([y for y, _ in ylab])
    ax0.set_yticklabels([label for _, label in ylab])
    ax0.set_xlabel("change from ambiguous to disambiguated (L1 $-$ L0)")
    ax0.set_title("(a) Specificity dial", loc="left", fontweight="bold")
    ax0.set_ylim(-1.2, 11.2)
    ax0.legend(frameon=False, loc="lower left", handletextpad=0.4, borderpad=0.1)

    # (b) width dial — paired-covered MoM with bootstrap CIs and the n
    for m in MODELS:
        w = width[m]
        ax1.errorbar(
            range(3),
            w["means"],
            yerr=[w["means"] - w["lo"], w["hi"] - w["means"]],
            fmt="o-",
            ms=3.4,
            lw=1.4,
            capsize=2,
            elinewidth=0.9,
            color=CMAP[m],
            label=f"{NICE[m]} (n={w['n']})",
        )
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["narrow", "production", "wide"])
    ax1.set_xlabel("paraphrase-generator width")
    ax1.set_ylabel("$\\rho_F$ (MoM, paired covered cells)")
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
        (
            "Dispersion",
            [
                "H_sem",
                "S_tau (Errica)",
                "TVD-sens  [M4]",
                "|A_q| observed",
                "variation ratio",
                "Var[FI_out]  [M4]",
                "FI_out_fixed",
            ],
        ),
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
    disp = [
        labels[i].replace("  [M1]", "").replace("  [M2]", "").replace("  [M4]", "") for i in order
    ]

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
    for gname, b in zip(gnames, bounds, strict=True):
        ax.text(
            len(order) - 0.25,
            (start + b - 1) / 2,
            gname,
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            linespacing=0.95,
        )
        start = b
    cb = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.30)
    cb.set_label("|Spearman| (mean within stratum)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Candidate metrics group into three blocks", loc="left", fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out / "fig2_metric_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_metric_structure.pdf")


def fig3(out: Path) -> None:
    """The underspecification head, in-dist and zero-shot, with holdout CIs."""
    in_head, in_base, ood = load_probe()
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    x = np.arange(len(MODELS))
    w = 0.2
    ax.bar(
        x - 1.5 * w,
        [in_head[m] for m in MODELS],
        w,
        label="head, in-distribution",
        color="#1b6ca8",
    )
    ax.bar(
        x - 0.5 * w,
        [in_base[m] for m in MODELS],
        w,
        label="best text baseline, in-dist.",
        color="#a8c8e0",
    )
    heads = [ood[m]["auroc_head"] for m in MODELS]
    los = [ood[m]["auroc_head"] - ood[m]["auroc_head_ci_lo"] for m in MODELS]
    his = [ood[m]["auroc_head_ci_hi"] - ood[m]["auroc_head"] for m in MODELS]
    ax.bar(
        x + 0.5 * w,
        heads,
        w,
        yerr=[los, his],
        capsize=2,
        error_kw={"elinewidth": 0.9},
        label="head, zero-shot holdout",
        color="#c1121f",
    )
    frozen_best = [
        max(
            ood[m]["auroc_length"],
            ood[m]["auroc_tfidf_word_frozen"],
            ood[m]["auroc_tfidf_char_frozen"],
            ood[m]["auroc_first_word_frozen"],
        )
        for m in MODELS
    ]
    ax.bar(
        x + 1.5 * w,
        frozen_best,
        w,
        label="best frozen text baseline, holdout",
        color="#e8a0a6",
    )
    ax.axhline(0.5, color="0.35", lw=0.9, ls="--", zorder=0)
    ax.text(-0.62, 0.505, "chance", fontsize=6.8, color="0.35", va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in MODELS])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Underspecification from one forward pass", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.25,
    )
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
