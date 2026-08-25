"""Generate the paper's figures as vector PDFs.

Every number is READ from a committed artifact at run time — there are no
hand-copied literals, so a regenerated artifact regenerates the figures:

  data/union_gold_*.parquet + data/specificity_v3_*.parquet
    + data/rho_f_hier_union_*.parquet -- per-model level means (Fig. 1a-c)
  data/width_dial_cells.parquet  -- the generator-width arms (Fig. 1d)
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


def _level_mean_ci(
    df: pd.DataFrame, col: str, *, n_boot: int = 2000, seed: int = 0
) -> dict[int, tuple[float, float, float]]:
    """Per-level mean of `col` with a question-clustered bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[float, float, float]] = {}
    for lvl, g in df.groupby("spec_level"):
        v = g[col].to_numpy(dtype=float)
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        means = v[idx].mean(axis=1)
        out[int(lvl)] = (
            float(v.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        )
    return out


def load_levels() -> dict[str, dict[str, dict[int, tuple[float, float, float]]]]:
    """Fig. 1a-c inputs: per-model level means from the committed parquets.

    accuracy: union_gold_*.parquet (f_graded_union_mean); dispersion:
    specificity_v3_*.parquet (h_sem_mean); formulation sensitivity:
    rho_f_hier_union_*.parquet (rho_f_hier). Values match the level means
    behind data/stats_hygiene.md; the paired significance tests live there.
    """
    out: dict[str, dict[str, dict[int, tuple[float, float, float]]]] = {
        "accuracy": {},
        "hsem": {},
        "rhof": {},
    }
    for m in MODELS:
        ug = pd.read_parquet(DATA / f"union_gold_{m}.parquet")
        v3 = pd.read_parquet(DATA / f"specificity_v3_{m}.parquet")
        hi = pd.read_parquet(DATA / f"rho_f_hier_union_{m}.parquet")
        out["accuracy"][m] = _level_mean_ci(ug, "f_graded_union_mean")
        out["hsem"][m] = _level_mean_ci(v3, "h_sem_mean")
        out["rhof"][m] = _level_mean_ci(hi, "rho_f_hier")
    return out


def load_width(*, n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Fig. 1d inputs: paired-covered MoM rho_F per arm, bootstrap CI, and n.

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
    """Four panels, one visual grammar: each axis's level under its intervention.

    (a)-(c) plot the ambiguous and disambiguated LEVEL of each axis per model,
    so the reader sees what moved and what stayed flat; (d) extends the same
    grammar to the three generator-width arms. Whiskers are question-clustered
    bootstrap 95% CIs of the level means; the paired significance tests live
    in the endpoint table.
    """
    levels = load_levels()
    width = load_width()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))

    panels = [
        ("accuracy", "(a) Competence", "accuracy (union gold)"),
        ("hsem", "(b) Output dispersion", "$H_{sem}$ (bits)"),
        ("rhof", "(c) Formulation sens.", "$\\rho_F$ (hierarchical)"),
    ]
    for ax, (key, title, ylab) in zip(axes[:3], panels, strict=False):
        for m in MODELS:
            pts = levels[key][m]
            ys = [pts[0][0], pts[1][0]]
            lo = [pts[0][0] - pts[0][1], pts[1][0] - pts[1][1]]
            hi = [pts[0][2] - pts[0][0], pts[1][2] - pts[1][0]]
            ax.errorbar(
                [0, 1],
                ys,
                yerr=[lo, hi],
                fmt="o-",
                ms=3.2,
                lw=1.4,
                capsize=2,
                elinewidth=0.9,
                color=CMAP[m],
                label=NICE[m] if key == "accuracy" else None,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ambig.", "disamb."])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
    # one shared model legend above the row, clear of all data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        handletextpad=0.4,
        columnspacing=1.4,
        handlelength=1.4,
    )

    ax = axes[3]
    for m in MODELS:
        w = width[m]
        ax.errorbar(
            range(3),
            w["means"],
            yerr=[w["means"] - w["lo"], w["hi"] - w["means"]],
            fmt="o-",
            ms=3.2,
            lw=1.4,
            capsize=2,
            elinewidth=0.9,
            color=CMAP[m],
            label=f"n={w['n']}",
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["narrow", "prod.", "wide"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_title("(d) Formulation sens.,\nwidth intervention", loc="left", fontweight="bold", fontsize=7.6)
    ax.set_ylabel("$\\rho_F$ (MoM, paired cells)")
    ax.legend(
        frameon=False,
        loc="upper left",
        handletextpad=0.3,
        borderpad=0.1,
        labelspacing=0.2,
        handlelength=1.2,
        fontsize=6.5,
    )

    fig.tight_layout(w_pad=0.8)
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
    # display names: match the paper's notation instead of code identifiers
    pretty = {
        "H_sem": "$H_{sem}$",
        "S_tau (Errica)": "$S_\\tau$ (Errica)",
        "TVD-sens  [M4]": "TVD consistency",
        "|A_q| observed": "$|\\mathcal{A}_q|$ observed",
        "variation ratio": "variation ratio",
        "Var[FI_out]  [M4]": "Var[$FI_{out}$]",
        "FI_out_fixed": "$FI_{out}^{fixed}$",
        "accuracy": "accuracy",
        "AUFI (graded)": "AUFI (graded)",
        "FI premium  [M2]": "$\\Delta$FI premium",
        "rho_F  [M1]": "$\\rho_F$",
        "rho_u (Cox)": "$\\rho_u$ (Cox)",
        "spread (Cao)": "spread (Cao)",
        "ESS_in": "$ESS_{in}$",
    }
    disp = [pretty[labels[i]] for i in order]

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
    """The underspecification probe, in-distribution and zero-shot.

    Bars are anchored at chance (0.5) so bar length encodes the margin over
    chance; the two text baselines are model-independent, so they are drawn
    as reference lines rather than repeated bars.
    """
    in_head, in_base, ood = load_probe()
    base_in = in_base[MODELS[0]]  # identical across models (same texts, same folds)
    base_ood = max(
        ood[MODELS[0]]["auroc_length"],
        ood[MODELS[0]]["auroc_tfidf_word_frozen"],
        ood[MODELS[0]]["auroc_tfidf_char_frozen"],
        ood[MODELS[0]]["auroc_first_word_frozen"],
    )
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    x = np.arange(len(MODELS))
    w = 0.3
    ax.bar(
        x - 0.5 * w,
        [in_head[m] - 0.5 for m in MODELS],
        w,
        bottom=0.5,
        label="probe, in-distribution",
        color="#1b6ca8",
    )
    heads = [ood[m]["auroc_head"] for m in MODELS]
    los = [ood[m]["auroc_head"] - ood[m]["auroc_head_ci_lo"] for m in MODELS]
    his = [ood[m]["auroc_head_ci_hi"] - ood[m]["auroc_head"] for m in MODELS]
    ax.bar(
        x + 0.5 * w,
        [h - 0.5 for h in heads],
        w,
        bottom=0.5,
        yerr=[los, his],
        capsize=2,
        error_kw={"elinewidth": 0.9},
        label="probe, holdout (zero-shot)",
        color="#c1121f",
    )
    ax.axhline(base_in, color="#1b6ca8", lw=1.0, ls=(0, (4, 2)), zorder=0, alpha=0.7)
    ax.axhline(base_ood, color="#c1121f", lw=1.0, ls=(0, (4, 2)), zorder=0, alpha=0.7)
    ax.text(
        2.62,
        base_in,
        "best text baseline,\nin-distribution",
        fontsize=6.3,
        color="#1b6ca8",
        va="center",
        ha="left",
    )
    ax.text(
        2.62,
        base_ood,
        "best frozen text\nbaseline, holdout",
        fontsize=6.3,
        color="#c1121f",
        va="center",
        ha="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in MODELS], fontsize=7.2)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 0.92)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.5\n(chance)", "0.6", "0.7", "0.8", "0.9"])
    ax.set_title("Underspecification from one forward pass", loc="left", fontweight="bold", pad=22)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.25,
        columnspacing=1.0,
        fontsize=6.8,
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
