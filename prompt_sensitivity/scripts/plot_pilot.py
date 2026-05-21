"""Plot the pilot MetricTuples for a supervisor presentation.

Reads a parquet of MetricTuples (default: data/e2e_metrics.parquet, the
output of `e2e-smoke`) and produces four PNGs + a markdown summary under
`data/plots/`:

  - 01_f_mean_by_level.png       headline: F-accuracy vs context level
  - 02_aufi_in_by_level.png       FI_in area-under-curve vs context level
  - 03_h_sem_by_level.png         Farquhar 2024 semantic entropy baseline
  - 04_three_ladder_envelope.png  gold_first / random / distractor_first bars
                                  (Research_Design_v3 V3 bound consistency)
  - 05_metric_correlations.png    Spearman correlation heatmap across the
                                  11 metric scalars (V1 validation lite)
  - REPORT.md                     summary text + embedded plot refs

Each plot uses one subplot per question + an aggregated panel; three lines
per ladder type (random / gold_first / distractor_first), color-coded
consistently. Small-N caveat is printed in REPORT.md.

The script gracefully handles MetricTuples without `f_mean` (older runs):
falls back to estimating F-mean from AUFI_in for binary F and flags it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no GUI required.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


# Consistent colors across all plots.
_LADDER_COLOURS: dict[str, str] = {
    "gold_first": "#2ca02c",         # green (best-case)
    "random": "#1f77b4",             # blue (expected)
    "distractor_first": "#d62728",   # red (worst-case)
}

_LADDER_ORDER = ["gold_first", "random", "distractor_first"]

_METRIC_COLS_FOR_HEATMAP = [
    "f_mean",
    "aufi_in",
    "fi_out_mean",
    "s_tau_mean",
    "consistency_mean",
    "spread",
    "variation_ratio",
    "ess_in",
    "rho_u",
    "h_sem_mean",
    "h_sem_var",
]


def _ensure_f_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Older runs lacked an `f_mean` column. Derive it from AUFI_in when missing.

    For binary F and uniform-fail (AUFI_in ~= log2(N+1)), F_mean is 0.
    For all-pass (AUFI_in = 0), F_mean is 1. Otherwise, infer from log2.
    This is approximate — fine for plotting an existing parquet, but the
    correct path is a re-run of e2e_smoke which now records f_mean directly.
    """
    if "f_mean" in df.columns and df["f_mean"].notna().any():
        return df
    logger.warning(
        "parquet has no f_mean; deriving an approximate value from aufi_in. "
        "Re-run e2e-smoke for exact values."
    )
    df = df.copy()
    n = df["n_paraphrases"].clip(lower=1)
    # Special cases: AUFI_in ≈ 0 -> all-pass -> F=1. AUFI_in near log2(N+1) -> all-fail -> F=0.
    # Otherwise, FI_in(k=1) ~= -log2(F) so F ~= 2^(-AUFI_in/integrand_width).
    # Keep it simple: snap to 0 or 1 when at extremes, else 0.5.
    aufi = df["aufi_in"].fillna(0.0)
    cap = np.log2(n + 1)
    derived = np.where(aufi <= 0.05, 1.0, np.where(aufi >= 0.9 * cap, 0.0, 0.5))
    df["f_mean"] = derived
    return df


# --------------------------------------------------------------------------- #
# Plot 1 — F-mean (accuracy) per level                                        #
# --------------------------------------------------------------------------- #


def _plot_metric_per_question(
    df: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Lineplot: one panel per question + one aggregated panel; lines per ladder.

    Works for any single-scalar MetricTuple column. Used for f_mean, aufi_in,
    h_sem_mean — each tells the same story shape but on a different y-axis.
    """
    questions = sorted(df["question_id"].unique())
    n_q = len(questions)
    if n_q == 0:
        logger.warning("no questions to plot for {}", metric)
        return

    n_cols = min(3, n_q + 1)
    n_rows = math.ceil((n_q + 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    # One panel per question.
    for ax, qid in zip(axes_flat, questions, strict=False):
        sub = df[df["question_id"] == qid]
        for ladder in _LADDER_ORDER:
            row = sub[sub["ladder_type"] == ladder].sort_values("level")
            if row.empty:
                continue
            ax.plot(
                row["level"],
                row[metric],
                marker="o",
                color=_LADDER_COLOURS[ladder],
                label=ladder if ax is axes_flat[0] else None,
                linewidth=2,
            )
        ax.set_title(f"qid={qid[:18]}...", fontsize=10)
        ax.set_xlabel("context level (#paragraphs)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Aggregated panel — mean across questions.
    agg_ax = axes_flat[n_q]
    grouped = (
        df.groupby(["ladder_type", "level"])[metric].mean().reset_index()
    )
    for ladder in _LADDER_ORDER:
        row = grouped[grouped["ladder_type"] == ladder].sort_values("level")
        if row.empty:
            continue
        agg_ax.plot(
            row["level"],
            row[metric],
            marker="s",
            color=_LADDER_COLOURS[ladder],
            label=ladder,
            linewidth=2.5,
        )
    agg_ax.set_title(f"AGGREGATE (mean across {n_q} questions)", fontsize=10, fontweight="bold")
    agg_ax.set_xlabel("context level (#paragraphs)")
    agg_ax.set_ylabel(ylabel)
    agg_ax.grid(True, alpha=0.3)
    if ylim is not None:
        agg_ax.set_ylim(*ylim)
    agg_ax.legend(loc="best", fontsize=9)

    # Hide any unused subplot slots.
    for ax in axes_flat[n_q + 1 :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("wrote {}", out_path)


def _plot_three_ladder_envelope(df: pd.DataFrame, out_path: Path) -> None:
    """At each level, bar chart per question with three bars (the ladders).

    Visualises V3 bound consistency: gold_first ≥ random ≥ distractor_first.
    """
    questions = sorted(df["question_id"].unique())
    levels = sorted(df["level"].unique())
    n_q = len(questions)
    n_l = len(levels)
    if n_q == 0 or n_l == 0:
        logger.warning("no data for three-ladder plot")
        return

    fig, axes = plt.subplots(n_q, 1, figsize=(max(8, 1.5 * n_l), 3.5 * n_q), squeeze=False)
    axes_flat = axes.flatten()

    bar_width = 0.25
    for ax, qid in zip(axes_flat, questions, strict=False):
        sub = df[df["question_id"] == qid]
        x = np.arange(n_l)
        for i, ladder in enumerate(_LADDER_ORDER):
            vals: list[float] = []
            for lvl in levels:
                row = sub[(sub["ladder_type"] == ladder) & (sub["level"] == lvl)]
                vals.append(float(row["f_mean"].iloc[0]) if not row.empty else float("nan"))
            ax.bar(
                x + (i - 1) * bar_width,
                vals,
                bar_width,
                label=ladder,
                color=_LADDER_COLOURS[ladder],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"L={lvl}" for lvl in levels])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F-mean (accuracy)")
        ax.set_title(f"qid={qid[:18]}... — three-ladder F-mean per level")
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes_flat[0]:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "V3 bound consistency: gold_first ≥ random ≥ distractor_first expected",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("wrote {}", out_path)


def _plot_metric_correlations(df: pd.DataFrame, out_path: Path) -> None:
    """Spearman heatmap across the metric scalars. V1 validation (small-N caveat)."""
    cols = [c for c in _METRIC_COLS_FOR_HEATMAP if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(sub) < 3:
        logger.warning("only {} usable rows; skipping correlation plot", len(sub))
        return
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(
        f"V1 validation (N={len(sub)} cells; tiny sample, treat as illustrative)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("wrote {}", out_path)


def _v3_check(df: pd.DataFrame) -> dict[str, float]:
    """Per-level V3 bound consistency stats: fraction of (q, level) cells
    where gold_first >= random >= distractor_first on F-mean.

    Returns {"checked": int, "consistent": int, "rate": float}.
    """
    pivot = df.pivot_table(
        index=["question_id", "level"], columns="ladder_type", values="f_mean"
    )
    if not {"gold_first", "random", "distractor_first"}.issubset(pivot.columns):
        return {"checked": 0, "consistent": 0, "rate": float("nan")}
    pivot = pivot.dropna(subset=list(_LADDER_ORDER))
    checked = len(pivot)
    consistent = int(
        (
            (pivot["gold_first"] >= pivot["random"])
            & (pivot["random"] >= pivot["distractor_first"])
        ).sum()
    )
    return {
        "checked": checked,
        "consistent": consistent,
        "rate": consistent / checked if checked > 0 else float("nan"),
    }


def _write_report(df: pd.DataFrame, plots_dir: Path) -> None:
    """REPORT.md ties the plots together with a short narrative."""
    questions = sorted(df["question_id"].unique())
    ladders = sorted(df["ladder_type"].unique())
    # Cast levels through int() so the report renders "[0, 4, 10]" rather
    # than "[np.int64(0), np.int64(4), np.int64(10)]".
    levels = sorted(int(x) for x in df["level"].unique())
    models = sorted(df["model_key"].unique())
    v3 = _v3_check(df)
    n_cells = len(df)

    # Per-question F-mean trajectory (random ladder) — the "deployer's view".
    rand = df[df["ladder_type"] == "random"]
    f_means_at_top = (
        rand[rand["level"] == levels[-1]].set_index("question_id")["f_mean"].to_dict()
    )

    lines: list[str] = []
    lines.append("# Pilot results — preliminary")
    lines.append("")
    lines.append(f"**Cells run:** {n_cells}  ")
    lines.append(f"**Questions:** {len(questions)} ({', '.join(qid[:18] + '...' for qid in questions)})  ")
    lines.append(f"**Ladders:** {', '.join(ladders)}  ")
    lines.append(f"**Levels:** {levels}  ")
    lines.append(f"**Models:** {', '.join(models)}  ")
    lines.append("")
    lines.append("> Sample size is intentionally small (this is a smoke-test pilot). ")
    lines.append("> All numbers should be treated as illustrative; the Sprint-5 full ")
    lines.append("> pilot (50 questions × 4 models) will produce statistically ")
    lines.append("> reportable values.")
    lines.append("")
    lines.append("## 1. Accuracy responds to context (the headline)")
    lines.append("")
    lines.append("![F-mean by level](01_f_mean_by_level.png)")
    lines.append("")
    lines.append("Each panel is one question; lines are the three ladders. Aggregated ")
    lines.append("panel shows the mean across questions. The expected pattern: ")
    lines.append("F-mean climbs with context level, with gold_first as upper envelope ")
    lines.append("and distractor_first as lower envelope. Per-question trajectories at ")
    lines.append(f"the top level (L={levels[-1]}, random ladder):")
    lines.append("")
    lines.append("| qid | F-mean at L=top |")
    lines.append("|---|---|")
    for qid, fm in f_means_at_top.items():
        lines.append(f"| {qid} | {fm:.2f} |")
    lines.append("")
    lines.append("## 2. The novel FI_in metric")
    lines.append("")
    lines.append("![AUFI_in by level](02_aufi_in_by_level.png)")
    lines.append("")
    lines.append("AUFI_in (Area under FI_in(k) curve) is the design doc's primary ")
    lines.append("scalar (Section_7 §7.3). Lower = more paraphrases pass the threshold ")
    lines.append("(less prompt-sensitivity). Should decrease as context grows for ")
    lines.append("questions the model can answer.")
    lines.append("")
    lines.append("## 3. Farquhar 2024 semantic entropy (baseline)")
    lines.append("")
    lines.append("![H_sem by level](03_h_sem_by_level.png)")
    lines.append("")
    lines.append("Farquhar's H_sem on cluster proportions. Expected to drop with ")
    lines.append("context (model converges to one answer when context is informative).")
    lines.append("")
    lines.append("## 4. V3 — three-ladder bound consistency")
    lines.append("")
    lines.append("![Three-ladder F-mean per level](04_three_ladder_envelope.png)")
    lines.append("")
    lines.append("**Bound check (gold_first ≥ random ≥ distractor_first on F-mean):**")
    lines.append("")
    if v3["checked"] > 0:
        lines.append(f"- Cells checked: {v3['checked']}")
        lines.append(f"- Cells consistent: {v3['consistent']} ({100 * v3['rate']:.0f}%)")
    else:
        lines.append("- Insufficient data (need all 3 ladders × overlapping levels).")
    lines.append("")
    lines.append("Failures (random > gold_first) would be interesting — they would ")
    lines.append("indicate that distractor paragraphs prime parametric retrieval more ")
    lines.append("effectively than direct gold facts for that question.")
    lines.append("")
    lines.append("## 5. V1 — metric inter-correlation (small-N illustrative)")
    lines.append("")
    lines.append("![Metric correlations](05_metric_correlations.png)")
    lines.append("")
    lines.append("Spearman ρ between the metric scalars. Design doc target: ρ ∈ ")
    lines.append("[0.4, 0.8] between AUFI_in and existing metrics (FI_in is a new ")
    lines.append("axis, not a re-derivation). Sample size here is too small to ")
    lines.append("conclude; the full Sprint-5 pilot is needed for the real V1 check.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Only one model in this pilot (`kit.gpt-4.1` via gateway). Adding ")
    lines.append("  the three open-weight models is straightforward once budget allows.")
    lines.append("- POSIX is `None` for kit.gpt-4.1 (no echo path on OpenAI chat models). ")
    lines.append("  Will populate when Llama/Teuken/Qwen are added.")
    lines.append("- ESS_in is small for context-heavy cells — this is a known mpnet ")
    lines.append("  encoder limitation (paraphrases project to nearby points). The ")
    lines.append("  own-encoder variant in Sprint 6 will not have this property.")
    lines.append("")
    out_path = plots_dir / "REPORT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote {}", out_path)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=str, default="data/e2e_metrics.parquet")
    parser.add_argument("--out", type=str, default="data/plots")
    args = parser.parse_args()

    configure_logging("plot_pilot")
    config = load_config()
    repo_root = config.repo_root()

    in_path = repo_root / args.inp
    if not in_path.exists():
        logger.error(
            "missing {} — run `make e2e-smoke` first (or pass --in PATH)", in_path
        )
        return 1
    df = pd.read_parquet(in_path)
    logger.info("loaded {} cells from {}", len(df), in_path)

    df = _ensure_f_mean(df)

    plots_dir = repo_root / args.out
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_per_question(
        df,
        metric="f_mean",
        title="F-mean (raw accuracy) vs context level — by question + aggregate",
        ylabel="F-mean (fraction of paraphrases correct)",
        out_path=plots_dir / "01_f_mean_by_level.png",
        ylim=(-0.05, 1.05),
    )
    _plot_metric_per_question(
        df,
        metric="aufi_in",
        title="AUFI_in (novel metric) vs context level — by question + aggregate",
        ylabel="AUFI_in (bits; lower = more uniformly pass)",
        out_path=plots_dir / "02_aufi_in_by_level.png",
    )
    _plot_metric_per_question(
        df,
        metric="h_sem_mean",
        title="H_sem (Farquhar 2024) vs context level — by question + aggregate",
        ylabel="H_sem (bits; lower = fewer semantic clusters)",
        out_path=plots_dir / "03_h_sem_by_level.png",
    )
    _plot_three_ladder_envelope(df, plots_dir / "04_three_ladder_envelope.png")
    _plot_metric_correlations(df, plots_dir / "05_metric_correlations.png")
    _write_report(df, plots_dir)

    # Console summary
    v3 = _v3_check(df)
    summary = {
        "cells": int(len(df)),
        "questions": sorted(df["question_id"].unique().tolist()),
        "ladders": sorted(df["ladder_type"].unique().tolist()),
        "levels": sorted(int(x) for x in df["level"].unique()),
        "models": sorted(df["model_key"].unique().tolist()),
        "v3_consistency_rate": (
            float(v3["rate"]) if not math.isnan(v3["rate"]) else None
        ),
        "v3_consistent_cells": int(v3["consistent"]),
        "v3_total_checked": int(v3["checked"]),
        "out_dir": str(plots_dir),
    }
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
