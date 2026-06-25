"""Quick results viewer for the (possibly partial) e2e / pilot parquet.

Because e2e_smoke now checkpoints after every cell, this works even while a
run is still going or after a crash — it shows whatever cells are done.

Prints:
  1. A per-cell table (f_mean = primary graded F, final_answer_f_mean = the
     secondary binary final-answer score, aufi_in = the novel metric).
  2. The v6 headline: mean chain-F by (ladder_family, level) — the
     context-vs-reasoning comparison.
  3. A graded-F sanity line (are scores strictly between 0 and 1?).

Optionally writes a PNG (--plot) of chain-F vs level, one line per ladder
family, the dual-ladder headline figure.

Usage:
  uv run python -m prompt_sensitivity.scripts.show_results [--in PATH] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


_FAMILY_COLOURS = {"context": "#1f77b4", "reasoning": "#d62728"}


def _plot_metric_by_family(
    df: pd.DataFrame,
    metric: str,
    out_png: Path,
    *,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Mean `metric` vs level, one line per ladder family. Works on partial data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(df["ladder_family"].dropna().unique()) if "ladder_family" in df else []

    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in families:
        sub = df[df["ladder_family"] == fam]
        grp = sub.groupby("level")[metric].mean().reset_index().sort_values("level")
        # Drop NaN points (e.g. a level not yet computed for this family).
        grp = grp[grp[metric].notna()]
        if grp.empty:
            continue
        ax.plot(grp["level"], grp[metric], marker="o", linewidth=2,
                label=f"{fam} ladder", color=_FAMILY_COLOURS.get(fam))
    ax.set_xlabel("ladder level  (context: #paragraphs | reasoning: #hops scaffolded)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    logger.info("wrote {}", out_png)


def _plot_accuracy_two_metrics(df: pd.DataFrame, out_png: Path) -> None:
    """Accuracy plot with BOTH metrics per family:
       solid + circles = chain-completion F (fraction of reasoning hops recovered),
       dashed + x      = final-answer F (binary correctness of the final span).
    Colour distinguishes ladder family; line style distinguishes the metric.
    The gap between the two lines = 'recovers the reasoning' vs 'nails the final answer'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    families = sorted(df["ladder_family"].dropna().unique()) if "ladder_family" in df else []
    have_final = "final_answer_f_mean" in df.columns

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for fam in families:
        sub = df[df["ladder_family"] == fam]
        colour = _FAMILY_COLOURS.get(fam)

        chain = sub.groupby("level")["f_mean"].mean().reset_index().sort_values("level")
        chain = chain[chain["f_mean"].notna()]
        if not chain.empty:
            ax.plot(chain["level"], chain["f_mean"], marker="o", linewidth=2,
                    color=colour)

        if have_final:
            fin = sub.groupby("level")["final_answer_f_mean"].mean().reset_index().sort_values("level")
            fin = fin[fin["final_answer_f_mean"].notna()]
            if not fin.empty:
                ax.plot(fin["level"], fin["final_answer_f_mean"], marker="x",
                        linewidth=2, linestyle="--", color=colour)

    ax.set_xlabel("ladder level  (context: #paragraphs | reasoning: #hops scaffolded)")
    ax.set_ylabel("mean F")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Accuracy vs level — chain-completion F (solid) vs final-answer F (dashed)")

    # Two-part legend: colour = family, style = metric.
    family_handles = [
        Line2D([0], [0], color=_FAMILY_COLOURS.get(f), linewidth=2, label=f"{f} ladder")
        for f in families
    ]
    style_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=2, label="chain-completion F"),
        Line2D([0], [0], color="black", marker="x", linewidth=2, linestyle="--",
               label="final-answer F"),
    ]
    leg1 = ax.legend(handles=family_handles, loc="upper left", title="ladder")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower right", title="metric")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    logger.info("wrote {}", out_png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    configure_logging("show_results")
    config = load_config()
    repo_root = config.repo_root()

    if args.inp:
        path = repo_root / args.inp
    else:
        path = _first_existing([
            repo_root / "data" / "pilot_musique.parquet",
            repo_root / "data" / "pilot_metrics.parquet",
            repo_root / "data" / "e2e_metrics.parquet",
        ])
    if path is None or not path.exists():
        logger.error("no results parquet found (looked for pilot_musique / pilot_metrics / e2e_metrics)")
        return 1

    df = pd.read_parquet(path)
    logger.info("loaded {} cells from {}", len(df), path)

    # 1. Per-cell table.
    cols = [c for c in [
        "question_id", "dataset", "ladder_family", "ladder_type_raw", "level",
        "model_key", "f_mean", "final_answer_f_mean", "aufi_in", "spread", "n_paraphrases",
    ] if c in df.columns]
    print()
    print("=" * 110)
    print(f"RESULTS  ({len(df)} cells)  from {path.name}")
    print("=" * 110)
    show = df[cols].copy()
    if "question_id" in show:
        show["question_id"] = show["question_id"].str.slice(0, 22)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 2. v6 headline: mean chain-F by (family, level).
    if {"ladder_family", "level", "f_mean"} <= set(df.columns):
        print()
        print("=" * 110)
        print("v6 HEADLINE — mean chain-F (accuracy) by (ladder_family, level)")
        print("=" * 110)
        pivot = (
            df.groupby(["ladder_family", "level"])["f_mean"]
            .mean()
            .reset_index()
            .pivot(index="level", columns="ladder_family", values="f_mean")
        )
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # 2b. FUNCTIONAL INFORMATION: mean AUFI_in (bits) by (family, level).
    #     AUFI_in = area under the FI_in(k) curve (Section_7 §7.3), in BITS.
    #     Higher = more prompt-sensitivity (only special phrasings recover the
    #     chain). It should DROP as context/reasoning is added — the FI mirror
    #     image of the rising accuracy curve.
    if {"ladder_family", "level", "aufi_in"} <= set(df.columns):
        print()
        print("=" * 110)
        print("FUNCTIONAL INFORMATION — mean AUFI_in [bits] by (ladder_family, level)")
        print("  (lower = more paraphrases recover the chain = less prompt-sensitive)")
        print("=" * 110)
        fi_pivot = (
            df.groupby(["ladder_family", "level"])["aufi_in"]
            .mean()
            .reset_index()
            .pivot(index="level", columns="ladder_family", values="aufi_in")
        )
        print(fi_pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # 3. graded-F sanity.
    if "f_mean" in df.columns:
        fvals = df["f_mean"].dropna().tolist()
        graded = any(0.0 < v < 1.0 for v in fvals)
        print()
        print(f"graded-F present (some f_mean strictly in (0,1))?  ->  {graded}")
        if fvals:
            print(f"f_mean range: [{min(fvals):.3f}, {max(fvals):.3f}], mean {sum(fvals)/len(fvals):.3f}")

    # P1-4: collinearity guard. If AUFI_in is just a monotone transform of mean F
    # at this sample, the headline must be the FI_in(k) curve, not the scalar
    # (AUFI_metrics_revisit §5.5/§6). Paired dropna so misaligned NaNs can't crash.
    if {"f_mean", "aufi_in"} <= set(df.columns):
        from scipy.stats import spearmanr
        pair = df[["f_mean", "aufi_in"]].dropna()
        rho = spearmanr(pair["f_mean"], pair["aufi_in"])[0] if len(pair) >= 5 else float("nan")
        print(f"\nCOLLINEARITY CHECK: Spearman(f_mean, aufi_in) = {rho:.3f}")
        if abs(rho) > 0.95:
            print("  WARNING: AUFI_in is essentially a transform of mean F at this sample.")
            print("  Headline should be the FI_in(k) curve, not the AUFI_in scalar.")
            print("  See AUFI_metrics_revisit_2026-06-25.docx §6.")

    if args.plot and "ladder_family" in df.columns:
        # Accuracy dual ladder: chain-F (solid) + final-answer F (dashed).
        acc_png = path.with_name(path.stem + "_dual_ladder.png")
        _plot_accuracy_two_metrics(df, acc_png)
        # Functional Information (AUFI_in, bits) dual ladder — the headline metric.
        fi_png = path.with_name(path.stem + "_fi_in.png")
        _plot_metric_by_family(
            df, "aufi_in", fi_png,
            ylabel="mean AUFI_in  [bits]  (lower = less prompt-sensitive)",
            title="Functional Information (AUFI_in) vs level — falls as context/reasoning is added",
        )
        print(f"\nplots -> {acc_png}\n         {fi_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
