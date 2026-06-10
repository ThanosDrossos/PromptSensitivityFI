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


def _plot_dual_ladder(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(df["ladder_family"].dropna().unique()) if "ladder_family" in df else []
    colours = {"context": "#1f77b4", "reasoning": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in families:
        sub = df[df["ladder_family"] == fam]
        grp = sub.groupby("level")["f_mean"].mean().reset_index().sort_values("level")
        ax.plot(grp["level"], grp["f_mean"], marker="o", linewidth=2,
                label=f"{fam} ladder", color=colours.get(fam))
    ax.set_xlabel("ladder level  (context: #paragraphs | reasoning: #hops scaffolded)")
    ax.set_ylabel("mean chain-completion F")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("v6 dual ladder — graded chain-F vs level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
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
        print("v6 HEADLINE — mean chain-F by (ladder_family, level)")
        print("=" * 110)
        pivot = (
            df.groupby(["ladder_family", "level"])["f_mean"]
            .mean()
            .reset_index()
            .pivot(index="level", columns="ladder_family", values="f_mean")
        )
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # 3. graded-F sanity.
    if "f_mean" in df.columns:
        fvals = df["f_mean"].dropna().tolist()
        graded = any(0.0 < v < 1.0 for v in fvals)
        print()
        print(f"graded-F present (some f_mean strictly in (0,1))?  ->  {graded}")
        if fvals:
            print(f"f_mean range: [{min(fvals):.3f}, {max(fvals):.3f}], mean {sum(fvals)/len(fvals):.3f}")

    if args.plot and "ladder_family" in df.columns:
        out_png = path.with_name(path.stem + "_dual_ladder.png")
        _plot_dual_ladder(df, out_png)
        print(f"\nplot -> {out_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
