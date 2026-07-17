"""Specificity results view (pivot spec §13.9): pivot on spec_level.

Puts f_mean, AUFI_in (FI_in), FI_out, H_sem, and FI_spec side by side per
specificity level (and per model when several are present), plus the per-question
level-1 minus level-0 deltas — the pivot's validation hypotheses read directly
off this table.

    uv run python -m prompt_sensitivity.scripts.show_specificity \
        --in data/specificity_metrics.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config

_COLS = ["f_mean", "f_mean_permissive", "f_graded_mean", "aufi_in", "aufi_in_graded",
         "fi_out_mean", "fi_out_fixed", "h_sem_mean", "a_q", "fi_spec"]


def add_fi_out_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Derived, FIXED-answer-space output FI: log2(m0) - H_sem.

    fi_out_mean uses the OBSERVED |A_q|, which itself shrinks with specificity
    (fewer output clusters at L1), so its sign is not interpretable for the
    specificity hypothesis. Holding the space at the dataset's m0 makes the
    output-side quantity comparable across levels (rises exactly when H_sem
    falls). The driver emits the column at run time since 2026-07-17; rows from
    older runs (or a mixed resume parquet) get the identical derivation filled
    in here, pipeline values taking precedence.
    """
    df = df.copy()
    if {"m0", "h_sem_mean"} <= set(df.columns):
        m0 = pd.to_numeric(df["m0"], errors="coerce").clip(lower=1)
        derived = np.log2(m0) - df["h_sem_mean"]
        if "fi_out_fixed" in df.columns:
            df["fi_out_fixed"] = df["fi_out_fixed"].combine_first(derived)
        else:
            df["fi_out_fixed"] = derived
    return df


def paired_deltas(
    df: pd.DataFrame, cols: list[str], *, exclude_mismatched: bool = True
) -> tuple[pd.DataFrame, int]:
    """Per-question (level 1 - level 0) deltas.

    N-mismatched pairs (different paraphrase counts at the two levels, e.g. a
    singleton universe at L0 vs 10 at L1) have DIFFERENT AUFI ceilings —
    log2(N+1) — so their FI_in deltas are artifacts (the v2 run's two "+2.4 bit"
    outliers were exactly this). They are excluded from the paired stats and
    reported as a count.
    """
    vals = [c for c in cols if c in df.columns] + ["n_paraphrases"]
    w = df.pivot_table(index=["question_id", "model_key"], columns="spec_level",
                       values=vals, aggfunc="mean")
    if not {0, 1} <= set(df["spec_level"].unique()):
        return pd.DataFrame(), 0
    mism = w[("n_paraphrases", 0)] != w[("n_paraphrases", 1)]
    n_excluded = 0
    if exclude_mismatched:
        n_excluded = int(mism.sum())
        w = w[~mism]
    deltas = pd.DataFrame({
        c: w[(c, 1)] - w[(c, 0)]
        for c in cols if (c, 0) in w.columns and (c, 1) in w.columns
    })
    return deltas, n_excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/specificity_metrics.parquet")
    args = ap.parse_args()

    config = load_config()
    path = Path(args.inp)
    if not path.is_absolute():
        path = config.repo_root() / path
    df = add_fi_out_fixed(pd.read_parquet(path))
    cols = [c for c in _COLS if c in df.columns]
    fmt = lambda x: f"{x:.3f}"  # noqa: E731

    print()
    print("=" * 100)
    print(f"SPECIFICITY — {len(df)} cells from {path.name} "
          "(level 0 = ambiguous, 1 = disambiguated; gold fixed)")
    print("=" * 100)
    group = ["model_key", "spec_level"] if df["model_key"].nunique() > 1 else ["spec_level"]
    print(df.groupby(group)[cols].mean().to_string(float_format=fmt))

    # Per-question level deltas (1 minus 0) — the validation hypotheses. NOTE:
    # fi_out_mean's sign is not interpretable (observed |A_q| shrinks with
    # specificity); fi_out_fixed = log2(m0) - H_sem carries the output-side
    # hypothesis instead (expected +).
    delta_cols = [c for c in ["f_mean", "f_graded_mean", "aufi_in", "aufi_in_graded",
                              "fi_out_fixed", "h_sem_mean", "fi_spec"]
                  if c in df.columns and df[c].notna().any()]
    deltas, n_excluded = paired_deltas(df, delta_cols)
    if not deltas.empty:
        print()
        print("per-question DELTAS (level 1 - level 0); expected signs: "
              "f_mean/f_graded +, aufi_in(_graded) -, fi_out_fixed +, h_sem -, fi_spec +")
        if n_excluded:
            print(f"  ({n_excluded} N-mismatched pair(s) excluded — unequal paraphrase "
                  "counts across levels make FI_in deltas artifacts)")
        print(deltas.describe().loc[["mean", "50%", "min", "max"]].to_string(float_format=fmt))
        frac_up = (deltas > 0).mean()
        print()
        print("fraction of questions with positive delta:")
        print(frac_up.to_string(float_format=fmt))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
