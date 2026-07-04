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

import pandas as pd

from ..config import load_config

_COLS = ["f_mean", "f_mean_permissive", "aufi_in", "fi_out_mean",
         "h_sem_mean", "a_q", "fi_spec"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/specificity_metrics.parquet")
    args = ap.parse_args()

    config = load_config()
    path = Path(args.inp)
    if not path.is_absolute():
        path = config.repo_root() / path
    df = pd.read_parquet(path)
    cols = [c for c in _COLS if c in df.columns]
    fmt = lambda x: f"{x:.3f}"  # noqa: E731

    print()
    print("=" * 100)
    print(f"SPECIFICITY — {len(df)} cells from {path.name} "
          "(level 0 = ambiguous, 1 = disambiguated; gold fixed)")
    print("=" * 100)
    group = ["model_key", "spec_level"] if df["model_key"].nunique() > 1 else ["spec_level"]
    print(df.groupby(group)[cols].mean().to_string(float_format=fmt))

    # Per-question level deltas (1 minus 0) — the validation hypotheses:
    # f_mean up, aufi_in down, h_sem down, fi_out up, fi_spec up.
    delta_cols = [c for c in ["f_mean", "aufi_in", "fi_out_mean", "h_sem_mean", "fi_spec"]
                  if c in df.columns]
    wide = df.pivot_table(index=["question_id", "model_key"], columns="spec_level",
                          values=delta_cols, aggfunc="mean")
    if not wide.empty and {0, 1} <= set(df["spec_level"].unique()):
        deltas = pd.DataFrame({c: wide[(c, 1)] - wide[(c, 0)] for c in delta_cols
                               if (c, 0) in wide.columns and (c, 1) in wide.columns})
        print()
        print("per-question DELTAS (level 1 - level 0); expected signs: "
              "f_mean +, aufi_in -, fi_out +, h_sem -, fi_spec +")
        print(deltas.describe().loc[["mean", "50%", "min", "max"]].to_string(float_format=fmt))
        frac_up = (deltas > 0).mean()
        print()
        print("fraction of questions with positive delta:")
        print(frac_up.to_string(float_format=fmt))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
