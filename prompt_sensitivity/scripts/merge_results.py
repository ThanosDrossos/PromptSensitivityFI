"""Merge the full run's per-model parquets into one, then summarise.

The full cluster run writes ONE parquet per model (`data/full_<model>.parquet`)
so the parallel SLURM array tasks never write the same file (no race). This
concatenates them into `data/full_run.parquet`, de-duplicating on cell identity
(so a resumed/re-run task can't double-count).

Run locally after `bash cluster/run.sh pull`:
  uv run python -m prompt_sensitivity.scripts.merge_results
  uv run python -m prompt_sensitivity.scripts.show_results --in data/full_run.parquet
  uv run python -m prompt_sensitivity.scripts.plot_pilot   --in data/full_run.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


_CELL_KEYS = ["question_id", "ladder_family", "ladder_type_raw", "level", "model_key"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/full_*.parquet",
                    help="glob of per-model parquets to merge (default data/full_*.parquet)")
    ap.add_argument("--out", default="data/full_run.parquet")
    args = ap.parse_args()

    configure_logging("merge_results")
    root = load_config().repo_root()
    out = (root / args.out).resolve()

    files = [Path(p) for p in sorted(glob.glob(str(root / args.glob)))]
    files = [f for f in files if f.resolve() != out]  # never merge the output into itself
    if not files:
        logger.error("no parquets match {} (under {})", args.glob, root)
        return 1

    frames = []
    for f in files:
        d = pd.read_parquet(f)
        logger.info("loaded {} rows from {}", len(d), f.name)
        frames.append(d)
    merged = pd.concat(frames, ignore_index=True)

    keys = [c for c in _CELL_KEYS if c in merged.columns]
    before = len(merged)
    if keys:
        merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    summary = {
        "files_merged": [f.name for f in files],
        "rows": int(len(merged)),
        "dropped_duplicate_cells": int(before - len(merged)),
        "models": sorted(merged["model_key"].unique().tolist()) if "model_key" in merged else [],
        "questions": int(merged["question_id"].nunique()) if "question_id" in merged else 0,
        "out": str(out),
    }
    logger.info("merged {} files -> {}", len(files), out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
