"""Backfill the dataset-side `target_collision` column into result parquets.

The flag (does the pinned target share an answer with another interpretation?
— e.g. Kriseman won both the 2013 and 2017 St. Petersburg races) is emitted by
the driver for every FUTURE run; this script adds it to parquets produced
before 2026-08-02 so collision cells can be split out or reweighted without a
re-run. Idempotent (recomputes + overwrites the column), dataset-side +
model-free (loader + seeded target choice only — no LLM anywhere).

    uv run python -m prompt_sensitivity.scripts.backfill_collisions data/specificity_v3_*.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.load_ambigqa import load_ambigqa
from ..logging_setup import configure_logging
from ..specificity.build_levels import choose_target_idx, target_has_collision


def collision_map(config) -> dict[str, bool]:
    """question_id -> collision flag for every ambiguous AmbigQA question."""
    acfg = config.sampling.ambigqa
    seed = (config.specificity.target_seed
            if config.specificity is not None else config.random_seed)
    out: dict[str, bool] = {}
    for q in load_ambigqa(
        hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
        min_interpretations=acfg.min_interpretations,
        include_single_answer_anchor=acfg.include_single_answer_anchor,
    ):
        idx = choose_target_idx(q.id, q.m0(), seed=seed)
        out[q.id] = target_has_collision(q, idx)
    return out


def backfill(path, cmap: dict[str, bool]) -> bool:
    df = pd.read_parquet(path)
    if "question_id" not in df.columns:
        logger.warning("{}: no question_id column — skipped", path)
        return False
    flags = df["question_id"].astype(str).map(cmap)
    n_missing = int(flags.isna().sum())
    if n_missing:
        logger.warning("{}: {} rows not in the AmbigQA map (left NA)", path, n_missing)
    df["target_collision"] = flags.astype("boolean")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    n_true = int(df["target_collision"].fillna(False).sum())
    logger.info("{}: target_collision written ({} of {} rows True)",
                path.name, n_true, len(df))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+", help="result parquet files to backfill")
    args = ap.parse_args()

    configure_logging("backfill_collisions")
    config = load_config()
    root = config.repo_root()
    cmap = collision_map(config)
    logger.info("collision map: {} questions, {} with collisions ({:.1%})",
                len(cmap), sum(cmap.values()), sum(cmap.values()) / len(cmap))
    n_ok = 0
    for p in args.parquets:
        path = root / p if not str(p).startswith(str(root)) else p
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            logger.warning("{}: not found — skipped", p)
            continue
        n_ok += int(backfill(path, cmap))
    logger.info("backfilled {} file(s)", n_ok)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
