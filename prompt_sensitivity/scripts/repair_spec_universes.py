"""Repair sub-target paraphrase universes for the specificity runs.

The v2 full run had 18 universes below the N=10 target (5 singleton fallbacks,
the 6-paraphrase gate leftovers, ...) -> 8 N-mismatched question pairs whose
FI_in deltas are artifacts (different AUFI ceilings). This utility makes the
next chain submission regenerate + re-evaluate exactly the affected questions:

  1. drop the paraphrase-cache rows of BOTH levels of every affected question
     (pair-consistent: both universes regenerate at the full target), including
     `singleton_fallback` rows so failed generations are RE-ATTEMPTED;
  2. drop those questions' cells from the metrics parquet so the resume loop
     recomputes them (generation cache makes re-eval cheap).

Dry-run by default; --apply writes (atomic replace). Run on the cluster before
resubmitting the chain:

    python -m prompt_sensitivity.scripts.repair_spec_universes \
        --metrics data/specificity_v2_metrics.parquet --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


def plan_repair(
    para: pd.DataFrame, metrics: pd.DataFrame | None, min_n: int
) -> tuple[set[str], pd.Series, pd.Series | None]:
    """Pure planner. Returns (affected question_ids,
    keep-mask for the paraphrase df, keep-mask for the metrics df | None)."""
    live = para[para["outcome"].isin(["accepted", "singleton_fallback"])]
    per_universe = live.groupby(["question_id", "spec_level"]).size()
    affected = {str(qid) for (qid, _lvl), n in per_universe.items() if n < min_n}
    para_keep = ~para["question_id"].astype(str).isin(affected)
    metrics_keep = None
    if metrics is not None:
        metrics_keep = ~metrics["question_id"].astype(str).isin(affected)
    return affected, para_keep, metrics_keep


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paraphrases", default="data/paraphrases_ambigqa.parquet")
    ap.add_argument("--metrics", default=None,
                    help="metrics parquet whose affected cells should be recomputed "
                         "(e.g. data/specificity_v2_metrics.parquet)")
    ap.add_argument("--min-n", type=int, default=None,
                    help="target universe size (default: config.paraphrases.n_per_question)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    configure_logging("repair_spec_universes")
    config = load_config()
    root = config.repo_root()
    min_n = args.min_n if args.min_n is not None else config.paraphrases.n_per_question

    ppath = root / args.paraphrases
    para = pd.read_parquet(ppath)
    metrics = None
    mpath = None
    if args.metrics:
        mpath = root / args.metrics
        if mpath.exists():
            metrics = pd.read_parquet(mpath)
        else:
            logger.warning("metrics parquet {} not found — repairing cache only", mpath)

    affected, para_keep, metrics_keep = plan_repair(para, metrics, min_n)
    logger.info("target N={} -> {} affected question(s)", min_n, len(affected))
    if not affected:
        print("nothing to repair — all universes at target size")
        return 0
    n_para_drop = int((~para_keep).sum())
    n_cell_drop = int((~metrics_keep).sum()) if metrics_keep is not None else 0
    print(f"affected questions ({len(affected)}):")
    for qid in sorted(affected):
        sizes = para[para.question_id.astype(str) == qid].groupby("spec_level").size().to_dict()
        print(f"  {qid}  universe sizes {sizes}")
    print(f"would drop {n_para_drop} paraphrase-cache rows"
          + (f" and {n_cell_drop} metric cells" if metrics is not None else ""))

    if not args.apply:
        print("\nDRY RUN — rerun with --apply to write, then resubmit the chain "
              "(bash cluster/submit_specificity_full.sh 6)")
        return 0
    _atomic_write(para[para_keep], ppath)
    logger.info("wrote {} ({} rows)", ppath, int(para_keep.sum()))
    if metrics is not None and mpath is not None:
        _atomic_write(metrics[metrics_keep], mpath)
        logger.info("wrote {} ({} rows)", mpath, int(metrics_keep.sum()))
    print("APPLIED — resubmit the chain to regenerate + re-evaluate the affected questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
