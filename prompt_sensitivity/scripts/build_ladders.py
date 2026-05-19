"""Sprint 3 driver — produce `data/ladders.parquet`.

For every sampled question and every (ladder_type, level), compute the
paragraph subset and persist as a row. Also prints the b_theo table for
the gate report.

Output shape (~2700 rows = 150 questions × 3 ladders × 6 levels):

    question_id, ladder_type, level_idx, level, paragraph_indices,
    paragraph_titles, gold_count, permutation

Sprint-3 brief edge case: drop any question that has fewer than 2 gold
paragraphs (the §4 design assumes exactly 2 gold per question).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..data import (
    MultiHopQuestion,
    load_hotpotqa_validation,
    load_twiki_validation,
)
from ..ladders import build_all_ladders, b_theo_table
from ..logging_setup import configure_logging


def _load_sample(path: Path) -> tuple[list[str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [r["id"] for r in raw["hotpotqa"]], [r["id"] for r in raw["twiki"]]


def _index_by_id(records: list[MultiHopQuestion]) -> dict[str, MultiHopQuestion]:
    return {q.id: q for q in records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/ladders.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configure_logging("build_ladders")
    config = load_config()
    repo_root = config.repo_root()
    k_gold = config.ladders.k_gold
    n_total = config.ladders.n_total_paragraphs
    levels = config.ladders.levels

    sample_path = repo_root / "data" / "sample_v1.json"
    if not sample_path.exists():
        logger.error("missing {} — run `make sample` first", sample_path)
        return 1
    hotpot_ids, twiki_ids = _load_sample(sample_path)

    logger.info("loading HotpotQA validation ...")
    hp_idx = _index_by_id(
        load_hotpotqa_validation(
            hf_dataset=config.sampling.hotpotqa.hf_dataset,
            hf_config=config.sampling.hotpotqa.hf_config or "distractor",
            split=config.sampling.hotpotqa.split,
        )
    )
    logger.info("loading 2WikiMultihopQA validation ...")
    tw_idx = _index_by_id(
        load_twiki_validation(
            hf_dataset=config.sampling.twiki.hf_dataset,
            hf_config=config.sampling.twiki.hf_config,
            split=config.sampling.twiki.split,
        )
    )

    work: list[MultiHopQuestion] = []
    skipped_missing: list[str] = []
    skipped_too_few_gold: list[str] = []
    skipped_wrong_paragraph_count: list[str] = []
    for qid in hotpot_ids + twiki_ids:
        q = hp_idx.get(qid) or tw_idx.get(qid)
        if q is None:
            skipped_missing.append(qid)
            continue
        if len(q.gold_paragraphs()) < k_gold:
            skipped_too_few_gold.append(qid)
            continue
        # The §4.4 b_theo formula is only valid when N = n_total_paragraphs
        # (typically 10). A handful of HotpotQA validation records ship with
        # fewer than 10 paragraphs; drop them so the ladder levels map
        # consistently to the b_theo table.
        if len(q.paragraphs) != n_total:
            skipped_wrong_paragraph_count.append(qid)
            continue
        work.append(q)

    if args.limit:
        work = work[: args.limit]

    rows: list[dict] = []
    for q in work:
        for lr in build_all_ladders(q):
            rows.append(lr.model_dump())

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # parquet handles list-typed columns natively via pyarrow.
    df.to_parquet(out_path, index=False)
    logger.info(
        "wrote {} rows ({} questions x {} ladders x {} levels) -> {}",
        len(df),
        df["question_id"].nunique(),
        df["ladder_type"].nunique(),
        df["level_idx"].nunique(),
        out_path,
    )

    # --- gate report: b_theo table for the random ladder -------------------
    print()
    print("=" * 70)
    print(f"b_theo table (N={n_total}, K={k_gold} gold, levels={levels})")
    print("=" * 70)
    table = b_theo_table(n_total, k_gold, levels)
    tbl_df = pd.DataFrame(table)
    print(tbl_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- ladder sanity invariants -----------------------------------------
    print()
    print("=" * 70)
    print("SANITY INVARIANTS")
    print("=" * 70)
    # At level 0, every ladder is empty.
    l0 = df[df["level"] == 0]
    assert (l0["paragraph_indices"].apply(len) == 0).all(), "level 0 not empty everywhere"
    print(f"  level 0: all rows empty ({len(l0)} rows) — OK")
    # At top level, every ladder has all N paragraphs.
    ltop = df[df["level"] == n_total]
    assert (ltop["paragraph_indices"].apply(len) == n_total).all(), f"level {n_total} not full"
    print(f"  level {n_total}: all rows contain {n_total} paragraphs ({len(ltop)} rows) — OK")
    # Same multiset at top: random/gold/distractor differ in ORDER, not set.
    for qid, sub in ltop.groupby("question_id"):
        sets = [tuple(sorted(idxs)) for idxs in sub["paragraph_indices"]]
        if len(set(sets)) != 1:
            print(f"  level {n_total}: qid={qid} has differing paragraph sets across ladders!")
    # Per-level gold counts: gold_first >= random_expected, distractor_first <= random_expected.
    for level_idx, level in enumerate(levels):
        sub = df[df["level_idx"] == level_idx]
        means = sub.groupby("ladder_type")["gold_count"].mean().to_dict()
        print(
            f"  level={level:>2}  mean gold_count: "
            f"gold_first={means.get('gold_first', float('nan')):.2f}  "
            f"random={means.get('random', float('nan')):.2f}  "
            f"distractor_first={means.get('distractor_first', float('nan')):.2f}"
        )

    print()
    print(json.dumps({
        "n_questions_processed": int(df["question_id"].nunique()),
        "n_rows": int(len(df)),
        "skipped_missing": len(skipped_missing),
        "skipped_too_few_gold": len(skipped_too_few_gold),
        "skipped_wrong_paragraph_count": len(skipped_wrong_paragraph_count),
        "skipped_ids": {
            "missing": skipped_missing,
            "too_few_gold": skipped_too_few_gold,
            "wrong_paragraph_count": skipped_wrong_paragraph_count,
        },
        "out_path": str(out_path),
    }, indent=2))

    if skipped_too_few_gold:
        logger.warning(
            "{} sampled questions had < {} gold paragraphs and were dropped; "
            "the sampler should backfill from sample_v1.json (Sprint 3 §3.1 rule)",
            len(skipped_too_few_gold),
            k_gold,
        )
    if skipped_wrong_paragraph_count:
        logger.warning(
            "{} sampled questions had paragraph_count != {} and were dropped; "
            "consider backfilling from the validation set",
            len(skipped_wrong_paragraph_count),
            n_total,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
