"""Sprint 2 driver — produce `data/paraphrases_v1.parquet`.

Walks `data/sample_v1.json`, looks up each question in HotpotQA or 2Wiki,
runs the full paraphrase pipeline, and persists accepted + rejected rows.

Flags:
  --limit N        : process only the first N sampled questions (for smoke testing)
  --dataset NAME   : restrict to "hotpotqa" or "twiki"
  --resume         : skip question_ids already present in the parquet output

Each row is a flattened paraphrase record (accepted OR rejected) — easier to
analyse in pandas / DuckDB than a nested per-question structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
from loguru import logger

from ..config import load_config
from ..data import (
    MultiHopQuestion,
    load_hotpotqa_validation,
    load_twiki_validation,
)
from ..logging_setup import configure_logging
from ..paraphrases.pipeline import build_paraphrase_set


def _load_sample(path: Path) -> tuple[list[str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    hotpot_ids = [r["id"] for r in raw["hotpotqa"]]
    twiki_ids = [r["id"] for r in raw["twiki"]]
    return hotpot_ids, twiki_ids


def _index_by_id(records: list[MultiHopQuestion]) -> dict[str, MultiHopQuestion]:
    return {q.id: q for q in records}


def _flatten_rows(pset) -> Iterator[dict]:
    """Yield one parquet row per accepted/rejected paraphrase + a header row."""
    for ap in pset.accepted:
        yield dict(
            question_id=ap.question_id,
            outcome="accepted",
            paraphrase_idx=ap.paraphrase_idx,
            role=ap.role,
            text=ap.text,
            nli_fwd=ap.nli_entail_fwd,
            nli_bwd=ap.nli_entail_bwd,
            jaccard=ap.constraint_jaccard,
            reason=None,
            generator_model_key=ap.generator_model_key,
            generator_seed=ap.generator_seed,
            request_hash=ap.request_hash,
            regeneration_attempts=pset.regeneration_attempts,
            nli_threshold_used=pset.nli_threshold_used,
            dropped=pset.dropped,
        )
    for rj in pset.rejected:
        yield dict(
            question_id=rj.question_id,
            outcome="rejected",
            paraphrase_idx=None,
            role=rj.role,
            text=rj.text,
            nli_fwd=rj.nli_entail_fwd,
            nli_bwd=rj.nli_entail_bwd,
            jaccard=rj.constraint_jaccard,
            reason=rj.reason,
            generator_model_key=None,
            generator_seed=None,
            request_hash=None,
            regeneration_attempts=pset.regeneration_attempts,
            nli_threshold_used=pset.nli_threshold_used,
            dropped=pset.dropped,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", choices=("hotpotqa", "twiki", "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=str, default="data/paraphrases_v1.parquet")
    args = parser.parse_args()

    configure_logging("generate_paraphrases")
    config = load_config()
    repo_root = config.repo_root()

    sample_path = repo_root / "data" / "sample_v1.json"
    if not sample_path.exists():
        logger.error("missing {} — run `make sample` first", sample_path)
        return 1
    hotpot_ids, twiki_ids = _load_sample(sample_path)

    work: list[tuple[str, MultiHopQuestion]] = []
    if args.dataset in {"hotpotqa", "both"} and hotpot_ids:
        logger.info("loading HotpotQA validation for paraphrase input ...")
        hp_idx = _index_by_id(
            load_hotpotqa_validation(
                hf_dataset=config.sampling.hotpotqa.hf_dataset,
                hf_config=config.sampling.hotpotqa.hf_config or "distractor",
                split=config.sampling.hotpotqa.split,
            )
        )
        for qid in hotpot_ids:
            if qid in hp_idx:
                work.append((qid, hp_idx[qid]))
    if args.dataset in {"twiki", "both"} and twiki_ids:
        logger.info("loading 2WikiMultihopQA validation for paraphrase input ...")
        tw_idx = _index_by_id(
            load_twiki_validation(
                hf_dataset=config.sampling.twiki.hf_dataset,
                hf_config=config.sampling.twiki.hf_config,
                split=config.sampling.twiki.split,
            )
        )
        for qid in twiki_ids:
            if qid in tw_idx:
                work.append((qid, tw_idx[qid]))

    if args.limit:
        work = work[: args.limit]
    logger.info("processing {} questions (--dataset={}, --limit={})", len(work), args.dataset, args.limit)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    existing_df: pd.DataFrame | None = None
    if args.resume and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        existing_ids = set(existing_df["question_id"].unique())
        logger.info("--resume: skipping {} already-processed question_ids", len(existing_ids))

    rows: list[dict] = []
    for qid, q in work:
        if qid in existing_ids:
            continue
        try:
            # Pass the dataset's known answer so the constraint filter can
            # use the gold-based judge (one yes/no per candidate) instead of
            # the fragile judge-vs-judge Jaccard path. Both HotpotQA and
            # framolfese/2WikiMultihopQA ship MultiHopQuestion.answer.
            pset = build_paraphrase_set(
                qid,
                q.question,
                config=config,
                gold_answer=q.answer,
            )
        except Exception:  # noqa: BLE001
            logger.exception("pipeline failed for qid={}", qid)
            continue
        rows.extend(_flatten_rows(pset))
        logger.info(
            "qid={} accepted={} dropped={} attempts={}",
            qid,
            pset.n_accepted(),
            pset.dropped,
            pset.regeneration_attempts,
        )

    if not rows:
        logger.warning("no new rows produced; nothing to write")
        return 0
    new_df = pd.DataFrame(rows)
    if existing_df is not None:
        new_df = pd.concat([existing_df, new_df], ignore_index=True)
    new_df.to_parquet(out_path, index=False)
    logger.info("wrote {} rows -> {}", len(new_df), out_path)

    # Summary
    n_accepted = (new_df["outcome"] == "accepted").sum()
    n_dropped_q = new_df.loc[new_df["dropped"], "question_id"].nunique()
    n_total_q = new_df["question_id"].nunique()
    print(json.dumps({
        "questions_processed": int(n_total_q),
        "questions_dropped": int(n_dropped_q),
        "accepted_paraphrases": int(n_accepted),
        "rows_total": int(len(new_df)),
        "out_path": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
