"""Sprint 2 gate — export 20 questions × all paraphrases as a CSV review sheet.

Thanos and Diener each annotate "is this paraphrase semantically equivalent
to the original?" (yes/no). Cohen's κ is computed afterwards via
`prompt_sensitivity.scripts.compute_kappa`.

The 20 questions are deterministically chosen (seed = config.random_seed + 2)
from the questions with full 30 accepted paraphrases — so the annotation
sample faithfully reflects the pipeline's typical output.

Output: `data/annotation_sample_v1.csv`. Columns:

    question_id, original_question, paraphrase_idx, role, paraphrase_text,
    nli_fwd, nli_bwd, jaccard, thanos, diener

`thanos` and `diener` are blank — humans fill them with 1/0.
"""

from __future__ import annotations

import argparse
import json
import random
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
from ..logging_setup import configure_logging


def _load_sample_ids(repo_root: Path) -> set[str]:
    raw = json.loads((repo_root / "data" / "sample_v1.json").read_text(encoding="utf-8"))
    return {r["id"] for r in raw["hotpotqa"]} | {r["id"] for r in raw["twiki"]}


def _index_questions(config) -> dict[str, MultiHopQuestion]:
    out: dict[str, MultiHopQuestion] = {}
    hp = load_hotpotqa_validation(
        hf_dataset=config.sampling.hotpotqa.hf_dataset,
        hf_config=config.sampling.hotpotqa.hf_config or "distractor",
        split=config.sampling.hotpotqa.split,
    )
    out.update({q.id: q for q in hp})
    tw = load_twiki_validation(
        hf_dataset=config.sampling.twiki.hf_dataset,
        hf_config=config.sampling.twiki.hf_config,
        split=config.sampling.twiki.split,
    )
    out.update({q.id: q for q in tw})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paraphrases", type=str, default="data/paraphrases_v1.parquet")
    parser.add_argument("--out", type=str, default="data/annotation_sample_v1.csv")
    parser.add_argument("--n-questions", type=int, default=20)
    args = parser.parse_args()

    configure_logging("export_annotation_sample")
    config = load_config()
    repo_root = config.repo_root()

    parq = repo_root / args.paraphrases
    if not parq.exists():
        logger.error("missing {} — run `make paraphrases` first", parq)
        return 1
    df = pd.read_parquet(parq)
    accepted = df[(df["outcome"] == "accepted") & (~df["dropped"])]

    # Keep only questions with the full target count.
    target = config.paraphrases.n_per_question
    per_q = accepted.groupby("question_id").size()
    eligible_ids = per_q[per_q >= target].index.tolist()
    logger.info(
        "{} eligible questions (each with >={} accepted paraphrases)",
        len(eligible_ids),
        target,
    )
    if len(eligible_ids) < args.n_questions:
        logger.error(
            "only {} eligible; need {} — generate more paraphrases first",
            len(eligible_ids),
            args.n_questions,
        )
        return 2

    rng = random.Random(config.random_seed + 2)
    picked = rng.sample(eligible_ids, args.n_questions)

    qmap = _index_questions(config)
    rows: list[dict] = []
    for qid in picked:
        if qid not in qmap:
            logger.warning("qid={} present in paraphrases but not in datasets; skipping", qid)
            continue
        original = qmap[qid].question
        sub = accepted[accepted["question_id"] == qid].sort_values("paraphrase_idx")
        for _, r in sub.iterrows():
            rows.append(dict(
                question_id=qid,
                original_question=original,
                paraphrase_idx=int(r["paraphrase_idx"]),
                role=r["role"],
                paraphrase_text=r["text"],
                nli_fwd=float(r["nli_fwd"]),
                nli_bwd=float(r["nli_bwd"]),
                jaccard=float(r["jaccard"]),
                thanos="",
                diener="",
            ))

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    logger.info("wrote {} rows -> {}", len(rows), out_path)
    print(json.dumps({
        "n_questions": len(picked),
        "n_paraphrases": len(rows),
        "out_path": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
