"""Sprint 1 §1.4 — write `data/sample_v1.json` with the 150 stratified IDs.

Output shape:
{
  "config_version": 1,
  "seed": 42,
  "hotpotqa": [
    {"id": "...", "level": "medium", "type": "bridge"},
    ...
  ],
  "twiki": [
    {"id": "...", "type": "comparison"},
    ...
  ]
}

The downstream pipeline never re-samples. If config.random_seed or
sampling.n_questions changes, bump `config_version`, regenerate, and version
the file as `sample_v2.json`.
"""

from __future__ import annotations

import json
import sys

from loguru import logger

from ..config import load_config
from ..data import (
    load_hotpotqa_validation,
    load_twiki_validation,
    stratified_sample,
)
from ..logging_setup import configure_logging


def main() -> int:
    configure_logging("sample_questions")
    config = load_config()
    repo_root = config.repo_root()

    logger.info("loading HotpotQA validation ...")
    hotpot = load_hotpotqa_validation(
        hf_dataset=config.sampling.hotpotqa.hf_dataset,
        hf_config=config.sampling.hotpotqa.hf_config or "distractor",
        split=config.sampling.hotpotqa.split,
    )
    hotpot_sample = stratified_sample(
        hotpot,
        n_total=config.sampling.hotpotqa.n_questions,
        stratify_by=config.sampling.hotpotqa.stratify_by,
        seed=config.random_seed,
        k_gold=config.ladders.k_gold,
    )

    logger.info("loading 2WikiMultihopQA validation ...")
    twiki = load_twiki_validation(
        hf_dataset=config.sampling.twiki.hf_dataset,
        hf_config=config.sampling.twiki.hf_config,
        split=config.sampling.twiki.split,
    )
    twiki_sample = stratified_sample(
        twiki,
        n_total=config.sampling.twiki.n_questions,
        stratify_by=config.sampling.twiki.stratify_by,
        seed=config.random_seed + 1,
        k_gold=config.ladders.k_gold,
    )

    out = {
        "config_version": config.config_version,
        "seed": config.random_seed,
        "hotpotqa": [
            {"id": q.id, "level": q.level, "type": q.question_type} for q in hotpot_sample
        ],
        "twiki": [
            {"id": q.id, "type": q.question_type} for q in twiki_sample
        ],
    }

    out_path = repo_root / "data" / "sample_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote {}", out_path)

    # Distribution summary on stdout.
    hotpot_levels: dict[str, int] = {}
    for r in out["hotpotqa"]:
        hotpot_levels[r["level"] or "<none>"] = hotpot_levels.get(r["level"] or "<none>", 0) + 1
    twiki_types: dict[str, int] = {}
    for r in out["twiki"]:
        twiki_types[r["type"]] = twiki_types.get(r["type"], 0) + 1
    print(json.dumps({
        "hotpotqa_n": len(out["hotpotqa"]),
        "hotpotqa_level_counts": hotpot_levels,
        "twiki_n": len(out["twiki"]),
        "twiki_type_counts": twiki_types,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
