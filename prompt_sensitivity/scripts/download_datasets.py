"""Download HotpotQA distractor + 2WikiMultihopQA validation. Sprint 1 §1.3.

The HF `datasets` library handles caching to ~/.cache/huggingface by default.
This script forces both downloads to happen up front and writes a tiny
`data/raw/dataset_manifest.json` recording row counts so downstream scripts can
fail fast if the cache disappears.
"""

from __future__ import annotations

import json
import sys

from loguru import logger

from ..config import load_config
from ..data import load_hotpotqa_validation, load_twiki_validation
from ..logging_setup import configure_logging


def main() -> int:
    configure_logging("download_datasets")
    config = load_config()
    repo_root = config.repo_root()

    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "dataset_manifest.json"

    logger.info("downloading HotpotQA distractor validation ...")
    hotpot = load_hotpotqa_validation(
        hf_dataset=config.sampling.hotpotqa.hf_dataset,
        hf_config=config.sampling.hotpotqa.hf_config or "distractor",
        split=config.sampling.hotpotqa.split,
    )
    # NB: `load_hotpotqa_validation` no longer accepts trust_remote_code —
    # HotpotQA is parquet-backed on HF as of 2024.
    logger.info("HotpotQA validation: {} questions parsed", len(hotpot))

    logger.info("downloading 2WikiMultihopQA validation (framolfese repack) ...")
    twiki = load_twiki_validation(
        hf_dataset=config.sampling.twiki.hf_dataset,
        hf_config=config.sampling.twiki.hf_config,
        split=config.sampling.twiki.split,
    )
    logger.info("2WikiMultihopQA validation: {} questions parsed", len(twiki))

    # Schema sanity stats — useful for the Sprint-1 report.
    hotpot_levels: dict[str, int] = {}
    for q in hotpot:
        hotpot_levels[q.level or "<none>"] = hotpot_levels.get(q.level or "<none>", 0) + 1

    twiki_types: dict[str, int] = {}
    for q in twiki:
        twiki_types[q.question_type] = twiki_types.get(q.question_type, 0) + 1

    manifest = {
        "config_version": config.config_version,
        "hotpotqa": {
            "hf_dataset": config.sampling.hotpotqa.hf_dataset,
            "hf_config": config.sampling.hotpotqa.hf_config,
            "split": config.sampling.hotpotqa.split,
            "n_records": len(hotpot),
            "level_counts": hotpot_levels,
        },
        "twiki": {
            "hf_dataset": config.sampling.twiki.hf_dataset,
            "split": config.sampling.twiki.split,
            "n_records": len(twiki),
            "type_counts": twiki_types,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote {}", manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
