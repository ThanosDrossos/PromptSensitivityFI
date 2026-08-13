"""R9 — the run manifest: everything a third party needs to reproduce the grid.

The review (2026-08-06 §3.10) found the 150-question selection, seeds and filter
funnel were recoverable only by re-executing the loader. This emits them as a
single JSON artifact, plus content hashes of the key inputs/outputs so any
downstream drift is detectable.

    uv run python -m prompt_sensitivity.scripts.make_run_manifest
"""

from __future__ import annotations

import hashlib
import json
import sys

from loguru import logger

from ..config import load_config
from ..data.load_ambigqa import load_ambigqa
from ..specificity.build_levels import target_in_evidence
from .run_specificity import load_spec_rows

_KEY_FILES = [
    "data/paraphrases_ambigqa.parquet",
    "data/specificity_v3_qwen_2_5_7b.parquet",
    "data/specificity_v3_llama_3_1_8b.parquet",
    "data/specificity_v3_mistral_7b_v03.parquet",
    "data/union_gold_qwen_2_5_7b.parquet",
    "data/union_gold_llama_3_1_8b.parquet",
    "data/union_gold_mistral_7b_v03.parquet",
    "data/paraphrases_width_narrow.parquet",
    "data/paraphrases_width_wide.parquet",
    "data/paraphrases_width_swap.parquet",
    "figures/v3_metric_corr.npy",
    "uv.lock",
]


def _sha256(path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    config = load_config()
    root = config.repo_root()
    acfg = config.sampling.ambigqa
    spec = config.specificity

    # reproduce the selection funnel exactly as the driver does
    ambiguous = [q for q in load_ambigqa(
        hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
        min_interpretations=acfg.min_interpretations,
        include_single_answer_anchor=acfg.include_single_answer_anchor,
    ) if q.is_ambiguous()]
    n_ambiguous = len(ambiguous)
    seed = spec.target_seed if spec is not None else config.random_seed
    covered = [q for q in ambiguous if target_in_evidence(q, seed=seed)]

    rows, questions, mode, ev_cap = load_spec_rows(config, n_questions=150)

    manifest = {
        "created": "2026-08-08",
        "dataset": {
            "hf_dataset": acfg.hf_dataset, "hf_config": acfg.hf_config,
            "split": acfg.split, "min_interpretations": acfg.min_interpretations,
        },
        "selection_funnel": {
            "split_rows": 2002,
            "ambiguous_min2_interpretations": n_ambiguous,
            "evidence_coverage_pass": len(covered),
            "selected_first_n": len(questions),
            "selection_rule": "first N of the deterministic loader order after the "
                              "evidence-coverage filter (dataset order; no stratification)",
        },
        "seeds": {
            "target_interpretation_seed": seed,
            "random_seed": config.random_seed,
            "target_rule": "sha256(question_id::seed) % m0",
            "hsem_sample_seed_layout": "10000 + paraphrase_idx*100 + sample_idx",
            "t0_seed": 42,
        },
        "design": {
            "levels": sorted({r.spec_level for r in rows}),
            "n_paraphrases_cap": 10, "k_samples": config.h_sem.n_samples_per_prompt,
            "context_mode": mode, "evidence_max_chars": ev_cap,
            "scoring": {"method": config.scoring.method,
                        "entail_threshold": config.scoring.entail_threshold,
                        "contradict_threshold": config.scoring.contradict_threshold,
                        "gold_sets": "target (pinned interpretation) AND union "
                                     "(all interpretations) — dual scoring since R1"},
        },
        "models": {k: {"provider": m.provider, "model_id": m.model_id}
                   for k, m in config.models.items()},
        "paraphrase_generator": {
            "model": config.paraphrases.generator_model,
            "temperature": config.paraphrases.generator_temperature,
            "templates": list(config.paraphrases.templates),
            "nli_model": config.paraphrases.nli.model,
            "nli_threshold": config.paraphrases.nli.bidirectional_threshold,
            "nli_fallback": config.paraphrases.nli.fallback_threshold,
            "judge_model": config.paraphrases.constraint_filter.judge_model,
            "note": "width-dial arms override roles/temperature/generator/judge "
                    "via prompts.WIDTH_ARMS; gates identical across arms",
        },
        "question_ids": sorted(q.id for q in questions),
        "file_hashes_sha256": {f: _sha256(root / f) for f in _KEY_FILES},
    }
    out = root / "data/run_manifest.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("wrote {} ({} questions, {} hashed files)", out.name,
                len(manifest["question_ids"]),
                sum(1 for v in manifest["file_hashes_sha256"].values() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
