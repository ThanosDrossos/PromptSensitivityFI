"""AmbigQA specificity driver (pivot spec §8) — thin, reuses the e2e cell helpers.

The manipulated axis is QUESTION SPECIFICITY: each AmbigQA question yields two
CLOSED-BOOK cells (spec_level 0 = ambiguous Q, 1 = disambiguated Q_i; empty
paragraph list -> `_assemble_messages` builds the no-context prompt). The
scoring gold a_i is FIXED across both levels (build_levels guardrail); scoring
is binary NLI-with-gold, OR'd over the interpretation's answer variants
(`f_score_batch_multi_gold`). FI_spec = log2(m0/m_valid) is attached from
`metrics.fi_spec` — dataset-side, not orchestrator-side.

Per-cell checkpoint to --out (resume key: question_id, spec_level, model_key).
Paraphrase universes are built per (question, level) over that level's question
text and persisted to data/paraphrases_ambigqa.parquet.

Cluster-only in practice (provider:local models); run via cluster sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.ambigqa_schemas import AmbigQuestion
from ..data.load_ambigqa import load_ambigqa
from ..ladders.schemas import LadderRow
from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple
from ..metrics.fi_spec import fi_spec_bits
from ..metrics.h_sem import cluster_responses_pooled
from ..models.embedding import encode_texts
from ..models.registry import get_client
from ..scoring.nli_with_gold import f_score_batch_multi_gold
from ..specificity.build_levels import SpecRow, build_spec_levels
from .e2e_smoke import _assemble_messages, _checkpoint, _clustering_inputs, _sample_response


_AMBIGQA_PARAPHRASE_PARQUET = "data/paraphrases_ambigqa.parquet"


# --------------------------------------------------------------------------- #
# Adapters (spec §8.5): closed-book cell via the EXISTING machinery            #
# --------------------------------------------------------------------------- #


class _SpecQuestionView:
    """Duck-typed stand-in for MultiHopQuestion, restricted to what the reused
    cell helpers touch on the closed-book path. MultiHopQuestion itself is NOT
    changed (scope guard §12): its validators demand paragraphs, which the
    specificity cells deliberately do not have.
    """

    def __init__(self, row: SpecRow) -> None:
        self.id = row.question_id
        self.dataset = "ambigqa"
        self.question = row.question_text
        self.answer = row.target_answers[0]   # primary gold (multi-gold OR in scoring)
        self.paragraphs: list = []            # closed book
        self.question_decomposition: list = []
        self.n_hops = None

    def has_decomposition(self) -> bool:      # -> binary NLI path, no CoT
        return False


def _ladder_row_for(row: SpecRow) -> LadderRow:
    """Minimal context-family LadderRow with NO paragraphs => closed-book prompt."""
    return LadderRow(
        question_id=row.question_id,
        ladder_type="random",
        ladder_family="context",
        level_idx=row.spec_level,
        level=row.spec_level,
        paragraph_indices=[],
        paragraph_titles=[],
        gold_count=0,
    )


# --------------------------------------------------------------------------- #
# Paraphrase universes per (question, spec level)                              #
# --------------------------------------------------------------------------- #


def _generate_spec_paraphrases(
    config, rows: list[SpecRow], max_paraphrases: int
) -> dict[tuple[str, int], list[str]]:
    """Build (and persist) the paraphrase universe per (question_id, spec_level).

    Each LEVEL has its own universe over its own question text — the NLI
    equivalence filter keeps specificity constant WITHIN a level. Cached in
    data/paraphrases_ambigqa.parquet keyed by (question_id, spec_level); a
    question that yields nothing falls back to its own text as a singleton.
    """
    from ..paraphrases.pipeline import build_paraphrase_set

    parquet_path = config.repo_root() / _AMBIGQA_PARAPHRASE_PARQUET

    persisted: dict[tuple[str, int], list[str]] = {}
    if parquet_path.exists():
        try:
            prev = pd.read_parquet(parquet_path)
            prev = prev[prev["outcome"] == "accepted"]
            for (qid, lvl), sub in prev.groupby(["question_id", "spec_level"]):
                persisted[(str(qid), int(lvl))] = (
                    sub.sort_values("paraphrase_idx")["text"].tolist()
                )
            logger.info("loaded persisted AmbigQA paraphrases for {} cells", len(persisted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} ({}); regenerating", parquet_path, exc)

    out: dict[tuple[str, int], list[str]] = {}
    new_rows: list[dict] = []
    for row in rows:
        key = (row.question_id, row.spec_level)
        if key in persisted and persisted[key]:
            out[key] = persisted[key][:max_paraphrases]
            logger.info("qid={} L{} paraphrases from cache ({})",
                        row.question_id, row.spec_level, len(out[key]))
            continue
        try:
            pset = build_paraphrase_set(
                f"{row.question_id}::L{row.spec_level}",
                row.question_text,
                config=config,
                gold_answer=row.target_answers[0],
            )
            texts = [ap.text for ap in pset.accepted][:max_paraphrases]
            for idx, text in enumerate(texts):
                new_rows.append({
                    "question_id": row.question_id, "spec_level": row.spec_level,
                    "outcome": "accepted", "paraphrase_idx": idx, "text": text,
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("paraphrase gen failed for {} L{}: {}",
                           row.question_id, row.spec_level, exc)
            texts = []
        if not texts:
            logger.warning("qid={} L{} no paraphrases; singleton fallback",
                           row.question_id, row.spec_level)
            texts = [row.question_text]
        out[key] = texts

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if parquet_path.exists():
            try:
                df_new = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
            except Exception:  # noqa: BLE001
                pass
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_parquet(parquet_path, index=False)
        logger.info("persisted {} new AmbigQA paraphrase rows -> {}", len(new_rows), parquet_path)
    return out


# --------------------------------------------------------------------------- #
# One closed-book specificity cell                                             #
# --------------------------------------------------------------------------- #


def _run_spec_cell(
    config,
    row: SpecRow,
    model_key: str,
    paraphrases: list[str],
    *,
    k_samples: int,
    fast: bool,
) -> dict:
    """One (SpecRow, model) cell -> flat result row. Mirrors the e2e binary
    (non-CoT) path but scores with the multi-gold OR and attaches FI_spec."""
    model_entry = config.models[model_key]
    client = get_client(model_key, config)
    view = _SpecQuestionView(row)
    lrow = _ladder_row_for(row)
    gen_max = config.generation.answer_max_tokens
    logger.info("cell qid={} spec_level={} model={} N={} m0={} m_valid={}",
                row.question_id, row.spec_level, model_key,
                len(paraphrases), row.m0, row.m_valid)

    prompt_messages = [
        _assemble_messages(view, p, lrow, use_cot=False) for p in paraphrases
    ]
    prompt_user_texts = [m[1].content for m in prompt_messages]

    # --- F(x) at T=0, multi-gold OR over the target's answer variants --------
    f_responses = [
        _sample_response(
            client, model_entry, msgs, temperature=0.0, seed=42,
            purpose=f"spec_f::{row.question_id}::L{row.spec_level}::{model_key}",
            max_tokens=gen_max,
        )
        for msgs in prompt_messages
    ]
    f_scores = [float(x) for x in f_score_batch_multi_gold(
        row.target_answers, f_responses, config=config
    )]
    f_scores_perm = [float(x) for x in f_score_batch_multi_gold(
        row.target_answers, f_responses, config=config, permissive=True
    )]
    logger.info("  F: mean={:.3f} (permissive {:.3f})",
                float(np.mean(f_scores)) if f_scores else 0.0,
                float(np.mean(f_scores_perm)) if f_scores_perm else 0.0)

    n = len(paraphrases)
    if fast:
        cluster_assignments: dict[int, list[int]] = {}
        prompt_embeddings = np.zeros((n, 1), dtype=np.float32)
        response_embeddings: dict[int, np.ndarray] = {}
    else:
        responses_per_paraphrase: dict[int, list[str]] = {}
        for i, msgs in enumerate(prompt_messages):
            responses_per_paraphrase[i] = [
                _sample_response(
                    client, model_entry, msgs,
                    temperature=config.h_sem.sampling_temperature,
                    seed=10000 + i * 100 + kk,
                    purpose=(f"spec_hsem::{row.question_id}::L{row.spec_level}"
                             f"::{model_key}::s{kk}"),
                    max_tokens=gen_max,
                )
                for kk in range(k_samples)
            ]
        clustering_inputs = _clustering_inputs(
            responses_per_paraphrase, config.h_sem.cluster_on
        )
        cluster_assignments = cluster_responses_pooled(clustering_inputs, config=config)
        prompt_embeddings = encode_texts(prompt_user_texts, config=config)
        response_embeddings = {
            i: encode_texts(samples, config=config)
            for i, samples in responses_per_paraphrase.items()
        }

    tup = build_metric_tuple(
        question_id=row.question_id,
        ladder_type="random",
        level=row.spec_level,
        model_key=model_key,
        scores=f_scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_embeddings,
        response_embeddings=response_embeddings,
        posix_log_p=None,
        posix_lengths=None,
        encoder_label="external_mpnet",
        config=config,
        fi_spec=fi_spec_bits(row.m0, row.m_valid),
        spec_level=row.spec_level,
        m_valid=row.m_valid,
        m0=row.m0,
        target_idx=row.target_idx,
    )
    row_dict = tup.model_dump()
    # Post-dump extras, e2e pattern (metrics/ untouched).
    row_dict["dataset"] = "ambigqa"
    row_dict["question_text"] = row.question_text
    row_dict["f_mean_permissive"] = float(np.mean(f_scores_perm)) if f_scores_perm else None
    return row_dict


def _cell_key(qid, spec_level, model_key) -> tuple:
    return (str(qid), int(spec_level), str(model_key))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=None,
                        help="default: config.sampling.ambigqa.n_questions (50)")
    parser.add_argument("--models", type=str, default="qwen_2_5_7b")
    parser.add_argument("--k-samples", type=int, default=None,
                        help="H_sem samples/prompt; default config.h_sem.n_samples_per_prompt")
    parser.add_argument("--max-paraphrases", type=int, default=10)
    parser.add_argument("--fast", action="store_true",
                        help="skip H_sem sampling/clustering/embeddings "
                             "(FI_in + accuracy + FI_spec only)")
    parser.add_argument("--out", type=str, default="data/specificity_metrics.parquet")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging("run_specificity")
    config = load_config()
    repo_root = config.repo_root()

    acfg = config.sampling.ambigqa
    if acfg is None:
        logger.error("config.sampling.ambigqa missing — add the block from the pivot spec §9")
        return 1
    spec_cfg = config.specificity
    target_seed = spec_cfg.target_seed if spec_cfg is not None else config.random_seed
    n_questions = args.n_questions if args.n_questions is not None else acfg.n_questions
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in config.models:
            logger.error("unknown model_key {!r}; available: {}", m, list(config.models))
            return 1
    k_samples = args.k_samples if args.k_samples is not None else config.h_sem.n_samples_per_prompt
    if not args.fast and k_samples < 5:
        logger.warning("H_sem k={} (<5): near-zero semantic-entropy resolution", k_samples)

    # --- questions -> spec rows ---------------------------------------------
    questions: list[AmbigQuestion] = [
        q for q in load_ambigqa(
            hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
            min_interpretations=acfg.min_interpretations,
            include_single_answer_anchor=acfg.include_single_answer_anchor,
        )
        if q.is_ambiguous()
    ][:n_questions]
    if not questions:
        logger.error("no ambiguous AmbigQA questions loaded; bailing")
        return 1
    rows: list[SpecRow] = []
    for q in questions:
        rows.extend(build_spec_levels(q, seed=target_seed))

    n_cells = len(rows) * len(models)
    per_cell_calls = args.max_paraphrases * (1 + (0 if args.fast else k_samples))
    logger.info("plan: {} questions -> {} spec rows -> {} cells, ~{} LLM calls (k={}, fast={})",
                len(questions), len(rows), n_cells, n_cells * per_cell_calls,
                k_samples, args.fast)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "n_questions": len(questions), "n_cells": n_cells,
            "estimated_llm_calls": n_cells * per_cell_calls,
            "models": models, "levels": sorted({r.spec_level for r in rows}),
            "m0_values": sorted({r.m0 for r in rows}),
        }, indent=2))
        return 0

    # --- paraphrase universes (per question x level) -------------------------
    paraphrases = _generate_spec_paraphrases(config, rows, args.max_paraphrases)

    # --- cells with per-cell checkpoint + resume ------------------------------
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    done: set[tuple] = set()
    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            rows_out = prev.to_dict("records")
            done = {
                _cell_key(r.get("question_id"), r.get("spec_level"), r.get("model_key"))
                for r in rows_out
            }
            logger.info("resume: {} cells already in {}", len(done), out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} for resume ({}); starting fresh", out_path, exc)
            rows_out, done = [], set()

    n_done = n_failed = n_skipped = 0
    for row in rows:
        for model_key in models:
            key = _cell_key(row.question_id, row.spec_level, model_key)
            if key in done:
                n_skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                row_dict = _run_spec_cell(
                    config, row, model_key,
                    paraphrases[(row.question_id, row.spec_level)],
                    k_samples=k_samples, fast=args.fast,
                )
            except Exception:  # noqa: BLE001 — isolate cell failures
                n_failed += 1
                logger.exception("cell FAILED qid={} L{} model={} — continuing",
                                 row.question_id, row.spec_level, model_key)
                continue
            rows_out.append(row_dict)
            done.add(key)
            n_done += 1
            logger.info("cell {} done in {:.1f}s — qid={} L{} model={}",
                        n_done, time.perf_counter() - t0,
                        row.question_id, row.spec_level, model_key)
            _checkpoint(rows_out, out_path)

    logger.info("done: {} new, {} skipped (resume), {} failed. total: {}",
                n_done, n_skipped, n_failed, len(rows_out))
    if not rows_out:
        logger.error("no cells produced")
        return 1
    _print_summary(pd.DataFrame(rows_out))
    return 0 if n_failed == 0 else 2


def _print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("SPECIFICITY PIVOT — per-level means")
    print("=" * 90)
    cols = [c for c in ["f_mean", "f_mean_permissive", "aufi_in", "fi_out_mean",
                        "h_sem_mean", "a_q", "fi_spec"] if c in df.columns]
    print(df.groupby("spec_level")[cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print()


if __name__ == "__main__":
    sys.exit(main())
