"""End-to-end smoke test — Sprints 1-4 + v6 dual-ladder wired together.

For every (question, ladder_type, level, model) cell it does the real work:

  1. Assemble prompts (one per accepted paraphrase + ladder content).
  2. F(x) per paraphrase:
       - MuSiQue (has decomposition): CoT prompt -> graded chain-completion
         fraction (v6 §2). Secondary binary final-answer F stored alongside.
       - HotpotQA / 2Wiki: baseline prompt -> binary NLI-with-gold F (unchanged).
  3. H_sem samples per paraphrase: k samples at T=h_sem_temp.
  4. Pool-cluster the (N x k) responses via DeBERTa NLI (Sprint-4 contract).
  5. Encode prompts + responses with the external mpnet.
  6. POSIX matrix (optional, --include-posix; context family only).
  7. build_metric_tuple -> one row, augmented with v6 columns
     (ladder_family, scoring_mode, final_answer_f_mean).

Two ways to select questions:
  - default: read qids from the paraphrase parquet (HotpotQA / 2Wiki / MuSiQue
    if those qids were paraphrased). Dataset is auto-detected per question.
  - --musique-direct N: sample N MuSiQue questions straight from the loader and
    generate their paraphrase universe live via the Sprint-2 pipeline (cached).
    This runs a full MuSiQue smoke without a pre-existing paraphrase parquet.

Ladder families (--families): "context" (paragraphs) and/or "reasoning"
(decomposition hops, MuSiQue-only). The reasoning ladder feeds hops 0..k-1 as
scaffold, withholding the final hop (v6 §5).

CLI knobs:
  --n-questions N         questions from the paraphrase parquet (default 5)
  --musique-direct N      sample N MuSiQue questions directly instead (default 0=off)
  --paraphrases PATH      paraphrase parquet (default smoke then v1)
  --models KEYS           comma list (default gpt_4o)
  --ladders TYPES         context ladders (default random)
  --families FAMS         context,reasoning (default context)
  --levels LIST           context-ladder levels (default 0,4,10)
  --k-samples K           H_sem samples per prompt (default 3)
  --max-paraphrases M     cap per question (default 10)
  --out PATH              parquet (default data/e2e_metrics.parquet)
  --include-posix         POSIX path on echo-capable models (context only)
  --paraphrase-only       MuSiQue-direct: build+persist paraphrases, then exit
  --own-encoder           ESS_in/rho_u from the model's own hidden states (local)
  --dry-run               print the plan, no model calls
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data import (
    HotpotParagraph,
    MultiHopQuestion,
    load_hotpotqa_validation,
    load_musique_validation,
    load_twiki_validation,
)
from ..ladders import (
    build_distractor_first_ladder,
    build_gold_first_ladder,
    build_random_ladder,
    build_reasoning_ladder,
    render_reasoning_scaffold,
)
from ..ladders.schemas import LadderRow
from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple
from ..models import LLMRequest
from ..models.embedding import encode_texts
from ..models.registry import get_client
from ..models.schemas import ChatMessage, CompletionRequest
from ..prompts import (
    assemble_qa_cot_messages,
    assemble_qa_messages,
    parse_answer_line,
)
from ..prompts.templates.qa_prompt import QA_COT_SYSTEM_PROMPT
from ..scoring import chain_completion_score_batch, f_score_batch


# --------------------------------------------------------------------------- #
# Data loading helpers                                                        #
# --------------------------------------------------------------------------- #


def _load_paraphrase_parquet(paths: list[Path]) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            logger.info("paraphrases source: {}", p)
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"none of these paraphrase parquets exist: {[str(p) for p in paths]}. "
        "Run `make paraphrases-smoke` or `make paraphrases` first, or use "
        "--musique-direct N."
    )


def _accepted_per_q(df: pd.DataFrame, max_per_q: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    accepted = df[df["outcome"] == "accepted"]
    for qid, sub in accepted.groupby("question_id"):
        sub = sub.sort_values("paraphrase_idx")
        out[str(qid)] = sub["text"].head(max_per_q).tolist()
    return out


def _index_questions(config, *, include_musique: bool = True) -> dict[str, MultiHopQuestion]:
    """{id -> MultiHopQuestion} across HotpotQA + 2Wiki (+ MuSiQue if available)."""
    idx: dict[str, MultiHopQuestion] = {}
    logger.info("loading HotpotQA validation ...")
    hp = load_hotpotqa_validation(
        hf_dataset=config.sampling.hotpotqa.hf_dataset,
        hf_config=config.sampling.hotpotqa.hf_config or "distractor",
        split=config.sampling.hotpotqa.split,
    )
    idx.update({q.id: q for q in hp})
    logger.info("loading 2WikiMultihopQA validation ...")
    tw = load_twiki_validation(
        hf_dataset=config.sampling.twiki.hf_dataset,
        hf_config=config.sampling.twiki.hf_config,
        split=config.sampling.twiki.split,
    )
    idx.update({q.id: q for q in tw})

    if include_musique and config.sampling.musique is not None:
        try:
            mq = _load_musique(config)
            idx.update({q.id: q for q in mq})
            logger.info("loaded {} MuSiQue questions into the index", len(mq))
        except Exception as exc:  # noqa: BLE001 — non-fatal for Hotpot-only runs
            logger.warning("MuSiQue not loaded into index: {}", exc)
    return idx


def _load_musique(config) -> list[MultiHopQuestion]:
    m = config.sampling.musique
    return load_musique_validation(
        hf_dataset=m.hf_dataset or None,
        hf_config=m.hf_config,
        split=m.split,
        local_path=m.local_path,
        repo_root=config.repo_root(),
    )


def _pick_musique_questions(config, n: int) -> list[MultiHopQuestion]:
    """Load MuSiQue and pick N questions (highest hop-count first for graded F)."""
    all_mq = _load_musique(config)
    all_mq.sort(key=lambda q: -(q.n_hops or 0))
    return all_mq[:n]


def _pick_musique_questions_stratified(
    config, per_stratum: int, seed: int
) -> list[MultiHopQuestion]:
    """Pick `per_stratum` questions from EACH hop stratum (2-, 3-, 4-hop).

    Deterministic (seeded shuffle) so the paraphrase-prep job and every per-model
    run select the IDENTICAL question set — the paraphrase universe must be
    shared across models for FI_in to be comparable. Used by the full cluster
    run (`--musique-strata`).
    """
    import random

    all_mq = _load_musique(config)
    by_hops: dict[int, list[MultiHopQuestion]] = {}
    for q in all_mq:
        by_hops.setdefault(int(q.n_hops or 0), []).append(q)

    rng = random.Random(seed)
    picked: list[MultiHopQuestion] = []
    for hops in sorted(by_hops):
        if hops < 2:  # MuSiQue is 2-4 hop; ignore anything degenerate
            continue
        group = sorted(by_hops[hops], key=lambda q: q.id)  # stable pre-shuffle order
        rng.shuffle(group)
        chosen = group[:per_stratum]
        picked.extend(chosen)
        logger.info("stratum {}hop: {} available -> picked {}", hops, len(by_hops[hops]), len(chosen))
    return picked


_MUSIQUE_PARAPHRASE_PARQUET = "data/paraphrases_musique.parquet"


def _generate_musique_paraphrases(
    config, questions: list[MultiHopQuestion], max_paraphrases: int
) -> dict[str, list[str]]:
    """Generate (and persist) each question's paraphrase universe.

    Persistence: accepted paraphrases are saved to data/paraphrases_musique.parquet.
    On a later run (e.g. resuming the full pilot), questions already in that
    parquet are loaded from disk instead of regenerated — this skips the
    ~1.5 h of local NLI filtering that paraphrase generation costs on CPU.

    Questions that yield no paraphrases fall back to the original question text
    as a singleton universe (graded chain scoring still produces a non-trivial F).
    """
    from ..paraphrases.pipeline import build_paraphrase_set

    repo_root = config.repo_root()
    parquet_path = repo_root / _MUSIQUE_PARAPHRASE_PARQUET

    # Load any previously-persisted paraphrases.
    persisted: dict[str, list[str]] = {}
    if parquet_path.exists():
        try:
            prev = pd.read_parquet(parquet_path)
            prev = prev[prev["outcome"] == "accepted"]
            for qid, sub in prev.groupby("question_id"):
                persisted[str(qid)] = sub.sort_values("paraphrase_idx")["text"].tolist()
            logger.info("loaded persisted MuSiQue paraphrases for {} questions", len(persisted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} ({}); regenerating", parquet_path, exc)

    paraphrases: dict[str, list[str]] = {}
    new_rows: list[dict] = []
    for q in questions:
        if q.id in persisted and persisted[q.id]:
            paraphrases[q.id] = persisted[q.id][:max_paraphrases]
            logger.info("qid={} paraphrases from cache ({})", q.id, len(paraphrases[q.id]))
            continue
        try:
            pset = build_paraphrase_set(q.id, q.question, config=config, gold_answer=q.answer)
            accepted = list(pset.accepted)
            texts = [ap.text for ap in accepted][:max_paraphrases]
            for idx, ap in enumerate(accepted):
                new_rows.append({
                    "question_id": q.id, "outcome": "accepted",
                    "paraphrase_idx": idx, "text": ap.text,
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("paraphrase gen failed for {}: {}", q.id, exc)
            texts = []
        if not texts:
            logger.warning("qid={} no paraphrases; using original question as singleton", q.id)
            texts = [q.question]
        paraphrases[q.id] = texts

    # Persist newly-generated paraphrases (append to any existing file).
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if parquet_path.exists():
            try:
                df_new = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
            except Exception:  # noqa: BLE001
                pass
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_parquet(parquet_path, index=False)
        logger.info("persisted {} new MuSiQue paraphrase rows -> {}", len(new_rows), parquet_path)

    return paraphrases


# --------------------------------------------------------------------------- #
# Ladder construction per family                                              #
# --------------------------------------------------------------------------- #


_CONTEXT_BUILDERS = {
    "random": build_random_ladder,
    "gold_first": build_gold_first_ladder,
    "distractor_first": build_distractor_first_ladder,
}


def _context_rows(question: MultiHopQuestion, ladder_type: str, levels: list[int]) -> list[LadderRow]:
    builder = _CONTEXT_BUILDERS[ladder_type]
    rows = builder(question)
    return [r for r in rows if r.level in levels]


def _paragraphs_for_row(question: MultiHopQuestion, row: LadderRow) -> list[HotpotParagraph]:
    return [question.paragraphs[i] for i in row.paragraph_indices]


# --------------------------------------------------------------------------- #
# Prompt assembly per cell                                                    #
# --------------------------------------------------------------------------- #


def _assemble_messages(
    question: MultiHopQuestion,
    paraphrase_text: str,
    row: LadderRow,
    *,
    use_cot: bool,
) -> list[ChatMessage]:
    """Build the chat messages for one (paraphrase, ladder-row) cell.

    - context family: paragraphs from the row.
    - reasoning family: scaffold of hops 0..k-1 injected as the context block.
    - use_cot: MuSiQue -> True (graded chain scoring needs the reasoning text);
               HotpotQA -> False (baseline brief-answer).
    """
    if row.ladder_family == "reasoning":
        scaffold = render_reasoning_scaffold(question, row.hops_provided or 0)
        # Reasoning ladder always uses the CoT system prompt; the scaffold is
        # the "known so far" block. Build the user message inline so we can
        # label it clearly as solved sub-steps rather than retrieved context.
        if scaffold:
            user = (
                "Known intermediate steps so far:\n"
                f"{scaffold}\n\n"
                f"Question: {paraphrase_text.strip()}\n\n"
                "Continue the reasoning to the final answer, then end with a "
                "line starting 'Answer:'."
            )
        else:
            user = (
                f"Question: {paraphrase_text.strip()}\n\n"
                "Reason step by step, then end with a line starting 'Answer:'."
            )
        return [
            ChatMessage(role="system", content=QA_COT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user),
        ]

    # context family
    paragraphs = _paragraphs_for_row(question, row)
    if use_cot:
        return assemble_qa_cot_messages(paraphrase_text, paragraphs)
    return assemble_qa_messages(paraphrase_text, paragraphs)


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #


def _sample_response(
    client, model_entry, messages, *, temperature: float, seed: int, purpose: str, max_tokens: int
) -> str:
    req = LLMRequest(
        provider=model_entry.provider,  # type: ignore[arg-type]
        model_id=model_entry.model_id,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        purpose=purpose,
    )
    return client.complete(req).text.strip()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--musique-direct", type=int, default=0,
                        help="Sample N MuSiQue questions directly (live paraphrase gen).")
    parser.add_argument("--musique-strata", type=int, default=0,
                        help="Stratified MuSiQue: N questions PER hop-stratum (2/3/4-hop), "
                             "seeded by config.random_seed so the paraphrase-prep job and all "
                             "per-model runs share ONE question set. Overrides --musique-direct.")
    parser.add_argument(
        "--singleton", action="store_true",
        help="MuSiQue-direct: skip paraphrase generation, use the original "
             "question as a 1-element universe. FI_in degenerates but the "
             "chain-F vs level curve appears in minutes (good for a first look).",
    )
    parser.add_argument("--paraphrases", type=str, default=None)
    parser.add_argument("--models", type=str, default="gpt_4o")
    parser.add_argument("--ladders", type=str, default="random")
    parser.add_argument("--families", type=str, default="context",
                        help="comma list: context,reasoning (reasoning is MuSiQue-only)")
    parser.add_argument("--levels", type=str, default="0,4,10")
    parser.add_argument("--k-samples", type=int, default=3)
    parser.add_argument("--max-paraphrases", type=int, default=10)
    parser.add_argument("--out", type=str, default="data/e2e_metrics.parquet")
    parser.add_argument("--include-posix", action="store_true")
    parser.add_argument(
        "--paraphrase-only", action="store_true",
        help="MuSiQue-direct only: build + persist the paraphrase universe to "
             "data/paraphrases_musique.parquet, then exit. The generator "
             "pre-pass — run once before the per-model eval passes so the "
             "(slow) generator model is loaded only in this step.",
    )
    parser.add_argument(
        "--own-encoder", action="store_true",
        help="Use each model's OWN last-layer hidden states (LocalHFClient."
             "embed_hidden) for ESS_in / rho_u instead of the external mpnet "
             "encoder. Requires provider:local models with has_hidden=true; "
             "falls back to mpnet otherwise. Tags rows encoder_label=own_<key>.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip H_sem clustering / embeddings / POSIX (FI_in + f_mean only). "
             "Removes the biggest CPU cost; ~10x faster per cell.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_cell_rows(
    question: MultiHopQuestion,
    families: list[str],
    ladders: list[str],
    levels: list[int],
) -> list[LadderRow]:
    """All ladder rows to evaluate for one question, across requested families."""
    rows: list[LadderRow] = []
    if "context" in families:
        for lt in ladders:
            rows.extend(_context_rows(question, lt, levels))
    if "reasoning" in families and question.has_decomposition():
        rows.extend(build_reasoning_ladder(question))
    return rows


def main() -> int:
    args = _parse_args()
    configure_logging("e2e_smoke")
    config = load_config()
    repo_root = config.repo_root()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ladders = [l.strip() for l in args.ladders.split(",") if l.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    for m in models:
        if m not in config.models:
            logger.error("unknown model_key {!r}; available: {}", m, list(config.models))
            return 1

    # --- select questions + paraphrases ------------------------------------
    musique_direct = args.musique_direct > 0 or args.musique_strata > 0
    if musique_direct:
        if args.musique_strata > 0:
            logger.info("MuSiQue stratified: {} questions/hop-stratum (seed={})",
                        args.musique_strata, config.random_seed)
            questions = _pick_musique_questions_stratified(
                config, args.musique_strata, config.random_seed
            )
        else:
            logger.info("MuSiQue-direct mode: sampling {} questions", args.musique_direct)
            questions = _pick_musique_questions(config, args.musique_direct)
        if not questions:
            logger.error("no MuSiQue questions loaded; bailing")
            return 1
        # Defer live paraphrase generation until past the dry-run gate. For the
        # plan estimate, assume the configured cap.
        accepted = {q.id: ["<paraphrase>"] * args.max_paraphrases for q in questions}
    else:
        if args.paraphrases:
            paths = [repo_root / args.paraphrases]
        else:
            paths = [
                repo_root / "data" / "paraphrases_smoke.parquet",
                repo_root / "data" / "paraphrases_v1.parquet",
            ]
        df = _load_paraphrase_parquet(paths)
        accepted = _accepted_per_q(df, args.max_paraphrases)
        if not accepted:
            logger.error("no accepted paraphrases found; nothing to do")
            return 1
        qids = list(accepted.keys())[: args.n_questions]
        q_idx = _index_questions(config)
        questions = []
        for qid in qids:
            q = q_idx.get(qid)
            if q is None:
                logger.warning("qid={} not found in datasets; skipping", qid)
                continue
            questions.append(q)
        if not questions:
            logger.error("no valid question records loaded; bailing")
            return 1

    # --- generator pre-pass: build + persist paraphrases, then stop ---------
    if args.paraphrase_only:
        if not musique_direct:
            logger.error("--paraphrase-only requires --musique-direct N")
            return 1
        logger.info("--paraphrase-only: generating paraphrase universe for {} questions", len(questions))
        paras = _generate_musique_paraphrases(config, questions, args.max_paraphrases)
        print(json.dumps({
            "paraphrase_only": True,
            "questions": [q.id for q in questions],
            "per_question_accepted": {k: len(v) for k, v in paras.items()},
            "total_accepted_paraphrases": int(sum(len(v) for v in paras.values())),
            "out": _MUSIQUE_PARAPHRASE_PARQUET,
        }, indent=2))
        return 0

    # --- plan ---------------------------------------------------------------
    cell_rows_by_q = {q.id: _build_cell_rows(q, families, ladders, levels) for q in questions}
    n_cells = sum(len(cell_rows_by_q[q.id]) for q in questions) * len(models)
    n_calls = sum(
        len(cell_rows_by_q[q.id]) * len(models) * len(accepted[q.id]) * (1 + args.k_samples)
        for q in questions
    )
    logger.info(
        "plan: {} cells, ~{} LLM calls (N_q={}, families={}, ladders={}, levels={}, models={}, k={})",
        n_cells, n_calls, len(questions), families, ladders, levels, models, args.k_samples,
    )
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "n_cells": n_cells,
            "estimated_llm_calls": n_calls,
            "questions": [q.id for q in questions],
            "datasets": sorted({q.dataset for q in questions}),
            "families": families,
            "models": models,
            "ladders": ladders,
            "levels": levels,
        }, indent=2))
        return 0

    # MuSiQue-direct: build the paraphrase universe now that we're past the
    # dry-run gate. --singleton skips live generation for a fast first look.
    if musique_direct:
        if args.singleton:
            logger.info("--singleton: using original question as the universe (no paraphrase gen)")
            accepted = {q.id: [q.question] for q in questions}
        else:
            accepted = _generate_musique_paraphrases(config, questions, args.max_paraphrases)
        cell_rows_by_q = {q.id: _build_cell_rows(q, families, ladders, levels) for q in questions}

    answer_max = config.generation.answer_max_tokens
    cot_max = config.generation.cot_max_tokens

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- RESUME: load any already-computed cells so a crash/restart skips them.
    rows_out: list[dict] = []
    done_keys: set[tuple] = set()
    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            rows_out = prev.to_dict("records")
            for r in rows_out:
                done_keys.add(_cell_key(
                    r.get("question_id"), r.get("ladder_family"),
                    r.get("ladder_type_raw"), r.get("level"), r.get("model_key"),
                ))
            logger.info("resume: {} cells already in {} — will skip them", len(done_keys), out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read existing {} for resume ({}); starting fresh", out_path, exc)
            rows_out, done_keys = [], set()

    n_done, n_failed, n_skipped = 0, 0, 0
    for q in questions:
        paraphrases = accepted[q.id]
        for row in cell_rows_by_q[q.id]:
            for model_key in models:
                key = _cell_key(q.id, row.ladder_family, row.ladder_type, row.level, model_key)
                if key in done_keys:
                    n_skipped += 1
                    continue
                t_cell = time.perf_counter()
                try:
                    row_dict = _run_cell(
                        config, q, row, model_key, paraphrases,
                        k_samples=args.k_samples, answer_max=answer_max, cot_max=cot_max,
                        include_posix=args.include_posix, fast=args.fast,
                        own_encoder=args.own_encoder,
                    )
                except Exception:  # noqa: BLE001 — isolate cell failures
                    n_failed += 1
                    logger.exception(
                        "cell FAILED qid={} family={} ladder={} L{} model={} — skipping, run continues",
                        q.id, row.ladder_family, row.ladder_type, row.level, model_key,
                    )
                    continue
                rows_out.append(row_dict)
                done_keys.add(key)
                n_done += 1
                # Per-cell timing so the first cell lets us extrapolate walltime.
                logger.info(
                    "cell {} done in {:.1f}s — qid={} family={} L{} model={}",
                    n_done, time.perf_counter() - t_cell, q.id,
                    row.ladder_family, row.level, model_key,
                )
                # CHECKPOINT after every cell so nothing is ever lost.
                _checkpoint(rows_out, out_path)

    logger.info(
        "done: {} new cells, {} skipped (resume), {} failed. total in parquet: {}",
        n_done, n_skipped, n_failed, len(rows_out),
    )
    if not rows_out:
        logger.error("no cells produced; nothing written")
        return 1

    _print_summary(pd.DataFrame(rows_out), out_path)
    return 0 if n_failed == 0 else 2


def _cell_key(qid, family, ladder_type, level, model_key) -> tuple:
    """Stable identity of a cell for resume/skip."""
    return (str(qid), str(family), str(ladder_type), int(level), str(model_key))


def _checkpoint(rows_out: list[dict], out_path: Path) -> None:
    """Atomically write the running results so a crash never loses progress."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame(rows_out).to_parquet(tmp, index=False)
    import os
    os.replace(tmp, out_path)


def _run_cell(
    config,
    q: MultiHopQuestion,
    row: LadderRow,
    model_key: str,
    paraphrases: list[str],
    *,
    k_samples: int,
    answer_max: int,
    cot_max: int,
    include_posix: bool,
    fast: bool = False,
    own_encoder: bool = False,
) -> dict:
    """Run one (question, ladder-row, model) cell -> a flat result row dict.

    `fast=True` skips the H_sem sampling + pooled NLI clustering + embeddings +
    POSIX. That removes the quadratic-in-(N*k) DeBERTa clustering — by far the
    biggest CPU cost — and keeps the headline metrics (FI_in / AUFI_in / f_mean
    / spread). The output-side metrics (H_sem, FI_out, S_tau, 1-TVD, rho_u,
    ESS_in, POSIX) come back as None/0 in fast mode.
    """
    from ..metrics.h_sem import cluster_responses_pooled

    model_entry = config.models[model_key]
    client = get_client(model_key, config)
    use_cot = q.has_decomposition()
    scoring_mode = "chain_completion" if use_cot else "binary_nli"
    logger.info(
        "cell qid={} ds={} family={} ladder={} level={} model={} N={} mode={} fast={}",
        q.id, q.dataset, row.ladder_family, row.ladder_type, row.level,
        model_key, len(paraphrases), scoring_mode, fast,
    )

    prompt_messages = [_assemble_messages(q, p, row, use_cot=use_cot) for p in paraphrases]
    prompt_user_texts = [m[1].content for m in prompt_messages]
    gen_max = cot_max if use_cot else answer_max

    # --- F(x) at T=0 -------------------------------------------------------
    f_responses: list[str] = []
    for i, msgs in enumerate(prompt_messages):
        f_responses.append(_sample_response(
            client, model_entry, msgs, temperature=0.0, seed=42,
            purpose=f"e2e_f::{q.id}::{row.ladder_type}::L{row.level}::{model_key}",
            max_tokens=gen_max,
        ))

    # Reasoning ladder: OR-credit hops the scaffold already supplies (P0-1).
    scaffold_text = (
        render_reasoning_scaffold(q, row.hops_provided or 0)
        if row.ladder_family == "reasoning" else None
    )
    if use_cot:
        f_scores = chain_completion_score_batch(
            q.question_decomposition, f_responses, config=config, scaffold_text=scaffold_text,
        )
        final_answers = [parse_answer_line(r) for r in f_responses]
        final_binary = f_score_batch(q.answer, final_answers, config=config)
        final_answer_f_mean = float(np.mean(final_binary)) if final_binary else None
        logger.info(
            "  chain F: mean={:.3f} values={}  final-answer F mean={:.3f}",
            float(np.mean(f_scores)) if f_scores else 0.0,
            [round(s, 2) for s in f_scores[:5]], final_answer_f_mean or 0.0,
        )
    else:
        f_scores = [float(x) for x in f_score_batch(q.answer, f_responses, config=config)]
        final_answer_f_mean = float(np.mean(f_scores)) if f_scores else None
        logger.info("  binary F: pass={}/{}", int(sum(f_scores)), len(f_scores))

    n = len(paraphrases)
    encoder_label = "external_mpnet"
    if fast:
        # Skip the quadratic H_sem path entirely.
        cluster_assignments = {}
        prompt_embeddings = np.zeros((n, 1), dtype=np.float32)
        response_embeddings = {}
        posix_log_p = posix_lengths = None
    else:
        # --- H_sem samples ------------------------------------------------
        responses_per_paraphrase: dict[int, list[str]] = {}
        for i, msgs in enumerate(prompt_messages):
            samples: list[str] = []
            for kk in range(k_samples):
                samples.append(_sample_response(
                    client, model_entry, msgs,
                    temperature=config.h_sem.sampling_temperature,
                    seed=10000 + i * 100 + kk,
                    purpose=f"e2e_hsem::{q.id}::{row.ladder_type}::L{row.level}::{model_key}::s{kk}",
                    max_tokens=gen_max,
                ))
            responses_per_paraphrase[i] = samples

        cluster_assignments = cluster_responses_pooled(responses_per_paraphrase, config=config)

        # Own-encoder path (Sprint 6): use the model's OWN last-layer hidden
        # states for ESS_in / rho_u instead of the external mpnet proxy. Only
        # for provider:local models exposing embed_hidden; mpnet otherwise.
        use_own = own_encoder and model_entry.has_hidden and hasattr(client, "embed_hidden")
        if use_own:
            encoder_label = f"own_{model_key}"
            prompt_embeddings = client.embed_hidden(prompt_user_texts)
            response_embeddings = {
                i: client.embed_hidden(samples) for i, samples in responses_per_paraphrase.items()
            }
        else:
            prompt_embeddings = encode_texts(prompt_user_texts, config=config)
            response_embeddings = {}
            for i, samples in responses_per_paraphrase.items():
                response_embeddings[i] = encode_texts(samples, config=config)

        posix_log_p = posix_lengths = None
        if include_posix and model_entry.echo_completions and row.ladder_family == "context":
            posix_log_p, posix_lengths = _posix_matrix(
                client, model_entry, prompt_messages, f_responses, q, row
            )

    tup = build_metric_tuple(
        question_id=q.id,
        ladder_type=row.ladder_type if row.ladder_family == "context" else "random",
        level=row.level,
        model_key=model_key,
        scores=[float(x) for x in f_scores],
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_embeddings,
        response_embeddings=response_embeddings,
        posix_log_p=posix_log_p,
        posix_lengths=posix_lengths,
        encoder_label=encoder_label,
    )
    row_dict = tup.model_dump()
    # v6 columns added AFTER model_dump so metrics/ stays untouched.
    row_dict["dataset"] = q.dataset
    row_dict["ladder_family"] = row.ladder_family
    row_dict["ladder_type_raw"] = row.ladder_type
    row_dict["scoring_mode"] = scoring_mode
    row_dict["final_answer_f_mean"] = final_answer_f_mean
    row_dict["n_hops"] = q.n_hops
    return row_dict


def _posix_matrix(client, model_entry, prompt_messages, f_responses, q, row):
    """Echo-mode POSIX matrix (unchanged from the v3 path). Context family only."""
    n = len(prompt_messages)
    log_p = np.zeros((n, n))
    lengths = np.zeros(n)
    for j, yj in enumerate(f_responses):
        lengths[j] = max(1, len(yj.split()))
        for i, msgs_i in enumerate(prompt_messages):
            rendered = "\n".join(f"{m.role}: {m.content}" for m in msgs_i)
            full = rendered + "\nassistant: " + yj
            try:
                resp = client.score_continuation(CompletionRequest(
                    provider=model_entry.provider,
                    model_id=model_entry.model_id,
                    prompt=full, max_tokens=0, echo=True, logprobs=1,
                    temperature=0.0,
                    purpose=f"e2e_posix::{q.id}::{row.ladder_type}::L{row.level}",
                ))
                if resp.token_logprobs:
                    tail = resp.token_logprobs[-int(lengths[j]):]
                    log_p[i, j] = sum(t.logprob for t in tail)
                else:
                    log_p[i, j] = math.nan
            except Exception as exc:  # noqa: BLE001
                logger.warning("POSIX echo failed at (i={}, j={}): {}", i, j, exc)
                log_p[i, j] = math.nan
    if not np.isnan(log_p).any():
        return log_p, lengths
    logger.warning("POSIX matrix has NaN — leaving posix_psi=None")
    return None, None


def _print_summary(out_df: pd.DataFrame, out_path: Path) -> None:
    print()
    print("=" * 100)
    print("END-TO-END METRIC TUPLES")
    print("=" * 100)
    cols = [
        "question_id", "dataset", "ladder_family", "ladder_type_raw", "level", "model_key",
        "f_mean", "final_answer_f_mean", "aufi_in", "spread", "h_sem_mean",
        "n_paraphrases",
    ]
    cols = [c for c in cols if c in out_df.columns]
    print(out_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Regression check for the v3 step-function bug: graded scores must appear.
    graded_present = False
    if "f_mean" in out_df.columns:
        fvals = out_df["f_mean"].dropna().tolist()
        graded_present = any(0.0 < v < 1.0 for v in fvals)
    print()
    print("=" * 100)
    print(f"GRADED-F CHECK: at least one f_mean strictly in (0,1)?  -> {graded_present}")
    if not graded_present:
        print("  WARNING: all f_mean are 0 or 1 — the step-function bug is NOT fixed for this run.")
        print("  (Expected for HotpotQA binary cells; MuSiQue chain cells should be graded.)")

    bad = []
    for _, r in out_df.iterrows():
        if r.get("n_paraphrases", 0) < 2:
            continue
        for col in ("aufi_in", "fi_out_mean", "s_tau_mean", "consistency_mean",
                    "spread", "variation_ratio", "h_sem_mean", "h_sem_var"):
            v = r.get(col)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                bad.append(f"{r['question_id']} {r.get('ladder_family')} L{r['level']} {col}={v!r}")

    print()
    print(json.dumps({
        "cells_run": int(len(out_df)),
        "datasets": sorted(out_df["dataset"].unique().tolist()) if "dataset" in out_df else [],
        "families": sorted(out_df["ladder_family"].unique().tolist()) if "ladder_family" in out_df else [],
        "graded_f_present": bool(graded_present),
        "plausibility_warnings": len(bad),
        "out_path": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
