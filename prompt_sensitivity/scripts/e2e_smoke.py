"""End-to-end smoke test — Sprints 1-4 wired together on N questions.

Reads paraphrases from the existing parquet (smoke or v1), builds ladders
on demand for those questions, and for every (q, ladder_type, level,
model) cell does the real Sprint-5 work:

  1. Assemble N prompts (one per accepted paraphrase + ladder paragraphs)
  2. F(x) per paraphrase: sample at T=0, score with NLI-with-gold
  3. H_sem samples per paraphrase: k samples at T=h_sem_temp
  4. Pool-cluster the (N x k) responses via DeBERTa NLI
  5. Encode prompts + responses with the external mpnet
  6. POSIX matrix: SKIPPED by default (echo path requires separate
     verification; pass --include-posix to enable on echo-capable models)
  7. build_metric_tuple -> one row in data/e2e_metrics.parquet

CLI knobs (all optional):
  --n-questions N      : default 5; uses first N from paraphrase parquet
  --paraphrases PATH   : default data/paraphrases_smoke.parquet (fallback
                         to data/paraphrases_v1.parquet)
  --models KEYS        : comma-separated model_keys; default gpt_4o
                         (the cheapest one)
  --ladders TYPES      : comma-separated; default random
  --levels LIST        : comma-separated ints; default 0,4,10
  --k-samples K        : H_sem samples per prompt; default 3 (brief says 10)
  --max-paraphrases M  : cap per question; default 10 (brief target is 30)
  --out PATH           : default data/e2e_metrics.parquet
  --include-posix      : enable POSIX echo path on echo-capable models
  --dry-run            : skip the gateway calls; report what WOULD run

Cost estimate (defaults):
  cells = N x |ladders| x |levels| x |models| = 5 x 1 x 3 x 1 = 15
  per cell = M paraphrases x (1 F-call + k H_sem samples)
           = 10 x 4 = 40 LLM calls
  total = ~600 LLM calls @ ~$0.0005 = ~$0.30 (kit.gpt-4.1)
  time  = ~15-20 min (network-bound, plus DeBERTa load + clustering)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data import (
    HotpotParagraph,
    MultiHopQuestion,
    load_hotpotqa_validation,
    load_twiki_validation,
)
from ..ladders import build_random_ladder, build_gold_first_ladder, build_distractor_first_ladder
from ..ladders.schemas import LadderRow
from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple
from ..models import LLMRequest
from ..models.embedding import encode_texts
from ..models.registry import get_client
from ..models.schemas import CompletionRequest
from ..prompts import assemble_qa_messages
from ..scoring import f_score_batch


# --------------------------------------------------------------------------- #
# Data loading helpers                                                        #
# --------------------------------------------------------------------------- #


def _load_paraphrase_parquet(paths: list[Path]) -> pd.DataFrame:
    """Return the first existing parquet from the prioritised list, or fail."""
    for p in paths:
        if p.exists():
            logger.info("paraphrases source: {}", p)
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"none of these paraphrase parquets exist: {[str(p) for p in paths]}. "
        "Run `make paraphrases-smoke` or `make paraphrases` first."
    )


def _accepted_per_q(df: pd.DataFrame, max_per_q: int) -> dict[str, list[str]]:
    """Return {qid -> [paraphrase texts]} for accepted rows, capped at max_per_q.

    A question's `dropped=True` flag means it COULDN'T reach the brief's
    target of 30, but rows marked outcome='accepted' on a dropped question
    are still valid paraphrases — we just have fewer of them than ideal.
    The E2E smoke happily processes whatever's available; the FI_in
    estimator's denominator is `|U_q|` so smaller N just means coarser
    resolution, not an error.
    """
    out: dict[str, list[str]] = {}
    accepted = df[df["outcome"] == "accepted"]
    for qid, sub in accepted.groupby("question_id"):
        sub = sub.sort_values("paraphrase_idx")
        out[str(qid)] = sub["text"].head(max_per_q).tolist()
    return out


def _index_questions(config) -> dict[str, MultiHopQuestion]:
    """Build {id -> MultiHopQuestion} across HotpotQA + 2Wiki validation."""
    logger.info("loading HotpotQA validation ...")
    hp = load_hotpotqa_validation(
        hf_dataset=config.sampling.hotpotqa.hf_dataset,
        hf_config=config.sampling.hotpotqa.hf_config or "distractor",
        split=config.sampling.hotpotqa.split,
    )
    logger.info("loading 2WikiMultihopQA validation ...")
    tw = load_twiki_validation(
        hf_dataset=config.sampling.twiki.hf_dataset,
        hf_config=config.sampling.twiki.hf_config,
        split=config.sampling.twiki.split,
    )
    idx: dict[str, MultiHopQuestion] = {}
    idx.update({q.id: q for q in hp})
    idx.update({q.id: q for q in tw})
    return idx


# --------------------------------------------------------------------------- #
# Ladder lookup                                                              #
# --------------------------------------------------------------------------- #


_LADDER_BUILDERS = {
    "random": build_random_ladder,
    "gold_first": build_gold_first_ladder,
    "distractor_first": build_distractor_first_ladder,
}


def _ladder_rows_for(question: MultiHopQuestion, ladder_type: str) -> list[LadderRow]:
    builder = _LADDER_BUILDERS.get(ladder_type)
    if builder is None:
        raise ValueError(f"unknown ladder_type {ladder_type!r}")
    return builder(question)


def _paragraphs_at_level(
    question: MultiHopQuestion, ladder_rows: list[LadderRow], level: int
) -> list[HotpotParagraph]:
    """Return the paragraphs (HotpotParagraph) selected at this ladder level."""
    row = next((r for r in ladder_rows if r.level == level), None)
    if row is None:
        raise ValueError(f"no ladder row at level={level}")
    return [question.paragraphs[i] for i in row.paragraph_indices]


# --------------------------------------------------------------------------- #
# Sampling helpers                                                            #
# --------------------------------------------------------------------------- #


def _sample_response(
    client,
    model_entry,
    messages,
    *,
    temperature: float,
    seed: int,
    purpose: str,
    max_tokens: int = 128,
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
    parser.add_argument(
        "--paraphrases",
        type=str,
        default=None,
        help="Paraphrase parquet path. Default tries smoke then v1.",
    )
    parser.add_argument("--models", type=str, default="gpt_4o")
    parser.add_argument("--ladders", type=str, default="random")
    parser.add_argument("--levels", type=str, default="0,4,10")
    parser.add_argument("--k-samples", type=int, default=3)
    parser.add_argument("--max-paraphrases", type=int, default=10)
    parser.add_argument("--out", type=str, default="data/e2e_metrics.parquet")
    parser.add_argument("--include-posix", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging("e2e_smoke")
    config = load_config()
    repo_root = config.repo_root()

    # --- 1. Load paraphrases -----------------------------------------------
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
    logger.info(
        "{} questions selected; paraphrase counts: {}",
        len(qids),
        {qid: len(accepted[qid]) for qid in qids},
    )

    # --- 2. Load question records (paragraphs + gold answer) ---------------
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

    # --- 3. Parse cell knobs ----------------------------------------------
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ladders = [l.strip() for l in args.ladders.split(",") if l.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    for m in models:
        if m not in config.models:
            logger.error(
                "unknown model_key {!r}; available: {}", m, list(config.models)
            )
            return 1
    for l in ladders:
        if l not in _LADDER_BUILDERS:
            logger.error("unknown ladder_type {!r}", l)
            return 1

    n_cells = len(questions) * len(ladders) * len(levels) * len(models)
    n_calls_estimate = sum(
        len(ladders) * len(levels) * len(models) * len(accepted[q.id]) * (1 + args.k_samples)
        for q in questions
    )
    logger.info(
        "plan: {} cells, ~{} LLM calls "
        "(N_q={}, ladders={}, levels={}, models={}, k_samples={}, max_paraphrases={})",
        n_cells,
        n_calls_estimate,
        len(questions),
        ladders,
        levels,
        models,
        args.k_samples,
        args.max_paraphrases,
    )
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "n_cells": n_cells,
            "estimated_llm_calls": n_calls_estimate,
            "questions": [q.id for q in questions],
            "models": models,
            "ladders": ladders,
            "levels": levels,
        }, indent=2))
        return 0

    # --- 4. Pre-build ladders -----------------------------------------------
    ladder_cache: dict[tuple[str, str], list[LadderRow]] = {}
    for q in questions:
        for lt in ladders:
            ladder_cache[(q.id, lt)] = _ladder_rows_for(q, lt)

    # --- 5. Run cells -------------------------------------------------------
    from ..metrics.h_sem import cluster_responses_pooled

    tuples: list[dict] = []
    for q in questions:
        paraphrases = accepted[q.id]
        for ladder_type in ladders:
            ladder_rows = ladder_cache[(q.id, ladder_type)]
            for level in levels:
                paragraphs = _paragraphs_at_level(q, ladder_rows, level)
                for model_key in models:
                    model_entry = config.models[model_key]
                    client = get_client(model_key, config)
                    logger.info(
                        "cell qid={} ladder={} level={} model={} N={} k={}",
                        q.id, ladder_type, level, model_key,
                        len(paraphrases), args.k_samples,
                    )

                    # --- prompts ------------------------------------------
                    prompt_messages = [
                        assemble_qa_messages(p, paragraphs) for p in paraphrases
                    ]
                    prompt_user_texts = [m[1].content for m in prompt_messages]

                    # --- F(x) at T=0 (deterministic single answer) ---------
                    f_answers: list[str] = []
                    for i, msgs in enumerate(prompt_messages):
                        ans = _sample_response(
                            client, model_entry, msgs,
                            temperature=0.0, seed=42,
                            purpose=f"e2e_f_score::{q.id}::{ladder_type}::L{level}::{model_key}",
                            max_tokens=64,
                        )
                        f_answers.append(ans)
                    f_scores = f_score_batch(q.answer, f_answers, config=config)
                    logger.info(
                        "  F(x): pass={}/{} answers={}",
                        sum(f_scores), len(f_scores),
                        [a[:40] for a in f_answers[:3]],
                    )

                    # --- H_sem samples at T=h_sem.sampling_temperature ----
                    responses_per_paraphrase: dict[int, list[str]] = {}
                    for i, msgs in enumerate(prompt_messages):
                        samples: list[str] = []
                        for k in range(args.k_samples):
                            samples.append(_sample_response(
                                client, model_entry, msgs,
                                temperature=config.h_sem.sampling_temperature,
                                seed=10000 + i * 100 + k,
                                purpose=f"e2e_h_sem::{q.id}::{ladder_type}::L{level}::{model_key}::s{k}",
                                max_tokens=64,
                            ))
                        responses_per_paraphrase[i] = samples

                    # --- pool-cluster (Sprint-4 contract!) ----------------
                    cluster_assignments = cluster_responses_pooled(
                        responses_per_paraphrase, config=config
                    )
                    n_unique = len({c for v in cluster_assignments.values() for c in v})
                    logger.info("  H_sem: |A_q estimate|={}", n_unique)

                    # --- embeddings --------------------------------------
                    prompt_embeddings = encode_texts(prompt_user_texts, config=config)
                    response_embeddings: dict[int, np.ndarray] = {}
                    for i, samples in responses_per_paraphrase.items():
                        response_embeddings[i] = encode_texts(samples, config=config)

                    # --- POSIX (optional) --------------------------------
                    posix_log_p = None
                    posix_lengths = None
                    if args.include_posix and model_entry.echo_completions:
                        logger.info("  POSIX: echo path enabled for this model")
                        # Build NxN matrix via echo on /v1/completions.
                        n = len(paraphrases)
                        log_p = np.zeros((n, n))
                        lengths = np.zeros(n)
                        for j, yj in enumerate(f_answers):
                            lengths[j] = max(1, len(yj.split()))
                            for i, msgs_i in enumerate(prompt_messages):
                                # Render the chat-formatted prompt naively as
                                # role-tagged text + the continuation y_j.
                                rendered = "\n".join(f"{m.role}: {m.content}" for m in msgs_i)
                                full = rendered + "\nassistant: " + yj
                                try:
                                    resp = client.score_continuation(CompletionRequest(
                                        provider=model_entry.provider,  # type: ignore[arg-type]
                                        model_id=model_entry.model_id,
                                        prompt=full,
                                        max_tokens=0,
                                        echo=True,
                                        logprobs=1,
                                        temperature=0.0,
                                        purpose=f"e2e_posix::{q.id}::{ladder_type}::L{level}",
                                    ))
                                    # Sum the last len(yj.split()) per-token logprobs.
                                    if resp.token_logprobs:
                                        tail = resp.token_logprobs[-int(lengths[j]):]
                                        log_p[i, j] = sum(t.logprob for t in tail)
                                    else:
                                        log_p[i, j] = math.nan
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("POSIX echo failed at (i={}, j={}): {}", i, j, exc)
                                    log_p[i, j] = math.nan
                        if not np.isnan(log_p).any():
                            posix_log_p = log_p
                            posix_lengths = lengths
                        else:
                            logger.warning(
                                "POSIX matrix has NaN entries — leaving posix_psi=None"
                            )

                    # --- build the MetricTuple ---------------------------
                    tup = build_metric_tuple(
                        question_id=q.id,
                        ladder_type=ladder_type,  # type: ignore[arg-type]
                        level=level,
                        model_key=model_key,
                        scores=[float(x) for x in f_scores],
                        cluster_assignments=cluster_assignments,
                        prompt_embeddings=prompt_embeddings,
                        response_embeddings=response_embeddings,
                        posix_log_p=posix_log_p,
                        posix_lengths=posix_lengths,
                        encoder_label="external_mpnet",
                    )
                    tuples.append(tup.model_dump())

    # --- 6. Write parquet + summary ---------------------------------------
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(tuples)
    out_df.to_parquet(out_path, index=False)
    logger.info("wrote {} cells -> {}", len(out_df), out_path)

    print()
    print("=" * 96)
    print("END-TO-END METRIC TUPLES")
    print("=" * 96)
    cols = [
        "question_id", "ladder_type", "level", "model_key",
        "aufi_in", "fi_out_mean", "s_tau_mean", "consistency_mean",
        "spread", "variation_ratio", "posix_psi",
        "ess_in", "rho_u", "h_sem_mean", "h_sem_var",
        "n_paraphrases", "n_samples_per_prompt",
    ]
    print(out_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Sanity assertions.
    #
    # ESS_in is allowed to be ~0 when using the external mpnet encoder with
    # context-heavy prompts: mpnet is trained to map paraphrases to nearby
    # points, so per-feature variance collapses by design. The own-encoder
    # variant from Sprint 6 (vLLM hidden states) will not have this property.
    # See metrics/ess_in.py docstring.
    bad = []
    for _, row in out_df.iterrows():
        if row["n_paraphrases"] < 2:
            continue  # too few for any reliable metric
        for col in ("aufi_in", "fi_out_mean", "s_tau_mean", "consistency_mean",
                    "spread", "variation_ratio", "ess_in", "rho_u",
                    "h_sem_mean", "h_sem_var"):
            v = row[col]
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                bad.append(f"{row['question_id']} {row['ladder_type']} L{row['level']} {col}={v!r}")
    if bad:
        logger.warning("plausibility issues in {} fields", len(bad))
        for line in bad[:10]:
            logger.warning("  {}", line)

    # Coarse summary that tells you whether the pipeline produced a sensible
    # context-amount curve, separate from the numeric MetricTuple dump.
    # F(x) pass-rates per level, averaged across questions:
    levels_summary: dict[int, list[float]] = {}
    for _, row in out_df.iterrows():
        levels_summary.setdefault(int(row["level"]), []).append(float(row.get("aufi_in") or 0.0))
    print()
    print("=" * 96)
    print("CONTEXT-AMOUNT TREND (mean AUFI_in by level — lower = paraphrases more uniformly pass)")
    print("=" * 96)
    for level in sorted(levels_summary):
        vals = levels_summary[level]
        print(f"  L={level:>2}  mean AUFI_in = {sum(vals) / len(vals):.3f}  ({len(vals)} cells)")

    print()
    print(json.dumps({
        "cells_run": len(out_df),
        "questions": sorted(out_df["question_id"].unique().tolist()),
        "levels": sorted(int(x) for x in out_df["level"].unique()),
        "models": sorted(out_df["model_key"].unique().tolist()),
        "ladders": sorted(out_df["ladder_type"].unique().tolist()),
        "plausibility_warnings": len(bad),
        "out_path": str(out_path),
    }, indent=2))

    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
