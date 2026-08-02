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

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.ambigqa_schemas import AmbigQuestion
from ..data.load_ambigqa import load_ambigqa
from ..ladders.schemas import LadderRow
from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple
from ..metrics.fi_in import aufi_in_from_scores
from ..metrics.fi_spec import fi_spec_bits
from ..metrics.sensitivity_v2 import compute_row_metrics
from ..metrics.h_sem import cluster_responses_pooled, entropy_from_assignment
from ..models.embedding import encode_texts
from ..models.registry import get_client
from ..scoring.nli_with_gold import f_score_batch_multi_gold
from ..data.schemas import HotpotParagraph
from ..specificity.build_levels import (
    SpecRow, build_spec_levels, build_spec_levels_multilevel, target_in_evidence)
from .e2e_smoke import _assemble_messages, _checkpoint, _clustering_inputs, _sample_response
from .local_check import _free_vram


_AMBIGQA_PARAPHRASE_PARQUET = "data/paraphrases_ambigqa.parquet"


# --------------------------------------------------------------------------- #
# Adapters (spec §8.5): closed-book cell via the EXISTING machinery            #
# --------------------------------------------------------------------------- #


class _SpecQuestionView:
    """Duck-typed stand-in for MultiHopQuestion, restricted to what the reused
    cell helpers touch. MultiHopQuestion itself is NOT changed (scope guard §12).

    v2: `paragraphs` carries the question's evidence snippets wrapped as
    HotpotParagraph, so the EXISTING context-block prompt path renders them;
    closed-book mode passes an empty list (v1 behaviour).
    """

    def __init__(self, row: SpecRow, paragraphs: list | None = None) -> None:
        self.id = row.question_id
        self.dataset = "ambigqa"
        self.question = row.question_text
        self.answer = row.target_answers[0]   # primary gold (multi-gold OR in scoring)
        self.paragraphs: list = paragraphs or []
        self.question_decomposition: list = []
        self.n_hops = None

    def has_decomposition(self) -> bool:      # -> binary NLI path, no CoT
        return False


def _evidence_paragraphs(row: SpecRow, max_chars: int) -> list[HotpotParagraph]:
    """Wrap the row's evidence snippets as context paragraphs.

    Whole snippets in dataset order until the char cap (median bundle ~4.8k,
    p75 ~6.3k). The SAME list is used at both levels and for every paraphrase —
    the v2 uniformity guardrail.
    """
    out: list[HotpotParagraph] = []
    used = 0
    for ev in row.evidence:
        n = len(ev.title) + len(ev.snippet)
        if out and used + n > max_chars:
            break
        out.append(HotpotParagraph(title=ev.title, sentences=[ev.snippet]))
        used += n
    return out


def _ladder_row_for(row: SpecRow, n_paragraphs: int = 0) -> LadderRow:
    """Context-family LadderRow over the (possibly empty) evidence block."""
    return LadderRow(
        question_id=row.question_id,
        ladder_type="random",
        ladder_family="context",
        level_idx=row.spec_level,
        level=row.spec_level,
        paragraph_indices=list(range(n_paragraphs)),
        paragraph_titles=[],
        gold_count=0,
    )


def load_spec_rows(
    config,
    *,
    n_questions: int | None = None,
    context_mode: str | None = None,
    ladder: str = "two",
    ml_m0_min: int = 3,
    ml_m0_max: int = 5,
    mid_cache: str | None = None,
    mid_generate: bool = True,
    evidence_fraction: float = 1.0,
) -> tuple[list[SpecRow], list[AmbigQuestion], str, int]:
    """Shared question -> SpecRow builder (deterministic: same filter, seed,
    order, evidence). Used by BOTH the run driver and the hidden-state dump so
    their prompts are bit-identical (FI probes join on (qid, level, para_idx)).

    ladder="multilevel" (C1): m0-in-[min,max] pool, gated L_mid rewrites from
    the mid-cache (generated on demand + persisted), 3 rows per question.
    evidence_fraction (C6): deterministic snippet trim, identical across
    levels + paraphrases.

    Returns (rows, questions, resolved_context_mode, evidence_max_chars).
    Raises RuntimeError on config/data problems (callers log + exit).
    """
    acfg = config.sampling.ambigqa
    if acfg is None:
        raise RuntimeError("config.sampling.ambigqa missing — add the block from the pivot spec §9")
    spec_cfg = config.specificity
    target_seed = spec_cfg.target_seed if spec_cfg is not None else config.random_seed
    n_q = n_questions if n_questions is not None else acfg.n_questions
    mode = context_mode or (
        spec_cfg.context_mode if spec_cfg is not None else "uniform_evidence"
    )
    evidence_max_chars = spec_cfg.evidence_max_chars if spec_cfg is not None else 6000
    require_cov = (
        mode == "uniform_evidence"
        and (spec_cfg.require_target_in_evidence if spec_cfg is not None else True)
    )

    ambiguous = [
        q for q in load_ambigqa(
            hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
            min_interpretations=acfg.min_interpretations,
            include_single_answer_anchor=acfg.include_single_answer_anchor,
        )
        if q.is_ambiguous()
    ]
    if require_cov:
        # v2 evidence-coverage filter: dataset-side + model-free (no selection on
        # model knowledge). ~52% of the ambiguous split passes.
        n_before = len(ambiguous)
        ambiguous = [q for q in ambiguous if target_in_evidence(q, seed=target_seed)]
        logger.info("evidence-coverage filter: {} -> {} questions (target answer in bundle)",
                    n_before, len(ambiguous))
    if ladder == "multilevel":
        # C1 pool: m0 in [ml_m0_min, ml_m0_max] — subset rewrites degrade beyond
        # ~5 interpretations; m0>=3 is structurally required for a middle level.
        n_before = len(ambiguous)
        ambiguous = [q for q in ambiguous if ml_m0_min <= q.m0() <= ml_m0_max]
        logger.info("multilevel m0-filter [{}..{}]: {} -> {} questions",
                    ml_m0_min, ml_m0_max, n_before, len(ambiguous))
    questions = ambiguous[:n_q]
    if not questions:
        raise RuntimeError("no AmbigQA questions left after filtering; bailing")
    rows: list[SpecRow] = []
    if ladder == "multilevel":
        from ..specificity.midlevel import (
            append_mid_cache, generate_mid_question, load_mid_cache)

        mid_path = config.repo_root() / (mid_cache or "data/midlevel_questions.parquet")
        mids = load_mid_cache(mid_path)
        n_gen = n_failed = 0
        kept: list = []
        for q in questions:
            mid = mids.get(q.id)
            if mid is None or mid.seed != target_seed:
                if not mid_generate:
                    # cache-only caller (hidden-state dump): a missing rewrite
                    # is a skip, never a Phi-4 load.
                    n_failed += 1
                    continue
                n_gen += 1
                logger.info("L_mid generate {}: qid={} (m0={})", n_gen, q.id, q.m0())
                mid = generate_mid_question(q, seed=target_seed, config=config)
                append_mid_cache(mid_path, mid)
                mids[q.id] = mid
            if mid.outcome != "accepted":
                n_failed += 1
                continue
            kept.append(q)
            rows.extend(build_spec_levels_multilevel(
                q, seed=target_seed, mid_text=mid.text,
                mid_subset=mid.subset,
                include_evidence=mode == "uniform_evidence",
            ))
        # Coverage discipline (mirrors rho_F): report exclusions, never hide them.
        logger.info("multilevel gate coverage: {}/{} questions have an accepted "
                    "L_mid ({} newly generated, {} failed -> excluded)",
                    len(kept), len(questions), n_gen, n_failed)
        questions = kept
        if not questions:
            raise RuntimeError("no questions with an accepted L_mid rewrite; bailing")
    else:
        for q in questions:
            rows.extend(build_spec_levels(
                q, seed=target_seed,
                include_evidence=mode == "uniform_evidence",
            ))
    if evidence_fraction < 1.0:
        # C6 evidence dial: keep the first ceil(f*n) snippets — deterministic
        # (dataset snippet order is stable) and IDENTICAL across levels +
        # paraphrases of a question, so evidence stays a controlled variable.
        import math as _math
        rows = [
            r.model_copy(update={"evidence": r.evidence[
                : _math.ceil(evidence_fraction * len(r.evidence))]})
            for r in rows
        ]
        logger.info("evidence-fraction {}: snippets trimmed per cell", evidence_fraction)
    logger.info("context_mode={} (evidence cap {} chars)", mode, evidence_max_chars)
    return rows, questions, mode, evidence_max_chars


def _graded_f_scores(
    golds: list[str],
    responses_per_paraphrase: dict[int, list[str]],
    config,
) -> list[float]:
    """GRADED per-paraphrase function F(x) = fraction of the k temperature
    samples (the H_sem samples — already generated, zero extra generation)
    that hit ANY gold variant.

    Why: the T=0 multi-gold score is BINARY per paraphrase here (no chain to
    grade, unlike MuSiQue), so FI_in(k) is a flat step for every cell (verified
    100/100 in the 2026-07-06 v2 run) and AUFI collapses to a monotone
    transform of accuracy. Estimating F(x) = P(correct | x) from the k samples
    restores a genuinely graded score, hence a real FI_in(k) curve to lead with.
    Ordered by paraphrase index; scored in ONE flattened multi-gold batch.
    """
    idxs = sorted(responses_per_paraphrase)
    flat: list[str] = []
    counts: list[int] = []
    for i in idxs:
        samples = responses_per_paraphrase[i]
        flat.extend(samples)
        counts.append(len(samples))
    if not flat:
        return []
    scores = [float(s) for s in f_score_batch_multi_gold(golds, flat, config=config)]
    out: list[float] = []
    pos = 0
    for c in counts:
        chunk = scores[pos:pos + c]
        out.append(float(np.mean(chunk)) if chunk else 0.0)
        pos += c
    return out


# --------------------------------------------------------------------------- #
# Paraphrase universes per (question, spec level)                              #
# --------------------------------------------------------------------------- #


def _generate_spec_paraphrases(
    config, rows: list[SpecRow], max_paraphrases: int,
    paraphrase_cache: str | None = None,
) -> dict[tuple[str, int], list[str]]:
    """Build (and persist) the paraphrase universe per (question_id, spec_level).

    Each LEVEL has its own universe over its own question text — the NLI
    equivalence filter keeps specificity constant WITHIN a level. Cached in
    `paraphrase_cache` (default data/paraphrases_ambigqa.parquet) keyed by
    (question_id, spec_level); a question that yields nothing falls back to its
    own text as a singleton. The multilevel ladder passes its OWN cache — the
    (qid, 1) key means "disambiguated" here but "L_mid" there.
    """
    from ..paraphrases.pipeline import build_paraphrase_set

    parquet_path = config.repo_root() / (paraphrase_cache or _AMBIGQA_PARAPHRASE_PARQUET)

    persisted: dict[tuple[str, int], list[str]] = {}
    if parquet_path.exists():
        try:
            prev = pd.read_parquet(parquet_path)
            # singleton_fallback rows count as cached too — a universe that
            # legitimately yielded nothing must not be re-attempted every resume
            # (each retry costs a full Phi-4 pipeline pass).
            prev = prev[prev["outcome"].isin(["accepted", "singleton_fallback"])]
            for (qid, lvl), sub in prev.groupby(["question_id", "spec_level"]):
                persisted[(str(qid), int(lvl))] = (
                    sub.sort_values("paraphrase_idx")["text"].tolist()
                )
            logger.info("loaded persisted AmbigQA paraphrases for {} cells", len(persisted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} ({}); regenerating", parquet_path, exc)

    out: dict[tuple[str, int], list[str]] = {}
    n_todo = sum(1 for r in rows if (r.question_id, r.spec_level) not in persisted)
    n_gen = 0
    for row in rows:
        key = (row.question_id, row.spec_level)
        if key in persisted and persisted[key]:
            out[key] = persisted[key][:max_paraphrases]
            continue
        n_gen += 1
        logger.info("paraphrase universe {}/{}: qid={} L{}",
                    n_gen, n_todo, row.question_id, row.spec_level)
        # The gold-constraint set is the answer SET the paraphrase must
        # preserve, matched to the level's meaning:
        #   L0 (ambiguous)     -> EVERY interpretation's answers. The ambiguous
        #      question's answer is the union; scoring a faithful L0 paraphrase
        #      against one interpretation rejected 100% of them (2026-07-06).
        #   L1 (disambiguated) -> the target answer's surface variants only.
        # The SCORING gold (row.target_answers, fixed across levels) is untouched.
        # Multi-level L_mid rows carry an explicit constraint (the admitted
        # subset's answer union); otherwise the two-level rule applies.
        gold_set = row.constraint_answers or (
            row.all_answers if row.spec_level == 0 else row.target_answers)
        try:
            pset = build_paraphrase_set(
                f"{row.question_id}::L{row.spec_level}",
                row.question_text,
                config=config,
                gold_answers=gold_set or row.target_answers,
            )
            texts = [ap.text for ap in pset.accepted][:max_paraphrases]
            outcome = "accepted"
        except Exception as exc:  # noqa: BLE001
            logger.warning("paraphrase gen failed for {} L{}: {}",
                           row.question_id, row.spec_level, exc)
            texts = []
            outcome = "accepted"
        if not texts:
            logger.warning("qid={} L{} no paraphrases; singleton fallback",
                           row.question_id, row.spec_level)
            texts = [row.question_text]
            outcome = "singleton_fallback"
        out[key] = texts
        # Persist THIS universe immediately (atomic replace) so a walltime kill
        # mid-prep loses at most the universe in flight — required for the
        # 30-min gpu_a100_short singleton-chain to make monotonic progress.
        _append_paraphrase_rows(parquet_path, [
            {"question_id": row.question_id, "spec_level": row.spec_level,
             "outcome": outcome, "paraphrase_idx": idx, "text": text}
            for idx, text in enumerate(texts)
        ])
    if n_gen:
        logger.info("generated + persisted {} new universes -> {}", n_gen, parquet_path)
    return out


def _append_paraphrase_rows(parquet_path, rows_new: list[dict]) -> None:
    """Append rows to the paraphrase cache with an atomic replace."""
    import os

    df_new = pd.DataFrame(rows_new)
    if parquet_path.exists():
        try:
            df_new = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    df_new.to_parquet(tmp, index=False)
    os.replace(tmp, parquet_path)


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
    inspect: bool = False,
    context_mode: str = "uniform_evidence",
    evidence_max_chars: int = 6000,
    posix: bool = False,
    ladder: str = "two",
    evidence_fraction: float = 1.0,
) -> tuple[dict, dict | None]:
    """One (SpecRow, model) cell -> (flat result row, inspection record | None).
    Mirrors the e2e binary (non-CoT) path but scores with the multi-gold OR and
    attaches FI_spec. v2: uniform evidence block unless context_mode=closed_book."""
    model_entry = config.models[model_key]
    client = get_client(model_key, config)
    ev_paragraphs = (
        _evidence_paragraphs(row, evidence_max_chars)
        if context_mode == "uniform_evidence" else []
    )
    view = _SpecQuestionView(row, paragraphs=ev_paragraphs)
    lrow = _ladder_row_for(row, n_paragraphs=len(ev_paragraphs))
    gen_max = config.generation.answer_max_tokens
    logger.info("cell qid={} spec_level={} model={} N={} m0={} m_valid={} evidence={}snip",
                row.question_id, row.spec_level, model_key,
                len(paraphrases), row.m0, row.m_valid, len(ev_paragraphs))

    prompt_messages = [
        _assemble_messages(view, p, lrow, use_cot=False) for p in paraphrases
    ]

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
    f_graded: list[float] | None = None
    posix_log_p = posix_lengths = None
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
        # ESS_in: embed the PARAPHRASE TEXT, not the full user message. The v2
        # user message is >90% the shared 6000-char evidence block, and mpnet
        # (trained to map same-meaning texts together) collapses the N nearly-
        # identical messages to one point — ESS_in was exactly 0 in 98/100
        # cells of the 2026-07-06 run. The paraphrase line is the only part of
        # the prompt that VARIES over U_q, which is the dispersion ESS_in is
        # meant to measure (design doc §3 Tier C C4).
        prompt_embeddings = encode_texts(list(paraphrases), config=config)
        response_embeddings = {
            i: encode_texts(samples, config=config)
            for i, samples in responses_per_paraphrase.items()
        }
        # GRADED F(x) from the same k samples (no extra generation; one batch
        # of multi-gold scorings). See _graded_f_scores for why.
        f_graded = _graded_f_scores(
            list(row.target_answers), responses_per_paraphrase, config
        )
        # POSIX (Chatterjee 2024) — opt-in: N*N teacher-forced forward passes
        # per cell via the echo path (~100 encodings of evidence-length prompts
        # for N=10). Same matrix builder as the e2e path.
        if posix and model_entry.echo_completions:
            from .e2e_smoke import _posix_matrix
            posix_log_p, posix_lengths = _posix_matrix(
                client, model_entry, prompt_messages, f_responses, view, lrow
            )

    tup = build_metric_tuple(
        question_id=row.question_id,
        ladder_type="random",
        level=row.spec_level,
        model_key=model_key,
        scores=f_scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_embeddings,
        response_embeddings=response_embeddings,
        posix_log_p=posix_log_p,
        posix_lengths=posix_lengths,
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
    row_dict["context_mode"] = context_mode
    row_dict["n_evidence_snippets"] = len(ev_paragraphs)
    # Arm bookkeeping (C1/C6): which ladder produced this row + how much of the
    # evidence bundle was kept — spec_level semantics depend on the ladder.
    row_dict["ladder"] = ladder
    row_dict["evidence_fraction"] = float(evidence_fraction)
    # FI_out over the FIXED answer space: log2(m0) - H_sem. fi_out_mean uses the
    # observed |A_q|, which itself shrinks with specificity, so its sign is not
    # interpretable for the hypothesis (2026-07-06 finding; mirrors
    # show_specificity.add_fi_out_fixed, now emitted at run time). May go
    # negative when the model disperses over MORE clusters than the question's
    # own m0 interpretations — that excess is informative, so no clamp.
    h_mean = row_dict.get("h_sem_mean")
    row_dict["fi_out_fixed"] = (
        float(np.log2(max(row.m0, 1)) - h_mean) if h_mean is not None else None
    )
    # GRADED input-space track: F(x) = P(correct|x) estimated from the k
    # temperature samples -> a real (non-step) FI_in(k) curve. The T=0 binary
    # columns stay primary for comparability with the earlier v2 cells.
    if f_graded is not None:
        row_dict["f_graded_per_paraphrase"] = list(f_graded)
        row_dict["f_graded_mean"] = float(np.mean(f_graded)) if f_graded else None
        row_dict["aufi_in_graded"] = aufi_in_from_scores(f_graded)
    else:
        row_dict["f_graded_per_paraphrase"] = None
        row_dict["f_graded_mean"] = None
        row_dict["aufi_in_graded"] = None
    # Sensitivity v2 (METRIC_PROPOSALS M1+M2): rho_F functional ICC + ΔFI
    # reliability premium — the accuracy-decoupled sensitivity scalars.
    row_dict.update(compute_row_metrics(f_graded, k_samples))
    # Per-paraphrase H_sem for the P3 probe: SEP predicts PER-PROMPT semantic
    # entropy, but only the cell mean/var was persisted — the probe label was a
    # cell constant. Ordered by paraphrase_idx (joins like f_graded).
    row_dict["h_sem_per_paraphrase"] = (
        [float(entropy_from_assignment(cluster_assignments[i]))
         for i in sorted(cluster_assignments)]
        if cluster_assignments else None
    )

    # Inspection capture (per-run audit deliverable): everything needed to check
    # this cell start->finish by hand. H_sem is summarised (distinct clusters +
    # one representative answer per cluster) rather than dumped in full.
    inspect_rec: dict | None = None
    if inspect:
        hsem_summary = None
        if not fast and cluster_assignments:
            reps: dict[int, str] = {}
            for i, assigns in cluster_assignments.items():
                for s_idx, cl in enumerate(assigns):
                    reps.setdefault(int(cl), responses_per_paraphrase[i][s_idx][:80])
            hsem_summary = {"k": k_samples, "n_clusters": len(reps), "representatives": reps}
        inspect_rec = {
            "question_id": row.question_id, "spec_level": row.spec_level,
            "question_text": row.question_text, "target_answers": list(row.target_answers),
            "m0": row.m0, "m_valid": row.m_valid, "target_idx": row.target_idx,
            "model_key": model_key,
            "context_mode": context_mode,
            "evidence": [f"{p.title}: {p.sentences[0][:120]}" for p in ev_paragraphs[:4]],
            "n_evidence": len(ev_paragraphs),
            "paraphrases": [
                {"idx": i, "paraphrase": p, "answer_t0": f_responses[i][:200],
                 "f": f_scores[i]}
                for i, p in enumerate(paraphrases)
            ],
            "hsem": hsem_summary,
            "metrics": {k: row_dict.get(k) for k in
                        ("f_mean", "aufi_in", "f_graded_mean", "aufi_in_graded",
                         "fi_out_mean", "fi_out_fixed", "h_sem_mean", "a_q", "fi_spec")},
        }
    return row_dict, inspect_rec


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
    parser.add_argument("--inspect-n", type=int, default=0,
                        help="write a start->finish audit bundle for the first N "
                             "questions (both levels) to data/inspect_<out-stem>.md")
    parser.add_argument("--context-mode", choices=["closed_book", "uniform_evidence"],
                        default=None,
                        help="override config.specificity.context_mode "
                             "(closed_book reproduces the v1 design)")
    parser.add_argument("--posix", action="store_true",
                        help="also compute POSIX (Chatterjee 2024) via the echo "
                             "path: N*N teacher-forced passes per cell — "
                             "~100 evidence-length encodings/cell at N=10")
    parser.add_argument("--prep-only", action="store_true",
                        help="build/extend the paraphrase-universe cache and exit "
                             "before loading any eval model (v3: run ONE prep "
                             "chain, then per-model eval chains in parallel)")
    parser.add_argument("--out", type=str, default="data/specificity_metrics.parquet")
    parser.add_argument("--dry-run", action="store_true")
    # ---- multi-level ladder (FINAL_PHASE_PLAN C1) ----
    parser.add_argument("--ladder", choices=["two", "multilevel"], default="two",
                        help="'multilevel' = L0/L_mid/L_top (spec_level 0/1/2) on "
                             "m0>=3 questions via gated partial disambiguation; "
                             "REQUIRES its own --out and --paraphrase-cache")
    parser.add_argument("--ml-m0-min", type=int, default=3)
    parser.add_argument("--ml-m0-max", type=int, default=5,
                        help="m0 range for the multilevel pool (subset-rewrite "
                             "quality degrades beyond ~5 interpretations)")
    parser.add_argument("--mid-cache", type=str,
                        default="data/midlevel_questions.parquet",
                        help="persisted L_mid rewrites + gate outcomes")
    parser.add_argument("--paraphrase-cache", type=str, default=None,
                        help="override the paraphrase-universe parquet. The "
                             "multilevel ladder MUST NOT share the two-level "
                             "cache: (qid, spec_level=1) means 'disambiguated' "
                             "there but 'L_mid' here (default auto-separates)")
    # ---- evidence dial (FINAL_PHASE_PLAN C6) ----
    parser.add_argument("--evidence-fraction", type=float, default=1.0,
                        help="keep the first ceil(f*n) evidence snippets, "
                             "identical across levels+paraphrases (0.0 = "
                             "closed-book). Use a DISTINCT --out per fraction: "
                             "the resume key ignores this knob")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging("run_specificity")
    config = load_config()
    repo_root = config.repo_root()

    if args.ladder == "multilevel" and args.paraphrase_cache is None:
        # Hard auto-separation: (qid, spec_level=1) means "disambiguated" in the
        # two-level cache but "L_mid" here — sharing would poison both ladders.
        args.paraphrase_cache = "data/paraphrases_ambigqa_ml.parquet"
        logger.info("multilevel ladder -> paraphrase cache {}", args.paraphrase_cache)
    try:
        rows, questions, context_mode, evidence_max_chars = load_spec_rows(
            config, n_questions=args.n_questions, context_mode=args.context_mode,
            ladder=args.ladder, ml_m0_min=args.ml_m0_min, ml_m0_max=args.ml_m0_max,
            mid_cache=args.mid_cache, evidence_fraction=args.evidence_fraction,
            mid_generate=not args.dry_run,  # a dry run must never load Phi-4
        )
    except RuntimeError as exc:
        logger.error("{}", exc)
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in config.models:
            logger.error("unknown model_key {!r}; available: {}", m, list(config.models))
            return 1
    k_samples = args.k_samples if args.k_samples is not None else config.h_sem.n_samples_per_prompt
    if not args.fast and k_samples < 5:
        logger.warning("H_sem k={} (<5): near-zero semantic-entropy resolution", k_samples)

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
            "context_mode": context_mode,
            "evidence_snippets_median": float(np.median([len(r.evidence) for r in rows])),
        }, indent=2))
        return 0

    # --- paraphrase universes (per question x level) -------------------------
    paraphrases = _generate_spec_paraphrases(
        config, rows, args.max_paraphrases, paraphrase_cache=args.paraphrase_cache)

    if args.prep_only:
        # v3 topology (FI_PROBES_PLAN.md §4): ONE prep chain builds every
        # universe, THEN per-model eval chains run in parallel reading the cache
        # read-only — without this barrier two eval chains that both find a
        # universe missing would generate it concurrently and race the cache's
        # read-modify-write append.
        n_missing = sum(
            1 for r in rows if not paraphrases.get((r.question_id, r.spec_level))
        )
        logger.info("prep-only: {} universes ready, {} missing — no eval",
                    len(paraphrases), n_missing)
        print(f"PREP DONE universes={len(paraphrases)} missing={n_missing}")
        return 0

    # Free the generator's VRAM before the eval model loads (job 5762430
    # post-mortem): the Phi-4 generator/judge (28 GiB bf16) stayed resident after
    # prep, and Phi-4 + Qwen (~15 GiB) exceed the 40 GB A100 on gpu_a100_short ->
    # CUDA OOM in every cell. The MuSiQue flow dodged this structurally (prep ran
    # as a separate job); this driver runs both phases in one process, so it must
    # drop the weights explicitly. DeBERTa (~1.6 GB, needed again for scoring +
    # clustering) lives in a different cache and survives. No-op when the
    # universes came from the parquet cache and no generator was ever loaded.
    _free_vram()

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

    inspect_qids = {q.id for q in questions[: args.inspect_n]} if args.inspect_n > 0 else set()
    inspect_recs: list[dict] = []
    n_done = n_failed = n_skipped = 0
    # MODEL-major order (2026-07-18): with models inner, a 3-model run keeps all
    # three ~15 GB models resident from the first row (lru_cache(2) loads the
    # third BEFORE evicting the first: ~46 GB peak > the 40 GB A100). Model-major
    # + an explicit free between models bounds peak VRAM at one eval model.
    # Resume is key-based, so cell order is free to change.
    for mi, model_key in enumerate(models):
        for row in rows:
            key = _cell_key(row.question_id, row.spec_level, model_key)
            if key in done:
                n_skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                row_dict, inspect_rec = _run_spec_cell(
                    config, row, model_key,
                    paraphrases[(row.question_id, row.spec_level)],
                    k_samples=k_samples, fast=args.fast,
                    inspect=row.question_id in inspect_qids,
                    context_mode=context_mode,
                    evidence_max_chars=evidence_max_chars,
                    posix=args.posix,
                    ladder=args.ladder,
                    evidence_fraction=args.evidence_fraction,
                )
            except Exception:  # noqa: BLE001 — isolate cell failures
                n_failed += 1
                logger.exception("cell FAILED qid={} L{} model={} — continuing",
                                 row.question_id, row.spec_level, model_key)
                continue
            rows_out.append(row_dict)
            if inspect_rec is not None:
                inspect_recs.append(inspect_rec)
                # Persist THIS record immediately. The end-of-run writer alone
                # loses the bundle on chained 30-min windows: every window that
                # COMPUTES inspect cells can die at walltime mid-loop, and the
                # surplus windows that do reach the writer have nothing to
                # append (all cells resumed) — exactly how all three v3 full-run
                # bundles were lost (2026-07-21) while smoke/v2 survived only by
                # finishing in-window. Ten inspect cells/model -> negligible IO.
                _persist_and_render_inspection(out_path, [inspect_rec])
            done.add(key)
            n_done += 1
            logger.info("cell {} done in {:.1f}s — qid={} L{} model={}",
                        n_done, time.perf_counter() - t0,
                        row.question_id, row.spec_level, model_key)
            _checkpoint(rows_out, out_path)
        if mi < len(models) - 1:
            _free_vram()

    logger.info("done: {} new, {} skipped (resume), {} failed. total: {}",
                n_done, n_skipped, n_failed, len(rows_out))
    if not rows_out:
        logger.error("no cells produced")
        return 1
    # Records were already persisted per cell; empty call = md regen only
    # (keeps the bundle fresh even in surplus no-op windows).
    _persist_and_render_inspection(out_path, [])
    _print_summary(pd.DataFrame(rows_out))
    return 0 if n_failed == 0 else 2


def _persist_and_render_inspection(out_path, new_recs: list[dict]) -> None:
    """Windowed-resume-safe inspection bundle (fix for the v1+v2 full runs, where
    the md was lost twice: records lived only in the window that computed the
    cells, and that window died at walltime before the writer ran).

    Every window APPENDS its records to inspect_<stem>.jsonl at the end, then
    re-renders inspect_<stem>.md from ALL jsonl records — so the surviving md is
    cumulative and even a surplus no-op window regenerates it. Records are
    deduped on (question_id, spec_level, model_key), last write wins.
    """
    jsonl = out_path.parent / f"inspect_{out_path.stem}.jsonl"
    if new_recs:
        with jsonl.open("a", encoding="utf-8") as fh:
            for rec in new_recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if not jsonl.exists():
        return
    dedup: dict[tuple, dict] = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue   # torn line from a walltime kill mid-write; skip
        dedup[(rec.get("question_id"), rec.get("spec_level"), rec.get("model_key"))] = rec
    if not dedup:
        return
    records = list(dedup.values())
    md = out_path.parent / f"inspect_{out_path.stem}.md"
    md.write_text(_render_spec_inspection_md(records), encoding="utf-8")
    logger.info("inspection bundle: {} new rec(s), {} total -> {}",
                len(new_recs), len(records), md)


def _render_spec_inspection_md(records: list[dict]) -> str:
    """Human-readable start->finish audit for the inspected questions: ambiguous
    vs disambiguated text, the fixed gold, every paraphrase with the model's T=0
    answer and its F, an H_sem cluster summary, and the cell metrics."""
    def t(s, n: int = 110) -> str:
        s = " ".join(str(s).split()).replace("|", "\\|")
        return s if len(s) <= n else s[:n] + "…"

    by_q: dict[str, list[dict]] = {}
    for r in records:
        by_q.setdefault(r["question_id"], []).append(r)

    out = ["# Specificity run — inspection bundle", "",
           f"{len(by_q)} question(s), {len(records)} cell(s). Gold is FIXED across "
           "levels (the guardrail); level 0 = ambiguous, level 1 = disambiguated.", ""]
    for qi, (qid, cells) in enumerate(by_q.items(), 1):
        cells = sorted(cells, key=lambda c: c["spec_level"])
        c0 = cells[0]
        out += [f"## {qi}. `{qid}` — m0={c0['m0']}, target_idx={c0['target_idx']}, "
                f"model `{c0['model_key']}`",
                f"**Fixed gold (a_i variants):** {', '.join(c0['target_answers'])}", ""]
        if c0.get("n_evidence"):
            out.append(f"**Uniform evidence** ({c0['n_evidence']} snippets, identical for "
                       "both levels & all paraphrases); first entries:")
            out += [f"- {t(e, 130)}" for e in c0.get("evidence", [])]
            out.append("")
        for c in cells:
            m = c["metrics"]
            out += [f"### Level {c['spec_level']} — “{t(c['question_text'])}”",
                    f"FI_spec={m['fi_spec']:.3f} · f_mean={_fmtf(m['f_mean'])} · "
                    f"AUFI_in={_fmtf(m['aufi_in'])} · FI_out={_fmtf(m['fi_out_mean'])} · "
                    f"H_sem={_fmtf(m['h_sem_mean'])} · |A_q|={m['a_q']}", "",
                    "| # | paraphrase | model answer (T=0) | F |",
                    "|--:|------------|--------------------|--:|"]
            for p in c["paraphrases"]:
                out.append(f"| {p['idx']} | {t(p['paraphrase'], 80)} | "
                           f"{t(p['answer_t0'], 80)} | {p['f']:.0f} |")
            out.append("")
            if c["hsem"]:
                h = c["hsem"]
                reps = " · ".join(f"c{cl}: “{t(txt, 40)}”"
                                  for cl, txt in sorted(h["representatives"].items())[:8])
                out += [f"H_sem (k={h['k']}): {h['n_clusters']} clusters — {reps}", ""]
    return "\n".join(out)


def _fmtf(v) -> str:
    return "—" if v is None else f"{v:.3f}"


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
