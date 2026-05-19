"""Full paraphrase pipeline orchestrator. Sprint 2 §3.1.

Flow per question:

  1. Generate 40 raw candidates (4 roles × 10 samples).
  2. Drop empties and trivially identical-to-original strings.
  3. NLI filter at τ = 0.9 bidirectional.
  4. Constraint filter: judge-elicited answer-set Jaccard >= 0.9.
  5. Edit-distance dedup at min_distance = 6.
  6. If we have >= 30 accepted: keep the first 30 (priority: NLI score desc,
     then role rotation for diversity). Return.
  7. If fewer than 30: extend sample_idxs by another 10/template per attempt,
     up to `max_regeneration_attempts` total raw candidates. Re-run filters
     incrementally (already-accepted candidates are kept; new ones go through
     the same gauntlet). After step 4 of the second pass, if still under 30,
     RELAX the NLI threshold to 0.85 (`nli.fallback_threshold`) and retry.
  8. If still under 30 after all attempts: mark `dropped=True`. Caller drops
     the question from the sample and picks a replacement from the dataset.

Each `RawParaphrase`, accept, and reject decision is logged for the audit
trail and surfaced in `ParaphraseSet.rejected` for inter-annotator review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from loguru import logger

from ..config import Config, load_config
from .constraint_filter import filter_by_constraint
from .deduplicate import deduplicate
from .generate import generate_raw_paraphrases
from .nli_filter import filter_by_nli
from .schemas import (
    AcceptedParaphrase,
    ParaphraseSet,
    RawParaphrase,
    RejectedParaphrase,
    RoleName,
)


@dataclass
class _Scored:
    """Internal struct for ranking surviving candidates before truncation."""

    raw: RawParaphrase
    nli_fwd: float
    nli_bwd: float
    jaccard: float

    def priority(self) -> tuple[float, float]:
        # Higher entail score and higher jaccard come first. Tuple-sort
        # descending via negation.
        return (-min(self.nli_fwd, self.nli_bwd), -self.jaccard)


def build_paraphrase_set(
    question_id: str,
    question_text: str,
    *,
    config: Config | None = None,
) -> ParaphraseSet:
    """Run the full pipeline for one question."""
    if config is None:
        config = load_config()
    pcfg = config.paraphrases

    target = pcfg.n_per_question
    samples_per_template = pcfg.samples_per_template
    max_total = pcfg.max_regeneration_attempts  # interpreted as max RAW candidates / template
    roles = list(pcfg.templates)  # type: ignore[assignment]

    pset = ParaphraseSet(question_id=question_id)
    all_raw: list[RawParaphrase] = []
    accepted: list[_Scored] = []
    threshold = pcfg.nli.bidirectional_threshold

    next_sample_start = 0
    while True:
        # ----- step 1: generate the next batch of raw candidates -----
        sample_idxs = list(range(next_sample_start, next_sample_start + samples_per_template))
        next_sample_start += samples_per_template
        logger.info(
            "qid={} attempt={} sample_idxs={}..{} threshold={:.2f}",
            question_id,
            pset.regeneration_attempts,
            sample_idxs[0],
            sample_idxs[-1],
            threshold,
        )
        new_raw = generate_raw_paraphrases(
            question_id,
            question_text,
            config=config,
            sample_idxs=sample_idxs,
            roles=roles,
        )
        if not new_raw:
            logger.warning("qid={} generated zero candidates this round", question_id)
        all_raw.extend(new_raw)

        # ----- step 2: drop trivially identical-to-original (case-folded) -----
        original_norm = question_text.strip().lower()
        round_candidates = [
            r for r in new_raw if r.text.strip().lower() != original_norm
        ]
        for r in new_raw:
            if r.text.strip().lower() == original_norm:
                pset.rejected.append(
                    RejectedParaphrase(
                        question_id=question_id,
                        role=r.role,
                        sample_idx=r.sample_idx,
                        text=r.text,
                        reason="exact_duplicate",
                    )
                )

        # ----- step 3: NLI filter -----
        nli_rows = filter_by_nli(
            question_text,
            [c.text for c in round_candidates],
            config=config,
            threshold=threshold,
        )
        # Diagnostic: log the score distribution for this batch so the user
        # can spot a "model never says yes" problem without re-running.
        if nli_rows:
            mins = [min(s.entail_fwd, s.entail_bwd) for _, s in nli_rows]
            n_pass = sum(1 for p, _ in nli_rows if p)
            logger.info(
                "qid={} NLI batch: pass={}/{} min(fwd,bwd) p50={:.2f} p90={:.2f} max={:.2f}",
                question_id,
                n_pass,
                len(nli_rows),
                float(sorted(mins)[len(mins) // 2]),
                float(sorted(mins)[max(0, int(len(mins) * 0.9) - 1)]),
                float(max(mins)),
            )
        post_nli: list[_Scored] = []
        for raw, (passed, scores) in zip(round_candidates, nli_rows, strict=True):
            if not passed:
                reason = (
                    "nli_one_direction"
                    if max(scores.entail_fwd, scores.entail_bwd) >= threshold
                    else "nli_low"
                )
                pset.rejected.append(
                    RejectedParaphrase(
                        question_id=question_id,
                        role=raw.role,
                        sample_idx=raw.sample_idx,
                        text=raw.text,
                        reason=reason,
                        nli_entail_fwd=scores.entail_fwd,
                        nli_entail_bwd=scores.entail_bwd,
                    )
                )
                continue
            post_nli.append(_Scored(raw=raw, nli_fwd=scores.entail_fwd, nli_bwd=scores.entail_bwd, jaccard=math.nan))

        # ----- step 4: constraint filter -----
        if post_nli:
            cf_rows = filter_by_constraint(
                question_text,
                [s.raw.text for s in post_nli],
                config=config,
            )
            survivors: list[_Scored] = []
            for s, (passed, j_res) in zip(post_nli, cf_rows, strict=True):
                s.jaccard = j_res.jaccard
                if not passed:
                    pset.rejected.append(
                        RejectedParaphrase(
                            question_id=question_id,
                            role=s.raw.role,
                            sample_idx=s.raw.sample_idx,
                            text=s.raw.text,
                            reason="constraint_mismatch",
                            nli_entail_fwd=s.nli_fwd,
                            nli_entail_bwd=s.nli_bwd,
                            constraint_jaccard=j_res.jaccard,
                        )
                    )
                    continue
                survivors.append(s)
            accepted.extend(survivors)

        # ----- step 5: dedup against all accepted so far -----
        accepted.sort(key=lambda s: s.priority())
        kept_idx = deduplicate(
            [s.raw.text for s in accepted],
            min_distance=pcfg.deduplication.min_edit_distance,
        )
        kept_set = set(kept_idx)
        for i, s in enumerate(accepted):
            if i in kept_set:
                continue
            pset.rejected.append(
                RejectedParaphrase(
                    question_id=question_id,
                    role=s.raw.role,
                    sample_idx=s.raw.sample_idx,
                    text=s.raw.text,
                    reason="edit_distance_close",
                    nli_entail_fwd=s.nli_fwd,
                    nli_entail_bwd=s.nli_bwd,
                    constraint_jaccard=s.jaccard,
                )
            )
        accepted = [accepted[i] for i in kept_idx]

        pset.regeneration_attempts += 1
        logger.info(
            "qid={} accepted_so_far={} target={} raw_so_far={}",
            question_id,
            len(accepted),
            target,
            len(all_raw),
        )

        if len(accepted) >= target:
            break

        # ----- regeneration policy -----
        if next_sample_start >= max_total:
            # We've exhausted attempts at the strict threshold. Try the
            # relaxed fallback once on ALL raw candidates we've already seen
            # (no extra API spend).
            if threshold > pcfg.nli.fallback_threshold + 1e-9:
                logger.info(
                    "qid={} relaxing NLI threshold {:.2f}->{:.2f} and re-evaluating",
                    question_id,
                    threshold,
                    pcfg.nli.fallback_threshold,
                )
                threshold = pcfg.nli.fallback_threshold
                pset.nli_threshold_used = threshold
                pset = _reset_for_retry(pset)
                accepted = []
                # Re-evaluate everything we have using cached NLI scores by
                # zero-ing next_sample_start back so the next loop iteration
                # re-generates nothing new but goes through the loop body
                # against `all_raw` ... easier: just call ourselves with the
                # cumulated raw set under the relaxed threshold. To keep the
                # control flow simple, jump out and let _retry_with_relaxed
                # finish the work.
                return _retry_with_relaxed(
                    pset,
                    question_id,
                    question_text,
                    all_raw,
                    config,
                )
            # Already at fallback threshold and still short: give up.
            pset.dropped = True
            logger.warning(
                "qid={} dropped: only {} accepted after {} attempts at threshold {:.2f}",
                question_id,
                len(accepted),
                pset.regeneration_attempts,
                threshold,
            )
            break

    pset.accepted = _materialise(accepted, target)
    return pset


def _reset_for_retry(pset: ParaphraseSet) -> ParaphraseSet:
    """Strip rejections that were threshold-driven; keep the constraint/dedup ones."""
    keep_reasons = {"constraint_mismatch", "edit_distance_close", "exact_duplicate"}
    return ParaphraseSet(
        question_id=pset.question_id,
        accepted=[],
        rejected=[r for r in pset.rejected if r.reason in keep_reasons],
        regeneration_attempts=pset.regeneration_attempts,
        dropped=False,
        nli_threshold_used=pset.nli_threshold_used,
    )


def _retry_with_relaxed(
    pset: ParaphraseSet,
    question_id: str,
    question_text: str,
    all_raw: Sequence[RawParaphrase],
    config: Config,
) -> ParaphraseSet:
    """Re-run filters at the fallback threshold on the cumulated raw candidates.

    The gateway cache makes this cheap: every call has the same request hash
    as the original attempt, so it is served from disk.
    """
    pcfg = config.paraphrases
    target = pcfg.n_per_question
    threshold = pcfg.nli.fallback_threshold

    # NLI filter at relaxed threshold.
    nli_rows = filter_by_nli(
        question_text,
        [r.text for r in all_raw],
        config=config,
        threshold=threshold,
    )
    post_nli: list[_Scored] = []
    for raw, (passed, scores) in zip(all_raw, nli_rows, strict=True):
        if not passed:
            pset.rejected.append(
                RejectedParaphrase(
                    question_id=question_id,
                    role=raw.role,
                    sample_idx=raw.sample_idx,
                    text=raw.text,
                    reason=(
                        "nli_one_direction"
                        if max(scores.entail_fwd, scores.entail_bwd) >= threshold
                        else "nli_low"
                    ),
                    nli_entail_fwd=scores.entail_fwd,
                    nli_entail_bwd=scores.entail_bwd,
                )
            )
            continue
        post_nli.append(_Scored(raw=raw, nli_fwd=scores.entail_fwd, nli_bwd=scores.entail_bwd, jaccard=math.nan))

    cf_rows = filter_by_constraint(
        question_text,
        [s.raw.text for s in post_nli],
        config=config,
    )
    survivors: list[_Scored] = []
    for s, (passed, j_res) in zip(post_nli, cf_rows, strict=True):
        s.jaccard = j_res.jaccard
        if not passed:
            pset.rejected.append(
                RejectedParaphrase(
                    question_id=question_id,
                    role=s.raw.role,
                    sample_idx=s.raw.sample_idx,
                    text=s.raw.text,
                    reason="constraint_mismatch",
                    nli_entail_fwd=s.nli_fwd,
                    nli_entail_bwd=s.nli_bwd,
                    constraint_jaccard=j_res.jaccard,
                )
            )
            continue
        survivors.append(s)

    survivors.sort(key=lambda s: s.priority())
    kept_idx = deduplicate(
        [s.raw.text for s in survivors],
        min_distance=pcfg.deduplication.min_edit_distance,
    )
    kept_set = set(kept_idx)
    for i, s in enumerate(survivors):
        if i in kept_set:
            continue
        pset.rejected.append(
            RejectedParaphrase(
                question_id=question_id,
                role=s.raw.role,
                sample_idx=s.raw.sample_idx,
                text=s.raw.text,
                reason="edit_distance_close",
                nli_entail_fwd=s.nli_fwd,
                nli_entail_bwd=s.nli_bwd,
                constraint_jaccard=s.jaccard,
            )
        )
    survivors = [survivors[i] for i in kept_idx]

    pset.accepted = _materialise(survivors, target)
    pset.dropped = len(pset.accepted) < target
    return pset


def _materialise(scored: Sequence[_Scored], target: int) -> list[AcceptedParaphrase]:
    """Convert internal _Scored list to ordered, indexed AcceptedParaphrase rows."""
    out: list[AcceptedParaphrase] = []
    for idx, s in enumerate(scored[:target]):
        out.append(
            AcceptedParaphrase(
                question_id=s.raw.question_id,
                paraphrase_idx=idx,
                role=s.raw.role,
                text=s.raw.text,
                nli_entail_fwd=s.nli_fwd,
                nli_entail_bwd=s.nli_bwd,
                constraint_jaccard=s.jaccard,
                generator_model_key=s.raw.generator_model_key,
                generator_seed=s.raw.generator_seed,
                request_hash=s.raw.request_hash,
            )
        )
    return out
