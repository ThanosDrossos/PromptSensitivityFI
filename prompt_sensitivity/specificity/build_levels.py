"""Specificity level builder: one AmbigQuestion -> two closed-book SpecRows.

The manipulation lives in the QUESTION TEXT, not in context paragraphs:

    level 0 (ambig):    text = the original ambiguous Q,  m_valid = m0
    level 1 (disambig): text = the target interpretation Q_i, m_valid = 1

THE GUARDRAIL: `target_answers` (the scoring gold a_i) is FIXED across both
levels — only the question text (and hence m_valid) changes, so the ground
truth never drifts; we measure whether the model hits the fixed target answer
more often as specificity rises.

Target choice is deterministic from (question_id, seed) via sha256 — NOT
Python's salted hash() — so the paraphrase-prep job and every model run pick
the identical target interpretation.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from ..data.ambigqa_schemas import AmbigQuestion, EvidenceSnippet


class SpecRow(BaseModel):
    """One (question, specificity level) cell descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    spec_level: int                 # two-level: 0=ambig, 1=disambig; multi-level: 0/1(mid)/2
    question_text: str              # Q at level 0, Q_mid / Q_i above
    target_answers: list[str]       # a_i variants (SCORING gold), FIXED across both levels
    # Union of EVERY interpretation's answers (the ambiguous question's full
    # answer SET). Used only as the L0 paraphrase gold-constraint set: a faithful
    # paraphrase of the ambiguous Q must preserve ANY interpretation's answer, so
    # scoring it against the single target answer wrongly rejected 100% of
    # NLI-valid L0 paraphrases (2026-07-06). NOT a scoring gold — the fixed
    # `target_answers` guardrail is unchanged.
    all_answers: list[str] = Field(default_factory=list)
    m_valid: int                    # level 0 -> m0, level 1 -> 1
    m0: int
    target_idx: int                 # which interpretation was chosen as target
    # v2 uniform-evidence: the SAME bundle at both levels and for every
    # paraphrase (guardrail #2, next to the fixed gold) — specificity stays the
    # only manipulated variable. Empty in closed-book mode / light config.
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    # Paraphrase gold-CONSTRAINT override for levels whose admitted answer set
    # is neither the full union (L0) nor the single target (top level) — i.e.
    # the multi-level ladder's L_mid, where a faithful paraphrase must preserve
    # ANY answer of the ADMITTED interpretation subset. None -> the driver's
    # default rule (all_answers at level 0, target_answers above). Never a
    # scoring gold — target_answers stays the fixed guardrail.
    constraint_answers: list[str] | None = None


def choose_target_idx(question_id: str, m0: int, *, seed: int) -> int:
    """Deterministic, process-stable target interpretation index."""
    if m0 <= 0:
        raise ValueError("m0 must be positive")
    digest = hashlib.sha256(f"{question_id}::{seed}".encode("utf-8")).hexdigest()
    return int(digest, 16) % m0


def build_spec_levels(
    q: AmbigQuestion, *, seed: int, include_evidence: bool = True
) -> list[SpecRow]:
    """[SpecRow(level 0), SpecRow(level 1)] for one AmbigQuestion."""
    m0 = q.m0()
    idx = choose_target_idx(q.id, m0, seed=seed)
    target = q.interpretations[idx]
    # Order-preserving dedup of every interpretation's answers -> the ambiguous
    # question's full answer set (target first so it leads the OR short-circuit).
    all_answers: dict[str, None] = {}
    for a in target.answers:
        all_answers.setdefault(a, None)
    for interp in q.interpretations:
        for a in interp.answers:
            all_answers.setdefault(a, None)
    common = dict(
        question_id=q.id,
        target_answers=list(target.answers),
        all_answers=list(all_answers),
        m0=m0,
        target_idx=idx,
        evidence=list(q.evidence) if include_evidence else [],
    )
    return [
        SpecRow(spec_level=0, question_text=q.question, m_valid=m0, **common),
        SpecRow(
            spec_level=1,
            question_text=target.disambiguated_question,
            m_valid=1,
            **common,
        ),
    ]


def build_spec_levels_multilevel(
    q: AmbigQuestion,
    *,
    seed: int,
    mid_text: str,
    mid_subset: tuple[int, ...],
    include_evidence: bool = True,
) -> list[SpecRow]:
    """[L0, L_mid, L_top] for one m0>=3 AmbigQuestion (FINAL_PHASE_PLAN C1).

    spec_level semantics in THIS ladder: 0 = ambiguous (m_valid=m0),
    1 = partially disambiguated (m_valid=|subset|), 2 = fully disambiguated
    (m_valid=1). Distinct from the two-level ladder's {0,1} — multi-level runs
    write to their own parquet + paraphrase cache, never mixed with v3 files.

    Guardrails identical to the two-level builder: `target_answers` fixed
    across ALL levels; evidence bundle identical across ALL levels. The target
    interpretation is chosen with the same seed -> the same target as a
    two-level build of the same question (comparability).
    """
    m0 = q.m0()
    idx = choose_target_idx(q.id, m0, seed=seed)
    if idx not in set(mid_subset):
        raise ValueError(
            f"mid_subset {mid_subset} must contain target_idx {idx} (qid={q.id})")
    if not 1 < len(mid_subset) < m0:
        raise ValueError(
            f"mid_subset size {len(mid_subset)} not strictly between 1 and m0={m0}")
    if not mid_text.strip():
        raise ValueError(f"empty mid_text for qid={q.id}")
    target = q.interpretations[idx]
    all_answers: dict[str, None] = {}
    for a in target.answers:
        all_answers.setdefault(a, None)
    for interp in q.interpretations:
        for a in interp.answers:
            all_answers.setdefault(a, None)
    # L_mid paraphrase constraint: any answer of the ADMITTED subset (target
    # first for the OR short-circuit) — the level-1 analogue of all_answers.
    subset_answers: dict[str, None] = {}
    for a in target.answers:
        subset_answers.setdefault(a, None)
    for i in mid_subset:
        for a in q.interpretations[i].answers:
            subset_answers.setdefault(a, None)
    common = dict(
        question_id=q.id,
        target_answers=list(target.answers),
        all_answers=list(all_answers),
        m0=m0,
        target_idx=idx,
        evidence=list(q.evidence) if include_evidence else [],
    )
    return [
        SpecRow(spec_level=0, question_text=q.question, m_valid=m0, **common),
        SpecRow(spec_level=1, question_text=mid_text.strip(),
                m_valid=len(mid_subset),
                constraint_answers=list(subset_answers), **common),
        SpecRow(spec_level=2, question_text=target.disambiguated_question,
                m_valid=1, **common),
    ]


def target_in_evidence(q: AmbigQuestion, *, seed: int) -> bool:
    """Evidence-coverage filter (v2): does the bundle contain the TARGET
    interpretation's answer (any variant, case-insensitive containment)?

    Dataset-side and model-free — unlike a model-based answerability pre-screen
    it introduces no selection-on-model-knowledge bias. Verbatim containment is
    a conservative lower bound for answerability-from-evidence (52% of the
    ambiguous validation split passes; measured 2026-07-06).
    """
    bundle = q.evidence_text()
    if not bundle:
        return False
    idx = choose_target_idx(q.id, q.m0(), seed=seed)
    return any(a.lower() in bundle for a in q.interpretations[idx].answers)
