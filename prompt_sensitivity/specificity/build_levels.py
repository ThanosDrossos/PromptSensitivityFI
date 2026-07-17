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
    spec_level: int                 # 0 = ambiguous, 1 = disambiguated
    question_text: str              # Q at level 0, Q_i at level 1
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
