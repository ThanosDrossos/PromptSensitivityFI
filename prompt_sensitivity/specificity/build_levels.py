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

from pydantic import BaseModel, ConfigDict

from ..data.ambigqa_schemas import AmbigQuestion


class SpecRow(BaseModel):
    """One (question, specificity level) cell descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    spec_level: int                 # 0 = ambiguous, 1 = disambiguated
    question_text: str              # Q at level 0, Q_i at level 1
    target_answers: list[str]       # a_i variants, FIXED across both levels
    m_valid: int                    # level 0 -> m0, level 1 -> 1
    m0: int
    target_idx: int                 # which interpretation was chosen as target


def choose_target_idx(question_id: str, m0: int, *, seed: int) -> int:
    """Deterministic, process-stable target interpretation index."""
    if m0 <= 0:
        raise ValueError("m0 must be positive")
    digest = hashlib.sha256(f"{question_id}::{seed}".encode("utf-8")).hexdigest()
    return int(digest, 16) % m0


def build_spec_levels(q: AmbigQuestion, *, seed: int) -> list[SpecRow]:
    """[SpecRow(level 0), SpecRow(level 1)] for one AmbigQuestion."""
    m0 = q.m0()
    idx = choose_target_idx(q.id, m0, seed=seed)
    target = q.interpretations[idx]
    common = dict(
        question_id=q.id,
        target_answers=list(target.answers),
        m0=m0,
        target_idx=idx,
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
