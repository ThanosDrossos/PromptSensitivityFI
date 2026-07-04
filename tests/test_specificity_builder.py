"""Specificity level builder (pivot spec §4): 2 rows, fixed gold, deterministic target."""

from __future__ import annotations

from prompt_sensitivity.data.ambigqa_schemas import AmbigInterpretation, AmbigQuestion
from prompt_sensitivity.specificity.build_levels import build_spec_levels, choose_target_idx


def _q(qid: str = "q1", n: int = 3) -> AmbigQuestion:
    return AmbigQuestion(
        id=qid,
        question="Who leads the newly established nation?",
        interpretations=[
            AmbigInterpretation(
                disambiguated_question=f"Who leads nation variant {i}?",
                answers=[f"leader-{i}", f"alias-{i}"],
            )
            for i in range(n)
        ],
    )


def test_builder_returns_two_rows_with_guardrail_gold():
    rows = build_spec_levels(_q(), seed=42)
    assert [r.spec_level for r in rows] == [0, 1]
    # THE GUARDRAIL: target_answers identical across both levels.
    assert rows[0].target_answers == rows[1].target_answers
    assert rows[0].target_idx == rows[1].target_idx
    # level 0 carries the ambiguous original; level 1 the target interpretation.
    assert rows[0].question_text == "Who leads the newly established nation?"
    assert rows[1].question_text == f"Who leads nation variant {rows[1].target_idx}?"
    # gold matches the chosen interpretation's variants
    assert rows[0].target_answers == [
        f"leader-{rows[0].target_idx}", f"alias-{rows[0].target_idx}"
    ]


def test_m_valid_semantics():
    rows = build_spec_levels(_q(n=4), seed=7)
    assert rows[0].m_valid == rows[0].m0 == 4
    assert rows[1].m_valid == 1 and rows[1].m0 == 4


def test_target_choice_deterministic_and_seed_sensitive():
    a = choose_target_idx("q1", 5, seed=42)
    assert a == choose_target_idx("q1", 5, seed=42)          # stable
    assert build_spec_levels(_q(n=5), seed=42)[0].target_idx == a
    idxs = {choose_target_idx("q1", 5, seed=s) for s in range(30)}
    assert len(idxs) > 1                                      # seed actually matters
    idxs_q = {choose_target_idx(f"q{i}", 5, seed=42) for i in range(30)}
    assert len(idxs_q) > 1                                    # id actually matters
