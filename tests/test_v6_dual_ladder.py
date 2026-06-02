"""CoT prompt + reasoning ladder + expected_hop_coverage (v6)."""

from __future__ import annotations

import pytest

from prompt_sensitivity.data.schemas import (
    DecompositionHop,
    HotpotParagraph,
    MultiHopQuestion,
)
from prompt_sensitivity.ladders import (
    build_reasoning_ladder,
    expected_hop_coverage,
    render_reasoning_scaffold,
)
from prompt_sensitivity.prompts import (
    QA_COT_SYSTEM_PROMPT,
    assemble_qa_cot_messages,
    parse_answer_line,
)


def _musique_q(n_hops: int = 4) -> MultiHopQuestion:
    paras = [
        HotpotParagraph(title=f"P{i}", sentences=[f"text {i}"], is_gold=(i < n_hops))
        for i in range(12)
    ]
    decomp = [
        DecompositionHop(
            hop_idx=i, sub_question=f"hop {i}?", sub_answer=f"ans{i}", supporting_paragraph_idx=i
        )
        for i in range(n_hops)
    ]
    return MultiHopQuestion(
        id=f"{n_hops}hop__a_b",
        dataset="musique",
        question="Full question?",
        answer=f"ans{n_hops - 1}",
        question_type=f"{n_hops}hop",  # type: ignore[arg-type]
        paragraphs=paras,
        supporting_facts=[],
        question_decomposition=decomp,
        n_hops=n_hops,
    )


# --- CoT prompt ------------------------------------------------------------


def test_cot_messages_use_cot_system_prompt():
    paras = [HotpotParagraph(title="P0", sentences=["Paris is the capital."])]
    msgs = assemble_qa_cot_messages("What is the capital of France?", paras)
    assert msgs[0].content == QA_COT_SYSTEM_PROMPT
    assert "step by step" in msgs[1].content.lower()
    assert "Answer:" in msgs[1].content


def test_cot_system_prompt_requests_intermediate_steps_and_answer_line():
    sp = QA_COT_SYSTEM_PROMPT.lower()
    assert "step by step" in sp
    assert "answer:" in sp


def test_parse_answer_line_extracts_final_answer():
    resp = "Step 1: X is A.\nStep 2: A's capital is B.\nAnswer: B"
    assert parse_answer_line(resp) == "B"


def test_parse_answer_line_takes_last_marker():
    resp = "Answer: draft\nmore reasoning\nAnswer: final"
    assert parse_answer_line(resp) == "final"


def test_parse_answer_line_falls_back_to_whole_response():
    resp = "Just a bare answer with no marker"
    assert parse_answer_line(resp) == "Just a bare answer with no marker"


def test_parse_answer_line_empty():
    assert parse_answer_line("") == ""


# --- reasoning ladder ------------------------------------------------------


def test_reasoning_ladder_withholds_final_hop():
    q = _musique_q(n_hops=4)
    rows = build_reasoning_ladder(q)
    # 4 hops -> rungs with 0,1,2,3 hops provided (never 4).
    assert [r.hops_provided for r in rows] == [0, 1, 2, 3]
    assert max(r.hops_provided for r in rows) == q.n_hops - 1
    assert all(r.ladder_family == "reasoning" for r in rows)
    assert all(r.ladder_type == "reasoning" for r in rows)


def test_reasoning_ladder_raises_without_decomposition():
    q = MultiHopQuestion(
        id="hp1",
        dataset="hotpotqa",
        question="Q?",
        answer="a",
        question_type="bridge",
        paragraphs=[HotpotParagraph(title="t", sentences=["s"], is_gold=True)],
        supporting_facts=[],
    )
    with pytest.raises(ValueError):
        build_reasoning_ladder(q)


def test_render_scaffold_shows_provided_hops_only():
    q = _musique_q(n_hops=4)
    scaffold = render_reasoning_scaffold(q, hops_provided=2)
    assert "hop 0?" in scaffold and "ans0" in scaffold
    assert "hop 1?" in scaffold and "ans1" in scaffold
    assert "hop 2?" not in scaffold  # not yet provided
    assert "hop 3?" not in scaffold  # final hop withheld


# --- expected_hop_coverage -------------------------------------------------


def test_expected_hop_coverage_linear():
    # l/N for each hop -> mean l/N.
    assert expected_hop_coverage(N=20, n_hops=4, l=0) == 0.0
    assert expected_hop_coverage(N=20, n_hops=4, l=10) == 0.5
    assert expected_hop_coverage(N=20, n_hops=4, l=20) == 1.0


def test_expected_hop_coverage_edge_cases():
    assert expected_hop_coverage(N=0, n_hops=4, l=5) == 0.0
    assert expected_hop_coverage(N=10, n_hops=0, l=5) == 0.0
    # l clamped to [0, N]
    assert expected_hop_coverage(N=10, n_hops=2, l=99) == 1.0
    assert expected_hop_coverage(N=10, n_hops=2, l=-3) == 0.0
