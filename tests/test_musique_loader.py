"""MuSiQue loader + schema extensions (v6 migration)."""

from __future__ import annotations

import pytest

from prompt_sensitivity.data import MultiHopQuestion, parse_musique_record
from prompt_sensitivity.data.schemas import DecompositionHop, HotpotParagraph


def _musique_record(n_hops: int = 3) -> dict:
    """A field-faithful MuSiQue-Ans record dict (canonical jsonl shape)."""
    paragraphs = []
    for i in range(8):  # MuSiQue pools are larger + variable; 8 here for the fixture
        paragraphs.append(
            {
                "idx": i,
                "title": f"Title {i}",
                "paragraph_text": f"Paragraph text number {i}.",
                "is_supporting": i < n_hops,  # first n_hops are gold
            }
        )
    decomposition = []
    for i in range(n_hops):
        decomposition.append(
            {
                "id": i,
                "question": f"hop {i} sub-question referencing #{i}?" if i else "hop 0 sub-question?",
                "answer": f"sub-answer-{i}",
                "paragraph_support_idx": i,
            }
        )
    return {
        "id": f"{n_hops}hop__1000_2000",
        "question": "The full multi-hop question?",
        "answer": f"sub-answer-{n_hops - 1}",
        "paragraphs": paragraphs,
        "question_decomposition": decomposition,
    }


def test_parse_populates_decomposition():
    q = parse_musique_record(_musique_record(n_hops=3))
    assert isinstance(q, MultiHopQuestion)
    assert q.dataset == "musique"
    assert q.n_hops == 3
    assert q.question_type == "3hop"
    assert len(q.question_decomposition) == 3
    assert q.has_decomposition()
    hop0 = q.question_decomposition[0]
    assert isinstance(hop0, DecompositionHop)
    assert hop0.hop_idx == 0
    assert hop0.sub_answer == "sub-answer-0"
    assert hop0.supporting_paragraph_idx == 0


def test_parse_sets_gold_from_is_supporting():
    """is_supporting -> is_gold, preserved through the validator (no supporting_facts)."""
    q = parse_musique_record(_musique_record(n_hops=4))
    gold = q.gold_paragraphs()
    assert len(gold) == 4
    # The first 4 paragraphs are the gold ones.
    assert {p.title for p in gold} == {f"Title {i}" for i in range(4)}
    # supporting_facts stays empty for MuSiQue.
    assert q.supporting_facts == []


def test_parse_handles_text_field_alias():
    """Some HF mirrors use `text` instead of `paragraph_text`."""
    rec = _musique_record(n_hops=2)
    for p in rec["paragraphs"]:
        p["text"] = p.pop("paragraph_text")
    q = parse_musique_record(rec)
    assert q.paragraphs[0].joined() == "Paragraph text number 0."


def test_two_hop_is_clamped_label():
    q = parse_musique_record(_musique_record(n_hops=2))
    assert q.question_type == "2hop"
    assert q.n_hops == 2


def test_paragraph_pool_not_hardcoded_to_10():
    """MuSiQue pools vary; nothing should assume 10."""
    rec = _musique_record(n_hops=3)
    # Trim to 5 paragraphs — loader must accept it.
    rec["paragraphs"] = rec["paragraphs"][:5]
    q = parse_musique_record(rec)
    assert len(q.paragraphs) == 5


def test_empty_supporting_facts_does_not_wipe_is_gold():
    """Regression: the validator must not clear loader-set is_gold for MuSiQue."""
    q = MultiHopQuestion(
        id="musique_x",
        dataset="musique",
        question="Q?",
        answer="a",
        question_type="2hop",
        paragraphs=[
            HotpotParagraph(title="g", sentences=["gold"], is_gold=True),
            HotpotParagraph(title="d", sentences=["distractor"], is_gold=False),
        ],
        supporting_facts=[],
        question_decomposition=[
            DecompositionHop(hop_idx=0, sub_question="q0?", sub_answer="a0"),
            DecompositionHop(hop_idx=1, sub_question="q1?", sub_answer="a1"),
        ],
        n_hops=2,
    )
    assert len(q.gold_paragraphs()) == 1  # the gold flag survived
