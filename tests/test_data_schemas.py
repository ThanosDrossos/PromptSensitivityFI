"""Pydantic schemas + dataset-record parsers."""

from __future__ import annotations

import pytest

from prompt_sensitivity.data import (
    MultiHopQuestion,
    parse_hotpotqa_record,
    parse_twiki_record,
)


def _make_hotpot_record(level: str = "medium") -> dict:
    """Synthetic but field-faithful HotpotQA record."""
    return {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": level,
        "context": {
            "title": [
                "Scott Derrickson",
                "Ed Wood",
                "Tim Burton",
                "Plan 9 from Outer Space",
                "Doctor Strange (film)",
                "Glen or Glenda",
                "American film director",
                "Sinister (film)",
                "B movie",
                "Ed Wood (film)",
            ],
            "sentences": [
                ["Scott Derrickson is an American filmmaker. ", "He was born in 1966."],
                ["Edward Davis Wood Jr. was an American filmmaker. ", "He died in 1978."],
                ["Burton is American. "],
                ["Plan 9 is a film. "],
                ["Doctor Strange is a 2016 film. "],
                ["Glen or Glenda is a 1953 film. "],
                ["American film director sentence. "],
                ["Sinister is a 2012 film. "],
                ["A B movie is a low budget film. "],
                ["Ed Wood is a 1994 film. "],
            ],
        },
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
    }


def _make_twiki_record() -> dict:
    return {
        "id": "5a7a06935542990198eaf050",
        "question": "Who is the maternal grandfather of Albert II of Monaco?",
        "answer": "John Kelly Sr.",
        "type": "inference",
        "context": {
            "title": [f"P{i}" for i in range(10)],
            "sentences": [
                [f"P{i} sentence one. "] for i in range(10)
            ],
        },
        "supporting_facts": {
            "title": ["P0", "P3"],
            "sent_id": [0, 0],
        },
    }


def test_hotpot_parse_round_trip():
    record = _make_hotpot_record()
    q = parse_hotpotqa_record(record)
    assert isinstance(q, MultiHopQuestion)
    assert q.dataset == "hotpotqa"
    assert q.id == "5a8b57f25542995d1e6f1371"
    assert q.question_type == "comparison"
    assert q.level == "medium"
    assert len(q.paragraphs) == 10
    # Exactly 2 gold paragraphs (matches supporting_facts titles).
    gold = q.gold_paragraphs()
    assert len(gold) == 2
    assert {p.title for p in gold} == {"Scott Derrickson", "Ed Wood"}
    # And exactly 8 distractors.
    assert len(q.distractor_paragraphs()) == 8


def test_hotpot_paragraph_join_preserves_text():
    record = _make_hotpot_record()
    q = parse_hotpotqa_record(record)
    derrickson = next(p for p in q.paragraphs if p.title == "Scott Derrickson")
    # Sentences are joined verbatim, NOT re-spaced — the dataset already includes trailing spaces.
    assert derrickson.joined() == "Scott Derrickson is an American filmmaker. He was born in 1966."


def test_twiki_parse_uses_2wiki_dataset_label():
    q = parse_twiki_record(_make_twiki_record())
    assert q.dataset == "2wikimultihopqa"
    assert q.level is None
    assert q.question_type == "inference"


def test_mismatched_context_arrays_raises():
    """Edge case: titles and sentences lists must align."""
    record = _make_hotpot_record()
    record["context"]["title"] = record["context"]["title"][:5]  # corrupt
    with pytest.raises(ValueError, match="context arrays mismatched"):
        parse_hotpotqa_record(record)
