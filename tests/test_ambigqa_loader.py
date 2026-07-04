"""AmbigQA loader — parsing both dataset forms against the fixture (no download)."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_sensitivity.data.load_ambigqa import accepts, parse_ambigqa_record

FIXTURE = Path(__file__).parent / "fixtures" / "ambigqa_sample.json"


def _records():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_multipleqas_record_parses_with_m0_3():
    q = parse_ambigqa_record(_records()[0])
    assert q is not None
    assert q.dataset == "ambigqa"
    assert q.m0() == 3 and q.is_ambiguous()
    assert q.question.startswith("Why did the st louis cardinals")
    # interpretations carry (disambiguated question, answer-variant list) pairs
    assert q.interpretations[1].disambiguated_question.startswith("What physical issue")
    assert q.interpretations[1].answers == ["old stadium"]


def test_single_answer_record_parses_as_m0_1_and_is_filtered():
    q = parse_ambigqa_record(_records()[1])
    assert q is not None
    assert q.m0() == 1 and not q.is_ambiguous()
    # answer VARIANTS land as a list on the single interpretation
    assert q.interpretations[0].answers == ["Tony Goldwyn", "Goldwyn"]
    # default filter (min_interpretations=2) drops it ...
    assert accepts(q, min_interpretations=2) is False
    # ... unless the single-answer anchor is explicitly enabled
    assert accepts(q, min_interpretations=2, include_single_answer_anchor=True) is True


def test_release_form_parses_identically_to_hf_form():
    hf = parse_ambigqa_record(_records()[0])
    rel = parse_ambigqa_record(_records()[2])
    assert hf is not None and rel is not None
    assert rel.m0() == hf.m0() == 3
    assert [i.disambiguated_question for i in rel.interpretations] == [
        i.disambiguated_question for i in hf.interpretations
    ]
    assert [i.answers for i in rel.interpretations] == [i.answers for i in hf.interpretations]


def test_ambiguous_record_passes_default_filter():
    q = parse_ambigqa_record(_records()[0])
    assert accepts(q, min_interpretations=2) is True


def test_unusable_record_returns_none():
    assert parse_ambigqa_record({"id": "x", "question": "", "annotations": {}}) is None
    assert parse_ambigqa_record({"id": "", "question": "q?", "annotations": None}) is None
