"""v2 uniform-evidence design: evidence parsing, guardrails, prompt wiring."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_sensitivity.data.ambigqa_schemas import AmbigQuestion, EvidenceSnippet
from prompt_sensitivity.data.load_ambigqa import parse_ambigqa_record
from prompt_sensitivity.scripts.e2e_smoke import _assemble_messages
from prompt_sensitivity.scripts.run_specificity import (
    _evidence_paragraphs,
    _ladder_row_for,
    _SpecQuestionView,
)
from prompt_sensitivity.specificity.build_levels import (
    build_spec_levels,
    target_in_evidence,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ambigqa_sample.json"


def _records():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------- loader: evidence -------------------------------


def test_evidence_parsed_and_cleaned_hf_form():
    q = parse_ambigqa_record(_records()[0])
    assert len(q.evidence) == 2
    first = q.evidence[0]
    assert first.title.startswith("History of the St. Louis Cardinals")
    # HTML tags stripped, entities unescaped, hard wraps collapsed
    assert "<b>" not in first.snippet and "&nbsp;" not in first.snippet
    assert "overall mediocrity of the Cardinals" in first.snippet
    assert "\n" not in first.snippet


def test_evidence_parsed_release_form_identically():
    hf = parse_ambigqa_record(_records()[0])
    rel = parse_ambigqa_record(_records()[2])
    assert [e.snippet for e in rel.evidence] == [e.snippet for e in hf.evidence]


def test_light_config_record_has_no_evidence():
    q = parse_ambigqa_record(_records()[1])   # dexter record: no used_queries
    assert q.evidence == []


# ----------------------- builder: guardrails --------------------------------


def _q_with_evidence():
    q = parse_ambigqa_record(_records()[0])
    assert q.is_ambiguous()
    return q


def test_evidence_identical_across_levels_and_optional():
    q = _q_with_evidence()
    r0, r1 = build_spec_levels(q, seed=42)
    assert r0.evidence == r1.evidence and len(r0.evidence) == 2   # guardrail #2
    c0, c1 = build_spec_levels(q, seed=42, include_evidence=False)
    assert c0.evidence == [] and c1.evidence == []                 # closed-book mode


def test_target_in_evidence_filter():
    q = _q_with_evidence()
    # the fixture snippet contains ALL three interpretation answers -> any target hits
    assert target_in_evidence(q, seed=42) is True
    bare = AmbigQuestion(id="x", question="Q?", interpretations=q.interpretations)
    assert target_in_evidence(bare, seed=42) is False              # no evidence -> filtered


# ----------------------- driver: prompt wiring ------------------------------


def test_uniform_evidence_reaches_the_prompt():
    q = _q_with_evidence()
    r0, _ = build_spec_levels(q, seed=42)
    paras = _evidence_paragraphs(r0, max_chars=6000)
    view = _SpecQuestionView(r0, paragraphs=paras)
    lrow = _ladder_row_for(r0, n_paragraphs=len(paras))
    msgs = _assemble_messages(view, "Some paraphrase?", lrow, use_cot=False)
    user = msgs[1].content
    assert "overall mediocrity of the Cardinals" in user   # snippet text present
    assert "Some paraphrase?" in user
    # closed-book still works
    view_cb = _SpecQuestionView(r0, paragraphs=[])
    msgs_cb = _assemble_messages(view_cb, "Some paraphrase?", _ladder_row_for(r0, 0),
                                 use_cot=False)
    assert "mediocrity" not in msgs_cb[1].content


def test_evidence_cap_keeps_whole_snippets_and_at_least_one():
    row_like = build_spec_levels(_q_with_evidence(), seed=42)[0]
    # tiny cap: still returns the first snippet whole
    paras = _evidence_paragraphs(row_like, max_chars=10)
    assert len(paras) == 1
    assert paras[0].sentences[0] == row_like.evidence[0].snippet
    # generous cap: everything
    assert len(_evidence_paragraphs(row_like, max_chars=10_000)) == 2


def test_evidence_snippet_roundtrip_schema():
    e = EvidenceSnippet(title="T", snippet="S")
    assert e.title == "T" and e.snippet == "S"
