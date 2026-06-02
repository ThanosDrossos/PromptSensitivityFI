"""Graded chain-completion scorer (v6). NLI is stubbed so no DeBERTa load."""

from __future__ import annotations

import pytest

from prompt_sensitivity.data.schemas import DecompositionHop
from prompt_sensitivity.scoring import chain_score
from prompt_sensitivity.scoring.chain_score import (
    build_fact_statements,
    chain_fraction,
    resolve_placeholders,
)
from prompt_sensitivity.scoring.nli_with_gold import NLIScoreResult


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


def test_chain_fraction_basic():
    assert chain_fraction([True, True, True, True]) == 1.0
    assert chain_fraction([False, False]) == 0.0
    assert chain_fraction([True, False, True, False]) == 0.5
    assert chain_fraction([True, False, False]) == pytest.approx(1 / 3)


def test_chain_fraction_empty_raises():
    with pytest.raises(ValueError):
        chain_fraction([])


def test_resolve_placeholders():
    decomp = [
        DecompositionHop(hop_idx=0, sub_question="Who founded Acme?", sub_answer="Jane Doe"),
        DecompositionHop(hop_idx=1, sub_question="Where was #1 born?", sub_answer="Berlin"),
    ]
    resolved = resolve_placeholders(decomp)
    assert resolved[0] == "Who founded Acme?"
    # #1 -> hop 0's sub_answer "Jane Doe"
    assert resolved[1] == "Where was Jane Doe born?"


def test_build_fact_statements_pairs_question_and_answer():
    decomp = [
        DecompositionHop(hop_idx=0, sub_question="Who founded Acme?", sub_answer="Jane Doe"),
    ]
    facts = build_fact_statements(decomp)
    assert facts == ["Who founded Acme? Jane Doe"]


def test_build_fact_statements_falls_back_to_answer_when_question_empty():
    decomp = [DecompositionHop(hop_idx=0, sub_question="", sub_answer="Jane Doe")]
    assert build_fact_statements(decomp) == ["Jane Doe"]


# --------------------------------------------------------------------------- #
# Scoring with a stubbed NLI                                                  #
# --------------------------------------------------------------------------- #


def _fake_nli_factory(good_marker: str = "GOOD"):
    """Return a stub for score_batch_nli_with_gold.

    A (premise, hypothesis) pair 'entails' iff `good_marker` appears in either
    string. Direction-agnostic, so a GOOD fact is recovered regardless of
    which side it's on.
    """

    def _fake(gold, answers, *, config=None, batch_size: int = 16):
        out = []
        for a in answers:
            hit = good_marker in (gold or "") or good_marker in (a or "")
            entail = 1.0 if hit else 0.0
            out.append(
                NLIScoreResult(
                    entail_prob=entail,
                    contradict_prob=0.0,
                    neutral_prob=1.0 - entail,
                    passes_entail=entail >= 0.7,
                    passes_contradict=True,
                )
            )
        return out

    return _fake


def _decomp(markers: list[str]) -> list[DecompositionHop]:
    """One hop per marker; the marker is baked into the sub_answer."""
    return [
        DecompositionHop(hop_idx=i, sub_question=f"hop {i}?", sub_answer=f"ans {m}")
        for i, m in enumerate(markers)
    ]


def test_all_hops_recovered_is_one(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli_factory())
    decomp = _decomp(["GOOD", "GOOD", "GOOD", "GOOD"])
    score = chain_score.chain_completion_score(decomp, "irrelevant response", config=_DummyCfg())
    assert score == 1.0


def test_no_hops_recovered_is_zero(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli_factory())
    decomp = _decomp(["x", "y", "z", "w"])  # no GOOD marker anywhere
    score = chain_score.chain_completion_score(decomp, "irrelevant", config=_DummyCfg())
    assert score == 0.0


def test_half_hops_recovered_is_half(monkeypatch):
    """The regression test for the v3 step-function bug: a STRICTLY-between F."""
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli_factory())
    decomp = _decomp(["GOOD", "x", "GOOD", "y"])  # 2 of 4 recover
    score = chain_score.chain_completion_score(decomp, "irrelevant", config=_DummyCfg())
    assert score == 0.5
    assert 0.0 < score < 1.0


def test_batch_variant_matches_single(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli_factory())
    decomp = _decomp(["GOOD", "x", "GOOD", "y"])
    responses = ["r0", "r1", "r2"]
    batch = chain_score.chain_completion_score_batch(decomp, responses, config=_DummyCfg())
    assert batch == [0.5, 0.5, 0.5]


def test_empty_decomposition_raises(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli_factory())
    with pytest.raises(ValueError):
        chain_score.chain_completion_score([], "resp", config=_DummyCfg())
    with pytest.raises(ValueError):
        chain_score.chain_completion_score_batch([], ["resp"], config=_DummyCfg())


class _DummyScoring:
    entail_threshold = 0.7
    contradict_threshold = 0.5


class _DummyCfg:
    """Minimal stand-in so the scorer doesn't call load_config()."""

    scoring = _DummyScoring()
