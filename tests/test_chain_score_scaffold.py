"""P0-1: the reasoning scaffold must OR-credit hops it already supplies.

Reasoning level L feeds hops 0..L-1 as scaffold and asks only for the final
answer; a terse response then omits the given hops and the scorer mechanically
under-counts (Smoke_Run §5.3). With the scaffold passed, those hops are credited.

The DeBERTa NLI primitive is mocked (substring => entailment) so the test is
fast and deterministic; the real model is exercised on the cluster.
"""

from __future__ import annotations

from types import SimpleNamespace

from prompt_sensitivity.config import load_config
from prompt_sensitivity.data.schemas import DecompositionHop
from prompt_sensitivity.scoring import chain_score


def _fake_nli(premise, hypotheses, *, config=None):
    """Entailment iff the hypothesis string is contained in the premise."""
    out = []
    for h in hypotheses:
        hit = h in premise
        out.append(SimpleNamespace(
            entail_prob=1.0 if hit else 0.0,
            contradict_prob=0.0,
            passes_entail=hit,
            passes_contradict=True,
        ))
    return out


def _decomp():
    # Empty sub_questions => build_fact_statements yields the bare sub_answers,
    # so the substring NLI mock is exact.
    return [
        DecompositionHop(hop_idx=i, sub_question="", sub_answer=a, supporting_paragraph_idx=None)
        for i, a in enumerate(["alpha", "beta", "gamma", "42"])
    ]


def test_scaffold_credits_given_hops(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli)
    cfg = load_config()
    decomp = _decomp()
    response = "Answer: 42"                 # terse: only the final hop
    scaffold = "alpha beta gamma"           # first 3 hops supplied

    # single-response path
    assert chain_score.chain_completion_score(decomp, response, config=cfg, scaffold_text=scaffold) == 1.0
    assert chain_score.chain_completion_score(decomp, response, config=cfg) == 0.25

    # batch path (identical OR-gating)
    assert chain_score.chain_completion_score_batch(
        decomp, [response], config=cfg, scaffold_text=scaffold) == [1.0]
    assert chain_score.chain_completion_score_batch(decomp, [response], config=cfg) == [0.25]


def test_empty_scaffold_is_noop(monkeypatch):
    monkeypatch.setattr(chain_score, "score_batch_nli_with_gold", _fake_nli)
    cfg = load_config()
    decomp = _decomp()
    # "" and None must behave identically to no scaffold (no spurious credit).
    for sc in (None, "", "   "):
        assert chain_score.chain_completion_score_batch(
            decomp, ["Answer: 42"], config=cfg, scaffold_text=sc) == [0.25]
