"""Pipeline orchestrator test — mocks the gateway + NLI to keep CI fast.

The mocks live in this file so the production code stays free of test-mode
branches. We patch four functions:

  - generate.generate_raw_paraphrases  -> returns a fixed list per attempt
  - nli_filter.filter_by_nli           -> deterministic per-paraphrase scores
  - constraint_filter.filter_by_constraint -> deterministic Jaccard scores

This is enough to exercise: dedup, regeneration, fallback-threshold relaxation,
and the dropped-question case.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence
from unittest.mock import patch

import pytest

from prompt_sensitivity.paraphrases import schemas as ps_schemas
from prompt_sensitivity.paraphrases.constraint_filter import JaccardResult
from prompt_sensitivity.paraphrases.nli_filter import NLIScores
from prompt_sensitivity.paraphrases.pipeline import build_paraphrase_set
from prompt_sensitivity.paraphrases.schemas import RawParaphrase


def _make_raw(qid: str, role: str, idx: int, text: str) -> RawParaphrase:
    return RawParaphrase(
        question_id=qid,
        role=role,  # type: ignore[arg-type]
        sample_idx=idx,
        text=text,
        generator_model_key="gpt_4o",
        generator_seed=1000 + idx,
        request_hash=f"h{idx:08x}",
    )


def _diverse_paraphrases(n: int) -> list[str]:
    """Generate n texts that are pairwise edit-distance >= 6 apart.

    We use SHA256 prefixes as the differentiator: any two prefixes differ in
    nearly every position, so pairwise Levenshtein distance is comfortably
    above the configured min_distance=6 even after the shared prefix wording.
    """
    import hashlib
    return [
        f"Variant: {hashlib.sha256(str(i).encode()).hexdigest()[:32]}"
        for i in range(n)
    ]


def _scripted_generate(*, _per_attempt: list[list[RawParaphrase]]):
    """Build a side_effect callable that returns successive batches."""
    state = {"i": 0}

    def _impl(question_id, question_text, *, config=None, sample_idxs=None, roles=None):
        i = state["i"]
        state["i"] += 1
        if i < len(_per_attempt):
            return _per_attempt[i]
        return []

    return _impl


def _high_nli(_original, candidates: Iterable[str], *, config=None, threshold=None):
    """All candidates pass NLI with fwd=bwd=0.95."""
    return [(True, NLIScores(entail_fwd=0.95, entail_bwd=0.95)) for _ in candidates]


def _low_nli(_original, candidates: Iterable[str], *, config=None, threshold=None):
    """All candidates fail NLI."""
    return [(False, NLIScores(entail_fwd=0.4, entail_bwd=0.4)) for _ in candidates]


def _high_constraint(_original, candidates: Iterable[str], *, config=None, threshold=None):
    return [
        (True, JaccardResult(jaccard=0.95, a_set=frozenset(), b_set=frozenset()))
        for _ in candidates
    ]


def test_happy_path_30_accepted(monkeypatch):
    """40 diverse candidates -> 30 accepted in one attempt, no regeneration."""
    raw = [_make_raw("q1", "neutral", i, t) for i, t in enumerate(_diverse_paraphrases(40))]
    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=[raw]),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_high_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=_high_constraint,
    ):
        pset = build_paraphrase_set("q1", "What is the capital of France?")
    assert pset.is_complete(30)
    assert len(pset.accepted) == 30
    # No NLI / constraint rejections in the happy path.
    reasons = {r.reason for r in pset.rejected}
    assert "nli_low" not in reasons
    assert "constraint_mismatch" not in reasons


def test_regeneration_until_target(monkeypatch):
    """First batch yields 20, second batch yields 20 -> 40 candidates, dedup to 30."""
    batch1 = [_make_raw("q2", "neutral", i, t) for i, t in enumerate(_diverse_paraphrases(20))]
    import hashlib
    batch2 = [
        _make_raw(
            "q2",
            "neutral",
            100 + i,
            f"Other-batch: {hashlib.sha256(f'b2-{i}'.encode()).hexdigest()[:32]}",
        )
        for i in range(20)
    ]
    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=[batch1, batch2]),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_high_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=_high_constraint,
    ):
        pset = build_paraphrase_set("q2", "Q?")
    assert pset.is_complete(30)
    assert pset.regeneration_attempts >= 2


def test_dropped_when_nli_never_passes(monkeypatch):
    """Even after exhausting attempts + fallback threshold, dropped=True."""
    # Provide 1 candidate per attempt so we DO exhaust the regeneration budget.
    batches = [[_make_raw("q3", "neutral", i, t)] for i, t in enumerate(_diverse_paraphrases(60))]
    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=batches),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_low_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=_high_constraint,
    ):
        pset = build_paraphrase_set("q3", "Q?")
    assert pset.dropped is True
    assert pset.n_accepted() < 30
    # NLI rejections must be captured for the audit log.
    nli_reject_count = sum(1 for r in pset.rejected if r.reason in ("nli_low", "nli_one_direction"))
    assert nli_reject_count > 0


def test_gold_answer_routes_to_gold_filter(monkeypatch):
    """When gold_answer is supplied, the gold-based filter is used and the
    legacy Jaccard filter is NOT called.
    """
    raw = [_make_raw("q5", "neutral", i, t) for i, t in enumerate(_diverse_paraphrases(40))]

    gold_calls: list[list[str]] = []
    jaccard_calls: list[list[str]] = []

    def fake_gold_filter(candidates, gold, *, original_question=None, config=None):
        gold_calls.append(list(candidates))
        return [True] * len(list(candidates))  # all pass

    def fake_jaccard_filter(original, candidates, *, config=None, threshold=None):
        jaccard_calls.append(list(candidates))
        return [(True, JaccardResult(jaccard=1.0, a_set=frozenset(), b_set=frozenset()))
                for _ in candidates]

    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=[raw]),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_high_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint_with_gold",
        side_effect=fake_gold_filter,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=fake_jaccard_filter,
    ):
        pset = build_paraphrase_set("q5", "What?", gold_answer="Paris")

    assert pset.is_complete(30)
    assert gold_calls, "gold filter must be used when gold_answer is supplied"
    assert not jaccard_calls, "jaccard filter must NOT be called when gold_answer is supplied"


def test_no_gold_answer_falls_back_to_jaccard(monkeypatch):
    """When gold_answer is None, the legacy Jaccard filter is used."""
    raw = [_make_raw("q6", "neutral", i, t) for i, t in enumerate(_diverse_paraphrases(40))]

    gold_calls: list[list[str]] = []
    jaccard_calls: list[list[str]] = []

    def fake_gold_filter(candidates, gold, *, original_question=None, config=None):
        gold_calls.append(list(candidates))
        return [True] * len(list(candidates))

    def fake_jaccard_filter(original, candidates, *, config=None, threshold=None):
        jaccard_calls.append(list(candidates))
        return [(True, JaccardResult(jaccard=1.0, a_set=frozenset(), b_set=frozenset()))
                for _ in candidates]

    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=[raw]),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_high_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint_with_gold",
        side_effect=fake_gold_filter,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=fake_jaccard_filter,
    ):
        pset = build_paraphrase_set("q6", "What?")  # no gold_answer

    assert pset.is_complete(30)
    assert jaccard_calls, "jaccard filter must be used when no gold_answer"
    assert not gold_calls, "gold filter must NOT be called when gold_answer is None"


def test_exact_duplicate_of_original_is_rejected(monkeypatch):
    """A candidate that's identical (modulo case) to the original is rejected pre-NLI."""
    raw = [
        _make_raw("q4", "neutral", 0, "What is the capital of France?"),
        _make_raw("q4", "neutral", 1, "what is the capital of france?"),  # case-only -> exact_duplicate
    ] + [_make_raw("q4", "neutral", 2 + i, t) for i, t in enumerate(_diverse_paraphrases(30))]
    with patch(
        "prompt_sensitivity.paraphrases.pipeline.generate_raw_paraphrases",
        side_effect=_scripted_generate(_per_attempt=[raw]),
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_nli",
        side_effect=_high_nli,
    ), patch(
        "prompt_sensitivity.paraphrases.pipeline.filter_by_constraint",
        side_effect=_high_constraint,
    ):
        pset = build_paraphrase_set("q4", "What is the capital of France?")
    dup_rejects = [r for r in pset.rejected if r.reason == "exact_duplicate"]
    assert len(dup_rejects) >= 2
