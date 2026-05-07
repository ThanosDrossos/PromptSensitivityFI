"""Stratified sampler unit tests. Sprint 1 §1.4."""

from __future__ import annotations

import random

from prompt_sensitivity.data import MultiHopQuestion, stratified_sample
from prompt_sensitivity.data.schemas import HotpotParagraph, HotpotSupportingFact


def _make_q(qid: str, level: str, qtype: str = "bridge", *, gold_count: int = 2) -> MultiHopQuestion:
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            HotpotParagraph(
                title=f"{qid}_p{i}",
                sentences=[f"sentence {i}."],
                is_gold=(i < gold_count),
            )
        )
    sf = [HotpotSupportingFact(title=p.title, sent_id=0) for p in paragraphs if p.is_gold]
    return MultiHopQuestion(
        id=qid,
        dataset="hotpotqa",
        question="?",
        answer="x",
        question_type=qtype,
        level=level,
        paragraphs=paragraphs,
        supporting_facts=sf,
    )


def test_balanced_split_by_level():
    """100 questions: 50 medium, 50 hard from a pool dominated by medium."""
    rng = random.Random(0)
    pool = [_make_q(f"med_{i}", "medium") for i in range(200)]
    pool += [_make_q(f"hard_{i}", "hard") for i in range(80)]
    rng.shuffle(pool)

    sample = stratified_sample(pool, n_total=100, stratify_by="level", seed=42)
    assert len(sample) == 100
    counts = {"medium": 0, "hard": 0}
    for q in sample:
        counts[q.level] += 1
    # Equal-share allocation: 50 medium, 50 hard.
    assert counts == {"medium": 50, "hard": 50}


def test_sampler_is_seeded():
    """Same seed -> same IDs in the same order."""
    pool = [_make_q(f"q{i}", "medium" if i % 2 == 0 else "hard") for i in range(80)]
    a = stratified_sample(pool, n_total=20, stratify_by="level", seed=42)
    b = stratified_sample(pool, n_total=20, stratify_by="level", seed=42)
    assert [q.id for q in a] == [q.id for q in b]


def test_drops_questions_with_too_few_gold_paragraphs():
    """Edge case Sprint 3 §3.1: <2 gold means exclude."""
    pool = [_make_q(f"q{i}", "medium", gold_count=1) for i in range(10)]
    pool += [_make_q(f"q{i+100}", "hard", gold_count=2) for i in range(10)]
    sample = stratified_sample(pool, n_total=10, stratify_by="level", seed=42, k_gold=2)
    # All survivors must have >=2 gold paragraphs.
    for q in sample:
        assert len(q.gold_paragraphs()) >= 2
    # And only `hard` records survived the filter.
    assert {q.level for q in sample} == {"hard"}


def test_stratify_by_type_4_groups():
    """2WikiMultihopQA has 4 question types — sampler must produce ~25 per type at n=100."""
    types = ["comparison", "inference", "compositional", "bridge_comparison"]
    pool = []
    for i, t in enumerate(types):
        for j in range(50):
            pool.append(_make_q(f"{t}_{j}", "medium", qtype=t))
    sample = stratified_sample(pool, n_total=100, stratify_by="type", seed=7)
    assert len(sample) == 100
    counts = {t: 0 for t in types}
    for q in sample:
        counts[q.question_type] += 1
    # Equal-share: 25 each.
    assert all(c == 25 for c in counts.values())
