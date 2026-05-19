"""Three-ladder construction tests. Research_Design_v3 §4.

Brief invariants we must enforce:
  1. At level 0, all three ladders produce empty context.
  2. At the top level (10 paragraphs), all three ladders produce the SAME
     paragraph set (sorted; only the ORDER differs).
  3. gold-first ladder has gold_count == K from level 2 upwards.
  4. distractor-first ladder has gold_count == 0 below the top, == K at top.
  5. Random ladder is deterministic given the same question_id.
"""

from __future__ import annotations

from prompt_sensitivity.data.schemas import (
    HotpotParagraph,
    HotpotSupportingFact,
    MultiHopQuestion,
)
from prompt_sensitivity.ladders import (
    build_distractor_first_ladder,
    build_gold_first_ladder,
    build_random_ladder,
    random_permutation,
)


def _make_question(qid: str, *, n_total: int = 10, gold_positions: tuple[int, ...] = (2, 7)) -> MultiHopQuestion:
    paragraphs = []
    for i in range(n_total):
        paragraphs.append(
            HotpotParagraph(
                title=f"P{i}",
                sentences=[f"Sentence for paragraph {i}."],
                is_gold=(i in gold_positions),
            )
        )
    sf = [HotpotSupportingFact(title=f"P{i}", sent_id=0) for i in gold_positions]
    return MultiHopQuestion(
        id=qid,
        dataset="hotpotqa",
        question="What is the test question?",
        answer="x",
        question_type="bridge",
        level="medium",
        paragraphs=paragraphs,
        supporting_facts=sf,
    )


# --- invariant 1: level 0 always empty -------------------------------------


def test_all_ladders_empty_at_level_zero():
    q = _make_question("q1")
    for build in (build_random_ladder, build_gold_first_ladder, build_distractor_first_ladder):
        rows = build(q)
        l0 = next(r for r in rows if r.level == 0)
        assert l0.paragraph_indices == []
        assert l0.gold_count == 0


# --- invariant 2: top level — same paragraph set across ladders -----------


def test_all_ladders_converge_at_top_level():
    q = _make_question("q2")
    sets = {}
    for ladder_name, build in [
        ("random", build_random_ladder),
        ("gold_first", build_gold_first_ladder),
        ("distractor_first", build_distractor_first_ladder),
    ]:
        rows = build(q)
        top = next(r for r in rows if r.level == 10)
        sets[ladder_name] = tuple(sorted(top.paragraph_indices))
    # All three must have the same paragraph set at the top level.
    assert sets["random"] == sets["gold_first"] == sets["distractor_first"]
    # And it must be all 10 indices.
    assert sets["random"] == tuple(range(10))


# --- invariant 3: gold-first ladder gold-count progression -----------------


def test_gold_first_ladder_gold_count_is_constant_at_K():
    """gold_first: 2 gold are in the prefix at every level >= 2."""
    q = _make_question("q3")
    rows = build_gold_first_ladder(q)
    counts = {r.level: r.gold_count for r in rows}
    assert counts[0] == 0
    for lvl in (2, 4, 6, 8, 10):
        assert counts[lvl] == 2, f"level {lvl}: expected 2 gold, got {counts[lvl]}"


# --- invariant 4: distractor-first ladder gold-count progression ----------


def test_distractor_first_ladder_holds_gold_until_top():
    """distractor_first: gold count is 0 below level 10, then 2."""
    q = _make_question("q4")
    rows = build_distractor_first_ladder(q)
    counts = {r.level: r.gold_count for r in rows}
    for lvl in (0, 2, 4, 6, 8):
        assert counts[lvl] == 0, f"level {lvl}: expected 0 gold, got {counts[lvl]}"
    assert counts[10] == 2


# --- invariant 5: random ladder is deterministic given question_id --------


def test_random_ladder_is_deterministic_per_question():
    q = _make_question("seeded_q")
    perm_a = random_permutation(q)
    perm_b = random_permutation(q)
    assert perm_a == perm_b
    # And different qids give different permutations (with very high prob).
    q2 = _make_question("different_q")
    perm_c = random_permutation(q2)
    assert perm_a != perm_c


def test_random_ladder_uses_full_paragraph_set():
    q = _make_question("q_full")
    rows = build_random_ladder(q)
    top = next(r for r in rows if r.level == 10)
    assert sorted(top.paragraph_indices) == list(range(10))
    # The permutation field is the per-question shuffle and must be a
    # permutation of [0, N).
    assert sorted(top.permutation) == list(range(10))


def test_ladder_preserves_paragraph_titles():
    """paragraph_titles must align with paragraph_indices in order."""
    q = _make_question("q_titles")
    for build in (build_random_ladder, build_gold_first_ladder, build_distractor_first_ladder):
        rows = build(q)
        for r in rows:
            for idx, title in zip(r.paragraph_indices, r.paragraph_titles, strict=True):
                assert title == f"P{idx}"
