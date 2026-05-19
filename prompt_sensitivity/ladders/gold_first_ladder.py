"""Gold-first ladder. Research_Design_v3 §4.1 (best-case bound).

Permutation puts the 2 gold paragraphs first, then the 8 distractors in
their original dataset order. The prefix at each level then deterministically
contains as many gold paragraphs as possible.

At levels {0, 2, 4, 6, 8, 10} (paragraph counts), the gold_count is
{0, 2, 2, 2, 2, 2}: at level 2 we have exactly the 2 gold; at any larger
level we still have both gold plus a growing distractor tail.
"""

from __future__ import annotations

from typing import Sequence

from ..config import load_config
from ..data import MultiHopQuestion
from .schemas import LadderRow, LevelSlice


def _gold_first_permutation(question: MultiHopQuestion) -> list[int]:
    """Indices: [gold0, gold1, distractor0, distractor1, ...] in original order."""
    gold = [i for i, p in enumerate(question.paragraphs) if p.is_gold]
    distractors = [i for i, p in enumerate(question.paragraphs) if not p.is_gold]
    return gold + distractors


def _slice_levels(
    permutation: Sequence[int],
    paragraph_titles: Sequence[str],
    gold_indices: set[int],
    levels: Sequence[int],
) -> list[LevelSlice]:
    out: list[LevelSlice] = []
    for size in levels:
        prefix = list(permutation[:size])
        titles = [paragraph_titles[i] for i in prefix]
        gold_count = sum(1 for i in prefix if i in gold_indices)
        out.append(
            LevelSlice(
                paragraph_indices=prefix,
                paragraph_titles=titles,
                gold_count=gold_count,
            )
        )
    return out


def build_gold_first_ladder(
    question: MultiHopQuestion,
    *,
    levels: Sequence[int] | None = None,
) -> list[LadderRow]:
    """Gold-first ladder. Returns 6 rows."""
    config = load_config()
    if levels is None:
        levels = config.ladders.levels

    perm = _gold_first_permutation(question)
    gold_indices = {i for i, p in enumerate(question.paragraphs) if p.is_gold}
    titles = [p.title for p in question.paragraphs]

    rows: list[LadderRow] = []
    for level_idx, slice_ in enumerate(_slice_levels(perm, titles, gold_indices, levels)):
        rows.append(
            LadderRow(
                question_id=question.id,
                ladder_type="gold_first",
                level_idx=level_idx,
                level=levels[level_idx],
                paragraph_indices=slice_.paragraph_indices,
                paragraph_titles=slice_.paragraph_titles,
                gold_count=slice_.gold_count,
                permutation=list(perm),
            )
        )
    return rows
