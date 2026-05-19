"""Distractor-first ladder. Research_Design_v3 §4.1 (worst-case bound).

Permutation puts the 8 distractor paragraphs first, then the 2 gold
paragraphs. Gold count at each level is {0, 0, 0, 0, 0, 2} for levels
{0, 2, 4, 6, 8, 10}: gold only enters at the very top of the ladder.
"""

from __future__ import annotations

from typing import Sequence

from ..config import load_config
from ..data import MultiHopQuestion
from .schemas import LadderRow, LevelSlice


def _distractor_first_permutation(question: MultiHopQuestion) -> list[int]:
    """Indices: [distractor0, distractor1, ..., gold0, gold1] in original order."""
    gold = [i for i, p in enumerate(question.paragraphs) if p.is_gold]
    distractors = [i for i, p in enumerate(question.paragraphs) if not p.is_gold]
    return distractors + gold


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


def build_distractor_first_ladder(
    question: MultiHopQuestion,
    *,
    levels: Sequence[int] | None = None,
) -> list[LadderRow]:
    """Distractor-first ladder. Returns 6 rows."""
    config = load_config()
    if levels is None:
        levels = config.ladders.levels

    perm = _distractor_first_permutation(question)
    gold_indices = {i for i, p in enumerate(question.paragraphs) if p.is_gold}
    titles = [p.title for p in question.paragraphs]

    rows: list[LadderRow] = []
    for level_idx, slice_ in enumerate(_slice_levels(perm, titles, gold_indices, levels)):
        rows.append(
            LadderRow(
                question_id=question.id,
                ladder_type="distractor_first",
                level_idx=level_idx,
                level=levels[level_idx],
                paragraph_indices=slice_.paragraph_indices,
                paragraph_titles=slice_.paragraph_titles,
                gold_count=slice_.gold_count,
                permutation=list(perm),
            )
        )
    return rows
