"""Random-permutation ladder. Research_Design_v3 §4.1 (headline ladder).

Per the brief: "one random permutation per question using seed
`42 + question_id_hash` for reproducibility. Levels are prefixes of size
{0, 2, 4, 6, 8, 10}".

We use a SHA-256-based hash because Python's built-in `hash()` is
non-deterministic across processes (PYTHONHASHSEED randomisation).
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from ..config import load_config
from ..data import MultiHopQuestion
from .schemas import LadderRow, LevelSlice


def _stable_hash(question_id: str) -> int:
    """SHA-256-derived 31-bit integer hash of the question_id."""
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def random_seed(question_id: str, base_seed: int = 42) -> int:
    """`42 + hash(qid)` per the Sprint-3 brief."""
    return base_seed + _stable_hash(question_id)


def random_permutation(question: MultiHopQuestion, *, base_seed: int = 42) -> list[int]:
    """Return one deterministic random permutation of [0, |paragraphs|)."""
    n = len(question.paragraphs)
    indices = list(range(n))
    rng = random.Random(random_seed(question.id, base_seed))
    rng.shuffle(indices)
    return indices


def _slice_levels(
    permutation: Sequence[int],
    paragraph_titles: Sequence[str],
    gold_indices: set[int],
    levels: Sequence[int],
) -> list[LevelSlice]:
    """For each level (paragraph count), return the prefix slice."""
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


def build_random_ladder(
    question: MultiHopQuestion,
    *,
    levels: Sequence[int] | None = None,
    base_seed: int | None = None,
) -> list[LadderRow]:
    """Random shuffle ladder. Returns 6 rows (one per level)."""
    config = load_config()
    if levels is None:
        levels = config.ladders.levels
    if base_seed is None:
        base_seed = config.random_seed

    perm = random_permutation(question, base_seed=base_seed)
    gold_indices = {i for i, p in enumerate(question.paragraphs) if p.is_gold}
    titles = [p.title for p in question.paragraphs]

    rows: list[LadderRow] = []
    for level_idx, slice_ in enumerate(_slice_levels(perm, titles, gold_indices, levels)):
        rows.append(
            LadderRow(
                question_id=question.id,
                ladder_type="random",
                level_idx=level_idx,
                level=levels[level_idx],
                paragraph_indices=slice_.paragraph_indices,
                paragraph_titles=slice_.paragraph_titles,
                gold_count=slice_.gold_count,
                permutation=list(perm),
            )
        )
    return rows
