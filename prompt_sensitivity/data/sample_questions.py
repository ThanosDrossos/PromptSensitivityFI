"""Stratified sampler. Sprint 1 §1.4 + §2.4 of Research_Design_v3.

Selects 100 HotpotQA validation questions balanced on `level` (medium / hard)
and 50 2WikiMultihopQA validation questions balanced on `type` (4 categories).
Seeded by `config.random_seed`. Drops questions with fewer than 2 gold paragraphs
per the §3.1 edge-case clause.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .schemas import MultiHopQuestion


def _eligible(questions: Iterable[MultiHopQuestion], k_gold: int) -> list[MultiHopQuestion]:
    """Drop records with fewer than k_gold gold paragraphs (Sprint 3 §3.1 rule)."""
    return [q for q in questions if len(q.gold_paragraphs()) >= k_gold]


def stratified_sample(
    questions: list[MultiHopQuestion],
    n_total: int,
    stratify_by: str,
    *,
    seed: int,
    k_gold: int = 2,
) -> list[MultiHopQuestion]:
    """Stratified sample without replacement on `stratify_by` attribute.

    `stratify_by` is one of "level" (HotpotQA) or "type" / "question_type"
    (2Wiki). We group, allocate counts proportional to group size capped at
    available, and round-robin any leftover slots.
    """
    eligible = _eligible(questions, k_gold)
    if not eligible:
        raise ValueError("no eligible questions after filtering")

    rng = random.Random(seed)
    groups: dict[str, list[MultiHopQuestion]] = defaultdict(list)
    for q in eligible:
        key = _strat_key(q, stratify_by)
        if key is None:
            continue
        groups[key].append(q)

    if not groups:
        raise ValueError(f"no groups found for stratify_by={stratify_by!r}")

    # Equal-share allocation: floor(n / |groups|) per stratum, distribute remainder
    # across the largest strata first.
    strata = sorted(groups.keys())
    base = n_total // len(strata)
    remainder = n_total - base * len(strata)
    sizes_by_stratum = sorted(strata, key=lambda s: -len(groups[s]))
    alloc: dict[str, int] = {s: base for s in strata}
    for s in sizes_by_stratum[:remainder]:
        alloc[s] += 1

    chosen: list[MultiHopQuestion] = []
    for s in strata:
        pool = list(groups[s])
        rng.shuffle(pool)
        want = min(alloc[s], len(pool))
        chosen.extend(pool[:want])

    # If any stratum was undersized, top up from the union of the remaining.
    if len(chosen) < n_total:
        chosen_ids = {q.id for q in chosen}
        leftover = [q for q in eligible if q.id not in chosen_ids]
        rng.shuffle(leftover)
        deficit = n_total - len(chosen)
        chosen.extend(leftover[:deficit])

    rng.shuffle(chosen)
    return chosen[:n_total]


def _strat_key(q: MultiHopQuestion, stratify_by: str) -> str | None:
    if stratify_by == "level":
        return q.level
    if stratify_by in ("type", "question_type"):
        return q.question_type
    raise ValueError(f"unsupported stratify_by={stratify_by!r}")
