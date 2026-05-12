"""Edit-distance deduplication. Sprint 2 §3.1 rule: edit distance > 5 between
accepted paraphrases (i.e. minimum distance >= 6 — configurable via
`config.paraphrases.deduplication.min_edit_distance`).

We accept candidates greedily: walk the candidate list in order, keep one,
reject the next if it is within distance `min_edit_distance - 1` of any
already-kept candidate. The order matters — pass the best-quality candidates
first if you want them prioritised.

The Levenshtein implementation is in-house (no extra dep). 40 raw candidates
of ~80 chars each gives 40*40*80*80 = ~10M cell ops worst case, which is
~50 ms in pure Python — fast enough that it's not worth a binding.
"""

from __future__ import annotations

from typing import Sequence


def levenshtein(a: str, b: str) -> int:
    """Classic two-row DP. O(len(a)*len(b)) time, O(min(len(a),len(b))) space.

    Returns the minimum number of single-character insertions, deletions, or
    substitutions to turn `a` into `b`. Case-sensitive — case differences
    count as substitutions. The brief says "edit distance > 5" without
    qualification; we use the raw count, including punctuation/case.
    """
    if a == b:
        return 0
    # Keep `a` as the shorter string to minimise memory.
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    prev = list(range(len(a) + 1))
    cur = [0] * (len(a) + 1)
    for j in range(1, len(b) + 1):
        cur[0] = j
        bj = b[j - 1]
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(
                prev[i] + 1,        # deletion
                cur[i - 1] + 1,     # insertion
                prev[i - 1] + cost, # substitution
            )
        prev, cur = cur, prev
    return prev[len(a)]


def deduplicate(
    candidates: Sequence[str],
    *,
    min_distance: int,
) -> list[int]:
    """Greedy dedup. Returns the indices of `candidates` that survive.

    A candidate is kept iff its Levenshtein distance to every previously-kept
    candidate is >= `min_distance`. With the configured default of 6 this
    eliminates near-identical rewrites while preserving genuine stylistic
    variants.

    Order-preserving: index i can only be kept if no kept index j < i is too
    close to it. The caller is expected to sort `candidates` by desired
    priority first (e.g. NLI score descending, then by role-diversity).
    """
    kept: list[int] = []
    kept_texts: list[str] = []
    for i, c in enumerate(candidates):
        c_strip = c.strip()
        if not c_strip:
            continue
        too_close = False
        for prior in kept_texts:
            if levenshtein(c_strip, prior) < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(i)
            kept_texts.append(c_strip)
    return kept
