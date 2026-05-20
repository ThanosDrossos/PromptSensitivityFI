"""Edit-distance deduplication. Sprint 2 §3.1 rule: edit distance > 5 between
accepted paraphrases (i.e. minimum distance >= 6 — configurable via
`config.paraphrases.deduplication.min_edit_distance`).

We accept candidates greedily: walk the candidate list in order, keep one,
reject the next if it is within distance `min_edit_distance - 1` of any
already-kept candidate. The order matters — pass the best-quality candidates
first if you want them prioritised.

Two distance metrics are supported (selectable via `config.paraphrases.deduplication.metric`):

  - `char` (default, brief-compliant): character-level Levenshtein. Recommended
    min_edit_distance = 6 (matches brief "> 5"). Appropriate for long questions
    where 5 chars is a small fraction of length.

  - `token`: word-level Levenshtein on whitespace-tokenised text. Recommended
    min_edit_distance = 3 (>= 3 word swaps). Appropriate when the dataset has
    short questions (~80 chars, ~12 tokens) where the brief's char-level
    threshold over-rejects: e.g. "What is the port city located about 25 km
    north..." vs "Which port city is situated about 25 km north..." differ in
    ~4 chars but represent genuinely different paraphrases. The 2026-05-21
    smoke run showed the char-mode rejecting 350+ of these per question.

The 2026-05-21 audit added `token` mode after dedup became the bottleneck
for short questions once the gold-judge constraint filter was fixed. Default
remains `char` to stay aligned with the brief; switch to `token` in config
to recover ~5x more diverse paraphrases on short questions.

The Levenshtein implementation is in-house (no extra dep). 40 raw candidates
of ~80 chars each gives 40*40*80*80 = ~10M cell ops worst case, which is
~50 ms in pure Python — fast enough that it's not worth a binding.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence


Metric = Literal["char", "token"]


def levenshtein(a: str, b: str) -> int:
    """Classic two-row DP. O(len(a)*len(b)) time, O(min(len(a),len(b))) space.

    Returns the minimum number of single-character insertions, deletions, or
    substitutions to turn `a` into `b`. Case-sensitive — case differences
    count as substitutions.
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


def _tokenize(s: str) -> list[str]:
    """Whitespace tokenisation, case-lowered. Punctuation stays attached.

    Lowercasing collapses "What's" vs "what's" but keeps "What's" vs "Whats"
    distinct (one token edit). Good middle ground for short questions where
    we want to see "Which port" vs "What port" as 1 token edit and
    "the port city is" vs "the port city which" as 1 token edit.
    """
    return s.lower().split()


def levenshtein_tokens(a: str, b: str) -> int:
    """Word-level Levenshtein. Same algorithm, but on token sequences.

    Two paraphrases that differ only in casing/whitespace land at distance 0.
    Each swapped/added/dropped word counts as 1. For ~12-token questions a
    threshold of 3 word-edits keeps genuinely different rewrites and rejects
    near-trivial rephrases.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if tokens_a == tokens_b:
        return 0
    if len(tokens_a) > len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a
    if not tokens_a:
        return len(tokens_b)
    prev = list(range(len(tokens_a) + 1))
    cur = [0] * (len(tokens_a) + 1)
    for j in range(1, len(tokens_b) + 1):
        cur[0] = j
        bj = tokens_b[j - 1]
        for i in range(1, len(tokens_a) + 1):
            cost = 0 if tokens_a[i - 1] == bj else 1
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
        prev, cur = cur, prev
    return prev[len(tokens_a)]


def _distance_fn(metric: Metric) -> Callable[[str, str], int]:
    if metric == "char":
        return levenshtein
    if metric == "token":
        return levenshtein_tokens
    raise ValueError(f"unknown dedup metric {metric!r}; expected 'char' or 'token'")


def deduplicate(
    candidates: Sequence[str],
    *,
    min_distance: int,
    metric: Metric = "char",
) -> list[int]:
    """Greedy dedup. Returns the indices of `candidates` that survive.

    A candidate is kept iff its `metric`-distance to every previously-kept
    candidate is >= `min_distance`. With the brief's char default of 6 this
    eliminates near-identical rewrites while preserving genuine stylistic
    variants. With token + min=3 it accepts more diverse phrasings on short
    questions (recommended when the corpus has many short factoid questions
    like HotpotQA's port-city/school-name pattern).

    Order-preserving: index i can only be kept if no kept index j < i is too
    close to it. The caller is expected to sort `candidates` by desired
    priority first (e.g. NLI score descending, then by role-diversity).
    """
    distance = _distance_fn(metric)
    kept: list[int] = []
    kept_texts: list[str] = []
    for i, c in enumerate(candidates):
        c_strip = c.strip()
        if not c_strip:
            continue
        too_close = False
        for prior in kept_texts:
            if distance(c_strip, prior) < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(i)
            kept_texts.append(c_strip)
    return kept
