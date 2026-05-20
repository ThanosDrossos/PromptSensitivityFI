"""Edit-distance dedup. Sprint 2 §3.1 rule: min_distance = 6 (i.e. > 5)."""

import pytest

from prompt_sensitivity.paraphrases.deduplicate import (
    deduplicate,
    levenshtein,
    levenshtein_tokens,
)


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("a", "") == 1
    assert levenshtein("kitten", "sitting") == 3   # canonical example
    assert levenshtein("flaw", "lawn") == 2


def test_deduplicate_drops_near_duplicates():
    candidates = [
        "What is the capital of France?",
        "what is the capital of france?",        # only case differs -> 6 substitutions
        "What's the capital of France?",          # 2 char diff
        "Which city is the capital of France?",   # >5 char diff
    ]
    kept = deduplicate(candidates, min_distance=6)
    # Index 0 is always kept. Index 1 differs by case (6 substitutions == 6, kept).
    # Index 2 differs by only ~2 chars from index 0 (dropped).
    # Index 3 is a full rephrase (kept).
    assert 0 in kept
    assert 2 not in kept
    assert 3 in kept


def test_deduplicate_keeps_first_seen():
    """Order matters: priority sorting is the caller's job."""
    a = "Short phrase A"
    b = "Short phrase B"            # 1 char diff
    c = "Completely different one here"
    kept = deduplicate([a, b, c], min_distance=6)
    assert 0 in kept and 1 not in kept and 2 in kept


def test_deduplicate_skips_empty():
    """Empty strings are filtered out rather than treated as duplicates of each other."""
    kept = deduplicate(["", "  ", "real text here"], min_distance=6)
    assert kept == [2]


def test_deduplicate_passes_when_min_distance_one():
    """min_distance=1 means only exact duplicates are dropped."""
    kept = deduplicate(["abc", "abc", "abd"], min_distance=1)
    assert kept == [0, 2]


# --- token-level dedup (2026-05-21: added for short-question handling) -----


def test_levenshtein_tokens_word_units():
    """Token-level: each swapped/added/dropped word = 1."""
    assert levenshtein_tokens("hi there", "hi there") == 0
    assert levenshtein_tokens("the cat sat", "the dog sat") == 1
    assert levenshtein_tokens("the cat", "the cat hat") == 1
    assert levenshtein_tokens("", "hello world") == 2


def test_levenshtein_tokens_case_insensitive_via_lower():
    """Casing-only differences collapse to 0 (intentional)."""
    assert levenshtein_tokens("What's the answer?", "what's THE ANSWER?") == 0


def test_dedup_modes_disagree_on_padding_edits():
    """A real case where the two modes give different verdicts.

    "What is X" vs "What is the X here today" — many CHARS added (filler
    words), but only a few TOKEN edits. Char mode at min=6 keeps both
    (12-char gap); token mode at min=4 rejects (3 token-insertions only).
    Establishes that the metrics are NOT redundant — they answer different
    questions.
    """
    candidates = [
        "What is X",
        "What is the X here today",
    ]
    # 12 char-edits between them; both kept at min=6.
    assert deduplicate(candidates, min_distance=6, metric="char") == [0, 1]
    # 3 token-edits (insert "the", "here", "today"); rejected at min=4.
    assert deduplicate(candidates, min_distance=4, metric="token") == [0]


def test_dedup_token_mode_still_drops_near_duplicates():
    """Token-mode at min=3 still drops genuine near-duplicates (1-word swaps)."""
    candidates = [
        "What is the capital of France today?",
        "What is the capital of France today?",            # exact dup
        "What's the capital of France today?",              # 1 token diff
    ]
    kept = deduplicate(candidates, min_distance=3, metric="token")
    assert kept == [0]


def test_dedup_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown dedup metric"):
        deduplicate(["a", "b"], min_distance=1, metric="kanji")  # type: ignore[arg-type]


def test_config_default_is_char():
    """Defensive: don't silently change dedup behavior."""
    from prompt_sensitivity.config import load_config

    cfg = load_config()
    assert cfg.paraphrases.deduplication.metric == "char"
    assert cfg.paraphrases.deduplication.min_edit_distance == 6
