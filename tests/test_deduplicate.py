"""Edit-distance dedup. Sprint 2 §3.1 rule: min_distance = 6 (i.e. > 5)."""

from prompt_sensitivity.paraphrases.deduplicate import deduplicate, levenshtein


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
