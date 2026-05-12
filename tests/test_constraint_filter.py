"""Constraint filter: JSON parsing + Jaccard + normaliser. No gateway needed."""

import pytest

from prompt_sensitivity.paraphrases.constraint_filter import (
    _norm,
    _parse_answers_json,
    jaccard,
)


def test_jaccard_basic():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"a"}) == 0.5


def test_jaccard_empty_pair_is_one():
    """If the judge returns no answers either side, treat as agreement, not violation."""
    assert jaccard(set(), set()) == 1.0


def test_jaccard_one_empty_side_is_zero():
    assert jaccard({"a"}, set()) == 0.0


def test_norm_strips_punctuation_and_case():
    assert _norm("  Paris.") == "paris"
    assert _norm("PARIS!") == "paris"
    assert _norm('"Paris"') == "paris"


def test_parse_answers_json_strips_fences():
    text = '```json\n{"answers": ["Paris", "Paris, France"]}\n```'
    assert _parse_answers_json(text) == ["Paris", "Paris, France"]


def test_parse_answers_json_handles_prose_preamble():
    text = "Here is the JSON:\n{\"answers\": [\"yes\", \"true\"]}\nThanks!"
    assert _parse_answers_json(text) == ["yes", "true"]


def test_parse_answers_json_returns_empty_on_garbage():
    assert _parse_answers_json("definitely not JSON") == []
    assert _parse_answers_json("") == []


def test_parse_answers_json_ignores_non_string_entries():
    text = '{"answers": ["ok", null, 42, ["nested"]]}'
    # Strings and primitives are kept; nested lists are skipped.
    out = _parse_answers_json(text)
    assert "ok" in out
    assert "42" in out
    assert all(not isinstance(x, list) for x in out)
