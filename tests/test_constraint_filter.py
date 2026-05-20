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


# --- gold-based constraint filter (preferred Sprint-2 path) ---------------


def test_parse_valid_json_accepts_clean_payload():
    from prompt_sensitivity.paraphrases.constraint_filter import _parse_valid_json

    assert _parse_valid_json('{"valid": true}') is True
    assert _parse_valid_json('{"valid": false}') is False


def test_parse_valid_json_strips_fences_and_prose():
    from prompt_sensitivity.paraphrases.constraint_filter import _parse_valid_json

    assert _parse_valid_json('```json\n{"valid": true}\n```') is True
    assert _parse_valid_json('Sure!\n{"valid": false}') is False


def test_parse_valid_json_accepts_bare_yes_no():
    """Tolerate the model returning a bare bool (some routes drop the JSON envelope)."""
    from prompt_sensitivity.paraphrases.constraint_filter import _parse_valid_json

    assert _parse_valid_json("true") is True
    assert _parse_valid_json("YES") is True
    assert _parse_valid_json("false") is False
    assert _parse_valid_json("no") is False


def test_parse_valid_json_returns_none_on_garbage():
    from prompt_sensitivity.paraphrases.constraint_filter import _parse_valid_json

    assert _parse_valid_json("maybe later") is None
    assert _parse_valid_json("") is None


def test_judge_contains_gold_passes_when_judge_says_true(monkeypatch):
    """Pipe-level test: judge_contains_gold returns True when the LLM responds {valid:true}."""
    from prompt_sensitivity.paraphrases import constraint_filter as cf

    class _FakeResp:
        text = '{"valid": true}'
        request_hash = "h"

    class _FakeClient:
        def complete(self, _req):
            return _FakeResp()

    monkeypatch.setattr(cf, "get_client", lambda *a, **k: _FakeClient())
    assert cf.judge_contains_gold("any paraphrase", "Paris") is True


def test_judge_contains_gold_returns_false_on_unparseable(monkeypatch):
    """Conservative: drop the candidate when we can't parse the verdict."""
    from prompt_sensitivity.paraphrases import constraint_filter as cf

    class _FakeResp:
        text = "completely unparseable garbage"
        request_hash = "h"

    class _FakeClient:
        def complete(self, _req):
            return _FakeResp()

    monkeypatch.setattr(cf, "get_client", lambda *a, **k: _FakeClient())
    assert cf.judge_contains_gold("any paraphrase", "Paris") is False


def test_filter_by_constraint_with_gold_returns_parallel_bools(monkeypatch):
    """3 candidates -> 3 yes/no decisions in order."""
    from prompt_sensitivity.paraphrases import constraint_filter as cf

    answers = iter([True, False, True])

    class _FakeResp:
        def __init__(self, v):
            self.text = '{"valid": ' + ("true" if v else "false") + "}"
            self.request_hash = "h"

    class _FakeClient:
        def complete(self, _req):
            return _FakeResp(next(answers))

    monkeypatch.setattr(cf, "get_client", lambda *a, **k: _FakeClient())
    out = cf.filter_by_constraint_with_gold(["a", "b", "c"], "GOLD")
    assert out == [True, False, True]
