"""Cache + LLMRequest hash determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_sensitivity.models import LLMCache, LLMRequest
from prompt_sensitivity.models.schemas import ChatMessage, LLMResponse


def _req(content: str = "hi", **overrides) -> LLMRequest:
    base = dict(
        provider="litellm",
        model_id="gpt-4o-2024-08-06",
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.0,
        max_tokens=8,
        seed=42,
        purpose="unit_test",
    )
    base.update(overrides)
    return LLMRequest(**base)


def test_cache_key_is_deterministic():
    a = _req("hi")
    b = _req("hi")
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_temperature():
    a = _req("hi", temperature=0.0)
    b = _req("hi", temperature=0.7)
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_with_purpose():
    """Sharding the cache by purpose must invalidate hits across purposes."""
    a = _req("hi", purpose="api_check")
    b = _req("hi", purpose="paraphrase_gen")
    assert a.cache_key() != b.cache_key()


def test_cache_round_trip(tmp_path: Path):
    cache = LLMCache(tmp_path / "test.sqlite")
    req = _req("hi")
    assert cache.get(req) is None
    resp = LLMResponse(request_hash=req.cache_key(), text="pong")
    cache.put(req, resp)
    fetched = cache.get(req)
    assert fetched is not None
    assert fetched.text == "pong"
    assert fetched.cached is True
    assert cache.size() == 1
    cache.close()


def test_cache_idempotent_put(tmp_path: Path):
    cache = LLMCache(tmp_path / "test.sqlite")
    req = _req("hi")
    resp = LLMResponse(request_hash=req.cache_key(), text="v1")
    cache.put(req, resp)
    cache.put(req, LLMResponse(request_hash=req.cache_key(), text="v2"))
    fetched = cache.get(req)
    assert fetched is not None
    assert fetched.text == "v2"  # latest wins
    assert cache.size() == 1
    cache.close()


def test_chat_messages_must_be_typed():
    """Pydantic should reject role values outside the Literal."""
    with pytest.raises(Exception):
        ChatMessage(role="root", content="hi")  # type: ignore[arg-type]
