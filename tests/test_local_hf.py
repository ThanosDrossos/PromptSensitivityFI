"""Unit tests for the in-process `local` (transformers) backend.

These exercise the pure-ish helpers + provider routing WITHOUT loading any
model weights (no GPU, no downloads) — the real generate/score/hidden-state
paths are validated on the cluster by `scripts/local_check.py`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from prompt_sensitivity.models.local_hf import (
    _apply_stop,
    _build_token_logprobs,
    _eos_id_set,
    _masked_mean_pool,
)
from prompt_sensitivity.models.schemas import ChatMessage, LLMRequest


# --- schema: provider literal now admits "local" ---------------------------


def test_llmrequest_accepts_local_provider():
    req = LLMRequest(
        provider="local",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.provider == "local"


# --- stop-string truncation -------------------------------------------------


def test_apply_stop_truncates_at_first_match():
    assert _apply_stop("hello\nAnswer: x", ["\nAnswer:"]) == ("hello", True)


def test_apply_stop_earliest_of_several():
    text = "aaaSTOP1bbbSTOP2ccc"
    out, stopped = _apply_stop(text, ["STOP2", "STOP1"])
    assert stopped is True
    assert out == "aaa"


def test_apply_stop_no_match_and_none():
    assert _apply_stop("no stop here", ["zzz"]) == ("no stop here", False)
    assert _apply_stop("text", None) == ("text", False)


# --- per-token logprob extraction from generate() scores --------------------


class _FakeTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_build_token_logprobs_values_and_topk():
    # step 0 logits favour token 1; step 1 logits favour token 0.
    scores = (
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
    )
    new_token_ids = torch.tensor([1, 0])
    out = _build_token_logprobs(_FakeTok(), scores, new_token_ids, top_logprobs=2)

    assert out is not None and len(out) == 2
    assert out[0].token == "<1>"
    expected0 = math.log(math.e**1 / (3 * math.e**0 + math.e**1))
    assert abs(out[0].logprob - expected0) < 1e-4
    # top_logprobs caps at the requested k.
    assert len(out[0].top_logprobs) == 2


def test_build_token_logprobs_truncates_to_min_length():
    # More generated ids than score steps -> only score-step count is used.
    scores = (torch.tensor([[0.0, 1.0]]),)
    new_token_ids = torch.tensor([1, 0, 1])
    out = _build_token_logprobs(_FakeTok(), scores, new_token_ids, top_logprobs=1)
    assert out is not None and len(out) == 1


# --- masked mean pooling ----------------------------------------------------


def test_masked_mean_pool_ignores_padding():
    last = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])  # (1, 3, 2)
    mask = torch.tensor([[1, 1, 0]])                              # last token padded
    pooled = _masked_mean_pool(last, mask)
    assert pooled.shape == (1, 2)
    assert pooled.dtype == np.float32
    assert np.allclose(pooled, [[2.0, 2.0]])


# --- EOS id set handling ----------------------------------------------------


def test_eos_id_set_scalar_and_list():
    class _S:
        eos_token_id = 5

    class _L:
        eos_token_id = [5, 6]

    class _N:
        eos_token_id = None

    assert _eos_id_set(_S()) == {5}
    assert _eos_id_set(_L()) == {5, 6}
    assert _eos_id_set(_N()) == set()


# --- provider routing: get_client -> LocalHFClient (no weights loaded) ------


def test_get_client_routes_local_provider(monkeypatch):
    from prompt_sensitivity.models import registry
    from prompt_sensitivity.models.cache import LLMCache
    from prompt_sensitivity.models.local_hf import LocalHFClient

    # In-memory cache so the test touches no disk and loads no model (the
    # transformers weights load lazily inside _raw_call, never at construction).
    monkeypatch.setattr(registry, "_get_cache", lambda cfg: LLMCache(":memory:"))
    registry.reset_clients()
    client = registry.get_client("qwen_2_5_7b")
    assert isinstance(client, LocalHFClient)
    assert client.entry.provider == "local"
    registry.reset_clients()
