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


def test_apply_stop_ignores_empty_strings():
    # Empty entries must be skipped, not truncate everything at index 0.
    assert _apply_stop("foo", ["", "bar"]) == ("foo", False)
    assert _apply_stop("foo", [""]) == ("foo", False)
    assert _apply_stop("fooXbar", ["", "X"]) == ("foo", True)


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


def test_build_token_logprobs_default_topk_clamps_to_vocab():
    # top_logprobs=None -> default 5, but vocab is only 3 -> clamp to 3.
    scores = (torch.tensor([[0.1, 0.2, 0.7]]),)
    out = _build_token_logprobs(_FakeTok(), scores, torch.tensor([2]), None)
    assert out is not None and len(out[0].top_logprobs) == 3


def test_build_token_logprobs_large_k_not_capped_at_20():
    # Local path has full vocab: asking for 25 returns 25 (no hard 20 cap).
    V = 25
    scores = (torch.arange(V, dtype=torch.float32).unsqueeze(0),)
    out = _build_token_logprobs(_FakeTok(), scores, torch.tensor([V - 1]), top_logprobs=25)
    assert out is not None and len(out[0].top_logprobs) == 25


def test_build_token_logprobs_empty_scores_returns_none():
    out = _build_token_logprobs(_FakeTok(), (), torch.tensor([0, 1]), top_logprobs=1)
    assert out is None


# --- _format_chat system->user fold fallback --------------------------------


class _FoldTok:
    """Fake tokenizer: rejects the first apply_chat_template call (as a template
    that disallows a standalone system role would), then succeeds on the folded
    layout. Records each call's messages for inspection."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if len(self.calls) == 1:
            raise ValueError("template does not support a system role")
        return {"input_ids": "OK", "attention_mask": "OK"}


def test_format_chat_folds_system_into_first_user():
    from prompt_sensitivity.models.local_hf import _format_chat

    tok = _FoldTok()
    out = _format_chat(tok, [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "USER2"},
    ])
    assert out == {"input_ids": "OK", "attention_mask": "OK"}
    assert len(tok.calls) == 2  # failed once, retried folded
    folded = tok.calls[1]
    assert folded[0]["role"] == "user"
    assert "SYS" in folded[0]["content"] and "USER1" in folded[0]["content"]
    # later turns preserved verbatim, system folded exactly once
    assert folded[1] == {"role": "assistant", "content": "A1"}
    assert folded[2] == {"role": "user", "content": "USER2"}


def test_format_chat_fold_accumulates_multiple_system():
    from prompt_sensitivity.models.local_hf import _format_chat

    tok = _FoldTok()
    _format_chat(tok, [
        {"role": "system", "content": "S1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "U"},
    ])
    folded = tok.calls[1]
    assert len(folded) == 1 and folded[0]["role"] == "user"
    assert all(s in folded[0]["content"] for s in ("S1", "S2", "U"))


def test_format_chat_fold_no_user_emits_system_as_user():
    from prompt_sensitivity.models.local_hf import _format_chat

    tok = _FoldTok()
    _format_chat(tok, [{"role": "system", "content": "SYS-ONLY"}])
    assert tok.calls[1] == [{"role": "user", "content": "SYS-ONLY"}]


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


# --- resolve_layer_fracs: fractional depth -> hidden_states indices ----------


def test_resolve_layer_fracs_model_agnostic():
    from prompt_sensitivity.models.local_hf import resolve_layer_fracs

    # qwen-2.5-7b has 28 transformer layers, llama/mistral 32.
    assert resolve_layer_fracs(28, (0.25, 0.5, 0.75, 1.0)) == [7, 14, 21, 28]
    assert resolve_layer_fracs(32, (0.25, 0.5, 0.75, 1.0)) == [8, 16, 24, 32]


def test_resolve_layer_fracs_clamps_dedups_and_validates():
    import pytest
    from prompt_sensitivity.models.local_hf import resolve_layer_fracs

    # tiny fracs clamp to layer 1 (never the index-0 embedding output) + dedup
    assert resolve_layer_fracs(28, (0.001, 0.01, 1.0)) == [1, 28]
    # near-identical fracs dedup after rounding
    assert resolve_layer_fracs(4, (0.5, 0.55, 1.0)) == [2, 4]
    with pytest.raises(ValueError):
        resolve_layer_fracs(28, (0.0,))     # 0 would select the embedding layer
    with pytest.raises(ValueError):
        resolve_layer_fracs(28, (1.5,))
    with pytest.raises(ValueError):
        resolve_layer_fracs(0, (0.5,))
