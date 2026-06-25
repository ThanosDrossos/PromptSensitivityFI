"""P1-1: embed_hidden output must be L2-normalized (unit rows).

Mocks _load_model with a tiny fake (random hidden states) so no weights load.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import prompt_sensitivity.models.local_hf as lh
from prompt_sensitivity.config import ModelEntry, load_config
from prompt_sensitivity.models.cache import LLMCache


class _Enc(dict):
    def to(self, _device):
        return self


class _FakeTok:
    pad_token_id = 0

    def __call__(self, batch, add_special_tokens=False, return_tensors=None,
                 padding=False, truncation=False, max_length=None):
        ids = [[7] * (3 + i) for i in range(len(batch))]   # lengths 3, 4, ...
        if return_tensors is None:                          # the length-check call
            return {"input_ids": ids}
        t = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), t, dtype=torch.long)
        attn = torch.zeros(len(ids), t, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            attn[i, : len(x)] = 1
        enc = _Enc()
        enc["input_ids"] = input_ids
        enc["attention_mask"] = attn
        return enc


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=2048)
        self.device = "cpu"

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        b, t = input_ids.shape
        hs = torch.randn(b, t, 4)
        return SimpleNamespace(hidden_states=[hs, hs])


def test_embed_hidden_rows_are_unit_norm(monkeypatch):
    monkeypatch.setattr(lh, "_load_model", lambda model_id: (_FakeTok(), _FakeModel()))
    entry = ModelEntry(provider="local", model_id="fake", chat_logprobs=False,
                       echo_completions=False, has_hidden=True)
    client = lh.LocalHFClient(entry, load_config(), LLMCache(":memory:"))

    v = client.embed_hidden(["a", "b"])
    assert v.shape == (2, 4) and v.dtype == np.float32
    norms = np.linalg.norm(v, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
