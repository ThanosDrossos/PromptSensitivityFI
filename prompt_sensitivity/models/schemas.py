"""Pydantic v2 schemas for LLM I/O.

The schemas double as the cache key (everything in `LLMRequest` participates in
the request hash) and as the on-disk record format.

Pre-cluster (Sprint 1-5) the only provider is `litellm` (KIT/DSI gateway). The
literal is left narrow so an accidentally-routed call (e.g. typo "openai")
fails Pydantic validation rather than reaching the gateway with a wrong base
URL. Add new providers explicitly when the cluster path lands in Sprint 6.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant"]
Provider = Literal["litellm"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str


class LLMRequest(BaseModel):
    """A single chat completion request. The hash of this object is the cache key."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model_id: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None  # OpenAI / LiteLLM: 0-20.
    stop: list[str] | None = None
    # Free-form `purpose` lets us shard the cache by call site (e.g. "F_score",
    # "h_sem_sample", "paraphrase_gen") so we can prune partial caches cleanly.
    purpose: str = "default"

    def cache_key(self) -> str:
        """SHA256 of canonical JSON. Must be deterministic across processes."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CompletionRequest(BaseModel):
    """A legacy `/v1/completions` request — only used for echo-mode scoring.

    POSIX (Chatterjee 2024 arXiv:2410.02185) needs `log P(y_j | x_i)` for an
    arbitrary continuation y_j given prompt x_i. Chat-completions cannot do
    this; the legacy completions endpoint with `echo=true, logprobs=N,
    max_tokens=0` can — provided the underlying provider supports it. Through
    our gateway, Together-hosted Llama/Mistral/Qwen support this; GPT-4o does
    not.

    Sprint 4 fills `score_continuation()` on the client to use this schema.
    Until then the schema exists but no call site emits it.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model_id: str
    prompt: str
    max_tokens: int = 0
    echo: bool = True
    logprobs: int = 1
    temperature: float = 0.0
    purpose: str = "posix_echo"

    def cache_key(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TokenLogprob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str
    logprob: float
    top_logprobs: dict[str, float] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Provider-agnostic response shape."""

    model_config = ConfigDict(extra="forbid")

    request_hash: str
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    token_logprobs: list[TokenLogprob] | None = None
    raw_provider_response: dict | None = None
    latency_ms: float | None = None
    cached: bool = False
