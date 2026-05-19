"""Single-provider LLM client routed through the KIT/DSI LiteLLM gateway.

Per the 2026-05-12 supervisor exchange and Research_Design_v3 §5.3 closed-weight
branch:

  - ALL traffic for Sprints 1-5 goes through one OpenAI-compatible endpoint
    (default https://ai-gateway.dsi-experimente.de/v1).
  - The supervisor's gateway speaks LiteLLM v1.82+ — `/v1/chat/completions`,
    `/v1/completions`, and `/v1/embeddings` are all available.
  - The OpenAI Python SDK is reused with `base_url=` pointed at the gateway.
    Same Auth Bearer header style.

Capability matrix (pre-cluster):

  | Capability                                  | Llama / Mistral / Qwen | GPT-4o |
  |---------------------------------------------|------------------------|--------|
  | chat-completions (text + token_logprobs<=20)| yes                    | yes    |
  | /v1/completions with echo+logprobs (POSIX)  | yes                    | NO     |
  | last-layer hidden states (ESS_in^own)       | NO                     | NO     |

Sprint 6 (KIT cluster) lifts ESS_in and the full-vocab logprob constraints by
running vLLM locally. Until then:
  - POSIX runs only for the three open-weight models via echo-mode scoring.
  - Errica's S_τ on free-form output is computed via MC over semantic clusters
    (effectively H_sem), not raw token entropy.
  - ESS_in is reported only with the external mpnet encoder.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from loguru import logger

from ..config import Config, ModelEntry, load_config, REPO_ROOT
from .cache import LLMCache
from .rate_limiter import TokenBucket
from .schemas import CompletionRequest, LLMRequest, LLMResponse, TokenLogprob


# Eagerly load .env so env vars are available even when called from REPL.
load_dotenv(REPO_ROOT / ".env", override=False)


# Global per-provider rate limiter (one bucket per process).
_BUCKETS: dict[str, TokenBucket] = {}


def _bucket(provider: str, qps: float) -> TokenBucket:
    if provider not in _BUCKETS:
        _BUCKETS[provider] = TokenBucket(rate=qps, capacity=qps)
    return _BUCKETS[provider]


# Exceptions classes considered transient for retry purposes.
_TRANSIENT: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)


class BaseLLMClient(ABC):
    """Common interface for all clients.

    `complete` (chat-completions) is the only method most call sites need. It
    enforces three invariants:
      1. Every call hits the cache first; cache misses are persisted.
      2. Calls go through the per-provider token bucket (avoids 429s).
      3. Failures are retried with exponential backoff per `config.api`.

    `score_continuation` is a future Sprint-4 hook for POSIX. The Sprint-1
    implementation raises NotImplementedError; the schema and routing exist so
    Sprint-4 only needs to add the body.
    """

    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        self.entry = model_entry
        self.config = config
        self.cache = cache
        qps = config.api.rate_limit_qps.get(model_entry.provider, 5.0)
        self._bucket = _bucket(model_entry.provider, qps)

    # ---------------- public API ----------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Force the registry's model_id and provider onto the request so call
        # sites cannot accidentally route a Llama prompt to GPT-4o.
        if request.model_id != self.entry.model_id or request.provider != self.entry.provider:
            request = request.model_copy(
                update={"model_id": self.entry.model_id, "provider": self.entry.provider}  # type: ignore[arg-type]
            )

        cached = self.cache.get(request)
        if cached is not None:
            logger.debug("cache hit {} {}", self.entry.provider, request.cache_key()[:8])
            return cached

        self._bucket.acquire(1.0)
        t0 = time.perf_counter()
        response = self._call_with_retry(request)
        response.latency_ms = (time.perf_counter() - t0) * 1000.0
        response.request_hash = request.cache_key()
        response.cached = False
        self.cache.put(request, response)
        return response

    def score_continuation(self, request: CompletionRequest) -> LLMResponse:
        """Echo-mode forced sequence scoring. Used by POSIX (Sprint 4).

        Calls `/v1/completions` with `echo=true, logprobs=N, max_tokens=0`
        and returns an LLMResponse whose `token_logprobs` is the prompt
        echo with one logprob per prompt token. POSIX sums the logprobs
        for the y-position tokens to recover `log P(y | x)`.

        IMPORTANT caveat — chat-template formatting is NOT applied. The
        caller passes raw text in `request.prompt`. For chat models this
        is an approximation of the chat-completion logprob; Sprint 6 will
        use vLLM with the model's tokenizer chat template applied
        directly.
        """
        if not self.entry.echo_completions:
            raise RuntimeError(
                f"model {self.entry.model_id!r} does not support echo-mode completions"
            )
        # Force registry-bound provider/model onto the request.
        if request.model_id != self.entry.model_id or request.provider != self.entry.provider:
            request = request.model_copy(
                update={"model_id": self.entry.model_id, "provider": self.entry.provider}  # type: ignore[arg-type]
            )

        cached = self._cache_get_completion(request)
        if cached is not None:
            logger.debug("echo cache hit {} {}", self.entry.provider, request.cache_key()[:8])
            return cached

        self._bucket.acquire(1.0)
        t0 = time.perf_counter()
        response = self._raw_completion(request)
        response.latency_ms = (time.perf_counter() - t0) * 1000.0
        response.request_hash = request.cache_key()
        response.cached = False
        self._cache_put_completion(request, response)
        return response

    # Subclass hook for /v1/completions echo. LiteLLMClient implements;
    # other providers raise NotImplementedError.
    def _raw_completion(self, request: CompletionRequest) -> LLMResponse:
        raise NotImplementedError(f"{type(self).__name__} does not implement echo completions")

    # The cache is keyed on LLMRequest.cache_key() but CompletionRequest
    # also has a cache_key() of the same SHA256-of-canonical-JSON shape.
    # We piggy-back on the existing SQLite by storing under the same
    # request_hash column; the provider+model_id columns let us prune.
    def _cache_get_completion(self, request: CompletionRequest) -> LLMResponse | None:
        key = request.cache_key()
        with self.cache._lock:  # noqa: SLF001 — intentional intra-package access
            row = self.cache._conn.execute(  # noqa: SLF001
                "SELECT response_json FROM llm_cache WHERE request_hash = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        response = LLMResponse.model_validate_json(row[0])
        response.cached = True
        return response

    def _cache_put_completion(self, request: CompletionRequest, response: LLMResponse) -> None:
        key = request.cache_key()
        if response.request_hash != key:
            response = response.model_copy(update={"request_hash": key})
        with self.cache._lock:  # noqa: SLF001
            self.cache._conn.execute(  # noqa: SLF001
                """
                INSERT OR REPLACE INTO llm_cache
                  (request_hash, provider, model_id, purpose, request_json, response_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    request.provider,
                    request.model_id,
                    request.purpose,
                    request.model_dump_json(),
                    response.model_dump_json(),
                ),
            )
            self.cache._conn.commit()  # noqa: SLF001

    # ---------------- subclass hooks ----------------

    @abstractmethod
    def _raw_call(self, request: LLMRequest) -> LLMResponse:
        """Provider-specific call. Must NOT touch cache or bucket."""

    # ---------------- retry wrapper ----------------

    def _call_with_retry(self, request: LLMRequest) -> LLMResponse:
        api = self.config.api

        @retry(
            stop=stop_after_attempt(api.max_retries),
            wait=wait_exponential(
                multiplier=api.initial_backoff_s,
                min=api.initial_backoff_s,
                max=api.max_backoff_s,
            ),
            retry=retry_if_exception_type(_TRANSIENT),
            before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
            reraise=True,
        )
        def _go() -> LLMResponse:
            return self._raw_call(request)

        return _go()


# --------------------------------------------------------------------------- #
# LiteLLM gateway provider                                                    #
# --------------------------------------------------------------------------- #


class LiteLLMClient(BaseLLMClient):
    """OpenAI-SDK client pointed at the KIT/DSI LiteLLM gateway.

    The gateway exposes the full OpenAI API surface (chat-completions,
    completions, embeddings) so the SDK is reused as-is with `base_url` set to
    the gateway. The API key starts with "sk-" and is supplied by the user's
    supervisor.
    """

    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        super().__init__(model_entry, config, cache)
        from openai import OpenAI  # noqa: WPS433 — lazy import keeps import-time cheap

        api_key = os.environ.get(config.api.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{config.api.api_key_env} not set; copy .env.example to .env"
            )
        base_url = (
            os.environ.get(config.api.base_url_env)
            or config.api.default_base_url
        )
        self._sdk = OpenAI(api_key=api_key, base_url=base_url)
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        return self._base_url

    def _raw_call(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.entry.model_id,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop:
            kwargs["stop"] = request.stop
        if request.logprobs:
            kwargs["logprobs"] = True
            if request.top_logprobs is not None:
                # OpenAI spec caps at 20. LiteLLM passes through.
                kwargs["top_logprobs"] = min(20, max(0, request.top_logprobs))

        completion = self._sdk.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        token_logprobs = _extract_chat_logprobs(choice) if request.logprobs else None

        return LLMResponse(
            request_hash=request.cache_key(),
            text=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_logprobs=token_logprobs,
            raw_provider_response=None,  # too large to store; re-derive if needed
        )


    def _raw_completion(self, request: CompletionRequest) -> LLMResponse:
        """`/v1/completions` echo path used by POSIX.

        LiteLLM routes this to the provider's text-completions endpoint.
        Together (which fronts our 3 open-weight models on this gateway)
        supports `echo=true, logprobs=N, max_tokens=0` and returns one
        logprob per prompt token. OpenAI (kit.gpt-4.1) does not support
        echo on modern chat models; the caller is expected to consult
        `entry.echo_completions` first.
        """
        kwargs: dict[str, Any] = dict(
            model=self.entry.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            logprobs=request.logprobs,
            echo=request.echo,
        )
        completion = self._sdk.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.text or ""
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        token_logprobs = _extract_text_logprobs(choice)
        return LLMResponse(
            request_hash=request.cache_key(),
            text=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_logprobs=token_logprobs,
            raw_provider_response=None,
        )


def _extract_text_logprobs(choice: Any) -> list[TokenLogprob] | None:
    """Legacy /v1/completions logprob shape: `choice.logprobs.tokens` + `.token_logprobs`."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    tokens = getattr(lp, "tokens", None) or []
    token_logprobs = getattr(lp, "token_logprobs", None) or []
    top_logprobs = getattr(lp, "top_logprobs", None) or [None] * len(tokens)
    out: list[TokenLogprob] = []
    for tok, logp, top in zip(tokens, token_logprobs, top_logprobs, strict=False):
        if logp is None:
            continue
        top_dict: dict[str, float] = {}
        if isinstance(top, dict):
            top_dict = {str(k): float(v) for k, v in top.items()}
        out.append(TokenLogprob(token=tok, logprob=float(logp), top_logprobs=top_dict))
    return out or None


def _extract_chat_logprobs(choice: Any) -> list[TokenLogprob] | None:
    """OpenAI-style chat-completions logprob extraction.

    Schema (when `logprobs=True`):
        choice.logprobs.content: list of {token, logprob, top_logprobs: [...]}
    LiteLLM's pass-through preserves this for any backend that supports it.
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None or not getattr(lp, "content", None):
        return None
    out: list[TokenLogprob] = []
    for item in lp.content:
        top: dict[str, float] = {}
        for alt in (item.top_logprobs or []):
            top[alt.token] = float(alt.logprob)
        out.append(TokenLogprob(token=item.token, logprob=float(item.logprob), top_logprobs=top))
    return out or None


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


_CLIENTS: dict[str, BaseLLMClient] = {}
_CACHE: LLMCache | None = None


def _get_cache(config: Config) -> LLMCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = LLMCache(config.cache_path())
    return _CACHE


_PROVIDER_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "litellm": LiteLLMClient,
}


def get_client(model_key: str, config: Config | None = None) -> BaseLLMClient:
    """Resolve a `config.models.<model_key>` entry to a singleton client."""
    if config is None:
        config = load_config()
    if model_key not in config.models:
        raise KeyError(f"unknown model key: {model_key} (available: {list(config.models)})")
    if model_key in _CLIENTS:
        return _CLIENTS[model_key]
    entry = config.models[model_key]
    if entry.provider not in _PROVIDER_REGISTRY:
        raise NotImplementedError(f"unknown provider: {entry.provider}")
    client = _PROVIDER_REGISTRY[entry.provider](entry, config, _get_cache(config))
    _CLIENTS[model_key] = client
    return client


def reset_clients() -> None:
    """Test helper: drop singleton state so a fresh `get_client` round occurs."""
    _CLIENTS.clear()
    global _CACHE
    if _CACHE is not None:
        _CACHE.close()
    _CACHE = None


def list_gateway_models(config: Config | None = None) -> list[str]:
    """Hit `GET /v1/models` on the gateway and return the model IDs it exposes.

    Useful when adjusting `config.yaml.models.*.model_id` to match whatever
    the supervisor's LiteLLM admin registered.
    """
    if config is None:
        config = load_config()
    from openai import OpenAI  # noqa: WPS433

    api_key = os.environ.get(config.api.api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.api.api_key_env} not set; copy .env.example to .env")
    base_url = (
        os.environ.get(config.api.base_url_env) or config.api.default_base_url
    )
    sdk = OpenAI(api_key=api_key, base_url=base_url)
    page = sdk.models.list()
    return sorted(m.id for m in page.data)
