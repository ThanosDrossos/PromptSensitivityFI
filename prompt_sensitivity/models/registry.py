"""Provider-agnostic LLM client registry.

`get_client(model_key)` resolves a `config.yaml` model entry to a concrete
`BaseLLMClient` (Together or OpenAI). Each call route flows through the SQLite
cache: cache hit -> no API call.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
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
from .schemas import LLMRequest, LLMResponse, TokenLogprob


# Global per-provider rate limiters. Lazily-built; one bucket per process.
_BUCKETS: dict[str, TokenBucket] = {}


def _bucket(provider: str, qps: float) -> TokenBucket:
    if provider not in _BUCKETS:
        _BUCKETS[provider] = TokenBucket(rate=qps, capacity=qps)
    return _BUCKETS[provider]


# Eagerly load .env so env vars are available even when called from REPL.
load_dotenv(REPO_ROOT / ".env", override=False)


class BaseLLMClient(ABC):
    """Common interface for all providers.

    The `complete` method is the only one the rest of the codebase needs. It
    enforces three invariants:

    1. Every call hits the cache first; cache misses are persisted.
    2. Calls go through the per-provider token bucket (avoids 429s).
    3. Failures are retried with exponential backoff per `config.api`.
    """

    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        self.entry = model_entry
        self.config = config
        self.cache = cache
        qps = config.api.rate_limit_qps.get(model_entry.provider, 5.0)
        self._bucket = _bucket(model_entry.provider, qps)

    # ---------------- public API ----------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Always force the registry's model_id and provider onto the request so
        # call sites cannot accidentally route a Llama prompt to GPT-4o.
        if request.model_id != self.entry.model_id or request.provider != self.entry.provider:
            request = request.model_copy(
                update={"model_id": self.entry.model_id, "provider": self.entry.provider}
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


# Exceptions classes considered transient for retry purposes. Imported lazily
# inside the providers; we keep a tuple here that the retry decorator can match.
_TRANSIENT: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)


# --------------------------------------------------------------------------- #
# Together provider                                                           #
# --------------------------------------------------------------------------- #


class TogetherClient(BaseLLMClient):
    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        super().__init__(model_entry, config, cache)
        from together import Together  # noqa: WPS433 — lazy import so import-time stays cheap

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set; check .env")
        self._sdk = Together(api_key=api_key)

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
            # Together exposes a numeric `logprobs` (top-N) parameter on /completions
            # but for /chat/completions only top-1 is supported in current SDKs.
            kwargs["logprobs"] = request.top_logprobs or 1

        completion = self._sdk.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        token_logprobs = _extract_together_logprobs(choice) if request.logprobs else None

        return LLMResponse(
            request_hash=request.cache_key(),
            text=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_logprobs=token_logprobs,
            raw_provider_response=None,  # full response not stored — too large
        )


def _extract_together_logprobs(choice: Any) -> list[TokenLogprob] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    tokens = getattr(lp, "tokens", None) or []
    token_logprobs = getattr(lp, "token_logprobs", None) or []
    out: list[TokenLogprob] = []
    for tok, logp in zip(tokens, token_logprobs, strict=False):
        if logp is None:
            continue
        out.append(TokenLogprob(token=tok, logprob=float(logp), top_logprobs={}))
    return out or None


# --------------------------------------------------------------------------- #
# OpenAI provider                                                              #
# --------------------------------------------------------------------------- #


class OpenAIClient(BaseLLMClient):
    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        super().__init__(model_entry, config, cache)
        from openai import OpenAI  # noqa: WPS433

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; check .env")
        self._sdk = OpenAI(api_key=api_key)

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
                kwargs["top_logprobs"] = request.top_logprobs

        completion = self._sdk.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None

        token_logprobs = _extract_openai_logprobs(choice) if request.logprobs else None

        return LLMResponse(
            request_hash=request.cache_key(),
            text=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_logprobs=token_logprobs,
            raw_provider_response=None,
        )


def _extract_openai_logprobs(choice: Any) -> list[TokenLogprob] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None or not getattr(lp, "content", None):
        return None
    out: list[TokenLogprob] = []
    for item in lp.content:
        top = {alt.token: float(alt.logprob) for alt in (item.top_logprobs or [])}
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
    "together": TogetherClient,
    "openai": OpenAIClient,
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
