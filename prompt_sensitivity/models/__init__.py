"""Model-provider wrappers and the (prompt, model) cache.

Public API:
- `LLMRequest` / `LLMResponse` Pydantic schemas.
- `BaseLLMClient` abstract interface (sync + async friendly).
- `get_client(model_key, config)` factory, returns a provider-specific client
  wrapping `together.Together` or `openai.OpenAI`.
- `LLMCache` SQLite cache, keyed by SHA256 of canonical request JSON.

§5.1-5.3 of `Research_Design_v3` constrain the model list and capability flags.
"""

from .schemas import LLMRequest, LLMResponse, TokenLogprob
from .cache import LLMCache
from .registry import get_client, BaseLLMClient

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "TokenLogprob",
    "LLMCache",
    "get_client",
    "BaseLLMClient",
]
