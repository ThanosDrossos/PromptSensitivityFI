"""Raw-paraphrase generation. Calls the gateway, returns `RawParaphrase` rows.

Sprint 2 §3.1: 4 templates × 10 samples per template per question = 40 raw
candidates. Each call is `n=1` with a deterministic seed so every
(question, role, sample_idx) pair is independently cacheable.

Per Section_7 §7.6.1 R3 (independence): the generator MUST be different from
the models being evaluated. Our generator is `kit.gpt-4.1` (mapped via the
`paraphrases.generator_model` key); the evaluated models are the three
open-weight Llama/Teuken/Qwen and `kit.gpt-4.1` itself. The last collision is
unavoidable — flag in the writeup as a known confound for GPT-4.1 results.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from loguru import logger

from ..config import Config, load_config
from ..models import LLMRequest
from ..models.registry import get_client
from .prompts import build_paraphrase_messages
from .schemas import RawParaphrase, RoleName


def _seed_for(question_id: str, role: RoleName, sample_idx: int) -> int:
    """Deterministic 31-bit seed per (question, role, sample_idx).

    The cache key already includes `seed` so we get reproducibility across
    runs. Hashing keeps the seed pseudo-random across questions even though
    sample_idx alone is small.
    """
    h = hashlib.sha256(f"{question_id}|{role}|{sample_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFF_FFFF


def generate_raw_paraphrases(
    question_id: str,
    question_text: str,
    *,
    config: Config | None = None,
    sample_idxs: Iterable[int] | None = None,
    roles: Iterable[RoleName] | None = None,
) -> list[RawParaphrase]:
    """Generate raw paraphrase candidates for one question.

    `sample_idxs` defaults to `range(config.paraphrases.samples_per_template)`.
    Re-running with a wider `sample_idxs` (e.g. range(10, 20)) generates fresh
    candidates while leaving cached ones untouched — that's the regeneration
    path used by the pipeline when the first 40 don't yield 30 accepted.
    """
    if config is None:
        config = load_config()
    pcfg = config.paraphrases

    if sample_idxs is None:
        sample_idxs = range(pcfg.samples_per_template)
    if roles is None:
        roles = pcfg.templates  # type: ignore[assignment]
    sample_idxs = list(sample_idxs)
    roles = list(roles)

    client = get_client(pcfg.generator_model, config)
    model_entry = config.models[pcfg.generator_model]

    out: list[RawParaphrase] = []
    for role in roles:
        messages = build_paraphrase_messages(question_text, role)  # type: ignore[arg-type]
        for s in sample_idxs:
            seed = _seed_for(question_id, role, s)  # type: ignore[arg-type]
            req = LLMRequest(
                provider=model_entry.provider,  # type: ignore[arg-type]
                model_id=model_entry.model_id,
                messages=messages,
                temperature=pcfg.generator_temperature,
                top_p=1.0,
                max_tokens=128,
                seed=seed,
                purpose="paraphrase_gen",
            )
            resp = client.complete(req)
            text = _clean_one_line(resp.text)
            if not text:
                logger.debug("empty paraphrase {} {} sample {}", question_id, role, s)
                continue
            out.append(
                RawParaphrase(
                    question_id=question_id,
                    role=role,  # type: ignore[arg-type]
                    sample_idx=s,
                    text=text,
                    generator_model_key=pcfg.generator_model,
                    generator_seed=seed,
                    request_hash=resp.request_hash,
                )
            )
    return out


def _clean_one_line(text: str) -> str:
    """Strip surrounding quotes/whitespace and collapse to the first line.

    Generators occasionally return wrapped quotes or stray markdown despite
    the system prompt. The cleaner is conservative — anything past the first
    \\n is dropped — but does not silently rewrite content.
    """
    if not text:
        return ""
    first = text.strip().splitlines()[0].strip()
    # Strip a single layer of surrounding quotes ("...", '...').
    if len(first) >= 2 and first[0] == first[-1] and first[0] in {'"', "'", "`"}:
        first = first[1:-1].strip()
    return first
