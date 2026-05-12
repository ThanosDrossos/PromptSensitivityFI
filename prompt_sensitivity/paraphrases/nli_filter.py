"""DeBERTa-v3-large-MNLI bidirectional entailment filter.

Section_7 §7.6.1 R1: "every x ∈ U_q must be semantically equivalent to the
canonical x_0. Operationalize as bidirectional NLI entailment under a strong
NLI model." Accept x iff
    NLI(x_0 ⊨ x) >= τ AND NLI(x ⊨ x_0) >= τ
with τ = 0.9 by default (`config.paraphrases.nli.bidirectional_threshold`)
and a relaxed fallback τ = 0.85 when too few candidates survive (§9 Sprint 2
risk-mitigation row in `Research_Design_v3`).

The model is `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. It
exposes three labels: 0=entailment, 1=neutral, 2=contradiction. We take the
softmax probability of the entailment class as the score.

The model is loaded lazily and cached at module scope so a single process
amortises the ~1.6 GB weight load over all calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from loguru import logger

from ..config import Config, load_config


@dataclass(frozen=True)
class NLIScores:
    """Bidirectional entailment for one (x_0, x) pair."""

    entail_fwd: float  # P(entailment | premise=x_0, hypothesis=x)
    entail_bwd: float  # P(entailment | premise=x,   hypothesis=x_0)

    def passes(self, threshold: float) -> bool:
        return self.entail_fwd >= threshold and self.entail_bwd >= threshold


# --------------------------------------------------------------------------- #
# Model loader                                                                #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_nli(model_name: str):  # type: ignore[no-untyped-def]
    """Lazy import of transformers + torch. Returns (tokenizer, model, device, id2label).

    Singleton: subsequent calls within the same process reuse the same
    in-RAM weights.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("loading NLI model {} (this may take ~30s on first call)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if "entailment" not in id2label.values():
        raise RuntimeError(f"unexpected label map for {model_name}: {id2label}")
    logger.info("NLI model loaded on {}", device)
    return tokenizer, model, device, id2label


def _entail_index(id2label: dict[int, str]) -> int:
    for k, v in id2label.items():
        if v == "entailment":
            return k
    raise RuntimeError("entailment label not found")


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def score_pair(
    original: str,
    paraphrase: str,
    *,
    config: Config | None = None,
) -> NLIScores:
    """Bidirectional entailment between `original` and `paraphrase`.

    The DeBERTa model is run twice: once with original as premise (entail_fwd)
    and once with paraphrase as premise (entail_bwd).
    """
    return score_batch(original, [paraphrase], config=config)[0]


def score_batch(
    original: str,
    paraphrases: Iterable[str],
    *,
    config: Config | None = None,
    batch_size: int = 16,
) -> list[NLIScores]:
    """Bidirectional entailment for many paraphrases against one original.

    Forward and backward passes are batched together: for N paraphrases we
    issue 2N tokenised pairs in chunks of `batch_size`. This is the only
    place that bulk-runs DeBERTa, so it's worth batching.
    """
    import torch

    if config is None:
        config = load_config()
    nli_cfg = config.paraphrases.nli
    tokenizer, model, device, id2label = _load_nli(nli_cfg.model)
    entail_idx = _entail_index(id2label)

    paraphrases = list(paraphrases)
    if not paraphrases:
        return []

    # Build the 2N pairs: forward = (original, p_i), backward = (p_i, original).
    pairs: list[tuple[str, str]] = []
    for p in paraphrases:
        pairs.append((original, p))
        pairs.append((p, original))

    entail_probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            enc = tokenizer(
                [a for a, _ in chunk],
                [b for _, b in chunk],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits  # shape: (chunk_size, n_labels)
            probs = logits.softmax(dim=-1).cpu().numpy()
            entail_probs.extend(float(p[entail_idx]) for p in probs)

    # Re-interleave into (fwd, bwd) per paraphrase.
    out: list[NLIScores] = []
    for i in range(len(paraphrases)):
        fwd = entail_probs[2 * i]
        bwd = entail_probs[2 * i + 1]
        out.append(NLIScores(entail_fwd=fwd, entail_bwd=bwd))
    return out


def filter_by_nli(
    original: str,
    candidates: Iterable[str],
    *,
    config: Config | None = None,
    threshold: float | None = None,
) -> list[tuple[bool, NLIScores]]:
    """Apply the bidirectional threshold to each (original, candidate) pair.

    Returns parallel list of (passed?, scores). Caller decides whether to
    relax the threshold and re-call.
    """
    if config is None:
        config = load_config()
    if threshold is None:
        threshold = config.paraphrases.nli.bidirectional_threshold
    scores = score_batch(original, candidates, config=config)
    return [(s.passes(threshold), s) for s in scores]
