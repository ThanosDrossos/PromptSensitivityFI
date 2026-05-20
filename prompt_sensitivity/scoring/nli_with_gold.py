"""NLI-with-gold F(x) scoring. Research_Design_v3 §8 risk-mitigation row 1.

> NEVER exact-match. F(x) computed via NLI-with-gold (DeBERTa-v3-large-MNLI
> asymmetric: gold entails answer >= 0.7 AND not-contradicts). Hua et al.
> 2025 EMNLP arXiv:2509.01790 shows exact-match inflates measured sensitivity.

For one (paraphrase, prompt) cell:

  1. Sample one response Y from the model at T=0.
  2. Encode the pair (premise=gold, hypothesis=Y) into DeBERTa.
  3. Read entail_prob and contradict_prob.
  4. F(x) = 1 iff entail_prob >= entail_threshold (0.7 by config)
                  AND contradict_prob < contradict_threshold (0.5 by config).

The DeBERTa loader is shared with `paraphrases/nli_filter` and
`metrics/h_sem` via lru_cache, so all three modules amortise the ~1.6 GB
weight load over one process.

Exact-match scoring is also exposed (`exact_match_score`) for the
appendix sanity check the brief asks for, but never used as the primary
F(x).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import Config, load_config


@dataclass(frozen=True)
class NLIScoreResult:
    """One (gold, answer) NLI scoring result + thresholds."""

    entail_prob: float
    contradict_prob: float
    neutral_prob: float
    passes_entail: bool
    passes_contradict: bool

    @property
    def f(self) -> int:
        """1 iff both checks pass — the binary F(x) used by FI_in."""
        return 1 if self.passes_entail and self.passes_contradict else 0


# --------------------------------------------------------------------------- #
# NLI scoring                                                                 #
# --------------------------------------------------------------------------- #


def _label_indices(id2label: dict[int, str]) -> tuple[int, int, int]:
    """Return (entail_idx, contradict_idx, neutral_idx)."""
    e_idx = c_idx = n_idx = -1
    for k, v in id2label.items():
        vlow = v.lower()
        if vlow == "entailment":
            e_idx = k
        elif vlow == "contradiction":
            c_idx = k
        elif vlow == "neutral":
            n_idx = k
    if e_idx < 0 or c_idx < 0:
        raise RuntimeError(f"unexpected label map: {id2label}")
    return e_idx, c_idx, n_idx


def score_nli_with_gold(
    gold: str,
    answer: str,
    *,
    config: Config | None = None,
) -> NLIScoreResult:
    """Score one (gold, answer) pair against the configured thresholds."""
    return score_batch_nli_with_gold(gold, [answer], config=config)[0]


def score_batch_nli_with_gold(
    gold: str,
    answers: Iterable[str],
    *,
    config: Config | None = None,
    batch_size: int = 16,
) -> list[NLIScoreResult]:
    """Score many candidates against ONE gold. Batched to amortise tokenizer cost."""
    import torch

    if config is None:
        config = load_config()
    cfg = config.scoring
    if cfg.method != "nli_with_gold":
        raise RuntimeError(
            f"config.scoring.method = {cfg.method!r}; expected 'nli_with_gold'. "
            f"Anti-pattern: exact_match is appendix-only."
        )

    # Reuse the shared DeBERTa loader so we don't pay another ~1.6 GB.
    from ..paraphrases.nli_filter import _load_nli

    tokenizer, model, device, id2label = _load_nli(cfg.nli_model)
    entail_idx, contradict_idx, neutral_idx = _label_indices(id2label)

    answers = [a if a is not None else "" for a in answers]
    if not answers:
        return []

    # Asymmetric: premise = gold, hypothesis = answer. Tests "does gold entail
    # the answer?" — i.e. is the answer consistent with the gold fact?
    premises = [gold] * len(answers)
    hypotheses = list(answers)

    out: list[NLIScoreResult] = []
    with torch.no_grad():
        for start in range(0, len(premises), batch_size):
            p_chunk = premises[start : start + batch_size]
            h_chunk = hypotheses[start : start + batch_size]
            enc = tokenizer(
                p_chunk,
                h_chunk,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = logits.softmax(dim=-1).cpu().numpy()
            for row in probs:
                entail = float(row[entail_idx])
                contradict = float(row[contradict_idx])
                neutral = (
                    float(row[neutral_idx]) if neutral_idx >= 0 else float(1.0 - entail - contradict)
                )
                out.append(
                    NLIScoreResult(
                        entail_prob=entail,
                        contradict_prob=contradict,
                        neutral_prob=neutral,
                        passes_entail=entail >= cfg.entail_threshold,
                        passes_contradict=contradict < cfg.contradict_threshold,
                    )
                )
    return out


def f_score(gold: str, answer: str, *, config: Config | None = None) -> int:
    """0/1 F(x) for the FI_in metric stack."""
    return score_nli_with_gold(gold, answer, config=config).f


def f_score_batch(
    gold: str,
    answers: Iterable[str],
    *,
    config: Config | None = None,
) -> list[int]:
    """Batched 0/1 scoring for one cell's paraphrases."""
    return [r.f for r in score_batch_nli_with_gold(gold, answers, config=config)]


# --------------------------------------------------------------------------- #
# Exact-match (appendix-only sanity check)                                   #
# --------------------------------------------------------------------------- #


def _normalise(s: str) -> str:
    """Standard QA normalisation: lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match_score(gold: str, answer: str) -> int:
    """Exact-match F(x) — appendix sanity check ONLY.

    Per `config.scoring.exact_match_appendix_only`, this MUST NOT be used as
    the primary F(x). Hua 2025 shows it inflates measured sensitivity.
    """
    return int(_normalise(gold) == _normalise(answer))
