"""F(x) scoring — NLI-with-gold (Sprint 5).

Per the brief's anti-pattern rule: NEVER exact-match. Use NLI-with-gold
asymmetric: gold entails answer >= 0.7 AND not-contradicts.

Exposed:
  - `score_nli_with_gold(gold, answer, *, config) -> NLIScoreResult`
  - `f_score(...) -> int`   # 0/1 wrapper for the metric stack
"""

from .nli_with_gold import (
    NLIScoreResult,
    score_nli_with_gold,
    score_batch_nli_with_gold,
    f_score,
    f_score_batch,
    exact_match_score,
)

__all__ = [
    "NLIScoreResult",
    "score_nli_with_gold",
    "score_batch_nli_with_gold",
    "f_score",
    "f_score_batch",
    "exact_match_score",
]
