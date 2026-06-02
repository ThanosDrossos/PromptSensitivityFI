"""F(x) scoring.

Two scoring paths:

  - BINARY final-answer (`nli_with_gold`): NLI-with-gold asymmetric, gold
    entails answer >= 0.7 AND not-contradicts. Used for HotpotQA / 2Wiki and
    kept as a secondary number for MuSiQue. NEVER exact-match (Hua 2025).

  - GRADED chain-completion (`chain_score`): fraction of MuSiQue reasoning
    hops the model recovers. Primary F for MuSiQue (v6 §2) — makes the
    FI_in curve graded instead of a step function.
"""

from .nli_with_gold import (
    NLIScoreResult,
    score_nli_with_gold,
    score_batch_nli_with_gold,
    f_score,
    f_score_batch,
    exact_match_score,
)
from .chain_score import (
    chain_completion_score,
    chain_completion_score_batch,
    chain_fraction,
    build_fact_statements,
    resolve_placeholders,
)

__all__ = [
    "NLIScoreResult",
    "score_nli_with_gold",
    "score_batch_nli_with_gold",
    "f_score",
    "f_score_batch",
    "exact_match_score",
    "chain_completion_score",
    "chain_completion_score_batch",
    "chain_fraction",
    "build_fact_statements",
    "resolve_placeholders",
]
