"""Paraphrase generation + filtering pipeline (Sprint 2).

Implements §3.1 of Research_Design_v3 + §7.6.1 of Section_7:
  1. Generate ~40 raw candidates per question via GPT-4.1 (gateway-routed) at
     T=0.8 with four role templates (neutral / journalist / casual_user /
     domain_expert), following the Razavi 2025 ECIR PromptSET pattern.
  2. DeBERTa-v3-large-MNLI bidirectional entailment filter, threshold 0.9
     (Section_7 §7.6.1 R1 condition).
  3. GPT-4.1-as-judge constraint-set filter: Jaccard >= 0.9 between the
     answer-sets the judge lists for x_0 vs x.
  4. Deduplicate by Levenshtein edit distance > 5.

Output: exactly 30 paraphrases per question (config.paraphrases.n_per_question).
"""

from .schemas import RawParaphrase, AcceptedParaphrase, ParaphraseSet
from .prompts import build_paraphrase_messages, ROLE_NAMES
from .constraint_filter import (
    judge_contains_gold,
    filter_by_constraint_with_gold,
    filter_by_constraint,
)

__all__ = [
    "RawParaphrase",
    "AcceptedParaphrase",
    "ParaphraseSet",
    "build_paraphrase_messages",
    "ROLE_NAMES",
    "judge_contains_gold",
    "filter_by_constraint_with_gold",
    "filter_by_constraint",
]
