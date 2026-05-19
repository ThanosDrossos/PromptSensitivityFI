"""Three-ladder context construction (Sprint 3, Research_Design_v3 §4).

Public API:
  - `LadderRow` / `LadderType`: Pydantic v2 schemas.
  - `build_random_ladder(q)`         - one random permutation per question.
  - `build_gold_first_ladder(q)`     - 2 gold paragraphs first, then 8 distractors.
  - `build_distractor_first_ladder(q)` - 8 distractors first, then 2 gold.
  - `build_all_ladders(q)`           - returns all 3 ladders × 6 levels.
  - `b_theo(N, K, l)`                - hypergeometric bit-cost for the random ladder.
  - `b_emp(u_above, u_below)`        - empirical bit-cost from F-score subsets.
"""

from .schemas import LadderRow, LadderType, LevelSlice
from .random_ladder import build_random_ladder, random_permutation
from .gold_first_ladder import build_gold_first_ladder
from .distractor_first_ladder import build_distractor_first_ladder
from .bit_cost import b_theo, b_theo_table, b_emp

__all__ = [
    "LadderRow",
    "LadderType",
    "LevelSlice",
    "build_random_ladder",
    "random_permutation",
    "build_gold_first_ladder",
    "build_distractor_first_ladder",
    "b_theo",
    "b_theo_table",
    "b_emp",
]


def build_all_ladders(question):  # noqa: ANN001 — circular-friendly type
    """Convenience: return [random, gold_first, distractor_first] ladder rows.

    Used by the build_ladders driver to produce a 3 × 6 = 18-row block per
    question.
    """
    return (
        build_random_ladder(question)
        + build_gold_first_ladder(question)
        + build_distractor_first_ladder(question)
    )
