"""Bit-cost estimators for the three-ladder design. Research_Design_v3 §4.4.

Two complementary estimators:

  - `b_theo(N, K, l)`  — closed-form *theoretical* bit-cost for the random
    ladder. Information content of "the random size-l subset of N paragraphs
    contains at least one of K gold paragraphs":

        b_theo(N, K, l) = -log2( 1 - C(N-K, l) / C(N, l) )

    Hypergeometric; matches scipy.stats.hypergeom analytically.

  - `b_emp(u_above, u_below)`  — *empirical* bit-cost from observed F-score
    subsets at consecutive ladder rungs:

        b_emp = log2( |U_above| / |U_below| )

    Where U_l is the set of prompts at level l that achieve F(x) >= k for
    some threshold k. Computable only after Sprint 4 produces F scores; for
    Sprint 3 we expose the function with a small caller surface.

The two should match within ~30% across levels if the random-shuffle
assumption (gold-effect dominates context effect) holds. Mismatch reveals
distractor-composition sensitivity beyond gold inclusion (§4.4 last
paragraph).
"""

from __future__ import annotations

import math
from typing import Iterable

from scipy.stats import hypergeom


def _comb(n: int, k: int) -> int:
    """Safe combinatorial: C(n, k) with C(n, k) = 0 for k > n or negative."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def prob_at_least_one_gold(N: int, K: int, l: int) -> float:
    """P(random size-l subset of N items contains >=1 of K gold) = 1 - C(N-K, l)/C(N, l).

    Edge cases:
      - l == 0:           prob = 0 (you sampled nothing)
      - l >= N - K + 1:   prob = 1 (pigeonhole: a gold must be included)
      - K == 0:           prob = 0 (no golds anywhere)
      - K == N:           prob = 1 for l >= 1
    """
    if l <= 0 or K <= 0 or N <= 0:
        return 0.0
    if l > N:
        raise ValueError(f"l={l} > N={N}")
    if l > N - K:
        return 1.0  # C(N-K, l) = 0
    denom = _comb(N, l)
    numer = _comb(N - K, l)
    return 1.0 - (numer / denom)


def b_theo(N: int, K: int, l: int) -> float:
    """Theoretical bit-cost. +inf when prob = 0 (the l=0 degenerate case)."""
    p = prob_at_least_one_gold(N, K, l)
    if p <= 0.0:
        return math.inf
    return -math.log2(p)


def b_theo_table(N: int, K: int, levels: Iterable[int]) -> list[dict]:
    """Convenience: per-level table used by the Sprint-3 gate report."""
    rows: list[dict] = []
    for l in levels:
        p = prob_at_least_one_gold(N, K, l)
        rows.append(
            {
                "level": l,
                "P(>=1 gold)": p,
                "b_theo (bits)": math.inf if p <= 0.0 else -math.log2(p),
            }
        )
    return rows


def b_emp(u_above: int, u_below: int) -> float:
    """Empirical bit-cost: log2(|U_above| / |U_below|).

    `u_above` is the number of successful prompts at the higher-context level;
    `u_below` is the same at the lower-context level. A positive value means
    moving up the ladder (adding context) shrinks the success set, which is
    counter-intuitive — used as a diagnostic, not a quality metric.
    Returns +inf if u_below == 0 (cannot quantify).
    """
    if u_below <= 0:
        return math.inf
    if u_above <= 0:
        return -math.inf
    return math.log2(u_above / u_below)


# --------------------------------------------------------------------------- #
# Sanity check against scipy.stats.hypergeom                                 #
# --------------------------------------------------------------------------- #


def _scipy_prob_at_least_one_gold(N: int, K: int, l: int) -> float:
    """Reference computation using scipy's hypergeometric distribution.

    scipy.stats.hypergeom(M=N, n=K, N=l) is the number of golds drawn when
    you sample l items without replacement from a population of N where K
    are gold. P(>=1) = 1 - P(0).
    """
    if l <= 0 or K <= 0 or N <= 0:
        return 0.0
    if l > N:
        raise ValueError(f"l={l} > N={N}")
    rv = hypergeom(N, K, l)
    return float(1.0 - rv.pmf(0))
