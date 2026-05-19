"""Errica et al. (NAACL 2025) two-number axis. Design doc §3 Tier B.

Two scalars per (query, ladder, level, model):

  - S_τ(x):  per-prompt sensitivity, entropy normalised to [0, 1].
              For free-form Q&A: H_sem(Y|X=x) / log2|A_q|.
              For multiple-choice: H(Y|X=x) / log2(C) over the class set.
  - 1 - TVD: pairwise consistency across paraphrases.

Errica's paper uses natural log; we use log2 internally to stay in bits but
the normalisation makes the result dimensionless either way. The MC variant
matches Errica Eq. 3 exactly (cross-checked on the TREC small example).

Free-form pre-cluster: we use the H_sem-derived MC variant (cluster
proportions as the empirical distribution), because the gateway does not
expose full-vocab token logprobs needed for the original token-entropy
formulation (see registry capability matrix).

CONTRACT: cluster_assignments fed to s_tau_freeform / tvd_consistency MUST
use POOLED cluster IDs across paraphrases of the cell (see h_sem
docstring). Independent per-paraphrase clusterings would produce
incompatible distributions and break TVD entirely.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .h_sem import entropy_from_assignment


# --------------------------------------------------------------------------- #
# S_tau                                                                       #
# --------------------------------------------------------------------------- #


def s_tau_freeform(cluster_assignment: Sequence[int], a_q_size: int) -> float:
    """Per-prompt sensitivity for free-form Q&A.

    S_τ(x) = H_sem(Y|X=x) / log2|A_q|  in [0, 1].

    Returns 0 when |A_q| <= 1 (the support is a single cluster, so no
    sensitivity to compute).
    """
    if a_q_size <= 1:
        return 0.0
    h = entropy_from_assignment(cluster_assignment)
    return float(h / math.log2(a_q_size))


def s_tau_multiple_choice(
    class_counts: Sequence[int] | np.ndarray,
    n_classes: int,
) -> float:
    """Per-prompt sensitivity for multiple-choice.

    S_τ(x) = H(Y|X=x) / log2(C) in [0, 1].

    `class_counts[c]` is the count of times class c was the model's answer
    across k samples.
    """
    counts = np.asarray(class_counts, dtype=float)
    n = counts.sum()
    if n <= 0 or n_classes <= 1:
        return 0.0
    probs = counts[counts > 0] / n
    h = float(-np.sum(probs * np.log2(probs)))
    return h / math.log2(n_classes)


def s_tau_summary(per_prompt: Mapping[int, float]) -> float:
    """Mean S_τ across paraphrases — the scalar that lands in MetricTuple."""
    if not per_prompt:
        return 0.0
    return float(np.mean(list(per_prompt.values())))


# --------------------------------------------------------------------------- #
# 1 - TVD consistency                                                         #
# --------------------------------------------------------------------------- #


def _proportions(assignment: Sequence[int], support: Sequence[int]) -> np.ndarray:
    """Empirical distribution over a fixed cluster-ID support."""
    counts = np.zeros(len(support), dtype=float)
    n = len(assignment)
    if n == 0:
        return counts
    pos = {cid: i for i, cid in enumerate(support)}
    for cid in assignment:
        if cid in pos:
            counts[pos[cid]] += 1
    return counts / n


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions on the same support."""
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    return float(0.5 * np.abs(p - q).sum())


def tvd_consistency(
    cluster_assignments: Mapping[int, Sequence[int]] | Sequence[Sequence[int]],
) -> float:
    """Mean (1 - TVD) across all pairs of paraphrases.

    Errica Eq. 4: consistency(x, x') = 1 - TVD(p(·|x), p(·|x')). We report
    the mean over the (N choose 2) pairs.
    """
    if isinstance(cluster_assignments, Mapping):
        items = list(cluster_assignments.values())
    else:
        items = list(cluster_assignments)

    if len(items) < 2:
        return 1.0  # singleton "pair" is trivially consistent

    support = sorted({cid for assign in items for cid in assign})
    if not support:
        return 1.0
    dists = [_proportions(a, support) for a in items]

    consistencies: list[float] = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            consistencies.append(1.0 - tvd(dists[i], dists[j]))
    return float(np.mean(consistencies))
