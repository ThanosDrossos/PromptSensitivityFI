"""ρ_u — Cox 2025 "Mapping from Meaning" epistemic-variance ratio.

Reference: github.com/xocelyk/paraphrase-uncertainty (Cox et al. 2025 AAAI).

The setup. For one (question, ladder, level, model) cell we have N
paraphrases × k sampled responses per paraphrase = N*k embeddings of the
*responses* via the external encoder. The law of total covariance
decomposes the total covariance of these embeddings into:

  Cov_total = Cov_within(within-paraphrase variability)
            + Cov_across(across-paraphrase mean variability)

Cox writes (paraphrasing Eq. 10 of the AAAI paper, slightly re-notated):

  U_t  = tr(Cov_total)      = total uncertainty (in encoder-trace units)
  U_a  = tr(Cov_within)     = aleatoric (within-paraphrase sampling noise)
  U_e  = tr(Cov_across)     = epistemic (prompt-induced)
  ρ_u  = U_e / (U_t + ε)    = fraction of variance attributable to prompt choice

ε > 0 stabilises the ratio when U_t collapses to 0 (e.g. degenerate where
every response embeds to the same point). Cox uses 1e-12; we expose it.

ρ_u ∈ [0, 1]:
  - 0   means all variance is within-paraphrase (the prompt does not move
        the response distribution; ideal robustness).
  - 1   means all variance is across-paraphrase (paraphrasing fully
        determines the output; maximally sensitive).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def rho_u(
    embeddings_per_paraphrase: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute U_t, U_a, U_e, ρ_u from per-paraphrase embedding samples.

    Each `embeddings_per_paraphrase[i]` is a (k_i, D) array of the k_i
    sampled responses' embeddings for paraphrase i. k_i can vary per
    paraphrase (Cox allows this) but D must match across paraphrases.

    Returns a dict so the orchestrator can record all four scalars for
    diagnostics, even though only ρ_u lands in the MetricTuple.
    """
    if isinstance(embeddings_per_paraphrase, Mapping):
        groups = [np.asarray(v, dtype=np.float64) for v in embeddings_per_paraphrase.values()]
    else:
        groups = [np.asarray(v, dtype=np.float64) for v in embeddings_per_paraphrase]

    groups = [g for g in groups if g.size > 0]
    if not groups:
        return {"U_t": 0.0, "U_a": 0.0, "U_e": 0.0, "rho_u": 0.0}

    dims = {g.shape[1] for g in groups}
    if len(dims) > 1:
        raise ValueError(f"embedding dims differ across paraphrases: {dims}")

    # Stack all (paraphrase, sample) -> embedding into a single (sum k_i, D).
    all_emb = np.vstack(groups)
    n_total = all_emb.shape[0]
    if n_total < 2:
        return {"U_t": 0.0, "U_a": 0.0, "U_e": 0.0, "rho_u": 0.0}

    # Total variance per feature (population convention here so within+across = total).
    u_t = float(all_emb.var(axis=0, ddof=0).sum())

    # Within-paraphrase mean variance (averaged across paraphrases, weighted by sample count).
    within_sum = 0.0
    within_weight = 0
    for g in groups:
        if g.shape[0] < 1:
            continue
        within_sum += float(g.var(axis=0, ddof=0).sum()) * g.shape[0]
        within_weight += g.shape[0]
    u_a = within_sum / within_weight if within_weight > 0 else 0.0

    # Across-paraphrase variance = variance of the per-paraphrase means
    # (weighted by sample count to recover the total-covariance decomposition).
    means = np.vstack([g.mean(axis=0) for g in groups])
    # Weighted mean (mean of all embeddings) for population variance.
    weights = np.asarray([g.shape[0] for g in groups], dtype=np.float64)
    overall_mean = (means * weights[:, None]).sum(axis=0) / weights.sum()
    centred = means - overall_mean
    weighted_sq = (weights[:, None] * centred**2).sum(axis=0) / weights.sum()
    u_e = float(weighted_sq.sum())

    rho = u_e / (u_t + epsilon)
    # Clamp to [0, 1] to guard against floating-point drift around the boundary.
    rho = max(0.0, min(1.0, rho))
    return {"U_t": u_t, "U_a": u_a, "U_e": u_e, "rho_u": rho}
