"""POSIX — Chatterjee et al. 2024 prompt sensitivity index.

Source: arXiv:2410.02185 §3, formula 4 (verified pp. 3-5).

    ψ_{M, X} = (1 / (N (N-1))) · Σ_i Σ_{j≠i}
                  (1 / L_{y_j}) · |log P_M(y_j | x_i) / P_M(y_j | x_j)|

where:
  - X = {x_1, ..., x_N} is the paraphrase universe at one (q, ladder, level)
  - y_j is the model's response to x_j (the "natural" response for paraphrase j)
  - P_M(y_j | x_i) is the probability of response y_j under prompt x_i
  - L_{y_j} is the token length of y_j (normalises for sequence length)

The metric is pure: it takes a precomputed NxN log-probability matrix `log_p`
where `log_p[i, j] = log P_M(y_j | x_i)` and a length vector `lengths[j] = L_{y_j}`.

Pre-cluster (Sprint 4-5): the log-probability matrix is collected via
`/v1/completions` echo (see registry.score_continuation). Only the 3
open-weight models on the gateway support echo; for GPT-4o we return None
upstream rather than computing a fake value.
"""

from __future__ import annotations

import math

import numpy as np


def posix(log_p: np.ndarray, lengths: np.ndarray) -> float:
    """ψ_{M, X} given the log P(y_j | x_i) matrix and y_j lengths.

    Both arguments use natural log to match Chatterjee's formula; the ratio
    inside the absolute value is dimensionless either way.

    Args:
        log_p: shape (N, N). `log_p[i, j]` = log P_M(y_j | x_i). Diagonal
               entries log P_M(y_j | x_j) are the natural-prompt scores.
        lengths: shape (N,). Token length of y_j.

    Returns:
        ψ in nats per token (consistent with Chatterjee's reporting).
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if log_p.ndim != 2 or log_p.shape[0] != log_p.shape[1]:
        raise ValueError(f"log_p must be square (N, N); got {log_p.shape}")
    n = log_p.shape[0]
    if lengths.shape != (n,):
        raise ValueError(f"lengths must have shape ({n},); got {lengths.shape}")
    if n < 2:
        return 0.0
    if (lengths <= 0).any():
        raise ValueError("lengths must be positive everywhere (L_{y_j} > 0)")

    # diag[i] = log P(y_i | x_i)
    diag = np.diag(log_p)
    # |log_p[i, j] - diag[j]| / lengths[j]; mask the diagonal.
    diff = np.abs(log_p - diag[np.newaxis, :])
    weighted = diff / lengths[np.newaxis, :]
    mask = ~np.eye(n, dtype=bool)
    total = float(weighted[mask].sum())
    return total / (n * (n - 1))


# --------------------------------------------------------------------------- #
# Helper: assemble the log-probability matrix from raw echo responses        #
# --------------------------------------------------------------------------- #


def log_p_from_token_logprobs(
    prompt_token_logprobs: list[float],
    y_token_count: int,
) -> float:
    """Sum the last `y_token_count` per-token logprobs from an echo response.

    The /v1/completions echo path returns one logprob per prompt token. When
    the echo prompt is the concatenation `x + y`, the last `len(y)` logprobs
    are the conditional log P(y | x) per token; summing gives log P(y | x).
    """
    if y_token_count <= 0:
        return 0.0
    relevant = prompt_token_logprobs[-y_token_count:]
    return float(sum(relevant))


def build_log_p_matrix(
    scores: list[list[float]],
    lengths: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: from a list-of-lists into the (N, N) matrix + length vector.

    `scores[i][j]` = log P(y_j | x_i). `lengths[j]` = L_{y_j}.
    """
    n = len(scores)
    if any(len(row) != n for row in scores):
        raise ValueError("scores must be a square NxN list of lists")
    if len(lengths) != n:
        raise ValueError(f"lengths must have length {n}")
    log_p = np.asarray(scores, dtype=np.float64)
    lens = np.asarray(lengths, dtype=np.float64)
    return log_p, lens
