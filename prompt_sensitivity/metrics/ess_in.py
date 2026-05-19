"""ESS_in — input-embedding dispersion. Design doc §3 Tier C C4.

    ESS_in = trace(Cov_x[e_M(x)])

where x ranges over the paraphrase universe U_q,l and e_M(x) is the
embedding of the *input prompt* at the model's final prompt token (or, for
black-box models, the external encoder's embedding of the prompt text).

Pre-cluster (Sprint 4-5), the gateway does not expose own-model hidden
states for any of our 4 models. We use sentence-transformers/all-mpnet-base-v2
as the external encoder for all models. Sprint 6 (KIT cluster + vLLM)
re-exposes the own-encoder variant; the orchestrator's `encoder_label`
field tracks which variant was used.

The metric itself is pure: it takes a (N, D) embedding matrix and returns
trace of its covariance.
"""

from __future__ import annotations

import numpy as np


def ess_in(embeddings: np.ndarray) -> float:
    """Trace of the per-feature covariance matrix.

    Equivalent to `np.trace(np.cov(emb.T))` but faster on tall matrices.
    Uses sample covariance (ddof=1) to match scipy / np.cov defaults.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D); got shape {emb.shape}")
    n, _ = emb.shape
    if n < 2:
        return 0.0
    # trace(Cov) = sum of per-feature variances.
    return float(emb.var(axis=0, ddof=1).sum())
