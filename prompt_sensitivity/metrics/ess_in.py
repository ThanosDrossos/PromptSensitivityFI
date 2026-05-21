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

KNOWN LIMITATION (documented in 2026-05-21 E2E smoke):
`sentence-transformers/all-mpnet-base-v2` is *trained* to map paraphrases
of the same sentence to nearby points — that is its core objective. So
when 10 paraphrases of one question are encoded together with a shared
~1500-char context block (e.g. ladder levels 4-10 where context
dominates >95% of the prompt text), the per-feature variance approaches
0 and ESS_in_ext collapses to ~0. This is the encoder doing its job
correctly, not a bug. The brief's own-encoder variant (e_M from the
model's last hidden state) does not have this property because it
projects through model-specific features that are not paraphrase-
invariant. Until cluster access lands, treat ESS_in_ext as a weak signal
that mostly tracks context length, not paraphrase diversity. The other
geometric metric, rho_u, uses RESPONSE embeddings (which do vary across
paraphrases) and is the more informative geometric signal pre-cluster.

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
