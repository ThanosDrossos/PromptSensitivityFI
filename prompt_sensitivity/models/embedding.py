"""Sentence-encoder wrapper. External encoder = sentence-transformers/all-mpnet-base-v2.

Pre-cluster (Sprint 4-5), all four models share this encoder for ESS_in^ext
and rho_u. The "own-encoder" variants of ESS_in / rho_u require last-layer
hidden states which the gateway does not expose; they land in Sprint 6 with
vLLM on the KIT cluster.

The model is ~440 MB. Lazy-loaded once per process via lru_cache, mirroring
the DeBERTa NLI loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from loguru import logger

from ..config import Config, load_config


@lru_cache(maxsize=2)
def _load_sentence_encoder(model_name: str):  # type: ignore[no-untyped-def]
    """Returns a sentence_transformers.SentenceTransformer pinned to CPU/GPU as available."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading sentence encoder {} on {}", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    return model


def encode_texts(
    texts: Sequence[str],
    *,
    config: Config | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    normalize: bool = False,
) -> np.ndarray:
    """Encode a list of strings -> (N, D) float32 numpy array.

    `normalize=True` returns unit-norm vectors (useful for cosine-distance
    consumers; we leave the default off so the raw covariance trace in
    ess_in / rho_u is interpretable in the encoder's native scale).
    """
    if config is None:
        config = load_config()
    if model_name is None:
        model_name = config.embedding.external_encoder
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    enc = _load_sentence_encoder(model_name)
    arr = enc.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(arr, dtype=np.float32)
