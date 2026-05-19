"""H_sem — semantic entropy via NLI clustering. Farquhar et al. 2024 Nature.

Pipeline (per Farquhar §Methods, Sprint-4 brief §2):

  1. Caller samples k responses {y_1, ..., y_k} from one prompt at T>0.
  2. We cluster the responses by bidirectional NLI entailment using the
     same DeBERTa-v3-large-MNLI as the paraphrase filter (Sprint 2).
     Two responses y_i and y_j land in the same cluster iff
     NLI(y_i ⊨ y_j) >= τ AND NLI(y_j ⊨ y_i) >= τ.
  3. Union-find collapses the pairwise links into equivalence classes.
  4. Semantic entropy is H = -Σ p_c log2 p_c over the cluster-proportion
     distribution.

The DeBERTa loader from `paraphrases/nli_filter.py` is reused — the model is
~1.6 GB and we don't want it loaded twice in one process.

CONTRACT — cluster ID coherence
-------------------------------
`cluster_responses(list[str]) -> list[int]` only clusters WITHIN the given
list. Cluster IDs from two independent calls are NOT comparable: ID 0 in
call A is not the same semantic cluster as ID 0 in call B.

For metrics that need comparable IDs across paraphrases of the same cell
(FI_out, S_τ_freeform, tvd_consistency, estimate_a_q, MetricTuple's
H_sem_mean), callers MUST use `cluster_responses_pooled` — it pools all
responses across paraphrases, clusters once, and slices back into per-
paraphrase assignments with shared IDs.

This is enforced by Sprint-5 driver code, not by the metric layer (which
remains a pure function over precomputed inputs).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from ..config import Config, load_config


def cluster_responses(
    responses: Sequence[str],
    *,
    config: Config | None = None,
    threshold: float | None = None,
) -> list[int]:
    """Return a cluster id (0..C-1) per response via bidirectional NLI.

    Pairs are evaluated in batch (2 * N*(N-1)/2 forward passes through
    DeBERTa). For k=10 samples that's ~90 NLI calls — manageable on CPU.
    """
    if config is None:
        config = load_config()
    if threshold is None:
        threshold = config.h_sem.cluster_threshold

    responses = list(responses)
    n = len(responses)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Lazy import: DeBERTa model is heavy, cached via paraphrases/nli_filter.
    from ..paraphrases.nli_filter import _entail_index, _load_nli
    import torch

    tokenizer, model, device, id2label = _load_nli(config.h_sem.cluster_nli_model)
    entail_idx = _entail_index(id2label)

    # Build all (i, j) pairs with i < j; we'll evaluate both directions in one batch.
    forward_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            forward_pairs.append((i, j))

    if not forward_pairs:
        return [0]

    # Premises for fwd: responses[i]; hypotheses: responses[j].
    # Premises for bwd: responses[j]; hypotheses: responses[i].
    premises = [responses[i] for i, _ in forward_pairs] + [responses[j] for _, j in forward_pairs]
    hypotheses = [responses[j] for _, j in forward_pairs] + [responses[i] for i, _ in forward_pairs]

    entail_probs: list[float] = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(premises), batch_size):
            p_chunk = premises[start : start + batch_size]
            h_chunk = hypotheses[start : start + batch_size]
            enc = tokenizer(
                p_chunk,
                h_chunk,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = logits.softmax(dim=-1).cpu().numpy()
            entail_probs.extend(float(p[entail_idx]) for p in probs)

    m = len(forward_pairs)
    fwd_probs = entail_probs[:m]
    bwd_probs = entail_probs[m:]

    # Union-find over responses; merge i, j iff both directions entail.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), fwd, bwd in zip(forward_pairs, fwd_probs, bwd_probs, strict=True):
        if fwd >= threshold and bwd >= threshold:
            union(i, j)

    # Re-label roots to contiguous 0..C-1.
    root_to_label: dict[int, int] = {}
    assignment: list[int] = []
    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = len(root_to_label)
        assignment.append(root_to_label[r])
    return assignment


def entropy_from_assignment(assignment: Iterable[int]) -> float:
    """Shannon entropy in bits of the cluster-proportion distribution."""
    assignment = list(assignment)
    n = len(assignment)
    if n == 0:
        return 0.0
    counts = np.bincount(np.asarray(assignment, dtype=int))
    probs = counts[counts > 0] / n
    return float(-np.sum(probs * np.log2(probs)))


def h_sem(
    responses: Sequence[str],
    *,
    config: Config | None = None,
    precomputed_clusters: Sequence[int] | None = None,
) -> tuple[float, list[int]]:
    """Return (H_sem in bits, cluster assignment per response).

    `precomputed_clusters` lets the caller skip the NLI pass when clusters
    are already known (e.g. when h_sem is called repeatedly for fi_out and
    we'd like to reuse one clustering).
    """
    if precomputed_clusters is not None:
        assignment = list(precomputed_clusters)
        if len(assignment) != len(responses):
            raise ValueError("precomputed_clusters length mismatch")
    else:
        assignment = cluster_responses(responses, config=config)
    return entropy_from_assignment(assignment), assignment


def n_unique_clusters(assignment: Iterable[int]) -> int:
    """|cluster set| — used as |A_q,x| building block for FI_out."""
    return len(set(assignment))


def cluster_responses_pooled(
    responses_per_prompt: Mapping[int, Sequence[str]],
    *,
    config: Config | None = None,
    threshold: float | None = None,
) -> dict[int, list[int]]:
    """Pool-cluster responses across paraphrases; return per-prompt assignments
    with cluster IDs that are comparable across prompts.

    This is the API that Sprint-5 pipeline code uses to feed FI_out / S_τ /
    1-TVD / |A_q|. The naive alternative — clustering each paraphrase's
    responses independently and hoping ID 0 means the same thing in both —
    over-counts |A_q| and mis-computes inter-paraphrase consistency.

    Implementation: concatenate all responses into one flat list, run a
    single union-find clustering, then slice the result back into the
    original per-paraphrase shape.

    Returns dict {paraphrase_idx -> [cluster_id, ...]} with the same shape
    as `responses_per_prompt`.
    """
    items = list(responses_per_prompt.items())
    if not items:
        return {}

    # Flatten while remembering each slice's range.
    flat: list[str] = []
    ranges: list[tuple[int, int, int]] = []  # (paraphrase_idx, start, end)
    for idx, resps in items:
        start = len(flat)
        flat.extend(resps)
        ranges.append((idx, start, len(flat)))

    pooled_assignment = cluster_responses(flat, config=config, threshold=threshold)

    out: dict[int, list[int]] = {}
    for idx, start, end in ranges:
        out[idx] = pooled_assignment[start:end]
    return out
