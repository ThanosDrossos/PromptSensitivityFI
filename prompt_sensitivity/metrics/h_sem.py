"""H_sem — semantic entropy via NLI clustering. Farquhar et al. 2024 Nature.

Pipeline (per Farquhar §Methods, Sprint-4 brief §2):

  1. Caller samples k responses {y_1, ..., y_k} from one prompt at T>0.
  2. We cluster the responses by bidirectional NLI entailment using the
     same DeBERTa-v3-large-MNLI as the paraphrase filter (Sprint 2).
     Two responses y_i and y_j land in the same cluster iff they entail each
     other in BOTH directions, where "entail" is `config.h_sem.cluster_criterion`:
       - "label" (default): argmax of the 3 NLI classes is `entailment`;
       - "prob": P(entailment) >= τ (`cluster_threshold`).
     Callers usually cluster the EXTRACTED ANSWER of each sample, not the full
     generation (`config.h_sem.cluster_on`), so style/verbosity don't over-merge.
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

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ..config import Config, load_config


def _nli_prob_vectors(
    premises: list[str], hypotheses: list[str], *, model_name: str
) -> tuple[list[np.ndarray], int]:
    """Run DeBERTa-MNLI on aligned (premise, hypothesis) pairs.

    Returns (one 3-class softmax vector per pair, entailment-column index). This
    is the ONLY heavy dependency of `cluster_responses`, isolated so the criterion
    + union-find logic is unit-testable by patching this function (no model load).
    """
    from ..paraphrases.nli_filter import _entail_index, _load_nli
    import torch

    tokenizer, model, device, id2label = _load_nli(model_name)
    entail_idx = _entail_index(id2label)
    out: list[np.ndarray] = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(premises), batch_size):
            p_chunk = premises[start : start + batch_size]
            h_chunk = hypotheses[start : start + batch_size]
            enc = tokenizer(
                p_chunk, h_chunk, truncation=True, padding=True,
                max_length=256, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            out.extend(logits.softmax(dim=-1).cpu().numpy())
    return out, entail_idx


def cluster_responses(
    responses: Sequence[str],
    *,
    config: Config | None = None,
    threshold: float | None = None,
    criterion: str | None = None,
) -> list[int]:
    """Return a cluster id (0..C-1) per response via bidirectional NLI.

    Pairs are evaluated in batch (2 * N*(N-1)/2 forward passes through
    DeBERTa). For k=10 samples that's ~90 NLI calls — manageable on CPU.

    `criterion` (default `config.h_sem.cluster_criterion`) decides the merge rule
    for an ordered pair, applied in BOTH directions:
      - "label": argmax over the 3 NLI classes is `entailment` (strict — the rule
        used by Farquhar et al. 2024; default).
      - "prob":  P(entailment) >= `threshold` (legacy, lenient — over-merges
        distinct-but-related answers at the default τ=0.5).
    Pure function: no config mutation, deterministic given the NLI weights.
    """
    if config is None:
        config = load_config()
    if threshold is None:
        threshold = config.h_sem.cluster_threshold
    if criterion is None:
        criterion = config.h_sem.cluster_criterion
    if criterion not in ("label", "prob"):
        raise ValueError(f"cluster_criterion must be 'label' or 'prob', got {criterion!r}")

    responses = list(responses)
    n = len(responses)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Perf: collapse exact-duplicate strings before the O(u^2) NLI pass. NLI(x, x)
    # is always entailment, so identical responses share a cluster trivially —
    # cluster only the UNIQUE strings (a large saving when answers repeat, e.g. a
    # consistent model) and map every response back to its representative. This
    # does not change the result, only the number of DeBERTa calls.
    uniq_index: dict[str, int] = {}
    items: list[str] = []
    for r in responses:
        if r not in uniq_index:
            uniq_index[r] = len(items)
            items.append(r)
    u = len(items)
    if u == 1:
        return [0] * n

    # Build all (i, j) pairs with i < j over the UNIQUE strings; both directions
    # in one batch.
    forward_pairs: list[tuple[int, int]] = []
    for i in range(u):
        for j in range(i + 1, u):
            forward_pairs.append((i, j))

    premises = [items[i] for i, _ in forward_pairs] + [items[j] for _, j in forward_pairs]
    hypotheses = [items[j] for _, j in forward_pairs] + [items[i] for i, _ in forward_pairs]

    # Single heavy/mockable seam: full 3-class softmax per directed pair + the
    # entailment column index. Tests patch `_nli_prob_vectors` to exercise the
    # criterion + union-find without loading DeBERTa.
    prob_vecs, entail_idx = _nli_prob_vectors(
        premises, hypotheses, model_name=config.h_sem.cluster_nli_model
    )

    m = len(forward_pairs)
    fwd_vecs = prob_vecs[:m]
    bwd_vecs = prob_vecs[m:]

    # Union-find over the unique strings; merge i, j iff both directions entail.
    parent = list(range(u))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), fwd, bwd in zip(forward_pairs, fwd_vecs, bwd_vecs, strict=True):
        if criterion == "label":
            merge = int(np.argmax(fwd)) == entail_idx and int(np.argmax(bwd)) == entail_idx
        else:  # "prob"
            merge = fwd[entail_idx] >= threshold and bwd[entail_idx] >= threshold
        if merge:
            union(i, j)

    # Contiguous 0..C-1 labels for the unique strings, then expand to all responses.
    root_to_label: dict[int, int] = {}
    item_label: list[int] = []
    for i in range(u):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = len(root_to_label)
        item_label.append(root_to_label[r])
    return [item_label[uniq_index[r]] for r in responses]


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
    criterion: str | None = None,
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

    pooled_assignment = cluster_responses(
        flat, config=config, threshold=threshold, criterion=criterion
    )

    out: dict[int, list[int]] = {}
    for idx, start, end in ranges:
        out[idx] = pooled_assignment[start:end]
    return out
