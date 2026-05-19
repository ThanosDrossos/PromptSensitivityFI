"""FI_out — Functional Information in OUTPUT space. Section_7 §7.4 Candidate B.

    FI_out(x) = log2|A_q| - H_sem(Y|X=x)

|A_q| (the semantic-answer-space size for query class q) is estimated from
the union of cluster IDs observed across all paraphrases of q at one
(ladder, level). This is the rarefaction-style estimator §7.8 P3 warns about;
we mitigate by exposing the count for the writeup to comment on.

CONTRACT: `cluster_assignments` MUST use POOLED cluster IDs — i.e. cluster
ID `c` means the same semantic cluster across every paraphrase in the cell.
Use `h_sem.cluster_responses_pooled` to obtain such IDs. Calling this with
independently-clustered IDs (one cluster_responses call per paraphrase) will
over-count |A_q| and silently inflate FI_out.

Identity check (Section_7 §7.4.1):  FI_out + H_sem = log2|A_q|.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .h_sem import entropy_from_assignment, n_unique_clusters


def fi_out_per_prompt(
    cluster_assignment: Sequence[int],
    a_q_size: int,
) -> float:
    """Restrictiveness of one prompt: log2|A_q| - H_sem(Y|X=x).

    Clamped at 0 from below per Section_7 §7.4.1 (negative values arise only
    under |A_q| under-estimation and are reported as 0).
    """
    if a_q_size <= 0:
        return 0.0
    h = entropy_from_assignment(cluster_assignment)
    val = math.log2(a_q_size) - h
    return max(0.0, val)


def fi_out(
    cluster_assignments: Mapping[int, Sequence[int]] | Sequence[Sequence[int]],
    *,
    a_q_size: int | None = None,
) -> dict[int, float]:
    """Compute FI_out for every prompt in U_q,l.

    `cluster_assignments` is a mapping `paraphrase_idx -> list[cluster_id]`
    (one cluster_id per sampled response). If `a_q_size` is omitted we
    estimate it from the union of cluster IDs across all paraphrases.
    """
    if isinstance(cluster_assignments, Mapping):
        items = list(cluster_assignments.items())
    else:
        items = list(enumerate(cluster_assignments))

    if a_q_size is None:
        # |A_q| via union of cluster IDs across all paraphrases.
        all_ids: set[int] = set()
        for _, assign in items:
            all_ids.update(assign)
        a_q_size = max(1, len(all_ids))

    out: dict[int, float] = {}
    for idx, assign in items:
        out[idx] = fi_out_per_prompt(assign, a_q_size)
    return out


def fi_out_summary(per_prompt: Mapping[int, float]) -> tuple[float, float]:
    """Return (mean, variance) of FI_out across paraphrases for the §3 reporting tuple."""
    if not per_prompt:
        return 0.0, 0.0
    vals = np.asarray(list(per_prompt.values()), dtype=float)
    return float(vals.mean()), float(vals.var(ddof=0))


def estimate_a_q(cluster_assignments: Mapping[int, Sequence[int]]) -> int:
    """|A_q| via union of unique cluster IDs across all paraphrases."""
    all_ids: set[int] = set()
    for assign in cluster_assignments.values():
        all_ids.update(assign)
    return max(1, len(all_ids))
