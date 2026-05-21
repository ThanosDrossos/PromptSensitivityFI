"""Glue layer: assemble the 11-scalar MetricTuple from precomputed inputs.

Pure function. Sprint 5 will write the pipeline that actually CALLS the
gateway to collect responses + scores + embeddings; this module only does
the math.

Caller responsibility — collect for one (q, ladder, level, model) cell:

  scores:                 list[float]                 # F(x) per paraphrase, length N
  cluster_assignments:    dict[paraphrase_idx, list[cluster_id]]
                                                       # per paraphrase, length k
  response_embeddings:    dict[paraphrase_idx, ndarray]
                                                       # (k, D) external-encoder embeddings
  prompt_embeddings:      ndarray                      # (N, D) external-encoder embeddings of prompts
  posix_log_p:            ndarray | None               # (N, N) log P(y_j|x_i); None for GPT-4o
  posix_lengths:          ndarray | None               # (N,) token counts of y_j
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errica import s_tau_freeform, s_tau_summary, tvd_consistency
from .ess_in import ess_in as ess_in_fn
from .fi_in import aufi_in_from_scores
from .fi_out import estimate_a_q, fi_out, fi_out_summary
from .h_sem import entropy_from_assignment
from .posix import posix as posix_fn
from .rho_u import rho_u as rho_u_fn
from .schemas import LadderType, MetricTuple
from .spread import spread as spread_fn
from .variation_ratio import variation_ratio as vr_fn


def _h_sem_mean_var(
    cluster_assignments: Mapping[int, Sequence[int]],
) -> tuple[float, float, int]:
    """Return (mean H_sem across prompts, variance, |A_q|)."""
    if not cluster_assignments:
        return 0.0, 0.0, 1
    a_q = estimate_a_q(cluster_assignments)
    h_values = [entropy_from_assignment(a) for a in cluster_assignments.values()]
    mean = float(np.mean(h_values))
    var = float(np.var(h_values, ddof=0))
    return mean, var, a_q


def build_metric_tuple(
    *,
    question_id: str,
    ladder_type: LadderType,
    level: int,
    model_key: str,
    scores: Sequence[float],
    cluster_assignments: Mapping[int, Sequence[int]],
    prompt_embeddings: np.ndarray,
    response_embeddings: Mapping[int, np.ndarray],
    posix_log_p: np.ndarray | None = None,
    posix_lengths: np.ndarray | None = None,
    encoder_label: str = "external_mpnet",
) -> MetricTuple:
    """Compute the 11-scalar MetricTuple for one (q, ladder, level, model) cell.

    All metrics that can be computed are; the ones whose inputs are missing
    (e.g. POSIX when posix_log_p is None) come back as None on the tuple.
    """
    scores = list(scores)
    n_paraphrases = len(scores)
    if n_paraphrases == 0:
        # Degenerate: no data. Return a tuple of Nones.
        return MetricTuple(
            question_id=question_id,
            ladder_type=ladder_type,
            level=level,
            model_key=model_key,
            n_paraphrases=0,
            n_samples_per_prompt=0,
            encoder_label=encoder_label,
        )

    # H_sem mean / var / |A_q|.
    h_mean, h_var, a_q = _h_sem_mean_var(cluster_assignments)

    # FI_out + S_τ per prompt (uses the same cluster assignments).
    fi_out_per = fi_out(cluster_assignments, a_q_size=a_q)
    fi_out_mean, _ = fi_out_summary(fi_out_per)

    s_tau_per = {
        idx: s_tau_freeform(assign, a_q) for idx, assign in cluster_assignments.items()
    }
    s_tau_mean = s_tau_summary(s_tau_per)

    consistency = tvd_consistency(cluster_assignments)

    spread_val = spread_fn(scores) if scores else None
    # variation_ratio uses the modal CLUSTER across paraphrases (one rep per
    # paraphrase): we pick the modal cluster of each paraphrase's k samples,
    # then compute variation across paraphrases.
    if cluster_assignments:
        from collections import Counter
        modes_per_prompt = []
        for assign in cluster_assignments.values():
            counts = Counter(assign)
            modes_per_prompt.append(counts.most_common(1)[0][0])
        var_ratio = vr_fn(modes_per_prompt)
    else:
        var_ratio = None

    # AUFI_in over k in [0, 1].
    aufi = aufi_in_from_scores(scores)

    # Mean F-score (raw accuracy). Useful for plots; AUFI_in is the integral
    # so the raw rate is recoverable but inconvenient.
    f_mean_val: float | None = float(np.mean(scores)) if scores else None

    # ESS_in on the (N, D) prompt-embedding matrix.
    ess_val = ess_in_fn(prompt_embeddings) if prompt_embeddings.size else None

    # ρ_u on the per-paraphrase response embeddings.
    if response_embeddings:
        rho_payload = rho_u_fn(response_embeddings)
        rho_val: float | None = rho_payload["rho_u"]
        # Sample count for the audit field.
        n_samples = next(iter(response_embeddings.values())).shape[0]
    else:
        rho_val = None
        n_samples = 0

    # POSIX — only when the caller supplies a matrix (i.e. echo-capable models).
    posix_val: float | None
    if posix_log_p is not None and posix_lengths is not None:
        posix_val = posix_fn(posix_log_p, posix_lengths)
    else:
        posix_val = None

    return MetricTuple(
        question_id=question_id,
        ladder_type=ladder_type,
        level=level,
        model_key=model_key,
        f_mean=f_mean_val,
        aufi_in=aufi,
        fi_out_mean=fi_out_mean,
        s_tau_mean=s_tau_mean,
        consistency_mean=consistency,
        spread=spread_val,
        variation_ratio=var_ratio,
        posix_psi=posix_val,
        ess_in=ess_val,
        rho_u=rho_val,
        h_sem_mean=h_mean,
        h_sem_var=h_var,
        n_paraphrases=n_paraphrases,
        n_samples_per_prompt=n_samples,
        encoder_label=encoder_label,
    )
