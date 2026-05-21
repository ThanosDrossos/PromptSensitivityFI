"""Orchestrator + Sprint-4 smoke test."""

from __future__ import annotations

import numpy as np

from prompt_sensitivity.metrics import build_metric_tuple
from prompt_sensitivity.metrics.schemas import MetricTuple


def test_build_tuple_sets_f_mean_correctly():
    """f_mean must equal sum(scores) / len(scores) — used by plot scripts."""
    import numpy as np

    rng = np.random.default_rng(0)
    scores = [1.0, 1.0, 0.0, 0.0, 1.0]   # 3/5 = 0.6
    cluster_assignments = {i: [0, 0, 0, 1] for i in range(5)}
    prompt_emb = rng.normal(size=(5, 8))
    response_emb = {i: rng.normal(size=(4, 8)) for i in range(5)}

    tup = build_metric_tuple(
        question_id="t_f_mean",
        ladder_type="random",
        level=4,
        model_key="llama_3_1_8b",
        scores=scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_emb,
        response_embeddings=response_emb,
        posix_log_p=None,
        posix_lengths=None,
    )
    assert tup.f_mean == 0.6


def test_build_tuple_returns_metric_tuple_with_all_scalars():
    rng = np.random.default_rng(0)
    n = 5
    k = 4
    scores = [1.0, 1.0, 0.0, 0.0, 1.0]
    cluster_assignments = {i: [0, 0, 0, 1] for i in range(n)}
    prompt_emb = rng.normal(size=(n, 8))
    response_emb = {i: rng.normal(size=(k, 8)) for i in range(n)}

    tup = build_metric_tuple(
        question_id="t1",
        ladder_type="random",
        level=4,
        model_key="llama_3_1_8b",
        scores=scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_emb,
        response_embeddings=response_emb,
        posix_log_p=None,
        posix_lengths=None,
    )
    assert isinstance(tup, MetricTuple)
    # All non-POSIX scalars populated.
    assert tup.aufi_in is not None
    assert tup.fi_out_mean is not None
    assert tup.s_tau_mean is not None
    assert tup.consistency_mean is not None
    assert tup.spread is not None
    assert tup.variation_ratio is not None
    assert tup.ess_in is not None
    assert tup.rho_u is not None
    assert tup.h_sem_mean is not None
    assert tup.h_sem_var is not None
    # POSIX null because no log-prob matrix supplied (mimics GPT-4o path).
    assert tup.posix_psi is None


def test_build_tuple_empty_scores_returns_nones():
    tup = build_metric_tuple(
        question_id="empty",
        ladder_type="random",
        level=0,
        model_key="llama_3_1_8b",
        scores=[],
        cluster_assignments={},
        prompt_embeddings=np.zeros((0, 8)),
        response_embeddings={},
    )
    assert tup.aufi_in is None
    assert tup.h_sem_mean is None
    assert tup.n_paraphrases == 0


def test_smoke_runs_end_to_end():
    """Imports the smoke script and runs its main() — must return 0."""
    from prompt_sensitivity.scripts.smoke_metrics import main

    assert main() == 0
