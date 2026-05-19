"""Sprint 4 gate — orchestrator smoke test with hand-built inputs.

Constructs a minimal 5-paraphrase × 4-sample-per-prompt synthetic cell,
runs the orchestrator, and prints the resulting MetricTuple. Asserts each
field is within a plausible range so a regression in any one metric is
caught here without needing a live gateway run.

No network calls. No DeBERTa load. Pure math sanity check.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from loguru import logger

from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple


def main() -> int:
    configure_logging("smoke_metrics")

    # 5 paraphrases, F-scores: 3/5 correct (0.6 acceptance at k=0.5).
    scores = [1.0, 1.0, 1.0, 0.0, 0.0]
    n_paraphrases = len(scores)

    # 4 samples per prompt; for paraphrases 0,1,2 (correct) most samples land
    # in cluster 0 (the "right" answer cluster); for paraphrases 3,4
    # (incorrect) samples scatter across clusters 1, 2, 3.
    cluster_assignments = {
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 1],   # one drift to a different cluster
        2: [0, 0, 1, 1],   # split across two clusters
        3: [2, 2, 2, 3],   # wrong-answer-prone
        4: [3, 3, 2, 2],   # different wrong cluster
    }
    n_samples = 4
    # |A_q| (union of cluster ids across all paraphrases) = {0, 1, 2, 3} -> 4

    # Synthetic prompt embeddings (N=5, D=8). Each paraphrase gets a slightly
    # different point; rng=42 for reproducibility.
    rng = np.random.default_rng(42)
    prompt_embeddings = rng.normal(loc=0.0, scale=1.0, size=(n_paraphrases, 8))

    # Synthetic response embeddings (k=4, D=8) per paraphrase. We inject
    # mean-shift across paraphrases so rho_u > 0 (some prompt-induced variance).
    response_embeddings = {
        idx: rng.normal(loc=idx * 0.3, scale=1.0, size=(n_samples, 8))
        for idx in range(n_paraphrases)
    }

    # POSIX matrix — simulate a 5x5 log-prob matrix where the diagonal
    # ("natural" prompt) is the highest, off-diagonals slightly lower.
    posix_log_p = -np.abs(rng.normal(loc=10.0, scale=2.0, size=(n_paraphrases, n_paraphrases)))
    np.fill_diagonal(posix_log_p, posix_log_p.diagonal() + 1.0)  # natural higher
    posix_lengths = np.array([6, 5, 7, 4, 8], dtype=float)

    tup = build_metric_tuple(
        question_id="smoke_q1",
        ladder_type="random",
        level=4,
        model_key="llama_3_1_8b",
        scores=scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_embeddings,
        response_embeddings=response_embeddings,
        posix_log_p=posix_log_p,
        posix_lengths=posix_lengths,
        encoder_label="external_mpnet",
    )

    print()
    print("=" * 70)
    print("SAMPLE METRIC TUPLE — Sprint 4 gate output")
    print("=" * 70)
    payload = tup.model_dump()
    print(json.dumps(payload, indent=2))

    # --- plausibility assertions ----------------------------------------
    # FI_in: 3 of 5 paraphrases pass k>=0.5, so FI_in(k=0.5) = -log2(3/5) ~ 0.74 bits.
    # AUFI_in (clamped) should be > 0 and < log2(N+1) = log2(6) ~ 2.58.
    assert tup.aufi_in is not None
    assert 0.0 < tup.aufi_in < 2.6, f"AUFI_in out of plausible range: {tup.aufi_in}"
    # FI_out >= 0 by construction (clamped).
    assert tup.fi_out_mean is not None
    assert tup.fi_out_mean >= 0.0
    # S_tau in [0, 1].
    assert tup.s_tau_mean is not None and 0.0 <= tup.s_tau_mean <= 1.0
    # 1 - TVD in [0, 1].
    assert tup.consistency_mean is not None and 0.0 <= tup.consistency_mean <= 1.0
    # Variation ratio in [0, 1].
    assert tup.variation_ratio is not None and 0.0 <= tup.variation_ratio <= 1.0
    # Spread for binary F-scores is exactly 1.0.
    assert tup.spread == 1.0
    # ρ_u in [0, 1].
    assert tup.rho_u is not None and 0.0 <= tup.rho_u <= 1.0
    # POSIX > 0 (off-diagonal logprobs differ from the diagonal).
    assert tup.posix_psi is not None and tup.posix_psi > 0
    # ESS_in > 0 (random embeddings have nonzero variance).
    assert tup.ess_in is not None and tup.ess_in > 0
    # H_sem fields populated.
    assert tup.h_sem_mean is not None and tup.h_sem_var is not None

    logger.info("smoke test passed — all 11 scalars in plausible range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
