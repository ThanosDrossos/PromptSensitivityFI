"""build_metric_tuple field-population tests (P0-3 / P0-4 / P1-3).

Pure-math: synthetic scores + cluster assignments + embeddings, no model load.
"""

from __future__ import annotations

import math

import numpy as np

from prompt_sensitivity.metrics.orchestrator import build_metric_tuple


def _cell(n=6, k=2, d=8, seed=0):
    rng = np.random.default_rng(seed)
    scores = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5][:n]
    cluster_assignments = {i: [int(rng.integers(0, 3)) for _ in range(k)] for i in range(n)}
    prompt_embeddings = rng.standard_normal((n, d)).astype(np.float32)
    response_embeddings = {i: rng.standard_normal((k, d)).astype(np.float32) for i in range(n)}
    return scores, cluster_assignments, prompt_embeddings, response_embeddings


def _build(**over):
    scores, ca, pe, re = _cell()
    kw = dict(
        question_id="q", ladder_type="random", level=0, model_key="m",
        scores=scores, cluster_assignments=ca,
        prompt_embeddings=pe, response_embeddings=re,
    )
    kw.update(over)
    return build_metric_tuple(**kw)


def test_p0_3_fi_in_curve_persisted():
    t = _build()
    assert t.fi_in_curve_ks is not None and t.fi_in_curve_vals is not None
    assert len(t.fi_in_curve_ks) == 21
    assert min(t.fi_in_curve_ks) == 0.0 and max(t.fi_in_curve_ks) == 1.0
    cap = math.log2(6 + 1)
    assert len(t.fi_in_curve_vals) == 21
    assert all(math.isfinite(v) and 0.0 <= v <= cap + 1e-9 for v in t.fi_in_curve_vals)


def test_p0_4_fi_out_var_non_negative():
    t = _build()
    assert t.fi_out_var is not None and t.fi_out_var >= 0.0
