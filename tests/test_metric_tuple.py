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


def test_p1_3_bootstrap_ci_brackets_curve():
    t = _build()
    assert t.fi_in_ci_lower is not None and t.fi_in_ci_upper is not None
    assert len(t.fi_in_ci_lower) == len(t.fi_in_curve_vals) == len(t.fi_in_ci_upper)
    for lo, v, hi in zip(t.fi_in_ci_lower, t.fi_in_curve_vals, t.fi_in_ci_upper):
        assert lo - 1e-9 <= v <= hi + 1e-9


def test_specificity_fields_default_none_and_pass_through():
    """AmbigQA pivot §7: fi_spec/spec_level/m_valid/m0/target_idx default None
    (old parquets keep loading) and pass through when the driver supplies them."""
    t = _build()
    assert t.fi_spec is None and t.spec_level is None
    assert t.m_valid is None and t.m0 is None and t.target_idx is None

    t2 = _build(fi_spec=2.0, spec_level=1, m_valid=1, m0=4, target_idx=2)
    assert t2.fi_spec == 2.0 and t2.spec_level == 1
    assert t2.m_valid == 1 and t2.m0 == 4 and t2.target_idx == 2

    # degenerate (no scores) still carries the dataset-side fields
    t3 = build_metric_tuple(
        question_id="q", ladder_type="random", level=0, model_key="m",
        scores=[], cluster_assignments={},
        prompt_embeddings=np.zeros((0, 8)), response_embeddings={},
        fi_spec=1.0, spec_level=0, m_valid=2, m0=2, target_idx=0,
    )
    assert t3.fi_spec == 1.0 and t3.spec_level == 0 and t3.m0 == 2
