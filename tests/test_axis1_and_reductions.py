"""R4/R4b: the metric-reduction checks and the axis-1 helpers.

What matters: (a) the identity checks would actually CATCH a broken identity,
(b) the censoring/islands statistics behave as designed, (c) the Williams test
matches a reference case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prompt_sensitivity.scripts.axis1_graded_curve import (
    censor_fraction,
    graded_curve_vals,
    islands_p_value,
    max_gap,
)
from prompt_sensitivity.scripts.metric_reductions import (
    verify_identities,
    williams_test,
)

K = 10


# ------------------------------------------------------------- axis-1 helpers


def test_censor_fraction_bounds():
    # every paraphrase perfect -> every threshold reachable -> 0
    assert censor_fraction([1.0] * 10) == 0.0
    # all zero -> only k=0 is reachable -> 20/21 censored
    assert censor_fraction([0.0] * 10) == pytest.approx(20 / 21)
    # half the range reachable
    frac = censor_fraction([0.5] * 10)
    assert 0.4 < frac < 0.55


def test_graded_curve_is_monotone_and_capped():
    vals = graded_curve_vals([0.1, 0.3, 0.5, 0.7, 0.9], cap=np.log2(6))
    assert np.all(np.diff(vals) >= -1e-12), "FI_in(k) must be non-decreasing in k"
    assert vals.max() <= np.log2(6) + 1e-12
    assert vals[0] == 0.0, "FI_in(0) = 0 by construction"


def test_max_gap_detects_islands():
    # two clumps -> big gap; uniform spread -> small gaps
    assert max_gap([0.0, 0.1, 0.9, 1.0]) == pytest.approx(0.8)
    assert max_gap([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.25)
    assert max_gap([0.5]) == 0.0


def test_islands_p_small_for_real_islands_large_for_binomial_noise():
    rng = np.random.default_rng(0)
    # genuine islands: half the paraphrases at 0, half at 1 — impossible under
    # a shared-rate binomial with p-bar = 0.5 without extreme luck
    p_islands = islands_p_value([0.0] * 5 + [1.0] * 5, K, rng=rng)
    assert p_islands < 0.01
    # pure binomial noise around one rate -> should NOT look like islands
    ps = []
    for _ in range(20):
        s = rng.binomial(K, 0.5, size=10) / K
        ps.append(islands_p_value(list(s), K, rng=rng))
    assert np.mean([p < 0.05 for p in ps]) < 0.3, "null cells must rarely fire"


def test_islands_p_degenerate_cell_is_one():
    rng = np.random.default_rng(1)
    assert islands_p_value([0.0] * 10, K, rng=rng) == pytest.approx(1.0)


# ------------------------------------------------------------- reductions


def _frame(n=40, seed=0):
    """Synthetic frame satisfying the identities exactly (as the pipeline does)."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.05, 2.0, size=n)
    a = rng.integers(2, 9, size=n).astype(float)
    m0 = rng.integers(2, 5, size=n).astype(float)
    return pd.DataFrame({
        "h_sem_mean": h,
        "a_q": a,
        "s_tau_mean": h / np.log2(a),
        "m0": m0,
        "fi_out_fixed": np.log2(m0) - h,
        "h_sem_var": rng.uniform(0, 0.5, size=n),
    }).assign(fi_out_var=lambda d: d.h_sem_var)


def test_verify_identities_passes_on_conforming_data():
    out = verify_identities(_frame())
    assert out["s_tau_max_abs_err"] < 1e-12
    assert out["fi_out_fixed_max_abs_err"] < 1e-12
    assert out["fi_out_var_max_abs_err"] < 1e-12


def test_verify_identities_catches_a_broken_identity():
    """The check must have teeth: perturb one relation and the error must show."""
    df = _frame()
    df.loc[3, "s_tau_mean"] += 0.05
    out = verify_identities(df)
    assert out["s_tau_max_abs_err"] > 0.04


def test_verify_identities_respects_the_degeneracy_rule():
    df = _frame()
    df.loc[0, "a_q"] = 1.0          # degenerate cell: S_tau undefined
    df.loc[0, "s_tau_mean"] = 0.0   # pipeline emits 0 there
    out = verify_identities(df)
    assert out["s_tau_n_checked"] == len(df) - 1
    assert out["s_tau_degenerate_frac"] == pytest.approx(1 / len(df))
    assert out["s_tau_max_abs_err"] < 1e-12, "degenerate cell must be excluded, not scored"


def test_williams_test_reference_behaviour():
    # equal correlations -> t == 0, p == 1
    t, p = williams_test(0.5, 0.5, 0.3, 100)
    assert t == pytest.approx(0.0)
    assert p == pytest.approx(1.0)
    # a large real difference at decent n -> significant
    t2, p2 = williams_test(0.8, 0.1, 0.2, 100)
    assert abs(t2) > 5 and p2 < 1e-4
    # symmetric: swapping r12/r13 flips the sign
    t3, _ = williams_test(0.1, 0.8, 0.2, 100)
    assert t3 == pytest.approx(-t2)
