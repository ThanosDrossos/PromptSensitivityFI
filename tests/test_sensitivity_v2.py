"""Sensitivity v2 (M1 rho_F + M2 fi_premium): formula correctness against
hand/reference computations, edge semantics, and backfill parity with the
driver path."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prompt_sensitivity.metrics.sensitivity_v2 import (
    compute_row_metrics,
    fi_premium,
    rho_f,
)
from prompt_sensitivity.scripts.backfill_sensitivity_v2 import backfill_frame


# ---------------------------------------------------------------- M1: rho_F


def _icc_reference(p_hats, k):
    """Independent reference: build the full N*k binary outcome matrix and run
    textbook one-way ANOVA. Valid because k*p_hat is integral in our cells."""
    groups = []
    for p in p_hats:
        ones = round(p * k)
        groups.append([1.0] * ones + [0.0] * (k - ones))
    y = np.array(groups)                       # (N, k)
    n, kk = y.shape
    grand = y.mean()
    ssb = kk * ((y.mean(axis=1) - grand) ** 2).sum()
    ssw = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum()
    msb, msw = ssb / (n - 1), ssw / (n * (kk - 1))
    return (msb - msw) / (msb + (kk - 1) * msw)


def test_rho_f_matches_full_anova_reference():
    cases = [
        [0.0, 0.0, 1.0, 1.0],          # perfectly phrasing-determined
        [0.5, 0.5, 0.5, 0.5],          # pure sampling noise
        [0.1, 0.3, 0.9, 0.6, 0.2],
        [0.2, 0.2, 0.2, 0.8],
    ]
    for p in cases:
        expect = max(0.0, min(1.0, _icc_reference(p, 10)))
        assert rho_f(p, 10) == pytest.approx(expect, abs=1e-12), p


def test_rho_f_semantics_and_edges():
    # success fully determined by paraphrase choice -> ICC ~ 1
    assert rho_f([0.0, 0.0, 1.0, 1.0], 10) > 0.95
    # identical rates -> all variability is decoding noise -> clamps to 0
    assert rho_f([0.5] * 10, 10) == 0.0
    # all-0 / all-1: NO variance anywhere -> NaN (unmeasurable, not zero)
    assert math.isnan(rho_f([0.0] * 10, 10))
    assert math.isnan(rho_f([1.0] * 10, 10))
    # degenerate inputs
    assert math.isnan(rho_f([0.7], 10))
    with pytest.raises(ValueError):
        rho_f([0.5, 0.6], 1)


# ---------------------------------------------------------------- M2: fi_premium


def test_fi_premium_binary_F_is_identically_zero():
    # the metric isolates GRADED shape info: any 0/1 vector gives 0
    for v in ([1, 0, 1, 0], [1] * 10, [0] * 10, [0, 0, 1]):
        assert fi_premium([float(x) for x in v]) == 0.0


def test_fi_premium_counts_usable_vs_perfect():
    # 6 of 10 usable (>=0.5), 2 of 10 perfect -> log2(6/2)
    scores = [1.0, 1.0, 0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0]
    assert fi_premium(scores) == pytest.approx(math.log2(6 / 2))
    # floor cell: nothing usable -> both thresholds at cap -> 0 (no shape info)
    assert fi_premium([0.1, 0.2, 0.3]) == 0.0
    # usable but never perfect: premium = cap - FI(0.5)
    s = [0.6, 0.7, 0.5, 0.9]
    assert fi_premium(s) == pytest.approx(math.log2(5) - (-math.log2(4 / 4)))
    # float-guard: means like 3/10 must not fall under the 0.5 test spuriously
    assert fi_premium([0.5, 0.5, 1.0, 0.0]) == pytest.approx(math.log2(3 / 1))


# ---------------------------------------------------------------- shared path


def test_backfill_uses_driver_compute_path_and_is_idempotent(tmp_path):
    df = pd.DataFrame([
        {"f_graded_per_paraphrase": [0.0, 0.0, 1.0, 1.0], "n_samples_per_prompt": 10},
        {"f_graded_per_paraphrase": None, "n_samples_per_prompt": 10},
        {"f_graded_per_paraphrase": [1.0, 1.0], "n_samples_per_prompt": 10},
    ])
    out = backfill_frame(df)
    d0 = compute_row_metrics([0.0, 0.0, 1.0, 1.0], 10)
    assert out.rho_f.iloc[0] == pytest.approx(d0["rho_f"])
    assert out.fi_premium.iloc[0] == pytest.approx(d0["fi_premium"])
    assert math.isnan(out.rho_f.iloc[1]) and math.isnan(out.fi_premium.iloc[1])
    assert math.isnan(out.rho_f.iloc[2])          # all-perfect -> unmeasurable
    out2 = backfill_frame(out)                     # idempotent overwrite
    pd.testing.assert_series_equal(out.rho_f, out2.rho_f)
