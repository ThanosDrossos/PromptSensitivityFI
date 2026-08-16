"""FI_in tests. Brief edge cases: uniform F -> 0; single magic phrasing -> log2(N)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prompt_sensitivity.metrics.fi_in import (
    aufi_in,
    aufi_in_from_scores,
    fi_in,
    fi_in_bootstrap,
    fi_in_curve,
)

# --- brief edge cases -----------------------------------------------------


def test_uniform_F_gives_zero_FI_in():
    """All paraphrases achieve F >= k -> FI_in = 0."""
    scores = [1.0] * 30
    assert fi_in(scores, k=0.5) == 0.0


def test_single_magic_phrasing_gives_log2_N():
    """Only one paraphrase achieves F >= k -> FI_in = log2(N)."""
    scores = [1.0] + [0.0] * 29
    assert fi_in(scores, k=0.5) == pytest.approx(math.log2(30))


def test_no_paraphrase_passes_gives_inf():
    scores = [0.0] * 5
    assert math.isinf(fi_in(scores, k=0.5))


def test_fi_in_curve_is_monotonic_in_k():
    """As k rises, fewer paraphrases pass, FI_in monotonically non-decreases."""
    scores = [0.2, 0.5, 0.7, 0.9, 1.0]
    curve = fi_in_curve(scores, ks=[0.1, 0.3, 0.6, 0.8, 1.0])
    vals = [curve[k] for k in sorted(curve.keys())]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12


def test_aufi_in_clamps_infinity():
    """AUFI_in must remain finite even when some k yields FI_in = inf."""
    scores = [0.1, 0.2]  # at k=1.0, no one passes -> inf
    curve = fi_in_curve(scores)
    result = aufi_in(curve, n=len(scores))
    assert math.isfinite(result)
    assert result > 0.0


def test_fi_in_bootstrap_returns_per_k_band():
    scores = [0.0, 0.0, 1.0, 1.0, 1.0]
    band = fi_in_bootstrap(scores, ks=[0.5, 1.0], n_iterations=100, seed=42)
    assert set(band.keys()) == {0.5, 1.0}
    for k, (lo, hi) in band.items():
        assert lo <= hi


def test_aufi_in_from_scores_helper_matches_step_by_step():
    scores = [0.0, 0.5, 1.0]
    curve = fi_in_curve(scores)
    direct = aufi_in(curve, n=3)
    via_helper = aufi_in_from_scores(scores)
    assert direct == pytest.approx(via_helper)


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        fi_in([], k=0.5)


def test_fi_in_grid_epsilon_exact_tenths_pass_linspace_thresholds():
    """Regression: F = 0.3 must pass the linspace threshold 0.30000000000000004.

    Graded F values are exact multiples of 1/10; np.linspace(0, 1, 21) contains
    0.30000000000000004, 0.6000000000000001 and 0.7000000000000001. Without the
    epsilon guard a paraphrase attaining a threshold exactly failed it at those
    grid points (fi_in was the only module comparing bare `>=`).
    """
    ks = np.linspace(0.0, 1.0, 21).tolist()
    for frac in (0.3, 0.6, 0.7):
        k = next(x for x in ks if abs(x - frac) < 1e-6)
        assert k != frac  # the grid value really is off by float error
        assert fi_in([frac], k) == 0.0  # the single paraphrase attains k
    band = fi_in_bootstrap([0.3, 0.3], ks=[0.30000000000000004], n_iterations=10, seed=0)
    assert band[0.30000000000000004] == (0.0, 0.0)
