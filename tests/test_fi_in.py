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
