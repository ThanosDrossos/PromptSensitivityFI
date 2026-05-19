"""ESS_in: trace(Cov) of input-prompt embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from prompt_sensitivity.metrics.ess_in import ess_in


def test_identical_embeddings_zero_dispersion():
    emb = np.ones((5, 8))
    assert ess_in(emb) == 0.0


def test_diagonal_unit_variance_equals_dimension():
    """Each feature has variance 1 -> trace(Cov) = D."""
    rng = np.random.default_rng(123)
    # Construct a (1000, 4) matrix with per-feature variance ~1 (population),
    # then sample variance (ddof=1) is ~1 too at large N.
    emb = rng.standard_normal(size=(2000, 4))
    val = ess_in(emb)
    assert 3.5 < val < 4.5


def test_too_few_rows_returns_zero():
    """1-row embedding cannot have nonzero variance."""
    assert ess_in(np.array([[1.0, 2.0, 3.0]])) == 0.0


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        ess_in(np.array([1.0, 2.0, 3.0]))


def test_known_2d_case():
    """Hand-computed: rows [(0,0), (2,0)] -> var(x)=2, var(y)=0, trace=2."""
    emb = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert ess_in(emb) == pytest.approx(2.0)
