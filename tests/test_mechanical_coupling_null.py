"""Tests for scripts/mechanical_coupling_null.py.

The load-bearing property: the vectorised MoM re-implementation must be
numerically identical to metrics.sensitivity_v2.rho_f (including the NaN rule on
zero-variance cells and the [0,1] clip), otherwise the null is simulated with a
different estimator than the observed numbers and the comparison is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from prompt_sensitivity.metrics.sensitivity_v2 import rho_f as rho_f_scalar
from prompt_sensitivity.scripts.mechanical_coupling_null import (
    rho_f_mom_vectorised,
    simulate_null,
)


def test_vectorised_matches_scalar_on_random_cells() -> None:
    rng = np.random.default_rng(0)
    k = 10
    for _ in range(200):
        n = int(rng.integers(2, 12))
        y = rng.binomial(k, rng.uniform(0.0, 1.0), size=n)
        expected = rho_f_scalar((y / k).tolist(), k)
        got = float(rho_f_mom_vectorised(y[None, :], k)[0])
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-12)


def test_vectorised_nan_on_degenerate_cells() -> None:
    k = 10
    all_wrong = np.zeros((1, 10), dtype=int)
    all_right = np.full((1, 10), k, dtype=int)
    assert np.isnan(rho_f_mom_vectorised(all_wrong, k)[0])
    assert np.isnan(rho_f_mom_vectorised(all_right, k)[0])


def test_vectorised_rejects_k_below_2() -> None:
    with pytest.raises(ValueError):
        rho_f_mom_vectorised(np.zeros((1, 5), dtype=int), 1)


def test_null_centred_near_zero_when_mu_constant() -> None:
    """With identical mu in every cell there is no prevalence gradient, so the
    mechanical coupling has nothing to attach to: the null Spearman must be
    centred near zero."""
    rng = np.random.default_rng(1)
    mu = np.full(120, 0.5)
    sim = simulate_null(mu, true_rho=0.2, n_paraphrases=10, k=10, n_sims=200, rng=rng)
    med = np.nanmedian(sim["r_null"])
    assert abs(med) < 0.1


def test_null_produces_coupling_under_prevalence_gradient() -> None:
    """With cell means spread over [0.05, 0.95] and constant true rho, the
    estimator alone must produce a non-degenerate null band — the effect the
    script exists to quantify. We assert the band has width, not its sign."""
    rng = np.random.default_rng(2)
    mu = np.linspace(0.05, 0.95, 120)
    sim = simulate_null(mu, true_rho=0.2, n_paraphrases=10, k=10, n_sims=200, rng=rng)
    r = sim["r_null"][np.isfinite(sim["r_null"])]
    assert r.size > 150
    assert np.quantile(r, 0.95) - np.quantile(r, 0.05) > 0.02
    # coverage must be below 1 at the extremes: degenerate cells exist under the null
    assert np.nanmean(sim["coverage_null"]) < 1.0
