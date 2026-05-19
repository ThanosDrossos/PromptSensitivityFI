"""ρ_u (Cox 2025) — within/across decomposition + edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from prompt_sensitivity.metrics.rho_u import rho_u


def test_rho_u_zero_when_all_paraphrases_share_distribution():
    """Within-paraphrase noise only; no across-paraphrase mean shift -> ρ_u = 0."""
    rng = np.random.default_rng(42)
    paraphrases = {
        i: rng.normal(loc=0.0, scale=1.0, size=(40, 5)) for i in range(4)
    }
    result = rho_u(paraphrases)
    assert result["rho_u"] < 0.2, f"expected ρ_u close to 0, got {result['rho_u']}"


def test_rho_u_one_when_each_paraphrase_is_a_constant():
    """Per-paraphrase clusters at fixed points (no within-noise) -> ρ_u ~ 1."""
    paraphrases = {
        0: np.tile([1.0, 0.0], (10, 1)),
        1: np.tile([0.0, 1.0], (10, 1)),
        2: np.tile([-1.0, 0.0], (10, 1)),
    }
    result = rho_u(paraphrases)
    assert result["rho_u"] == pytest.approx(1.0)
    assert result["U_a"] == pytest.approx(0.0)


def test_rho_u_decomposition_sums_within_eps():
    """U_t = U_a + U_e to floating-point precision."""
    rng = np.random.default_rng(7)
    paraphrases = {
        i: rng.normal(loc=i, scale=1.0, size=(30, 8)) for i in range(5)
    }
    result = rho_u(paraphrases)
    assert result["U_t"] == pytest.approx(result["U_a"] + result["U_e"], rel=1e-6)


def test_rho_u_empty_input_returns_zeros():
    result = rho_u({})
    assert result == {"U_t": 0.0, "U_a": 0.0, "U_e": 0.0, "rho_u": 0.0}


def test_rho_u_dimension_mismatch_raises():
    paraphrases = {
        0: np.zeros((3, 4)),
        1: np.zeros((3, 5)),
    }
    with pytest.raises(ValueError):
        rho_u(paraphrases)
